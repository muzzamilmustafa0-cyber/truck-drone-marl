"""Ablation study for GAT-MARL.

Re-trains the full model and four reduced variants from scratch under an identical
protocol (200 episodes, 3 seeds each) and logs per-episode training curves.

Variants:
  Full         complete model
  A1-NoGAT     graph-attention encoder replaced by mean pooling
  A2-FixedEnt  fixed entropy coefficient (no annealing)
  A3-NoCritic  no trained value baseline (pure REINFORCE)
  A4-NoMasks   stored action masks not applied in the policy update

All three seeds are reported individually so that seed variance is visible.
Outputs: ablation_results.csv (final metrics) and ablation_curves.csv (curves).
"""

import os, sys, math, time, random
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
CODE         = ROOT / "code"
DATA         = ROOT / "data"
# Solomon benchmark files live one level above project root
SOLOMON_DIR  = ROOT / "data" / "solomon"
RESULTS_DIR   = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CODE))

from truck_drone_env import TruckDroneEnv, load_solomon_instance
from gat_marl  import (
    GATMARLAgent, train_episode, evaluate_episode,
    GATEncoder, Adam
)
from neural_base  import relu, Linear, MLP, ActorNet, CriticNet, layer_norm

# ── Experiment configuration ────────────────────────────────────────────────
ABLATION_INSTANCES = ["C101", "C201", "R101", "RC101"]
N_EPISODES  = 200 # same budget as the main experiment
N_SEEDS     = 3 # all 3 seeds reported individually
EVAL_FREQ   = 5 # evaluate every N episodes for training curve
VERBOSE     = True

# Combinations to skip entirely (variant, instance).
# A1-NoGAT trains ~4x slower (no attention -> longer episodes). We already have
# 9 A1-NoGAT seeds across C101/C201/R101 — sufficient to characterize the
# no-GAT variant — so RC101 is skipped to save ~1.8h. All GAT-based variants
# (Full, A2, A3, A4) still run the full 4-instance set.
SKIP_COMBOS = {("A1-NoGAT", "RC101")}

OUTPUT_FINAL    = RESULTS_DIR / "ablation_results.csv"
OUTPUT_CURVES   = RESULTS_DIR / "ablation_curves.csv"
PARTIAL_FINAL   = RESULTS_DIR / "ablation_partial.csv"
PARTIAL_CURVES  = RESULTS_DIR / "ablation_curves_partial.csv"

# Seed assignment: each instance gets 3 seeds spaced 200 apart
# instance_offset ensures different base per instance to avoid seed aliasing
INSTANCE_OFFSET = {"C101": 0, "C201": 3, "R101": 1, "RC101": 2}

BKS_TABLE = {
    "C101":828.9,"C102":828.9,"C103":828.1,"C104":824.8,"C105":828.9,
    "C106":828.9,"C107":828.9,"C108":828.9,"C109":828.9,
    "C201":589.1,"C202":589.1,"C203":591.6,"C204":590.6,"C205":586.4,
    "C206":586.0,"C207":585.8,"C208":585.8,
    "R101":1645.9,"R102":1486.1,"R103":1292.7,"R104":1007.3,
    "R105":1377.1,"R106":1251.1,"R107":1104.7,"R108":960.9,
    "R109":1194.7,"R110":1118.6,"R111":1096.7,"R112":982.1,
    "R201":1252.4,"R202":1191.7,"R203":939.5,"R204":825.5,
    "R205":994.4,"R206":906.1,"R207":890.6,"R208":727.0,
    "R209":909.2,"R210":939.3,"R211":892.7,
    "RC101":1696.9,"RC102":1554.8,"RC103":1261.7,"RC104":1135.5,
    "RC105":1629.4,"RC106":1424.7,"RC107":1230.5,"RC108":1139.8,
    "RC201":1406.9,"RC202":1369.6,"RC203":1049.6,"RC204":798.5,
    "RC205":1297.7,"RC206":1146.3,"RC207":1057.1,"RC208":828.1,
}


# ════════════════════════════════════════════════════════════════════════════
# A1: Simple Mean-Pooling Encoder (no graph attention)
# ════════════════════════════════════════════════════════════════════════════
class SimpleMeanEncoder:
    """
    Replaces HeteroGAT: averages all node features and projects to 128-dim.
    Tests whether the graph attention mechanism is necessary.
    """
    EMBED_DIM = 128

    def __init__(self, seed: int = 0):
        self._proj  = None
        self._seed  = seed

    def encode(self, graph: Dict):
        nodes = graph.get("nodes", [])
        if not nodes:
            zeros = np.zeros(self.EMBED_DIM, dtype=np.float32)
            return zeros[None], zeros

        feats = [np.array(n["features"], dtype=np.float32)
                 for n in nodes if "features" in n]
        if not feats:
            zeros = np.zeros(self.EMBED_DIM, dtype=np.float32)
            return zeros[None], zeros

        feat_mat  = np.stack(feats, axis=0)
        mean_feat = feat_mat.mean(axis=0)
        feat_dim  = mean_feat.shape[0]

        if self._proj is None or self._proj.W.shape[0] != feat_dim:
            rng = np.random.RandomState(self._seed)
            self._proj = Linear(feat_dim, self.EMBED_DIM, seed=self._seed)
            self._proj.W = (rng.randn(feat_dim, self.EMBED_DIM) * 0.02
                            ).astype(np.float32)
            self._proj.b = np.zeros(self.EMBED_DIM, dtype=np.float32)

        graph_emb = relu(self._proj.forward(mean_feat[None]))[0]
        node_embs = np.tile(graph_emb[None], (feat_mat.shape[0], 1))
        return node_embs, graph_emb


# ════════════════════════════════════════════════════════════════════════════
# Ablation-aware Agent
# ════════════════════════════════════════════════════════════════════════════
class GATMARLAgentAblation(GATMARLAgent):
    """
    GATMARLAgent with four ablation switches:
      no_gat        — use SimpleMeanEncoder instead of HeteroGAT
      fixed_entropy — entropy_coef fixed at 0.05, no annealing
      no_critic     — critic forward only, optimizer step skipped
      no_mask_reuse — stored action masks not applied in policy update
    """
    def __init__(self, *args,
                 no_gat=False, fixed_entropy=False,
                 no_critic=False, no_mask_reuse=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.no_gat        = no_gat
        self.fixed_entropy = fixed_entropy
        self.no_critic     = no_critic
        self.no_mask_reuse = no_mask_reuse

        if self.no_gat:
            self.encoder = SimpleMeanEncoder(seed=kwargs.get("seed", 42))

        if self.fixed_entropy:
            self.entropy_coef_init  = 0.05
            self.entropy_coef_final = 0.05
            self.entropy_coef       = 0.05

    def _train_critic(self, returns_per_agent, agent_ids):
        if self.no_critic:
            T       = len(self._trajectory)
            states  = np.stack([s["graph_emb"] for s in self._trajectory],
                               axis=0).astype(np.float32)
            targets = np.array(
                [np.mean([returns_per_agent[i][t] for i in agent_ids])
                 for t in range(T)], dtype=np.float32)
            preds = self.critic.net.forward(states)[:, 0]
            loss  = float(np.mean((preds - targets) ** 2))
            for g in self.critic.grads():
                g[:] = 0.0
            return loss
        return super()._train_critic(returns_per_agent, agent_ids)

    def update_reinforce(self, episode=0, n_total_episodes=200):
        if self.fixed_entropy:
            self.entropy_coef_init  = 0.05
            self.entropy_coef_final = 0.05

        if not self.no_mask_reuse:
            return super().update_reinforce(episode, n_total_episodes)

 # ── A4: reproduce parent logic without mask application ──────────────
        if len(self._trajectory) < 2:
            self._trajectory.clear()
            return {}

        T         = len(self._trajectory)
        agent_ids = list(self._trajectory[0]["obs"].keys())
        n_ag      = len(agent_ids)

        returns_raw: Dict[int, np.ndarray] = {}
        for i in agent_ids:
            G = 0.0
            ret = np.zeros(T, dtype=np.float32)
            for t in reversed(range(T)):
                G = self._trajectory[t]["rewards"].get(i, 0.0) + self.gamma * G
                ret[t] = G
            returns_raw[i] = ret

        critic_loss = self._train_critic(returns_raw, agent_ids)

        returns: Dict[int, np.ndarray] = {}
        for i in agent_ids:
            mu, std = returns_raw[i].mean(), returns_raw[i].std() + 1e-8
            returns[i] = (returns_raw[i] - mu) / std

        N       = T * n_ag
        obs_dim = list(self._trajectory[0]["obs"].values())[0].shape[0]
        emb_dim = self._trajectory[0]["graph_emb"].shape[0]
        in_dim  = obs_dim + emb_dim

        X           = np.empty((N, in_dim), dtype=np.float32)
        actions_arr = np.empty(N,           dtype=np.int32)
        adv_arr     = np.empty(N,           dtype=np.float32)

        idx = 0
        for t, step in enumerate(self._trajectory):
            gemb     = step["graph_emb"]
            baseline = self.critic.forward(step["graph_emb"])
            for i in agent_ids:
                X[idx, :obs_dim] = step["obs"][i]
                X[idx, obs_dim:] = gemb
                actions_arr[idx] = step["actions"].get(i, 0)
                adv_arr[idx]     = returns[i][t] - baseline
                idx += 1

 # No mask applied — gradient computed over full action space
        logits = self.actor.net.forward(X)
        lmax   = logits.max(axis=1, keepdims=True)
        lse    = lmax + np.log(np.exp(logits - lmax).sum(axis=1, keepdims=True) + 1e-9)
        log_p  = logits - lse
        probs  = np.exp(log_p)
        probs /= probs.sum(axis=1, keepdims=True) + 1e-9

        grad  = adv_arr[:, None] * probs
        grad[np.arange(N), actions_arr] -= adv_arr

        frac = min(episode / max(n_total_episodes - 1, 1), 1.0)
        self.entropy_coef = (self.entropy_coef_init * (1.0 - frac)
                             + self.entropy_coef_final * frac)
        grad += self.entropy_coef * probs * (np.log(probs + 1e-9) + 1.0)
        grad /= N

        self.actor.net.backward(grad)
        actor_grads = self.actor.grads()
        gnorm = math.sqrt(sum(np.sum(g**2) for g in actor_grads))
        if gnorm > self.max_grad_norm:
            scale = self.max_grad_norm / gnorm
            for g in actor_grads:
                g *= scale
        self.opt_actor.step(actor_grads)

        log_probs_taken = np.log(probs[np.arange(N), actions_arr] + 1e-9)
        actor_loss = float(-np.mean(adv_arr * log_probs_taken))
        entropy    = float(-np.mean(np.sum(probs * np.log(probs + 1e-9), axis=1)))

        stats = {"actor_loss": actor_loss, "critic_loss": critic_loss,
                 "entropy": entropy, "entropy_coef": self.entropy_coef,
                 "grad_norm": gnorm, "n_steps": T}
        self.train_stats.append(stats)
        self.episode_count += 1
        self._trajectory.clear()
        return stats


# ════════════════════════════════════════════════════════════════════════════
# Ablation variant table
# ════════════════════════════════════════════════════════════════════════════
# Full uses default kwargs (all flags False) — re-trained from scratch
ABLATION_CONFIGS = {
    "Full":     dict(no_gat=False, fixed_entropy=False, no_critic=False, no_mask_reuse=False),
    "A1-NoGAT":    dict(no_gat=True,  fixed_entropy=False, no_critic=False, no_mask_reuse=False),
    "A2-FixedEnt": dict(no_gat=False, fixed_entropy=True,  no_critic=False, no_mask_reuse=False),
    "A3-NoCritic": dict(no_gat=False, fixed_entropy=False, no_critic=True,  no_mask_reuse=False),
    "A4-NoMasks":  dict(no_gat=False, fixed_entropy=False, no_critic=False, no_mask_reuse=True),
}


# ════════════════════════════════════════════════════════════════════════════
# Single-seed training run with curve logging
# ════════════════════════════════════════════════════════════════════════════
def run_one_seed(inst_name: str, variant_name: str, ablation_kwargs: dict,
                 seed: int, seed_idx: int,
                 inst_data: dict, n_cust: int, n_veh: int) -> tuple:
    """
    Trains one seed for n_episodes.
    Returns:
      final_row  : dict with final performance metrics
      curve_rows : list of dicts {instance, variant, seed, episode, service_rate, co2_kg}
    """
    np.random.seed(seed)
    random.seed(seed)

    env = TruckDroneEnv(inst_data, n_trucks=n_veh, seed=seed)
    obs_dict  = env.reset()
    n_agents  = env.n_agents
    obs_dim   = list(obs_dict.values())[0].shape[0]
    n_actions = len(env.get_action_mask(0))

    agent = GATMARLAgentAblation(
        n_agents=n_agents, obs_dim=obs_dim, n_actions=n_actions,
        seed=seed, **ablation_kwargs
    )

    curve_rows = []
    t0 = time.time()

    for ep in range(N_EPISODES):
        train_episode(agent, env, episode=ep, n_total_episodes=N_EPISODES)

 # Log training curve at EVAL_FREQ intervals and at final episode
        if (ep + 1) % EVAL_FREQ == 0 or ep == N_EPISODES - 1:
 # Single greedy rollout for curve (fast — no averaging)
            snapshot = evaluate_episode(agent, env, n_rollouts=1)
            sr = int(snapshot.get("n_served", 0)) / max(n_cust, 1)
            co2 = float(snapshot.get("total_co2_kg", 0.0))
            curve_rows.append({
                "instance":     inst_name,
                "variant":      variant_name,
                "seed":         seed,
                "episode":      ep + 1,
                "service_rate": sr,
                "co2_kg":       co2,
            })

 # Final evaluation (10 rollouts for reliable estimate)
    final = evaluate_episode(agent, env, n_rollouts=10)
    elapsed = time.time() - t0

    n_served = int(final.get("n_served", 0))
    final_row = {
        "instance":       inst_name,
        "variant":        variant_name,
        "seed":           seed,
        "seed_idx":       seed_idx,
        "n_customers":    n_cust,
        "n_vehicles":     n_veh,
        "bks_distance":   BKS_TABLE.get(inst_name, 0.0),
        "n_served":       n_served,
        "service_rate":   n_served / max(n_cust, 1),
        "total_co2_kg":   float(final.get("total_co2_kg", 0.0)),
        "makespan":       float(final.get("makespan", 0.0)),
        "total_distance": float(final.get("total_distance", 0.0)),
        "runtime_s":      elapsed,
    }

    if VERBOSE:
        print(f"    {variant_name} | seed={seed} ({seed_idx+1}/{N_SEEDS}) "
              f"served={n_served}/{n_cust} "
              f"co2={final_row['total_co2_kg']:.1f} "
              f"t={elapsed:.0f}s")

    return final_row, curve_rows


# ════════════════════════════════════════════════════════════════════════════
# Resume support
# ════════════════════════════════════════════════════════════════════════════
def load_partial_results():
    rows = pd.read_csv(PARTIAL_FINAL).to_dict("records") if PARTIAL_FINAL.exists() else []
    return rows

def load_partial_curves():
    rows = pd.read_csv(PARTIAL_CURVES).to_dict("records") if PARTIAL_CURVES.exists() else []
    return rows

def is_seed_done(rows, inst, variant, seed):
    return any(r["instance"] == inst and r["variant"] == variant
               and int(r["seed"]) == seed for r in rows)

def save_partial(final_rows, curve_rows):
    pd.DataFrame(final_rows).to_csv(PARTIAL_FINAL, index=False)
    pd.DataFrame(curve_rows).to_csv(PARTIAL_CURVES, index=False)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("GAT-MARL — Robust Ablation Study")
    print("=" * 70)
    print(f"  Instances  : {ABLATION_INSTANCES}")
    print(f"  Variants   : {list(ABLATION_CONFIGS.keys())}")
    print(f"  Episodes   : {N_EPISODES} per seed")
    print(f"  Seeds      : {N_SEEDS} (all reported individually)")
    print(f"  Eval freq  : every {EVAL_FREQ} episodes (training curves)")
    print(f"  Note       : Full re-trained from scratch (fair comparison)")
    print("=" * 70)

 # Total jobs: 5 variants × 4 instances × 3 seeds = 60 seeds
    total_seeds = len(ABLATION_CONFIGS) * len(ABLATION_INSTANCES) * N_SEEDS
    print(f"  Total seed runs : {total_seeds}")
    est_h = total_seeds * 6.0 / 60 # rough: 6 min per seed
    print(f"  Estimated time  : ~{est_h:.1f} h\n")

    all_final  = load_partial_results()
    all_curves = load_partial_curves()
    done_seeds = {(r["instance"], r["variant"], int(r["seed"]))
                  for r in all_final}

    done_count = len(done_seeds)
    print(f"  Already done : {done_count}/{total_seeds} seed runs\n")

 # Pre-load all instances (avoid repeated disk reads)
    inst_cache = {}
    for iname in ABLATION_INSTANCES:
        sol_path = SOLOMON_DIR / f"{iname}.txt"
        if not sol_path.exists():
            print(f"  ERROR: {sol_path} not found — skipping {iname}")
            continue
        inst_data = load_solomon_instance(str(sol_path))
        n_cust    = len(inst_data["customers"])
        n_veh     = inst_data.get("n_vehicles", 25)
        inst_cache[iname] = (inst_data, n_cust, n_veh)
        print(f"  Loaded {iname}: {n_cust} customers, {n_veh} vehicles")

    print()

 # ── Main training loop ───────────────────────────────────────────────────
    for variant_name, kwargs in ABLATION_CONFIGS.items():
        for inst_name in ABLATION_INSTANCES:
            if inst_name not in inst_cache:
                continue

            if (variant_name, inst_name) in SKIP_COMBOS:
                print(f"  SKIP-COMBO [{inst_name}] {variant_name} (user decision — sufficient data)")
                continue

            inst_data, n_cust, n_veh = inst_cache[inst_name]
            offset = INSTANCE_OFFSET.get(inst_name, 0)

            for seed_idx in range(N_SEEDS):
                seed = seed_idx * 200 + offset

                if (inst_name, variant_name, seed) in done_seeds:
                    print(f"  SKIP  [{inst_name}] {variant_name} seed={seed}")
                    continue

                print(f"\n  [{inst_name}] {variant_name} seed={seed} ({seed_idx+1}/{N_SEEDS})")

                final_row, curve_rows = run_one_seed(
                    inst_name, variant_name, kwargs,
                    seed, seed_idx, inst_data, n_cust, n_veh
                )

                all_final.append(final_row)
                all_curves.extend(curve_rows)
                done_seeds.add((inst_name, variant_name, seed))
                save_partial(all_final, all_curves)
                print(f"    -> Saved partial ({len(all_final)} seed-rows, "
                      f"{len(all_curves)} curve-points)")

 # ── Write final outputs ──────────────────────────────────────────────────
    final_df  = pd.DataFrame(all_final)
    curves_df = pd.DataFrame(all_curves)
    final_df.to_csv(OUTPUT_FINAL,  index=False)
    curves_df.to_csv(OUTPUT_CURVES, index=False)

    print(f"\n{'='*70}")
    print(f"ROBUST ABLATION COMPLETE.")
    print(f"  Final results : {OUTPUT_FINAL}")
    print(f"  Training curves: {OUTPUT_CURVES}")
    print(f"{'='*70}\n")

 # ── Summary table ────────────────────────────────────────────────────────
    print("=== SUMMARY (mean across seeds) ===")
    summary = (final_df
               .groupby(["instance", "variant"])
               .agg(
                   mean_sr=("service_rate", lambda x: f"{x.mean()*100:.1f}%"),
                   std_sr =("service_rate", lambda x: f"±{x.std()*100:.1f}%"),
                   mean_co2=("total_co2_kg", lambda x: f"{x.mean():.1f}"),
                   n_collapse=("service_rate", lambda x: int((x < 0.80).sum()))
               ))
    print(summary.to_string())
    print()
    print("=== KEY SIGNAL: seed collapses (service_rate < 80%) ===")
    collapses = final_df[final_df["service_rate"] < 0.80][
        ["instance","variant","seed","service_rate"]
    ]
    if collapses.empty:
        print("  None — all seeds achieved >= 80% service rate.")
    else:
        print(collapses.to_string(index=False))
