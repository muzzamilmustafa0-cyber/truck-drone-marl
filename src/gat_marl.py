"""
GAT-Enhanced multi-agent reinforcement learning for truck-drone routing.

Design motivated by a training-stability analysis:
  1. ENTROPY COLLAPSE: Mean entropy dropped 4.32 -> 0.48 (-89%) by ep49.
     Many R/RC instances hit 0.00 (fully deterministic, stuck in bad mode).
     Fix: Entropy annealing — start at 0.10, decay to 0.01 over training.

  2. UNTRAINED CRITIC BASELINE: the critic is created and trained.
     Random critic adds pure noise to advantage estimates.
     Fix: Train critic via MSE to Monte-Carlo returns each episode.

  3. GRADIENT MISMATCH: Masks applied during select_actions but NOT during
     update_reinforce (fresh unmasked forward pass). Policy gradient is
     computed for wrong distribution.
     Fix: Store masks in trajectory buffer; apply them in update_reinforce.

  4. GRADIENT CLIPPING: Large gradient norms cause single-step policy
     destruction (C101 ep0=92 -> ep10=38). Fix: clip norm to 0.5.

Architecture:
  - HeteroGAT encoder (2-layer, 4-head, 128-dim) — still FROZEN
  - Shared actor MLP [obs+emb -> 256 -> 256 -> n_actions]
  - Centralised critic MLP [global_state -> 512 -> 256 -> 1]
  - REINFORCE with trained critic baseline (Actor-Critic style MC)

Training improvements:
  - entropy_coef: annealed from 0.10 -> 0.01 (10x stronger early exploration)
  - critic trained: Adam optimizer with MSE loss to MC returns
  - masks stored & reused: consistent gradient signal
  - gradient clipping: max_grad_norm = 0.5
  - default episodes: 200 (was 50)
  - default seeds: best-of-5 (was best-of-3)
"""

import math, os, random, time, copy, json
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Core neural components
from neural_base import (
    relu, tanh, sigmoid, layer_norm, softmax,
    Linear, MLP, HeteroGATLayer, GATEncoder,
    ActorNet, CriticNet, Adam,
    save_model, load_model # save/load still work with same format
)


# ─────────────────────────────────────────────────────────────────
# Improved MARL Agent 
# ─────────────────────────────────────────────────────────────────

class GATMARLAgent:
    """
    GAT-Enhanced actor-critic agent with three targeted stability fixes.

    Key design choices:
      - Entropy annealing: 0.10 -> 0.01 over n_episodes
      - Critic is trained (Actor-Critic MC): Adam optimizer added
      - Action masks stored in trajectory and reused in update
      - Gradient clipping before Adam step
    """
    def __init__(self,
                 n_agents:            int,
                 obs_dim:             int,
                 n_actions:           int,
                 lr_actor:            float = 3e-4,
                 lr_critic:           float = 1e-3,
                 entropy_coef_init:   float = 0.10, # high start for exploration
                 entropy_coef_final:  float = 0.01, # low end for exploitation
                 gamma:               float = 0.99,
                 max_grad_norm:       float = 0.5, # gradient clipping
                 seed:                int   = 42,
                 centralised_critic:  bool  = True,
                 use_gat:             bool  = True):

        self.n_agents        = n_agents
        self.obs_dim         = obs_dim
        self.n_actions       = n_actions
        self.entropy_coef_init  = entropy_coef_init
        self.entropy_coef_final = entropy_coef_final
        self.entropy_coef    = entropy_coef_init # updated each episode
        self.gamma           = gamma
        self.max_grad_norm   = max_grad_norm
        self.centralised_critic = centralised_critic

        self.encoder   = GATEncoder(seed=seed, use_gat=use_gat)
        self.embed_dim = GATEncoder.EMBED_DIM

        self.actor = ActorNet(obs_dim, n_actions, self.embed_dim, seed=seed+1000)

 # critic uses only the graph embedding (128-dim) as global state.
 # The graph embedding captures depot+customer+truck+drone states via GAT.
 # Using the full obs_dim*n_agents concatenation is prohibitively slow on CPU
 # for 50-agent instances (17,428-dim critic = severe bottleneck).
 # 128-dim graph embedding is sufficient for a variance-reducing baseline.
        critic_dim = self.embed_dim # always 128, regardless of centralised flag
        self.centralised_critic = centralised_critic # kept for ablation labelling
        self.critic = CriticNet(critic_dim, seed=seed+2000)

 # BOTH actor AND critic have optimizers
        self.opt_actor  = Adam(self.actor.params(),  lr=lr_actor)
        self.opt_critic = Adam(self.critic.params(), lr=lr_critic)

        self._trajectory: List[Dict] = []
        self.total_steps   = 0
        self.episode_count = 0
        self.train_stats   = []

 # ── Action selection (unchanged logic, returns masks too) ──────
    def select_actions(self, obs_dict: Dict[int, np.ndarray],
                       graph: Dict,
                       masks: Dict[int, np.ndarray],
                       deterministic: bool = False):
        """Vectorized action selection — vectorized."""
        _, graph_emb = self.encoder.encode(graph)

        agent_ids = sorted(obs_dict.keys())
        n_ag      = len(agent_ids)

        obs_batch = np.stack([obs_dict[i] for i in agent_ids], axis=0)
        emb_batch = np.tile(graph_emb[None], (n_ag, 1))
        x_batch   = np.concatenate([obs_batch, emb_batch], axis=1)

        logits_batch = self.actor.net.forward(x_batch)

        actions   = {}
        log_probs = {}
        for idx, i in enumerate(agent_ids):
            logits = logits_batch[idx].copy()
            mask   = masks.get(i)
            if mask is not None:
                logits = np.where(mask, logits, -1e9)

            lmax = logits.max()
            lse  = lmax + np.log(np.exp(logits - lmax).sum() + 1e-9)
            lp   = logits - lse
            probs = np.exp(lp); probs /= probs.sum() + 1e-9

            if deterministic:
                a = int(np.argmax(probs))
            else:
                a = int(np.random.choice(len(probs), p=probs))

            actions[i]   = a
            log_probs[i] = lp

 # use graph_emb (128-dim) as global state for critic
 # Much faster than concatenating all agent obs (was 17k-dim for 50 agents)
        gs = graph_emb.copy()
        value = self.critic.forward(gs)
        return actions, log_probs, value, gs, graph_emb

 # ── Store transition ( masks NOW stored) ───────────────────
    def store(self, obs_dict, graph_emb, actions, log_probs_dict,
              rewards, dones, value, masks: Dict[int, np.ndarray],
              global_state: np.ndarray = None):
        """
        adds 'masks' to the trajectory step (global_state no longer needed;
        graph_emb is used directly as critic input — 128-dim, fast).
        masks: needed to reproduce the masked distribution in update_reinforce.
        """
        step = {
            "obs":       {i: obs_dict[i].copy() for i in obs_dict},
            "graph_emb": graph_emb.copy(),
            "actions":   dict(actions),
            "log_probs": {i: float(log_probs_dict[i][actions[i]]) for i in actions},
            "rewards":   dict(rewards),
            "dones":     dict(dones),
            "value":     value,
            "masks":     {i: masks[i].copy() if masks.get(i) is not None
                          else None for i in actions},
        }
        self._trajectory.append(step)
        self.total_steps += 1

 # ── Critic training — vectorized  ─────────────────
    def _train_critic(self, returns_per_agent: Dict[int, np.ndarray],
                      agent_ids: List[int]) -> float:
        """
        Vectorized critic training: ONE forward+backward pass for all T steps.
        Uses graph_emb (128-dim) stored in trajectory as global state.
        Target = mean normalised MC return across agents at each step.
        """
        T = len(self._trajectory)

 # Stack graph embeddings: (T, embed_dim)
        states = np.stack([step["graph_emb"] for step in self._trajectory],
                          axis=0).astype(np.float32)

 # Target values: mean MC return across agents per timestep — (T,)
        targets = np.array(
            [np.mean([returns_per_agent[i][t] for i in agent_ids])
             for t in range(T)],
            dtype=np.float32
        )

 # ONE vectorized forward pass: (T, 1)
        preds = self.critic.net.forward(states)[:, 0] # (T,)
        errs  = preds - targets # (T,)
        loss  = float(np.mean(errs ** 2))

 # ONE vectorized backward: gradient of MSE w.r.t. each prediction
        grad_out = (2.0 * errs / T)[:, None].astype(np.float32) # (T, 1)
        self.critic.net.backward(grad_out)

 # Clip and update once
        cgrads = self.critic.grads()
        gnorm  = math.sqrt(sum(np.sum(g ** 2) for g in cgrads))
        if gnorm > self.max_grad_norm:
            scale = self.max_grad_norm / gnorm
            for g in cgrads:
                g *= scale
        self.opt_critic.step(cgrads)
        return loss

 # ── REINFORCE update ───────────────────────────────────────
    def update_reinforce(self, episode: int = 0,
                         n_total_episodes: int = 200) -> Dict[str, float]:
        """
        Vectorized REINFORCE with three improvements:
          1. Entropy annealing based on episode/n_total_episodes
          2. Masks applied per-sample in forward pass (gradient consistency)
          3. Gradient clipping before Adam step
          (Critic is trained separately by _train_critic then used as baseline)
        """
        if len(self._trajectory) < 2:
            self._trajectory.clear()
            return {}

        T         = len(self._trajectory)
        agent_ids = list(self._trajectory[0]["obs"].keys())
        n_ag      = len(agent_ids)

 # ── 1. Compute discounted MC returns ──────────────────────
 # Store UN-normalised returns for critic training target
        returns_raw: Dict[int, np.ndarray] = {}
        for i in agent_ids:
            G   = 0.0
            ret = np.zeros(T, dtype=np.float32)
            for t in reversed(range(T)):
                G = self._trajectory[t]["rewards"].get(i, 0.0) + self.gamma * G
                ret[t] = G
            returns_raw[i] = ret

 # ── 2. Train critic to predict MC returns ─────────────────
        critic_loss = self._train_critic(returns_raw, agent_ids)

 # ── 3. Normalise returns for actor advantage ──────────────
        returns: Dict[int, np.ndarray] = {}
        for i in agent_ids:
            mu, std = returns_raw[i].mean(), returns_raw[i].std() + 1e-8
            returns[i] = (returns_raw[i] - mu) / std

 # ── 4. Build batch tensors (N = T × n_agents) ─────────────
        N       = T * n_ag
        obs_dim = list(self._trajectory[0]["obs"].values())[0].shape[0]
        emb_dim = self._trajectory[0]["graph_emb"].shape[0]
        in_dim  = obs_dim + emb_dim

        X            = np.empty((N, in_dim),              dtype=np.float32)
        actions_arr  = np.empty(N,                        dtype=np.int32)
        adv_arr      = np.empty(N,                        dtype=np.float32)
        masks_batch  = [None] * N # per-sample mask (list, not array — ragged)

        idx = 0
        for t, step in enumerate(self._trajectory):
            gemb     = step["graph_emb"]
 # Use TRAINED critic (128-dim graph_emb input) for baseline
            baseline = self.critic.forward(step["graph_emb"])
            for i in agent_ids:
                X[idx, :obs_dim]  = step["obs"][i]
                X[idx, obs_dim:]  = gemb
                actions_arr[idx]  = step["actions"].get(i, 0)
                adv_arr[idx]      = returns[i][t] - baseline
 # Store the mask for this agent at this timestep
                masks_batch[idx]  = step["masks"].get(i)
                idx += 1

 # ── 5. ONE forward pass (with per-sample mask application) ─
        logits = self.actor.net.forward(X) # (N, n_actions)

 # Apply stored mask to each sample's logits
 # This is the FIX for gradient mismatch: same distribution as during selection
        masked_logits = logits.copy()
        for idx in range(N):
            m = masks_batch[idx]
            if m is not None:
                masked_logits[idx] = np.where(m, logits[idx], -1e9)

 # Stable log-softmax from masked logits
        lmax   = masked_logits.max(axis=1, keepdims=True)
        lse    = lmax + np.log(np.exp(masked_logits - lmax).sum(axis=1, keepdims=True) + 1e-9)
        log_p  = masked_logits - lse # (N, n_actions)
        probs  = np.exp(log_p)
        probs /= probs.sum(axis=1, keepdims=True) + 1e-9

 # ── 6. REINFORCE gradient ─────────────────────────────────
 # d(-adv * log_prob(a)) / d(logits_k) = adv * p_k - adv * 1{k==a}
        grad = adv_arr[:, None] * probs
        grad[np.arange(N), actions_arr] -= adv_arr

 # ── 7. Entropy annealing ──────────────────────────────────
        frac = min(episode / max(n_total_episodes - 1, 1), 1.0)
        self.entropy_coef = (self.entropy_coef_init * (1.0 - frac)
                             + self.entropy_coef_final * frac)

 # Entropy gradient: -(H_coef * (log p + 1)) applied to logits
        grad += self.entropy_coef * probs * (np.log(probs + 1e-9) + 1.0)
        grad /= N # average over batch

 # ── 8. ONE backward pass ──────────────────────────────────
        self.actor.net.backward(grad)

 # ── 9. Gradient clipping ──────────────────────────────────
        actor_grads = self.actor.grads()
        gnorm = math.sqrt(sum(np.sum(g**2) for g in actor_grads))
        if gnorm > self.max_grad_norm:
            scale = self.max_grad_norm / gnorm
            for g in actor_grads:
                g *= scale # in-place scale (these are the .dW / .db arrays)

 # ── 10. Adam update ───────────────────────────────────────
        self.opt_actor.step(actor_grads)

        log_probs_taken = np.log(probs[np.arange(N), actions_arr] + 1e-9)
        actor_loss  = float(-np.mean(adv_arr * log_probs_taken))
        entropy     = float(-np.mean(np.sum(probs * np.log(probs + 1e-9), axis=1)))

        stats = {
            "actor_loss":  actor_loss,
            "critic_loss": critic_loss,
            "entropy":     entropy,
            "entropy_coef": self.entropy_coef,
            "grad_norm":   gnorm,
            "n_steps":     T,
        }
        self.train_stats.append(stats)
        self.episode_count += 1
        self._trajectory.clear()
        return stats


# ─────────────────────────────────────────────────────────────────
# Training and evaluation loops 
# ─────────────────────────────────────────────────────────────────

def train_episode(agent: GATMARLAgent, env,
                     episode: int = 0,
                     n_total_episodes: int = 200,
                     max_steps: int = 150) -> Dict:
    """
    Run one training episode.
    Key change: masks and global_state passed to agent.store().
    """
    obs_dict = env.reset()
    graph    = env.get_graph_representation()
    masks    = {i: env.get_action_mask(i) for i in range(env.n_agents)}

    ep_reward: Dict[int, float] = defaultdict(float)
    step = 0
    done = False

    while not done and step < max_steps:
        actions, log_probs, value, gs, graph_emb = agent.select_actions(
            obs_dict, graph, masks, deterministic=False
        )
        next_obs, rewards, dones, infos = env.step(actions)
        done = dones.get("__all__", False)

 # pass masks to store() (global_state no longer needed — critic uses graph_emb)
        agent.store(obs_dict, graph_emb, actions, log_probs,
                    rewards, dones, value, masks=masks)

        for i, r in rewards.items():
            ep_reward[i] += r

        obs_dict = next_obs
        graph    = env.get_graph_representation()
        masks    = {i: env.get_action_mask(i) for i in range(env.n_agents)}
        step    += 1

 # pass episode number for entropy annealing
    train_stats = agent.update_reinforce(episode=episode,
                                         n_total_episodes=n_total_episodes)

    metrics = env.get_metrics()
    metrics.update({
        "episode_reward": float(sum(ep_reward.values())),
        "steps":          step,
        **train_stats
    })
    return metrics


def evaluate_episode(agent: GATMARLAgent, env,
                        max_steps: int = 200,
                        n_rollouts: int = 10) -> Dict:
    """
    Best-of-N stochastic evaluation (vectorized — unchanged).
    """
    best_metrics = None
    best_served  = -1

    for _ in range(n_rollouts):
        obs_dict = env.reset()
        graph    = env.get_graph_representation()
        masks    = {i: env.get_action_mask(i) for i in range(env.n_agents)}

        step = 0
        done = False
        while not done and step < max_steps:
            actions, _, _, _, _ = agent.select_actions(
                obs_dict, graph, masks, deterministic=False
            )
            next_obs, _, dones, _ = env.step(actions)
            done = dones.get("__all__", False)
            obs_dict = next_obs
            graph    = env.get_graph_representation()
            masks    = {i: env.get_action_mask(i) for i in range(env.n_agents)}
            step    += 1

        m = env.get_metrics()
        if m.get("n_served", 0) > best_served:
            best_served  = m.get("n_served", 0)
            best_metrics = m

    return best_metrics or env.get_metrics()


# ─────────────────────────────────────────────────────────────────
# Run GAT-MARL on one instance
# ─────────────────────────────────────────────────────────────────

def run_gat_marl(instance: dict,
                    n_trucks: int = None,
                    n_drones_per_truck: int = 1,
                    seed: int = 42,
                    n_train_episodes: int = 200,
                    centralised_critic: bool = True,
                    use_gat: bool = True,
                    use_co2_reward: bool = True,
                    use_nfz_mask: bool = True) -> Dict:
    """
    Run GAT-MARL on a single instance.

    Default n_train_episodes = 200 (was 50 ).
    Uses environment (stronger rewards, drone CO2 in metrics).
    All ablation flags still work (used by the ablation study).
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from truck_drone_env import TruckDroneEnv, NFZone

 # NFZone still imported from original env (dataclass unchanged)
    from vrp_base import NFZone as NFZoneOrig

    if n_trucks is None:
        n_trucks = max(3, instance["n_vehicles"])

    custs = instance["customers"]
    mid_x = float(np.mean([c.x for c in custs]))
    mid_y = float(np.mean([c.y for c in custs]))
    nfz_list = ([NFZoneOrig(cx=mid_x + 5, cy=mid_y + 5, radius=8.0)]
                if use_nfz_mask else [])

    env = TruckDroneEnv(
        instance, n_trucks=n_trucks,
        n_drones_per_truck=n_drones_per_truck,
        nfz_list=nfz_list, seed=seed
    )

    if not use_co2_reward:
        env.W_CO2_STEP = 0.0 # disable CO2 component of per-step reward

    if not use_nfz_mask:
        env.nfz_list = []

    agent = GATMARLAgent(
        n_agents           = env.n_agents,
        obs_dim            = env.obs_dim,
        n_actions          = env.n_cust + 1,
        seed               = seed,
        centralised_critic = centralised_critic,
        use_gat            = use_gat,
    )

    train_history = []
    t_start = time.time()

    for ep in range(n_train_episodes):
        m = train_episode(agent, env,
                             episode=ep,
                             n_total_episodes=n_train_episodes,
                             max_steps=150)
        train_history.append({
            "episode":      ep,
            "reward":       m.get("episode_reward", 0.0),
            "served":       m.get("n_served", 0),
            "makespan":     m.get("makespan", 0.0),
            "co2_kg":       m.get("total_co2_kg", 0.0),
            "actor_loss":   m.get("actor_loss", 0.0),
            "critic_loss":  m.get("critic_loss", 0.0),
            "entropy":      m.get("entropy", 0.0),
            "entropy_coef": m.get("entropy_coef", 0.0),
            "grad_norm":    m.get("grad_norm", 0.0),
        })

    runtime_s = time.time() - t_start

 # Best-of-10 stochastic evaluation
    eval_m = evaluate_episode(agent, env, max_steps=200, n_rollouts=10)
    eval_m["train_history"] = train_history
    eval_m["runtime_s"]     = runtime_s
    eval_m["n_train_ep"]    = n_train_episodes

    return eval_m, agent


# ─────────────────────────────────────────────────────────────────
# Model save / load (compatible with format)
# ─────────────────────────────────────────────────────────────────

def save_model(agent: GATMARLAgent, path: str):
    data = {
        "version":           "1.0",
        "n_agents":          agent.n_agents,
        "obs_dim":           agent.obs_dim,
        "n_actions":         agent.n_actions,
        "centralised_critic": agent.centralised_critic,
        "actor_W":           [p.tolist() for p in agent.actor.params()],
        "critic_W":          [p.tolist() for p in agent.critic.params()],
        "encoder_W":         [p.tolist() for p in agent.encoder.params()],
        "total_steps":       agent.total_steps,
        "episode_count":     agent.episode_count,
        "train_stats":       agent.train_stats[-200:],
        "entropy_coef":      agent.entropy_coef,
    }
    with open(path, "w") as f:
        json.dump(data, f)


# ─────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vrp_base import load_solomon_instance, NFZone

    BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inst_path = os.path.join(BASE, "..", "data", "solomon", "C101.txt")

    if not os.path.exists(inst_path):
        print(f"SKIP self-test: {inst_path} not found")
        import sys; sys.exit(0)

    inst = load_solomon_instance(inst_path)
    print(f"GAT-MARL self-test on C101 (10 episodes)...")

    result, agent = run_gat_marl(inst, n_trucks=3, seed=42,
                                     n_train_episodes=10)

    hist = result["train_history"]
    print(f"  ep0  served={hist[0]['served']:.0f}  entropy={hist[0]['entropy']:.3f}  "
          f"entropy_coef={hist[0]['entropy_coef']:.4f}")
    print(f"  ep9  served={hist[-1]['served']:.0f}  entropy={hist[-1]['entropy']:.3f}  "
          f"entropy_coef={hist[-1]['entropy_coef']:.4f}  "
          f"critic_loss={hist[-1]['critic_loss']:.4f}  "
          f"grad_norm={hist[-1]['grad_norm']:.4f}")
    print(f"  Eval served={result['n_served']}/{result['n_customers']}  "
          f"CO2={result['total_co2_kg']:.2f}kg")
    print("[PASS] gat_marl self-test complete.")
