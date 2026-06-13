"""Carbon-preference sweep.

Trains a separate GAT-MARL policy at each carbon-preference weight lambda and
records the resulting time-window-feasible service rate and CO2, tracing the
service-CO2 trade-off. The environment scales its CO2 reward terms by lambda:
    env.W_CO2_STEP = lambda          (per-trip CO2 penalty)
    env.W_CO2      = 0.5 * lambda    (terminal CO2 term)
lambda = 0 ignores carbon; larger lambda increases carbon aversion; lambda = 1
reproduces the baseline configuration. Output: carbon_sweep.csv.
"""
import sys, time, random
from pathlib import Path
import numpy as np
import pandas as pd

ROOT        = Path(__file__).parent.parent
CODE        = ROOT / "code"
SOLOMON_DIR = ROOT / "data" / "solomon"
RESULTS_DIR  = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE))

from truck_drone_env import TruckDroneEnv, load_solomon_instance
from gat_marl  import GATMARLAgent, train_episode, evaluate_episode

# ── Configuration ────────────────────────────────────────────────────────────
CARBON_WEIGHTS = [0.0, 1.0, 4.0, 10.0, 25.0] # lambda dial (1.0 == baseline configuration)
INSTANCES      = ["C101", "C201", "R101", "RC101"]
N_EPISODES     = 200
N_SEEDS        = 3
VERBOSE        = True

OUTPUT   = RESULTS_DIR / "carbon_sweep.csv"
PARTIAL  = RESULTS_DIR / "carbon_sweep_partial.csv"
INSTANCE_OFFSET = {"C101": 0, "C201": 3, "R101": 1, "RC101": 2}

BKS = {"C101":828.9,"C201":589.1,"R101":1645.9,"RC101":1696.9}


def run_one(inst_name, inst_data, n_cust, n_veh, lam, seed, seed_idx):
    np.random.seed(seed); random.seed(seed)
    env = TruckDroneEnv(inst_data, n_trucks=n_veh, seed=seed)

 # ── set the carbon dial (overrides defaults set in __init__) ─────────────
    env.W_CO2_STEP = lam
    env.W_CO2      = 0.5 * lam

    obs = env.reset()
 # reset() must not wipe the dial; re-assert just in case
    env.W_CO2_STEP = lam
    env.W_CO2      = 0.5 * lam

    n_agents  = env.n_agents
    obs_dim   = list(obs.values())[0].shape[0]
    n_actions = len(env.get_action_mask(0))

    agent = GATMARLAgent(n_agents=n_agents, obs_dim=obs_dim,
                           n_actions=n_actions, seed=seed)

    t0 = time.time()
    for ep in range(N_EPISODES):
 # re-assert the dial every episode (reset() is called inside training)
        env.W_CO2_STEP = lam
        env.W_CO2      = 0.5 * lam
        train_episode(agent, env, episode=ep, n_total_episodes=N_EPISODES)

    env.W_CO2_STEP = lam; env.W_CO2 = 0.5 * lam
    final = evaluate_episode(agent, env, n_rollouts=10)
    elapsed = time.time() - t0

    n_served = int(final.get("n_served", 0))
    row = {
        "instance":      inst_name,
        "lambda":        lam,
        "seed":          seed,
        "seed_idx":      seed_idx,
        "n_customers":   n_cust,
        "n_served":      n_served,
        "service_rate":  n_served / max(n_cust, 1), # time-window feasible
        "total_co2_kg":  float(final.get("total_co2_kg", 0.0)),
        "makespan":      float(final.get("makespan", 0.0)),
        "total_distance":float(final.get("total_distance", 0.0)),
        "runtime_s":     elapsed,
    }
    if VERBOSE:
        print(f"    lam={lam:5.1f} seed={seed} ({seed_idx+1}/{N_SEEDS}) "
              f"served={n_served}/{n_cust} co2={row['total_co2_kg']:.1f} "
              f"t={elapsed:.0f}s")
    return row


def load_partial():
    return pd.read_csv(PARTIAL).to_dict("records") if PARTIAL.exists() else []


if __name__ == "__main__":
    print("=" * 70)
    print("Controllable Carbon Trade-off (scalarization sweep)")
    print("=" * 70)
    print(f"  lambda values : {CARBON_WEIGHTS}")
    print(f"  instances     : {INSTANCES}")
    print(f"  episodes/seed : {N_EPISODES}    seeds: {N_SEEDS}")
    total = len(CARBON_WEIGHTS) * len(INSTANCES) * N_SEEDS
    print(f"  total runs    : {total}   (~{total*5/60:.1f} h @ 5 min/run)")
    print("=" * 70)

    rows = load_partial()
    done = {(r["instance"], float(r["lambda"]), int(r["seed"])) for r in rows}
    print(f"  already done  : {len(done)}/{total}\n")

    cache = {}
    for iname in INSTANCES:
        sf = SOLOMON_DIR / f"{iname}.txt"
        if not sf.exists():
            print(f"  ERROR missing {sf}"); continue
        inst = load_solomon_instance(str(sf))
        cache[iname] = (inst, len(inst["customers"]), inst.get("n_vehicles", 25))
        print(f"  loaded {iname}: {cache[iname][1]} customers")
    print()

    for iname in INSTANCES:
        if iname not in cache: continue
        inst_data, n_cust, n_veh = cache[iname]
        off = INSTANCE_OFFSET.get(iname, 0)
        for lam in CARBON_WEIGHTS:
            for si in range(N_SEEDS):
                seed = si * 200 + off
                if (iname, float(lam), seed) in done:
                    print(f"  SKIP [{iname}] lam={lam} seed={seed}")
                    continue
                print(f"\n  [{iname}] lambda={lam} seed={seed} ({si+1}/{N_SEEDS})")
                row = run_one(iname, inst_data, n_cust, n_veh, lam, seed, si)
                rows.append(row)
                done.add((iname, float(lam), seed))
                pd.DataFrame(rows).to_csv(PARTIAL, index=False)
                print(f"    -> saved partial ({len(rows)} rows)")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT, index=False)
    print(f"\n{'='*70}\nSWEEP COMPLETE -> {OUTPUT}\n{'='*70}")

    print("\n=== Mean over seeds: feasible service & CO2 by (instance, lambda) ===")
    s = (df.groupby(["instance","lambda"])
           .agg(service=("service_rate", lambda x: f"{x.mean()*100:.1f}%"),
                co2=("total_co2_kg","mean"))
           .round(1))
    print(s.to_string())
