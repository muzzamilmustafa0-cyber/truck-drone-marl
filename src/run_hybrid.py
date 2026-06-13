"""
run_hybrid.py — GAT-MARL-LS over all 56 Solomon instances.

For each instance: load the saved GAT-MARL model, roll out (best-of-5) to get
the feasible served set + truck/drone split, then re-route the TRUCK-served
customers via fill-first consolidation + TW-aware 2-opt/Or-opt. Drone deliveries
untouched.

Reports per instance:
  service_rate_ls  (time-window feasible; = served after consolidation / n_cust)
  total_co2_kg     (consolidated truck COPERT + drone electricity CO2)
  makespan, total_distance, n_trucks_used, dropped

No training — pure inference + classical routing. Output: hybrid_results.csv
"""
import sys, random, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "code"))
from local_search import (load_model, edist, route_distance,
                                     route_co2_g, is_tw_feasible, two_opt, or_opt,
                                     rollout_routes, SPEED)
from consolidation import consolidate
from truck_drone_env import TruckDroneEnv, load_solomon_instance

SOLOMON = ROOT / "data" / "solomon"
MODELS  = ROOT / "data" / "models"
RESULTS = ROOT / "results"
OUT     = RESULTS / "hybrid_results.csv"

BKS = pd.read_csv(RESULTS / "main_comparison.csv")[
        ["instance","bks_distance","n_customers"]].drop_duplicates().set_index("instance")


def route_time(depot, route):
    t = 0.0; cur = depot
    for c in route:
        t += edist(cur, c) / SPEED
        t = max(t, c.ready_time) + c.service_time
        cur = c
    return t + edist(cur, depot) / SPEED


def main():
    rows = []
    insts = sorted(p.stem for p in SOLOMON.glob("*.txt"))
    for k, inst_name in enumerate(insts):
        models = sorted(MODELS.glob(f"gatmarl_{inst_name}_seed*.json"))
        if not models:
            print(f"  {inst_name}: no model, skip"); continue
        inst = load_solomon_instance(str(SOLOMON / f"{inst_name}.txt"))
        depot = inst["depot"]; cap = inst["capacity"]
        by_id = {c.id: c for c in inst["customers"]}
        n_cust = len(inst["customers"]); n_veh = inst.get("n_vehicles", 25)

        np.random.seed(0); random.seed(0)
        env = TruckDroneEnv(inst, n_trucks=n_veh, seed=0); env.reset()
        od = env._get_obs(0).shape[0]
        ag = load_model(str(models[0]), env.n_agents, od, len(env.get_action_mask(0)))
 # truck routes, drone-served ids, and drone CO2 all from the SAME best rollout
        m, truck_routes_ids, drone_ids, drone_co2_g = rollout_routes(ag, env)

 # original (pre-consolidation) truck routes from this same rollout
        routes0 = [[by_id[cid] for cid in r if cid in by_id] for r in truck_routes_ids]
        served  = [c for r in routes0 for c in r]
        co2_pre  = (sum(route_co2_g(depot, r, cap) for r in routes0) + drone_co2_g) / 1000.0
        dist_pre = sum(route_distance(depot, r) for r in routes0)
        nserved_pre = len(served) + len(drone_ids)
        trucks_pre  = sum(1 for r in routes0 if r)

 # truck-served customers -> consolidate + local search
        cons, dropped = consolidate(depot, served, cap, n_veh)
        cons = [or_opt(depot, two_opt(depot, r)) for r in cons]

        truck_served = sum(len(r) for r in cons)
        n_served_ls  = truck_served + len(drone_ids)
        co2_ls = (sum(route_co2_g(depot, r, cap) for r in cons) + drone_co2_g) / 1000.0
        dist_ls = sum(route_distance(depot, r) for r in cons)
        mkspan_ls = max([route_time(depot, r) for r in cons] + [0.0])

        rows.append({
            "instance":      inst_name,
            "class":         inst_name[:2] if inst_name[1].isalpha() else inst_name[0],
            "method":        "GAT-MARL-LS",
            "n_customers":   n_cust,
            "n_served":      n_served_ls,
            "service_rate":  n_served_ls / n_cust,
            "total_co2_kg":  co2_ls,
            "makespan":      mkspan_ls,
            "total_distance":dist_ls,
            "n_trucks_used": len(cons),
            "dropped":       dropped,
            "bks_distance":  float(BKS.loc[inst_name,"bks_distance"]) if inst_name in BKS.index else 0.0,
 # matched pre-consolidation (same rollout) for clean before/after
            "n_served_pre":  nserved_pre,
            "co2_pre_kg":    co2_pre,
            "dist_pre":      dist_pre,
            "trucks_pre":    trucks_pre,
        })
        print(f"  [{k+1}/56] {inst_name}: served={n_served_ls}/{n_cust} "
              f"co2={co2_ls:.1f}kg trucks={len(cons)} dropped={dropped}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nSaved: {OUT} ({len(df)} rows)")
    print("\n=== GAT-MARL-LS summary (mean over 56) ===")
    print(f"  feasible service : {df.service_rate.mean()*100:.1f}%")
    print(f"  CO2 (kg)         : {df.total_co2_kg.mean():.1f}")
    print(f"  trucks used      : {df.n_trucks_used.mean():.1f}")
    print(f"  customers dropped: {df.dropped.sum()} total across 56 instances")
    print(f"  dist / BKS       : {(df.total_distance/df.bks_distance).mean():.2f}x")


if __name__ == "__main__":
    main()
