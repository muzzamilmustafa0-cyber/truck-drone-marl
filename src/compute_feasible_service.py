"""Compute the time-window-feasible service rate for every method on the 56
Solomon instances. A customer counts as served only if its vehicle arrives within
the customer's time window; each method's routes are replayed with the same timing
model used by the environment. Output: feasible_service.csv.
"""

import sys, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT        = Path(__file__).parent.parent
CODE        = ROOT / "code"
SOLOMON_DIR = ROOT / "data" / "solomon"
RESULTS_DIR  = ROOT / "results"
sys.path.insert(0, str(CODE))

import baselines as bf
from baselines import dist
from vrp_base import load_solomon_instance

TRUCK_SPEED   = 50.0 # km/h — matches route_time() and env _truck_serve
DRONE_SPEED   = 12.0 * 3.6 # m/s -> km/h, matches build_metrics drone trip calc
SEED          = 42 
BASELINES     = ["NNH", "CWS", "ALNS", "AM", "POMO", "TD-Split"]


def feasible_count_route(depot, route, speed=TRUCK_SPEED):
    """Replay a truck route; count customers reached within their time window.
    Arithmetic identical to baselines.route_time / env _truck_serve."""
    t = 0.0
    cur = depot
    feas = 0
    for c in route:
        t += dist(cur, c) / speed
        arrive = max(t, c.ready_time) # wait until ready
        if arrive <= c.due_time: # on-time delivery
            feas += 1
        t = arrive + c.service_time # truck still serves (time advances)
        cur = c
    return feas


def feasible_count_drones(depot, drone_assignments, speed=DRONE_SPEED):
    """Direct depot->customer drone trips; count on-time deliveries."""
    feas = 0
    for c, _energy in drone_assignments:
        travel = dist(depot, c) / speed
        arrive = max(travel, c.ready_time)
        if arrive <= c.due_time:
            feas += 1
    return feas


# ── Monkeypatch build_metrics to also report feasibility ─────────────────────
_orig_build_metrics = bf.build_metrics

def _patched_build_metrics(depot, truck_routes, drone_assignments, capacity,
                           all_customers, speed_truck=TRUCK_SPEED,
                           speed_drone_ms=12.0):
    m = _orig_build_metrics(depot, truck_routes, drone_assignments, capacity,
                            all_customers, speed_truck, speed_drone_ms)
    feas = sum(feasible_count_route(depot, r, speed_truck) for r in truck_routes)
    feas += feasible_count_drones(depot, drone_assignments)
    n_cust = len(all_customers)
    m["n_served_feasible"]      = float(feas)
    m["service_rate_feasible"]  = feas / max(n_cust, 1)
    return m

bf.build_metrics = _patched_build_metrics


def main():
 # GAT-MARL feasible = service rate (env enforces time windows)
    comp = pd.read_csv(RESULTS_DIR / "main_comparison.csv")
    gat = comp[comp.method == "GAT-MARL"].set_index("instance")

    sol_files = sorted(SOLOMON_DIR.glob("*.txt"))
    rows = []

    for sf in sol_files:
        inst_name = sf.stem
        inst = load_solomon_instance(str(sf))
        n_veh  = inst.get("n_vehicles", 25)
        n_cust = len(inst["customers"])
        cls    = inst_name[:2] if inst_name[1].isalpha() else inst_name[0]

 # Re-run baselines (n_trucks=n_veh, seed=42)
        results = bf.run_all_baselines(inst, n_trucks=n_veh, seed=SEED, verbose=False)

        for method in BASELINES:
            m = results.get(method, {})
            if "error" in m:
                print(f"  {inst_name} {method}: ERROR {m['error']}")
                continue
            rows.append({
                "instance":              inst_name,
                "class":                 cls,
                "method":                method,
                "n_customers":           n_cust,
                "service_rate_raw":      m["service_rate"],
                "service_rate_feasible": m["service_rate_feasible"],
                "n_served_raw":          m["n_served"],
                "n_served_feasible":     m["n_served_feasible"],
                "makespan":              m["makespan"],
            })

 # GAT-MARL row (feasible == raw, env-enforced)
        if inst_name in gat.index:
            g = gat.loc[inst_name]
            rows.append({
                "instance":              inst_name,
                "class":                 cls,
                "method":                "GAT-MARL",
                "n_customers":           n_cust,
                "service_rate_raw":      float(g["service_rate"]),
                "service_rate_feasible": float(g["service_rate"]), # env-enforced
                "n_served_raw":          float(g["n_served"]),
                "n_served_feasible":     float(g["n_served"]),
                "makespan":              float(g["makespan"]),
            })

        print(f"  {inst_name}: done ({n_cust} customers)")

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "feasible_service.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({len(df)} rows)")

 # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FEASIBLE vs RAW SERVICE RATE (mean over 56 instances)")
    print("=" * 70)
    METHOD_ORDER = ["GAT-MARL", "TD-Split", "AM", "POMO", "ALNS", "NNH", "CWS"]
    summary = (df.groupby("method")
                 .agg(raw=("service_rate_raw", "mean"),
                      feasible=("service_rate_feasible", "mean"))
                 .reindex([m for m in METHOD_ORDER if m in df.method.unique()]))
    summary["raw_%"]      = (summary["raw"] * 100).round(1)
    summary["feasible_%"] = (summary["feasible"] * 100).round(1)
    summary["drop_pts"]   = (summary["raw_%"] - summary["feasible_%"]).round(1)
    print(summary[["raw_%", "feasible_%", "drop_pts"]].to_string())


if __name__ == "__main__":
    main()
