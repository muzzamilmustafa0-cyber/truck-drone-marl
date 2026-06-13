"""Recompute CO2 for every method under a single consistent accounting:
load-dependent truck COPERT emissions plus drone electricity
(Wh / 1000 * 0.258 kg CO2 per kWh). Outputs: harmonized_co2.csv and
harmonized_pareto.csv.
"""
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "code"))
import baselines as bf
from local_search import route_co2_g
from vrp_base import load_solomon_instance

SOLOMON = ROOT / "data" / "solomon"
RESULTS = ROOT / "results"
ITALY_GRID = 0.258
SEED = 42
BASELINES = ["NNH", "CWS", "ALNS", "AM", "POMO", "TD-Split"]

# ── Monkeypatch build_metrics: add harmonized CO2 (truck route_co2_g + drone elec) ──
_orig = bf.build_metrics
def _patched(depot, truck_routes, drone_assignments, capacity, all_customers,
             speed_truck=50.0, speed_drone_ms=12.0):
    m = _orig(depot, truck_routes, drone_assignments, capacity, all_customers,
              speed_truck, speed_drone_ms)
    truck_g = sum(route_co2_g(depot, r, capacity) for r in truck_routes) # grams
    drone_kg = sum((e / 1000.0) * ITALY_GRID for _c, e in drone_assignments) # kg
    m["co2_harmonized_kg"] = truck_g / 1000.0 + drone_kg
    return m
bf.build_metrics = _patched


def main():
    feas = pd.read_csv(RESULTS / "feasible_service.csv")
    hyb  = pd.read_csv(RESULTS / "hybrid_results.csv")

    rows = []
    for sf in sorted(SOLOMON.glob("*.txt")):
        inst_name = sf.stem
        inst = load_solomon_instance(str(sf))
        n_veh = inst.get("n_vehicles", 25)
        res = bf.run_all_baselines(inst, n_trucks=n_veh, seed=SEED, verbose=False)
        for method in BASELINES:
            m = res.get(method, {})
            if "error" in m: continue
            rows.append({"instance": inst_name, "method": method,
                         "co2_harmonized_kg": m["co2_harmonized_kg"]})
        print(f"  {inst_name} done")

    base = pd.DataFrame(rows)
    base.to_csv(RESULTS / "harmonized_co2.csv", index=False)

 # ── Assemble per-method comparison: feasible service + harmonized CO2 ────────
    svc = feas.groupby("method")["service_rate_feasible"].mean() * 100
    bco2 = base.groupby("method")["co2_harmonized_kg"].mean()

    table = []
    for mth in BASELINES:
        table.append((mth, round(svc[mth], 1), round(bco2[mth], 1)))
 # GAT-MARL-LS and raw GAT-MARL from hybrid csv (already harmonized definition)
    table.append(("GAT-MARL-LS", round(hyb.service_rate.mean()*100, 1),
                  round(hyb.total_co2_kg.mean(), 1)))
    table.append(("GAT-MARL", round(hyb.n_served_pre.sum()/hyb.n_customers.sum()*100, 1),
                  round(hyb.co2_pre_kg.mean(), 1)))

    df = pd.DataFrame(table, columns=["method", "feas_service_%", "co2_harmonized_kg"])
    df = df.sort_values("feas_service_%", ascending=False)
    df.to_csv(RESULTS / "harmonized_pareto.csv", index=False)

    print("\n" + "=" * 60)
    print("HARMONIZED: feasible service vs CO2 (truck COPERT + drone elec)")
    print("=" * 60)
    print(df.to_string(index=False))

 # Pareto domination for GAT-MARL-LS
    gs, gc = df[df.method == "GAT-MARL-LS"].iloc[0][["feas_service_%", "co2_harmonized_kg"]]
    dom = [r.method for _, r in df.iterrows()
           if r.method != "GAT-MARL-LS" and gs >= r["feas_service_%"] and gc <= r["co2_harmonized_kg"]]
    S = df["feas_service_%"].values; C = df["co2_harmonized_kg"].values; M = df["method"].values
    nd = []
    for _, r in df.iterrows():
        rs, rc = r["feas_service_%"], r["co2_harmonized_kg"]
        dominated = any((S[i] >= rs and C[i] <= rc and (S[i] > rs or C[i] < rc))
                        for i in range(len(df)) if M[i] != r.method)
        if not dominated: nd.append(r.method)
    print(f"\nGAT-MARL-LS dominates: {dom}")
    print(f"Pareto-optimal set: {nd}")


if __name__ == "__main__":
    main()
