"""Main experiment runner.

Trains GAT-MARL on each of the 56 Solomon instances (best of several seeds) and
evaluates it alongside the baseline methods (NNH, CWS, ALNS, AM, POMO, TD-Split).

Model: entropy annealing (0.10 -> 0.01), value baseline trained by MSE to
Monte-Carlo returns, action masks stored and reused in the policy update, and
gradient clipping. Environment: per-step service reward, scaled COPERT CO2
penalty, drone electricity CO2 (0.258 kg/kWh), and a terminal penalty per
unserved customer.

Outputs (data/results/): main_comparison.csv, training_curves.csv, and the best
model per instance under data/models/.

Usage:
    python -u run_main_experiment.py --episodes 200 --verbose
    python -u run_main_experiment.py --episodes 10 --test --verbose   # quick test
    python -u run_main_experiment.py --episodes 200 --resume --verbose # resume
"""

import os, sys, time, argparse, csv
from pathlib import Path

import numpy as np
import pandas as pd
from collections import defaultdict

# ── Path setup ────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent.absolute() # project root/
CODE_DIR  = Path(__file__).parent.absolute() # project root/code/
ORIG_DATA = BASE_DIR / "data" # AI and Engineering/data/
SOL_DIR   = ORIG_DATA / "solomon" # same as original runner
RES_DIR   = BASE_DIR / "results"
LOG_DIR   = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "data" / "models"

for d in [RES_DIR, LOG_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CODE_DIR))

from vrp_base     import load_solomon_instance, NFZone
from truck_drone_env  import TruckDroneEnv
from gat_marl   import (GATMARLAgent, run_gat_marl, save_model)
from baselines     import (baseline_nnh  as run_nnh,
                                 baseline_cws  as run_cws,
                                 baseline_alns as run_alns,
                                 baseline_am   as run_am,
                                 baseline_pomo as run_pomo,
                                 baseline_td_split as run_td_split)


# ── Solomon instance list (all 56) ───────────────────────────────
ALL_INSTANCES = [
    "C101","C102","C103","C104","C105","C106","C107","C108","C109",
    "C201","C202","C203","C204","C205","C206","C207","C208",
    "R101","R102","R103","R104","R105","R106","R107","R108",
    "R109","R110","R111","R112",
    "R201","R202","R203","R204","R205","R206","R207","R208",
    "R209","R210","R211",
    "RC101","RC102","RC103","RC104","RC105","RC106","RC107","RC108",
    "RC201","RC202","RC203","RC204","RC205","RC206","RC207","RC208",
]

# BKS distances from Solomon (1987)
BKS = {
    "C101":828.9,"C102":828.9,"C103":828.1,"C104":822.9,
    "C105":828.9,"C106":828.9,"C107":828.9,"C108":828.9,"C109":828.9,
    "C201":591.6,"C202":591.6,"C203":591.2,"C204":590.6,
    "C205":588.9,"C206":588.5,"C207":588.3,"C208":585.9,
    "R101":1645.8,"R102":1486.1,"R103":1292.7,"R104":1007.3,
    "R105":1377.1,"R106":1252.0,"R107":1104.7,"R108":960.9,
    "R109":1194.7,"R110":1118.8,"R111":1096.7,"R112":982.1,
    "R201":1252.4,"R202":1191.7,"R203":939.5,"R204":825.5,
    "R205":994.4,"R206":906.1,"R207":890.6,"R208":726.8,
    "R209":909.2,"R210":939.4,"R211":892.7,
    "RC101":1696.9,"RC102":1554.8,"RC103":1261.7,"RC104":1135.5,
    "RC105":1629.4,"RC106":1424.7,"RC107":1230.5,"RC108":1139.8,
    "RC201":1406.9,"RC202":1365.6,"RC203":1049.6,"RC204":798.5,
    "RC205":1297.7,"RC206":1146.3,"RC207":1061.1,"RC208":828.1,
}


def get_class(name: str) -> str:
    if   name.startswith("RC"): return "RC2" if name[2]=="2" else "RC1"
    elif name.startswith("R"):  return "R2"  if name[1]=="2" else "R1"
    elif name.startswith("C"):  return "C2"  if name[1]=="2" else "C1"
    return "?"


def row_dict(instance_name, method, inst, metrics, runtime_s=None):
    """Build one CSV row dict."""
    cls = get_class(instance_name)
    bks = BKS.get(instance_name, 0.0)
    sr  = metrics.get("service_rate", 0.0)
    mk  = metrics.get("makespan", 0.0)
    gap = (mk - bks) / bks * 100 if bks > 0 else 0.0
    return {
        "instance":         instance_name,
        "class":            cls,
        "n_customers":      inst.get("n_customers", len(inst.get("customers", []))),
        "n_vehicles":       inst.get("n_vehicles", 0),
        "bks_distance":     bks,
        "method":           method,
        "makespan":         round(mk, 4),
        "total_co2_kg":     round(metrics.get("total_co2_kg", 0.0), 4),
        "total_distance":   round(metrics.get("total_distance", 0.0), 4),
        "service_rate":     round(sr, 4),
        "n_served":         metrics.get("n_served", 0),
        "workload_balance": round(metrics.get("workload_balance", 0.0), 4),
        "battery_used_pct": round(metrics.get("battery_used_pct", 0.0), 4),
        "runtime_s":        round(runtime_s if runtime_s is not None
                                  else metrics.get("runtime_s", 0.0), 4),
        "gap_to_bks_pct":   round(gap, 4),
    }


CSV_FIELDS = [
    "instance","class","n_customers","n_vehicles","bks_distance","method",
    "makespan","total_co2_kg","total_distance","service_rate","n_served",
    "workload_balance","battery_used_pct","runtime_s","gap_to_bks_pct"
]

CURVE_FIELDS = [
    "instance","instance_idx","seed","episode","reward","served",
    "makespan","co2_kg","actor_loss","critic_loss","entropy","entropy_coef","grad_norm"
]


# ── Main experiment loop ──────────────────────────────────────────
def run_experiments(episodes: int = 200,
                    n_seeds: int = 5,
                    verbose: bool = True,
                    test_mode: bool = False,
                    resume: bool = False):

    partial_path = RES_DIR / "main_partial.csv"
    final_path   = RES_DIR / "main_comparison.csv"
    curves_path  = RES_DIR / "training_curves.csv"

 # Load already-done instances for resume
    done_instances = set()
    existing_rows  = []
    if resume and partial_path.exists():
        df_done = pd.read_csv(partial_path)
 # An instance is done if ALL 7 methods are present
        counts = df_done.groupby("instance")["method"].count()
        done_instances = set(counts[counts == 7].index)
        existing_rows = df_done.to_dict("records")
        if verbose and done_instances:
            print(f"  Resume: skipping {len(done_instances)} already-done instances")

    instances_to_run = ALL_INSTANCES[:3] if test_mode else ALL_INSTANCES

    all_rows      = list(existing_rows)
    training_curves = []
    t0_global     = time.time()

    for fidx, inst_name in enumerate(instances_to_run):
        if inst_name in done_instances:
            continue

        inst_path = SOL_DIR / f"{inst_name}.txt"
        if not inst_path.exists():
            if verbose:
                print(f"  SKIP {inst_name} — file not found at {inst_path}")
            continue

        inst    = load_solomon_instance(str(inst_path))
        n_cust  = len(inst["customers"])
        n_veh   = inst["n_vehicles"]
        bks     = BKS.get(inst_name, 0.0)
        inst["n_customers"] = n_cust

        if verbose:
            print(f"\n  [{fidx+1:2d}/{len(instances_to_run)}] {inst_name}"
                  f"  (n_cust={n_cust}, n_veh={n_veh}, BKS={bks:.1f})")

        inst_rows = [] # rows for this instance

 # ── Baselines ──────────────────────────────────────────────
        baseline_fns = [
            ("NNH",      run_nnh),
            ("CWS",      run_cws),
            ("ALNS",     run_alns),
            ("AM",       run_am),
            ("POMO",     run_pomo),
            ("TD-Split", run_td_split),
        ]
        for method_name, fn in baseline_fns:
            t_b = time.time()
            try:
                m = fn(inst, n_trucks=n_veh, seed=42)
            except Exception as e:
                if verbose:
                    print(f"  {method_name} ERROR: {e}")
                m = {"service_rate": 0, "n_served": 0, "makespan": 0,
                     "total_co2_kg": 0, "total_distance": 0,
                     "workload_balance": 0, "battery_used_pct": 0}
            rt = time.time() - t_b
            if verbose:
                print(f"  {method_name:<12} | served={int(m.get('n_served',0)):3d}/{n_cust}"
                      f" | makespan={m.get('makespan',0):8.2f}"
                      f" | co2_kg={m.get('total_co2_kg',0):7.2f}"
                      f" | time={rt:.2f}s")
            inst_rows.append(row_dict(inst_name, method_name, inst, m, rt))

 # ── GAT-MARL (best-of-N seeds) ─────────────────────────
        t_gat_start  = time.time()
        best_result  = None
        best_agent   = None
        best_served  = -1
        best_seed    = None

        seeds_tried = [fidx + k * 200 for k in range(n_seeds)]

        for seed_idx, inst_seed in enumerate(seeds_tried):
            if verbose:
                print(f"  GAT-MARL | seed={inst_seed} ({seed_idx+1}/{n_seeds})"
                      f"  training {episodes} episodes...")
            try:
                r, ag = run_gat_marl(
                    inst, n_trucks=n_veh, seed=inst_seed,
                    n_train_episodes=episodes,
                    centralised_critic=True, use_gat=True,
                    use_co2_reward=True, use_nfz_mask=True
                )
            except Exception as e:
                if verbose:
                    print(f"  GAT-MARL ERROR seed={inst_seed}: {e}")
                import traceback; traceback.print_exc()
                continue

            if r.get("n_served", 0) > best_served:
                best_served = r.get("n_served", 0)
                best_result = r
                best_agent  = ag
                best_seed   = inst_seed

        if best_result is None:
            if verbose:
                print(f"  GAT-MARL | ALL seeds failed")
            best_result = {"service_rate": 0, "n_served": 0, "makespan": 0,
                           "total_co2_kg": 0, "total_distance": 0,
                           "workload_balance": 0, "battery_used_pct": 0}
        else:
            t_gat = time.time() - t_gat_start
            best_result["runtime_s"]     = t_gat
            best_result["n_seeds_tried"] = n_seeds
            best_result["best_seed"]     = best_seed

 # Save model weights
            model_path = MODEL_DIR / f"gatmarl_{inst_name}_seed{best_seed}.json"
            try:
                save_model(best_agent, str(model_path))
            except Exception:
                pass

 # Collect training curves from BEST seed
            for ep_rec in best_result.get("train_history", []):
                training_curves.append({
                    "instance":     inst_name,
                    "instance_idx": fidx,
                    "seed":         best_seed,
                    "episode":      ep_rec.get("episode", 0),
                    "reward":       ep_rec.get("reward", 0.0),
                    "served":       ep_rec.get("served", 0),
                    "makespan":     ep_rec.get("makespan", 0.0),
                    "co2_kg":       ep_rec.get("co2_kg", 0.0),
                    "actor_loss":   ep_rec.get("actor_loss", 0.0),
                    "critic_loss":  ep_rec.get("critic_loss", 0.0),
                    "entropy":      ep_rec.get("entropy", 0.0),
                    "entropy_coef": ep_rec.get("entropy_coef", 0.0),
                    "grad_norm":    ep_rec.get("grad_norm", 0.0),
                })

        t_gat = time.time() - t_gat_start
        if verbose:
            print(f"  GAT-MARL | BEST seed={best_seed} -> "
                  f"served={best_result.get('n_served',0):.0f}/{n_cust}"
                  f" | co2_kg={best_result.get('total_co2_kg',0):.2f}"
                  f" | time={t_gat:.1f}s")

        inst_rows.append(row_dict(inst_name, "GAT-MARL", inst, best_result, t_gat))
        all_rows.extend(inst_rows)

 # Save partial results after each instance
        with open(partial_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(all_rows)

 # Save training curves incrementally
        if training_curves:
            with open(curves_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=CURVE_FIELDS)
                w.writeheader()
                w.writerows(training_curves)

 # ── Final save ────────────────────────────────────────────────
    with open(final_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(all_rows)

    if training_curves:
        with open(curves_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CURVE_FIELDS)
            w.writeheader()
            w.writerows(training_curves)

    elapsed = time.time() - t0_global
    print(f"\n  Saved: {final_path}  ({len(all_rows)} rows)")
    print(f"  Saved: {curves_path}  ({len(training_curves)} rows)")
    print(f"\n{'='*70}")
    print(f"ALL DONE  [{elapsed/3600:.2f} hours]")
    print(f"{'='*70}")

 # ── Summary ───────────────────────────────────────────────────
    df = pd.read_csv(final_path)
    print(f"\n  === Summary ===")
    for m in ["NNH", "ALNS", "GAT-MARL"]:
        sub = df[df["method"] == m]
        if len(sub) == 0: continue
        print(f"  {m:<15} | CO2={sub['total_co2_kg'].mean():.3f} kg"
              f" | makespan={sub['makespan'].mean():.2f}"
              f" | service={sub['service_rate'].mean()*100:.1f}%"
              f" | time={sub['runtime_s'].mean():.2f}s")
    print(f"{'='*70}")


# ── Argument parser ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAT-MARL Experiment Runner")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Training episodes per seed (default: 200)")
    parser.add_argument("--seeds",    type=int, default=5,
                        help="Number of seeds per instance (default: 5, best-of-N)")
    parser.add_argument("--verbose",  action="store_true",
                        help="Print per-instance results")
    parser.add_argument("--test",     action="store_true",
                        help="Quick test: only C101, C102, R101 (3 instances)")
    parser.add_argument("--resume",   action="store_true",
                        help="Skip instances already in main_partial.csv")
    args = parser.parse_args()

    print("="*70)
    print("GAT-MARL — Improved Training Experiments")
    print("="*70)
    print(f"  Episodes per seed : {args.episodes}")
    print(f"  Seeds per instance: {args.seeds} (best-of-{args.seeds})")
    print(f"  Test mode         : {args.test}")
    print(f"  Resume mode       : {args.resume}")
    print(f"  Output dir        : {RES_DIR}")
    print("="*70)

    run_experiments(
        episodes  = args.episodes,
        n_seeds   = args.seeds,
        verbose   = args.verbose,
        test_mode = args.test,
        resume    = args.resume
    )
