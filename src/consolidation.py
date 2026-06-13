"""Fill-first inter-route consolidation for truck routes.

Re-packs a set of customers into the minimum number of capacity- and
time-window-feasible routes using nearest-neighbour insertion, reducing the
number of vehicles and the total distance travelled.
"""
import sys, random, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "code"))
from local_search import (load_model, edist, route_distance,
                                     route_co2_g, is_tw_feasible, two_opt, or_opt,
                                     rollout_routes, SPEED)
from truck_drone_env import TruckDroneEnv, load_solomon_instance

SOLOMON = ROOT / "data" / "solomon"
MODELS  = ROOT / "data" / "models"
TEST = [("C101","gatmarl_C101_seed0.json"),
        ("C201","gatmarl_C201_seed209.json"),
        ("R101","gatmarl_R101_seed417.json"),
        ("RC101","gatmarl_RC101_seed240.json")]


def consolidate(depot, served, capacity, n_trucks):
    """Fill-first NN: pack each truck full before opening the next."""
    unserved = list(served)
    routes = []
    while unserved and len(routes) < n_trucks:
        route = []; load = 0.0; t = 0.0; cur = depot
        while True:
            best = None; bestd = math.inf
            for c in unserved:
                if load + c.demand > capacity: continue
                arrive = max(t + edist(cur, c) / SPEED, c.ready_time)
                if arrive > c.due_time: continue
                if edist(cur, c) < bestd:
                    bestd = edist(cur, c); best = c
            if best is None: break
            route.append(best); load += best.demand
            t = max(t + edist(cur, best) / SPEED, best.ready_time) + best.service_time
            cur = best; unserved.remove(best)
        if not route: break
        routes.append(route)
    return routes, len(unserved) # unplaced = dropped customers


def main():
    print(f"{'inst':6s} {'trk0':>4s} {'trkC':>4s} {'dist0':>7s} {'distC':>7s} {'cut%':>5s} "
          f"{'co2_0':>6s} {'co2_C':>6s} {'cut%':>5s} {'dropped':>7s}")
    print("-" * 74)
    for inst_name, mf in TEST:
        inst = load_solomon_instance(str(SOLOMON / f"{inst_name}.txt"))
        depot = inst["depot"]; cap = inst["capacity"]
        by_id = {c.id: c for c in inst["customers"]}
        n_veh = inst.get("n_vehicles", 25)

        np.random.seed(0); random.seed(0)
        env = TruckDroneEnv(inst, n_trucks=n_veh, seed=0); env.reset()
        od = env._get_obs(0).shape[0]
        ag = load_model(str(MODELS / mf), env.n_agents, od, len(env.get_action_mask(0)))
        m, truck_routes_ids, _drone_ids, _drone_co2 = rollout_routes(ag, env)

        routes0 = [[by_id[cid] for cid in r if cid in by_id] for r in truck_routes_ids]
        trk0  = sum(1 for r in routes0 if r)
        dist0 = sum(route_distance(depot, r) for r in routes0)
        co2_0 = sum(route_co2_g(depot, r, cap) for r in routes0) / 1000.0

        served = [c for r in routes0 for c in r]
        cons, dropped = consolidate(depot, served, cap, n_veh)
        cons = [two_opt(depot, r) for r in cons]
        cons = [or_opt(depot, r) for r in cons]
        feas = all(is_tw_feasible(depot, r) for r in cons)

        trkC  = len(cons)
        distC = sum(route_distance(depot, r) for r in cons)
        co2_C = sum(route_co2_g(depot, r, cap) for r in cons) / 1000.0

        dcut = 100*(dist0-distC)/dist0 if dist0 else 0
        ccut = 100*(co2_0-co2_C)/co2_0 if co2_0 else 0
        print(f"{inst_name:6s} {trk0:4d} {trkC:4d} {dist0:7.0f} {distC:7.0f} {dcut:4.0f}% "
              f"{co2_0:6.1f} {co2_C:6.1f} {ccut:4.0f}% {dropped:7d}  feas={feas}")


if __name__ == "__main__":
    main()
