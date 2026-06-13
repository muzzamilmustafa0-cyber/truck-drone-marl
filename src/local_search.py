"""Time-window-aware 2-opt and Or-opt local search for truck routes.

Improves the intra-route ordering of truck routes while preserving time-window
feasibility (the served customer set is unchanged). Also provides helpers to load
a trained policy, roll it out to obtain routes, and recompute route distance and
load-dependent COPERT CO2.
"""
import sys, json, math, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "code"))
from truck_drone_env import TruckDroneEnv, load_solomon_instance, COPERTModel
from gat_marl  import GATMARLAgent

SOLOMON = ROOT / "data" / "solomon"
MODELS  = ROOT / "data" / "models"
SPEED   = 50.0 # TruckState.speed_kmh default

TEST = [("C101","gatmarl_C101_seed0.json"),
        ("C201","gatmarl_C201_seed209.json"),
        ("R101","gatmarl_R101_seed417.json"),
        ("RC101","gatmarl_RC101_seed240.json")]


def load_model(path, n_agents, obs_dim, n_actions):
    with open(path) as f:
        data = json.load(f)
    ag = GATMARLAgent(n_agents=n_agents, obs_dim=obs_dim, n_actions=n_actions, seed=0)
    def restore(params, saved):
        for p, s in zip(params, saved):
            p[:] = np.array(s, dtype=np.float32)
    restore(ag.actor.params(),   data["actor_W"])
    restore(ag.critic.params(),  data["critic_W"])
    restore(ag.encoder.params(), data["encoder_W"])
    return ag


def edist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def route_distance(depot, route):
    if not route: return 0.0
    nodes = [depot] + route + [depot]
    return sum(edist(nodes[i], nodes[i+1]) for i in range(len(nodes)-1))


def route_co2_g(depot, route, capacity):
    """Replay route leg-by-leg with load-dependent COPERT, as in the env."""
    if not route: return 0.0
    load = 0.0
    co2 = 0.0
    cur = depot
    for c in route:
        d = edist(cur, c)
        d_km = d / 1000.0 if d > 100 else d
        lr = load / capacity if capacity else 0.0
        co2 += COPERTModel.co2_for_trip(d_km, SPEED, lr)
        load += c.demand
        cur = c
 # return to depot
    d = edist(cur, depot); d_km = d / 1000.0 if d > 100 else d
    co2 += COPERTModel.co2_for_trip(d_km, SPEED, load / capacity if capacity else 0.0)
    return co2


def is_tw_feasible(depot, route):
    """All customers reached within their time window (env arithmetic)."""
    t = 0.0; cur = depot
    for c in route:
        t += edist(cur, c) / SPEED
        arrive = max(t, c.ready_time)
        if arrive > c.due_time:
            return False
        t = arrive + c.service_time
        cur = c
    return True


def two_opt(depot, route):
    """TW-aware 2-opt: reverse segments; accept if shorter and feasible."""
    improved = True
    best = route[:]
    while improved:
        improved = False
        n = len(best)
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if route_distance(depot, cand) + 1e-9 < route_distance(depot, best) \
                   and is_tw_feasible(depot, cand):
                    best = cand; improved = True
    return best


def or_opt(depot, route, max_seg=3):
    """TW-aware Or-opt: relocate chains of length 1..max_seg."""
    improved = True
    best = route[:]
    while improved:
        improved = False
        n = len(best)
        for seg in range(1, max_seg + 1):
            for i in range(n - seg + 1):
                chain = best[i:i+seg]
                rest  = best[:i] + best[i+seg:]
                for k in range(len(rest) + 1):
                    cand = rest[:k] + chain + rest[k:]
                    if route_distance(depot, cand) + 1e-9 < route_distance(depot, best) \
                       and is_tw_feasible(depot, cand):
                        best = cand; improved = True; break
                if improved: break
            if improved: break
    return best


def rollout_routes(agent, env, n_rollouts=5):
    """Best-of-N rollout; return (best_env_metrics, truck_routes as id-lists)."""
    best_served = -1; best_routes = None; best_m = None
    for _ in range(n_rollouts):
        env.reset()
        graph = env.get_graph_representation()
        masks = {i: env.get_action_mask(i) for i in range(env.n_agents)}
        done = False; step = 0
        while not done and step < 200:
            actions, _, _, _, _ = agent.select_actions(
                {i: env._get_obs(i) for i in range(env.n_agents)}, graph, masks,
                deterministic=False)
            _, _, dones, _ = env.step(actions)
            done = dones.get("__all__", False)
            graph = env.get_graph_representation()
            masks = {i: env.get_action_mask(i) for i in range(env.n_agents)}
            step += 1
        m = env.get_metrics()
        if m.get("n_served", 0) > best_served:
            best_served = m.get("n_served", 0)
            best_routes = [list(t.route) for t in env.trucks]
 # capture drone-served ids + drone CO2 from the SAME (best) rollout
            best_drone_ids = [c.id for c in env.customers
                              if c.served_by is not None and c.served_by >= env.n_trucks]
            truck_co2 = sum(t.co2_emitted for t in env.trucks)
            best_drone_co2_g = max(m["total_co2_kg"] * 1000.0 - truck_co2, 0.0)
            best_m = m
    return best_m, best_routes, best_drone_ids, best_drone_co2_g


def main():
    print(f"{'inst':6s} {'served':>7s} {'dist0':>8s} {'distLS':>8s} {'cut%':>6s} "
          f"{'co2_0':>7s} {'co2_LS':>7s} {'cut%':>6s} {'feas_ok':>8s}")
    print("-" * 78)
    for inst_name, model_file in TEST:
        inst = load_solomon_instance(str(SOLOMON / f"{inst_name}.txt"))
        depot = inst["depot"]; cap = inst["capacity"]
        by_id = {c.id: c for c in inst["customers"]}
        n_veh = inst.get("n_vehicles", 25)

        np.random.seed(0); random.seed(0)
        env = TruckDroneEnv(inst, n_trucks=n_veh, seed=0)
        env.reset()
        agent = load_model(str(MODELS / model_file),
                              env.n_agents, list({i: env._get_obs(i) for i in range(env.n_agents)}.values())[0].shape[0],
                              len(env.get_action_mask(0)))

        m, truck_routes_ids, _drone_ids, drone_co2_g = rollout_routes(agent, env)

 # Build Customer-object routes, drop depot id (0) if present
        routes = [[by_id[cid] for cid in r if cid in by_id] for r in truck_routes_ids]

        dist0 = sum(route_distance(depot, r) for r in routes)
        co2_0 = sum(route_co2_g(depot, r, cap) for r in routes) # recomputed truck grams

 # Apply local search (2-opt then Or-opt) per route
        improved = []
        feas_ok = True
        for r in routes:
            if len(r) >= 2:
                r2 = two_opt(depot, r)
                r2 = or_opt(depot, r2)
                if not is_tw_feasible(depot, r2):
                    feas_ok = False
                improved.append(r2)
            else:
                improved.append(r)

        distLS = sum(route_distance(depot, r) for r in improved)
        co2LS  = sum(route_co2_g(depot, r, cap) for r in improved)

 # Total CO2 (kg) = improved truck + unchanged drone
        total_co2_ls_kg = (co2LS + drone_co2_g) / 1000.0
        total_co2_0_kg  = (co2_0 + drone_co2_g) / 1000.0

        served = int(m["n_served"])
        dcut = 100 * (dist0 - distLS) / dist0 if dist0 else 0
        ccut = 100 * (total_co2_0_kg - total_co2_ls_kg) / total_co2_0_kg if total_co2_0_kg else 0
        print(f"{inst_name:6s} {served:7d} {dist0:8.0f} {distLS:8.0f} {dcut:5.1f}% "
              f"{total_co2_0_kg:7.1f} {total_co2_ls_kg:7.1f} {ccut:5.1f}% {str(feas_ok):>8s}")


if __name__ == "__main__":
    main()
