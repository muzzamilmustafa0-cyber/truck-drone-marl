"""
TruckDroneEnvBase — Improved reward shaping for GAT-MARL.

Changes (vrp_base.py):
  1. Per-step service reward raised to +5.0 (was +1.0/+1.2)
  2. CO2 penalty properly scaled so it matters (~10% of service reward)
  3. Drone CO2 counted via Italy grid electricity (0.258 kg CO2/kWh)
  4. Terminal reward: explicit -1.0 per unserved customer penalty
  5. get_metrics() now includes drone_co2_kg and total combined CO2
  6. Late-arrival penalty reduced (-1.0 instead of -2.0) — allows time-window
     violation learning to be gradual (not cliff-edge penalty)

All physics models (COPERT5, Stolaroff 2018) unchanged.
All kinematics unchanged.
"""

import math
import random
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import IntEnum

# Re-use unchanged data structures and physics 
from vrp_base import (
    AgentType, Customer, NFZone, TruckState, DroneState,
    COPERTModel, DroneEnergyModel,
    load_solomon_instance, load_cvrp_instance
)

# Italy grid average CO2 intensity (ENTSO-E 2023, Italy)
ITALY_GRID_KG_CO2_PER_KWH = 0.258 # kg CO2 per kWh


class TruckDroneEnv:
    """
    Improved truck-drone environment with stronger reward shaping.

    Key reward changes:
    - Per-step customer service reward: +5.0 (was +1.0 truck / +1.2 drone)
    - CO2 penalty: -1.5 * co2_kg per truck trip (was -0.001 * co2_g/1000)
    - Drone energy penalty: -0.5 * drone_co2_kg per delivery (was -0.002 * Wh)
    - Terminal: -1.0 per unserved customer (new), keeps existing multi-obj terms
    """

 # Reward weights (as in the base model)
    W_MAKESPAN  = 1.0
    W_CO2       = 0.5
    W_BALANCE   = 0.3
    W_BATTERY   = 0.2

    def __init__(self,
                 instance: Dict,
                 n_trucks: int = 3,
                 n_drones_per_truck: int = 1,
                 nfz_list: Optional[List[NFZone]] = None,
                 seed: int = 42):

        self.instance = instance
        self.n_trucks = n_trucks
        self.n_drones_per_truck = n_drones_per_truck
        self.n_drones = n_trucks * n_drones_per_truck
        self.n_agents = n_trucks + self.n_drones
        self.nfz_list = nfz_list or []
        self.rng = random.Random(seed)
        np.random.seed(seed)

        self.depot      = instance["depot"]
        self.customers  = instance["customers"]
        self.n_cust     = len(self.customers)
        self.capacity   = instance["capacity"]

        all_x = [c.x for c in self.customers] + [self.depot.x]
        all_y = [c.y for c in self.customers] + [self.depot.y]
        self.x_min, self.x_max = min(all_x), max(all_x)
        self.y_min, self.y_max = min(all_y), max(all_y)
        self.coord_scale = max(self.x_max - self.x_min, self.y_max - self.y_min, 1e-9)

        self.k_nearest = 20
        self.obs_dim = 6 + self.k_nearest * 7 + self.n_agents * 4

 # Ablation flag: can set to 0.0 to disable CO2 reward
        self.W_CO2_STEP = 1.0

        self.reset()

    def reset(self) -> Dict[int, np.ndarray]:
        for c in self.customers:
            c.served    = False
            c.served_by = None
            c.served_at = None

        self.trucks = []
        for i in range(self.n_trucks):
            self.trucks.append(TruckState(
                id=i, x=self.depot.x, y=self.depot.y,
                capacity=self.capacity, load=0.0
            ))

        self.drones = []
        for tid in range(self.n_trucks):
            for did in range(self.n_drones_per_truck):
                drone_id = tid * self.n_drones_per_truck + did
                self.drones.append(DroneState(
                    id=drone_id, truck_id=tid,
                    x=self.depot.x, y=self.depot.y
                ))

        self.time = 0.0
        self.step_count = 0
        self.done_agents = set()

        return {i: self._get_obs(i) for i in range(self.n_agents)}

 # ── Observation (vectorized) ─────────────────────────────
    def _norm_x(self, x): return (x - self.x_min) / self.coord_scale
    def _norm_y(self, y): return (y - self.y_min) / self.coord_scale

    def _get_obs(self, agent_id: int) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        ptr = 0

        is_truck = agent_id < self.n_trucks
        if is_truck:
            t = self.trucks[agent_id]
            obs[ptr:ptr+6] = [
                self._norm_x(t.x), self._norm_y(t.y),
                t.load / self.capacity, t.time / 1500.0,
                t.total_distance / (self.coord_scale * 10),
                float(t.done)
            ]
        else:
            d_idx = agent_id - self.n_trucks
            d = self.drones[d_idx]
            obs[ptr:ptr+6] = [
                self._norm_x(d.x), self._norm_y(d.y),
                d.battery_wh / d.battery_max, d.time / 1500.0,
                d.total_distance / (self.coord_scale * 10),
                float(d.done)
            ]
        ptr += 6

        cur_x = self.trucks[agent_id].x if is_truck else self.drones[agent_id - self.n_trucks].x
        cur_y = self.trucks[agent_id].y if is_truck else self.drones[agent_id - self.n_trucks].y

        unserved = [c for c in self.customers if not c.served]
        dists = sorted([(math.hypot(c.x - cur_x, c.y - cur_y), i, c)
                        for i, c in enumerate(unserved)])[:self.k_nearest]

        for i, (dist, _, c) in enumerate(dists):
            base = ptr + i * 7
            obs[base:base+7] = [
                self._norm_x(c.x), self._norm_y(c.y),
                c.demand / self.capacity,
                c.ready_time / 1500.0,
                c.due_time / 1500.0,
                dist / self.coord_scale,
                1.0 if not self._nfz_blocks(cur_x, cur_y, c.x, c.y) else 0.0
            ]
        ptr += self.k_nearest * 7

        for i in range(self.n_agents):
            base = ptr + i * 4
            if i < self.n_trucks:
                t2 = self.trucks[i]
                obs[base:base+4] = [self._norm_x(t2.x), self._norm_y(t2.y),
                                    t2.load/self.capacity, float(t2.done)]
            else:
                d2 = self.drones[i - self.n_trucks]
                obs[base:base+4] = [self._norm_x(d2.x), self._norm_y(d2.y),
                                    d2.battery_wh/d2.battery_max, float(d2.done)]
        return obs

    def _nfz_blocks(self, x1, y1, x2, y2) -> bool:
        return any(nfz.path_intersects(x1, y1, x2, y2) for nfz in self.nfz_list)

 # ── Action mask (vectorized — capacity + drone battery + NFZ) ──
    def get_action_mask(self, agent_id: int) -> np.ndarray:
        mask = np.zeros(self.n_cust + 1, dtype=bool)
        mask[0] = True # depot always valid

        is_truck = agent_id < self.n_trucks
        if is_truck:
            t = self.trucks[agent_id]
            if t.done: return mask
            for i, c in enumerate(self.customers):
                if c.served: continue
                if c.demand > (self.capacity - t.load): continue
                mask[i + 1] = True
        else:
            d_idx = agent_id - self.n_trucks
            d = self.drones[d_idx]
            if d.done: return mask
            for i, c in enumerate(self.customers):
                if c.served: continue
                dist = math.hypot(c.x - d.x, c.y - d.y) / 1000.0
                if not DroneEnergyModel.feasible(dist, d.speed_ms, c.demand/100.0, d.battery_wh):
                    continue
                if self._nfz_blocks(d.x, d.y, c.x, c.y):
                    continue
                mask[i + 1] = True
        return mask

 # ── Step ─────────────────────────────────────────────────────
    def step(self, actions: Dict[int, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        rewards = {i: 0.0 for i in range(self.n_agents)}
        infos   = {}

        for agent_id, action in actions.items():
            if agent_id in self.done_agents:
                continue

            is_truck = agent_id < self.n_trucks

            if action == 0:
                if is_truck:
                    self._truck_goto_depot(agent_id)
                else:
                    self._drone_goto_depot(agent_id - self.n_trucks)
                self.done_agents.add(agent_id)
                continue

            cust_idx = action - 1
            if cust_idx < 0 or cust_idx >= self.n_cust:
                continue
            c = self.customers[cust_idx]
            if c.served:
                rewards[agent_id] -= 5.0 # penalty for invalid re-serve
                continue

            if is_truck:
                r = self._truck_serve(agent_id, c)
            else:
                r = self._drone_serve(agent_id - self.n_trucks, c)

            rewards[agent_id] += r

        all_served = all(c.served for c in self.customers)
        all_agents_done = (all(t.done for t in self.trucks) and
                           all(d.done for d in self.drones))

        if all_served:
            for i in range(self.n_trucks):
                if not self.trucks[i].done: self._truck_goto_depot(i)
            for i in range(self.n_drones):
                if not self.drones[i].done: self._drone_goto_depot(i)

        done_flag = all_served or all_agents_done

        if done_flag:
            terminal_reward = self._compute_terminal_reward()
            for i in range(self.n_agents):
                rewards[i] += terminal_reward

        self.step_count += 1
        obs   = {i: self._get_obs(i) for i in range(self.n_agents)}
        dones = {i: done_flag for i in range(self.n_agents)}
        dones["__all__"] = done_flag

        return obs, rewards, dones, infos

 # ── Truck movement (reward strengthened) ─────────────────────
    def _truck_serve(self, truck_id: int, customer: Customer) -> float:
        t = self.trucks[truck_id]
        dist = math.hypot(customer.x - t.x, customer.y - t.y)
        dist_km = dist / 1000.0 if dist > 100 else dist

        travel_time = dist / t.speed_kmh
        arrive_time = t.time + travel_time
        arrive_time = max(arrive_time, customer.ready_time)

        if arrive_time > customer.due_time:
 # Softer late penalty: only -1.0 ( -2.0)
 # This allows the model to still serve late customers early in training
            return -1.0

        load_ratio = t.load / self.capacity
        co2_g = COPERTModel.co2_for_trip(
            dist_km if dist < 200 else dist/1000.0,
            t.speed_kmh, load_ratio
        )
        co2_kg = co2_g / 1000.0

        t.x, t.y = customer.x, customer.y
        t.time = arrive_time + customer.service_time
        t.load += customer.demand
        t.total_distance += dist
        t.co2_emitted += co2_g
        t.route.append(customer.id)

        customer.served    = True
        customer.served_by = truck_id
        customer.served_at = t.time

 # service reward +5.0, CO2 penalty scaled to ~5-10% of service reward
 # For typical 3km truck trip: co2 ~0.3 kg -> penalty -0.45 -> ~9% of +5.0
        r = 5.0 - self.W_CO2_STEP * 1.5 * co2_kg
        return r

    def _truck_goto_depot(self, truck_id: int):
        t = self.trucks[truck_id]
        dist = math.hypot(self.depot.x - t.x, self.depot.y - t.y)
        dist_km = dist / 1000.0 if dist > 100 else dist
        travel_time = dist / t.speed_kmh
        co2_g = COPERTModel.co2_for_trip(dist_km, t.speed_kmh, t.load / self.capacity)
        t.x, t.y = self.depot.x, self.depot.y
        t.time += travel_time
        t.total_distance += dist
        t.co2_emitted += co2_g
        t.done = True

 # ── Drone movement (reward strengthened) ─────────────────────
    def _drone_serve(self, drone_idx: int, customer: Customer) -> float:
        d = self.drones[drone_idx]
        dist = math.hypot(customer.x - d.x, customer.y - d.y)
        dist_km = dist / 1000.0 if dist > 100 else dist

        if self._nfz_blocks(d.x, d.y, customer.x, customer.y):
            return -3.0

        energy_wh = DroneEnergyModel.energy_wh(dist_km, d.speed_ms, d.payload_kg)
        if energy_wh > d.battery_wh:
            return -4.0

        travel_time = dist / (d.speed_ms * 3.6)

        d.x, d.y = customer.x, customer.y
        d.time += travel_time
        d.total_distance += dist
        d.battery_wh -= energy_wh
        d.energy_used += energy_wh
        d.payload_kg = max(0.0, d.payload_kg - customer.demand / 1000.0)
        d.route.append(customer.id)

        customer.served    = True
        customer.served_by = self.n_trucks + drone_idx
        customer.served_at = d.time

 # Drone CO2 from grid electricity (Italy: 0.258 kg CO2/kWh)
        drone_co2_kg = energy_wh / 1000.0 * ITALY_GRID_KG_CO2_PER_KWH

 # same +5.0 base as trucks; drone CO2 penalty ~100x smaller than truck
 # For typical 1km drone trip: 6.23 Wh -> 0.0016 kg CO2 -> penalty -0.0024
 # This naturally incentivises drones over trucks for short-range deliveries
        r = 5.0 - self.W_CO2_STEP * 1.5 * drone_co2_kg
        return r

    def _drone_goto_depot(self, drone_idx: int):
        d = self.drones[drone_idx]
        dist = math.hypot(self.depot.x - d.x, self.depot.y - d.y)
        dist_km = dist / 1000.0 if dist > 100 else dist
        energy_wh = DroneEnergyModel.energy_wh(dist_km, d.speed_ms, 0.0)
        travel_time = dist / (d.speed_ms * 3.6)
        d.x, d.y = self.depot.x, self.depot.y
        d.time += travel_time
        d.total_distance += dist
        d.battery_wh -= energy_wh
        d.energy_used += energy_wh
        d.done = True

 # ── Terminal reward (unserved penalty added) ──────────────────
    def _compute_terminal_reward(self) -> float:
        truck_times = [t.time for t in self.trucks]
        drone_times = [d.time for d in self.drones]
        makespan = max(truck_times + drone_times)
        makespan_norm = makespan / 1500.0

        total_co2_g = sum(t.co2_emitted for t in self.trucks)
        drone_energy_kwh = sum(d.energy_used for d in self.drones) / 1000.0
        drone_co2_kg = drone_energy_kwh * ITALY_GRID_KG_CO2_PER_KWH
        total_co2_kg = total_co2_g / 1000.0 + drone_co2_kg
        co2_norm = min(total_co2_kg / 100.0, 1.0)

        all_times = truck_times + drone_times
        balance = np.std(all_times) / (np.mean(all_times) + 1e-9)
        balance_norm = min(balance, 1.0)

        battery_ok = sum(1 for d in self.drones if d.battery_wh / d.battery_max >= 0.2)
        battery_ratio = battery_ok / max(self.n_drones, 1)

        served = sum(1 for c in self.customers if c.served)
        service_rate = served / self.n_cust
        n_unserved = self.n_cust - served

        reward = (
            - self.W_MAKESPAN  * makespan_norm
            - self.W_CO2       * co2_norm
            - self.W_BALANCE   * balance_norm
            + self.W_BATTERY   * battery_ratio
            + 10.0 * service_rate # strong service incentive (same as)
            - 1.0  * n_unserved # NEW: explicit penalty per unserved customer
        )
        return reward

 # ── Metrics (drone CO2 now included) ─────────────────────────
    def get_metrics(self) -> Dict[str, float]:
        truck_times  = [t.time for t in self.trucks]
        drone_times  = [d.time for d in self.drones]
        all_times    = truck_times + drone_times

        makespan        = max(all_times)
        truck_co2_g     = sum(t.co2_emitted for t in self.trucks)
        drone_energy_kwh = sum(d.energy_used for d in self.drones) / 1000.0
        drone_co2_kg    = drone_energy_kwh * ITALY_GRID_KG_CO2_PER_KWH
        total_co2_kg    = truck_co2_g / 1000.0 + drone_co2_kg

        total_dist      = (sum(t.total_distance for t in self.trucks) +
                           sum(d.total_distance for d in self.drones))
        battery_used_pct = np.mean([(d.energy_used / d.battery_max) * 100
                                    for d in self.drones])
        balance_std     = float(np.std(all_times))
        served          = sum(1 for c in self.customers if c.served)
        n_trucks_used   = sum(1 for t in self.trucks if len(t.route) > 0)
        n_drones_used   = sum(1 for d in self.drones if len(d.route) > 0)

        return {
            "makespan":           makespan,
            "total_co2_grams":    truck_co2_g,
            "total_co2_kg":       total_co2_kg, # now includes drone electricity CO2
            "truck_co2_kg":       truck_co2_g / 1000.0,
            "drone_co2_kg":       drone_co2_kg,
            "total_distance":     total_dist,
            "battery_used_pct":   battery_used_pct,
            "workload_balance":   balance_std,
            "service_rate":       served / self.n_cust,
            "n_served":           served,
            "n_customers":        self.n_cust,
            "n_trucks_used":      n_trucks_used,
            "n_drones_used":      n_drones_used,
            "truck_co2_per_km":   truck_co2_g / max(
                                    sum(t.total_distance for t in self.trucks), 1),
        }

 # ── Graph representation (vectorized) ────────────────────
    def get_graph_representation(self) -> Dict:
        nodes = []
        nodes.append({"type": 0, "x": self._norm_x(self.depot.x),
                       "y": self._norm_y(self.depot.y),
                       "demand": 0, "ready": 0, "due": 1.0, "service": 0,
                       "load": 0, "battery": 1.0})
        for c in self.customers:
            nodes.append({
                "type": 1,
                "x": self._norm_x(c.x), "y": self._norm_y(c.y),
                "demand": c.demand / self.capacity,
                "ready": c.ready_time / 1500.0,
                "due":   c.due_time / 1500.0,
                "service": c.service_time / 100.0,
                "load": 0, "battery": 1.0,
                "served": float(c.served)
            })
        for t in self.trucks:
            nodes.append({"type": 2, "x": self._norm_x(t.x), "y": self._norm_y(t.y),
                           "demand": 0, "ready": 0, "due": 1.0, "service": 0,
                           "load": t.load / self.capacity, "battery": 1.0})
        for d in self.drones:
            nodes.append({"type": 3, "x": self._norm_x(d.x), "y": self._norm_y(d.y),
                           "demand": 0, "ready": 0, "due": 1.0, "service": 0,
                           "load": 0, "battery": d.battery_wh / d.battery_max})

        edges = []
        cust_start = 1
        k = min(5, len(self.customers) - 1)
        for i, c1 in enumerate(self.customers):
            dists = [(math.hypot(c1.x - c2.x, c1.y - c2.y), j)
                     for j, c2 in enumerate(self.customers) if j != i]
            dists.sort()
            for dist, j in dists[:k]:
                edges.append((cust_start + i, cust_start + j,
                               dist / self.coord_scale, 0))

        truck_start = 1 + self.n_cust
        for ti, t in enumerate(self.trucks):
            dists = sorted([(math.hypot(t.x - c.x, t.y - c.y), j)
                            for j, c in enumerate(self.customers) if not c.served])[:k]
            for dist, j in dists:
                edges.append((truck_start + ti, cust_start + j,
                               dist / self.coord_scale, 1))

        drone_start = truck_start + self.n_trucks
        for di, d in enumerate(self.drones):
            dists = sorted([(math.hypot(d.x - c.x, d.y - c.y), j)
                            for j, c in enumerate(self.customers)
                            if not c.served and not self._nfz_blocks(d.x, d.y, c.x, c.y)])[:k]
            for dist, j in dists:
                edges.append((drone_start + di, cust_start + j,
                               dist / self.coord_scale, 2))

        return {"nodes": nodes, "edges": edges, "n_nodes": len(nodes)}
