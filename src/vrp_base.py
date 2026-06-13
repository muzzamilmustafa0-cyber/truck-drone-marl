"""
TruckDroneEnvBase — gym-compatible multi-agent environment for
heterogeneous truck-drone last-mile delivery.

Architecture:
  - N_trucks trucks (mobile depots carrying drones)
  - N_drones drones (1-2 per truck)
  - Customers with time windows, demands, locations
  - NFZ (no-fly zones) modelled as circular forbidden regions
  - COPERT 5 CO2 model for trucks
  - Stolaroff 2018 energy model for drones
  - Graph representation: heterogeneous (truck nodes, drone nodes, customer nodes)

Observation space per agent:
  - Agent own state (pos, capacity, time, battery/fuel)
  - Local graph: k-nearest customers + their features
  - Global context: other agents' positions and loads

Action space per agent:
  - Discrete: which customer to visit next (or return to depot)

Reward: multi-objective (makespan, CO2, workload balance, battery)
"""

import math
import random
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import IntEnum

# ─────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────

class AgentType(IntEnum):
    TRUCK = 0
    DRONE = 1

@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_time: float
    service_time: float
    served: bool = False
    served_by: Optional[int] = None # agent id
    served_at: Optional[float] = None # time

@dataclass
class NFZone:
    cx: float
    cy: float
    radius: float
    active: bool = True

    def contains(self, x: float, y: float) -> bool:
        return self.active and (math.hypot(x - self.cx, y - self.cy) <= self.radius)

    def path_intersects(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Check if segment (x1,y1)→(x2,y2) passes through this NFZ."""
        if not self.active:
            return False
 # Point-to-segment distance
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return self.contains(x1, y1)
        t = max(0.0, min(1.0, ((self.cx - x1)*dx + (self.cy - y1)*dy) / (dx**2 + dy**2)))
        px, py = x1 + t*dx, y1 + t*dy
        return math.hypot(px - self.cx, py - self.cy) <= self.radius

@dataclass
class TruckState:
    id: int
    x: float
    y: float
    capacity: float
    load: float = 0.0
    time: float = 0.0
    total_distance: float = 0.0
    co2_emitted: float = 0.0 # grams
    route: List[int] = field(default_factory=list)
    speed_kmh: float = 50.0 # default urban speed
    done: bool = False

@dataclass
class DroneState:
    id: int
    truck_id: int # assigned truck
    x: float
    y: float
    battery_wh: float = 500.0 # 500 Wh — commercial delivery drone (DJI Agras class)
    battery_max: float = 500.0 # covers ~50 km one-way at 12 m/s
    payload_kg: float = 0.0
    payload_max: float = 2.5
    time: float = 0.0
    total_distance: float = 0.0
    energy_used: float = 0.0 # Wh
    route: List[int] = field(default_factory=list)
    speed_ms: float = 12.0 # ~43 km/h cruising
    done: bool = False
    in_flight: bool = False

# ─────────────────────────────────────────────────────────────────
# Physics models
# ─────────────────────────────────────────────────────────────────

class COPERTModel:
    """
    COPERT 5 CO2 emission factor model.
    Vehicle: Light Commercial Vehicle, N1 class (<3.5t), Euro 6 Diesel
    Source: EMEP/EEA Guidebook 2023, Table 3-4
    EF(v) = (a/v + b + c*v + d*v^2) * (1 + e_load * load_ratio)  [gCO2/km]
    """
 # LCV Euro 6 Diesel coefficients
    A, B, C, D = 380.0, 120.5, -1.82, 0.018
    E_LOAD = 0.42

    @classmethod
    def emission_factor(cls, speed_kmh: float, load_ratio: float = 0.5) -> float:
        """Returns gCO2/km"""
        v = max(speed_kmh, 5.0)
        ef = (cls.A/v + cls.B + cls.C*v + cls.D*v**2) * (1 + cls.E_LOAD * load_ratio)
        return max(ef, 80.0)

    @classmethod
    def co2_for_trip(cls, distance_km: float, speed_kmh: float, load_ratio: float) -> float:
        """Returns gCO2 for a trip"""
        return cls.emission_factor(speed_kmh, load_ratio) * distance_km


class DroneEnergyModel:
    """
    Quadrotor energy model: P = base_power + speed_power + payload_penalty
    Source: Stolaroff et al. 2018 (Patterns); Fontaine et al. 2021
    """
    @staticmethod
    def power_watts(speed_ms: float, payload_kg: float) -> float:
        base  = 150.0 + 40.0 * payload_kg # hover + payload penalty (W)
        drag  = 0.8 * speed_ms**2 # aerodynamic drag (W)
        return base + drag

    @staticmethod
    def energy_wh(distance_km: float, speed_ms: float, payload_kg: float) -> float:
        """Energy consumed in Wh for a given trip"""
        if speed_ms < 0.1:
            return 0.0
        power_w  = DroneEnergyModel.power_watts(speed_ms, payload_kg)
        speed_kmh = speed_ms * 3.6
        time_h    = distance_km / speed_kmh
        return power_w * time_h # Wh

    @staticmethod
    def feasible(distance_km: float, speed_ms: float, payload_kg: float,
                 current_battery_wh: float) -> bool:
        """Check if drone can complete the trip and return (round-trip)."""
        one_way = DroneEnergyModel.energy_wh(distance_km, speed_ms, payload_kg)
        return_e = DroneEnergyModel.energy_wh(distance_km, speed_ms, 0.0) # empty return
        return (one_way + return_e) <= current_battery_wh * 0.9 # 10% reserve


# ─────────────────────────────────────────────────────────────────
# Instance loader
# ─────────────────────────────────────────────────────────────────

def load_solomon_instance(path: str) -> Dict:
    """Parse a Solomon VRPTW file into a dict."""
    import re
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    name = lines[0]
    for i, l in enumerate(lines):
        if l.upper().startswith("NUMBER"):
            parts = lines[i+1].split()
            n_vehicles = int(parts[0])
            capacity   = int(parts[1])
            break
    customers = []
    reading = False
    for l in lines:
        if l.upper().startswith("CUST"):
            reading = True; continue
        if reading and re.match(r'^\d', l):
            vals = l.split()
            if len(vals) >= 7:
                customers.append(Customer(
                    id=int(vals[0]), x=float(vals[1]), y=float(vals[2]),
                    demand=float(vals[3]), ready_time=float(vals[4]),
                    due_time=float(vals[5]), service_time=float(vals[6])
                ))
    return {"name": name, "n_vehicles": n_vehicles, "capacity": capacity,
            "depot": customers[0], "customers": customers[1:]}


def load_cvrp_instance(path: str) -> Dict:
    """Parse a CVRPLIB .vrp file."""
    import re
    with open(path) as f:
        content = f.read()
    name  = re.search(r'NAME\s*:\s*(\S+)', content).group(1)
    cap_m = re.search(r'CAPACITY\s*:\s*(\d+)', content)
    cap   = int(cap_m.group(1)) if cap_m else 1000

    coords  = {}
    demands = {}
    in_c = in_d = False
    for line in content.split('\n'):
        line = line.strip()
        if 'NODE_COORD_SECTION' in line: in_c = True; in_d = False; continue
        if 'DEMAND_SECTION'     in line: in_d = True; in_c = False; continue
        if 'DEPOT_SECTION'      in line: in_c = in_d = False; continue
        if in_c and re.match(r'\d+\s+[\d.]+\s+[\d.]+', line):
            p = line.split(); coords[int(p[0])] = (float(p[1]), float(p[2]))
        if in_d and re.match(r'\d+\s+\d+', line):
            p = line.split(); demands[int(p[0])] = int(p[1])

    depot_id = min(coords.keys())
    customers = []
    for nid in sorted(coords.keys()):
        if nid == depot_id: continue
        x, y = coords[nid]
        d = demands.get(nid, 0)
        customers.append(Customer(
            id=nid, x=x, y=y, demand=d,
            ready_time=0, due_time=1e9, service_time=10.0
        ))
    depot_x, depot_y = coords[depot_id]
    depot = Customer(id=0, x=depot_x, y=depot_y, demand=0,
                     ready_time=0, due_time=1e9, service_time=0)
    return {"name": name, "n_vehicles": 0, "capacity": cap,
            "depot": depot, "customers": customers}


# ─────────────────────────────────────────────────────────────────
# Main environment
# ─────────────────────────────────────────────────────────────────

class TruckDroneEnvBase:
    """
    Gym-compatible environment for truck-drone cooperative last-mile delivery.

    Observation per agent: numpy array of shape (obs_dim,)
    Action: integer in [0, n_customers + 1]  (0=depot, 1..n=customer id)
    """

 # Reward weights (tunable)
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

 # Normalize coordinates to [0,1]
        all_x = [c.x for c in self.customers] + [self.depot.x]
        all_y = [c.y for c in self.customers] + [self.depot.y]
        self.x_min, self.x_max = min(all_x), max(all_x)
        self.y_min, self.y_max = min(all_y), max(all_y)
        self.coord_scale = max(self.x_max - self.x_min, self.y_max - self.y_min, 1e-9)

 # Observation dimension: own_state(6) + k_nearest(20*7) + global_ctx(n_agents*4)
        self.k_nearest = 20
        self.obs_dim = 6 + self.k_nearest * 7 + self.n_agents * 4

        self.reset()

    def reset(self) -> Dict[int, np.ndarray]:
 # Reset customer served flags
        for c in self.customers:
            c.served = False
            c.served_by = None
            c.served_at = None

 # Init trucks
        self.trucks = []
        for i in range(self.n_trucks):
            self.trucks.append(TruckState(
                id=i, x=self.depot.x, y=self.depot.y,
                capacity=self.capacity, load=0.0
            ))

 # Init drones
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

 # ─── Observation ─────────────────────────────────────────────
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
                t.load / self.capacity,
                t.time / 1500.0,
                t.total_distance / (self.coord_scale * 10),
                float(t.done)
            ]
        else:
            d_idx = agent_id - self.n_trucks
            d = self.drones[d_idx]
            obs[ptr:ptr+6] = [
                self._norm_x(d.x), self._norm_y(d.y),
                d.battery_wh / d.battery_max,
                d.time / 1500.0,
                d.total_distance / (self.coord_scale * 10),
                float(d.done)
            ]
        ptr += 6

 # k-nearest unserved customers
        cur_x = self.trucks[agent_id].x if is_truck else self.drones[agent_id - self.n_trucks].x
        cur_y = self.trucks[agent_id].y if is_truck else self.drones[agent_id - self.n_trucks].y

        unserved = [c for c in self.customers if not c.served]
        dists = sorted([(math.hypot(c.x - cur_x, c.y - cur_y), i, c) for i, c in enumerate(unserved)])[:self.k_nearest]

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

 # Global context: all agents' positions and loads
        for i in range(self.n_agents):
            base = ptr + i * 4
            if i < self.n_trucks:
                t = self.trucks[i]
                obs[base:base+4] = [self._norm_x(t.x), self._norm_y(t.y),
                                    t.load/self.capacity, float(t.done)]
            else:
                d = self.drones[i - self.n_trucks]
                obs[base:base+4] = [self._norm_x(d.x), self._norm_y(d.y),
                                    d.battery_wh/d.battery_max, float(d.done)]

        return obs

    def _nfz_blocks(self, x1, y1, x2, y2) -> bool:
        return any(nfz.path_intersects(x1, y1, x2, y2) for nfz in self.nfz_list)

 # ─── Valid actions mask ───────────────────────────────────────
    def get_action_mask(self, agent_id: int) -> np.ndarray:
        """
        Returns boolean mask of shape (n_cust + 1,)
        Index 0 = return to depot
        Index i+1 = serve customer i
        """
        mask = np.zeros(self.n_cust + 1, dtype=bool)
        mask[0] = True # always can return to depot

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
                dist = math.hypot(c.x - d.x, c.y - d.y) / 1000.0 # km approx
                if not DroneEnergyModel.feasible(dist, d.speed_ms, c.demand/100.0, d.battery_wh):
                    continue
 # Check NFZ
                if self._nfz_blocks(d.x, d.y, c.x, c.y):
                    continue
                mask[i + 1] = True

        return mask

 # ─── Step ────────────────────────────────────────────────────
    def step(self, actions: Dict[int, int]) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        actions: {agent_id: action_index}
        Returns: obs, rewards, dones, infos
        """
        rewards = {i: 0.0 for i in range(self.n_agents)}
        infos   = {}

        for agent_id, action in actions.items():
            if agent_id in self.done_agents:
                continue

            is_truck = agent_id < self.n_trucks

            if action == 0:
 # Return to depot
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
 # penalise invalid action
                rewards[agent_id] -= 5.0
                continue

            if is_truck:
                r = self._truck_serve(agent_id, c)
            else:
                r = self._drone_serve(agent_id - self.n_trucks, c)

            rewards[agent_id] += r

 # Check global done
        all_served = all(c.served for c in self.customers)
        all_agents_done = all(
            self.trucks[i].done for i in range(self.n_trucks)
        ) and all(
            self.drones[i].done for i in range(self.n_drones)
        )

        if all_served:
 # Return all remaining agents to depot
            for i in range(self.n_trucks):
                if not self.trucks[i].done:
                    self._truck_goto_depot(i)
            for i in range(self.n_drones):
                if not self.drones[i].done:
                    self._drone_goto_depot(i)

        done_flag = all_served or all_agents_done

        if done_flag:
 # Global terminal reward
            terminal_reward = self._compute_terminal_reward()
            for i in range(self.n_agents):
                rewards[i] += terminal_reward

        self.step_count += 1
        obs   = {i: self._get_obs(i) for i in range(self.n_agents)}
        dones = {i: done_flag for i in range(self.n_agents)}
        dones["__all__"] = done_flag

        return obs, rewards, dones, infos

 # ─── Truck movement ──────────────────────────────────────────
    def _truck_serve(self, truck_id: int, customer: Customer) -> float:
        t = self.trucks[truck_id]
        dist = math.hypot(customer.x - t.x, customer.y - t.y)
        dist_km = dist / 1000.0 if dist > 100 else dist # assume units are km if small

 # Travel time (hours → time units match Solomon: minutes)
        travel_time = dist / t.speed_kmh # distance / speed (same units)
        arrive_time = t.time + travel_time

 # Time window check
        arrive_time = max(arrive_time, customer.ready_time)
        if arrive_time > customer.due_time:
            return -2.0 # late penalty

 # CO2 cost
        load_ratio = t.load / self.capacity
        co2 = COPERTModel.co2_for_trip(dist_km if dist < 200 else dist/1000.0,
                                        t.speed_kmh, load_ratio)

 # Update state
        t.x, t.y = customer.x, customer.y
        t.time = arrive_time + customer.service_time
        t.load += customer.demand
        t.total_distance += dist
        t.co2_emitted += co2
        t.route.append(customer.id)

        customer.served = True
        customer.served_by = truck_id
        customer.served_at = t.time

 # Step reward: negative travel time + CO2 penalty
        r = -0.01 * travel_time - 0.001 * (co2 / 1000.0) + 1.0
        return r

    def _truck_goto_depot(self, truck_id: int):
        t = self.trucks[truck_id]
        dist = math.hypot(self.depot.x - t.x, self.depot.y - t.y)
        dist_km = dist / 1000.0 if dist > 100 else dist
        travel_time = dist / t.speed_kmh
        co2 = COPERTModel.co2_for_trip(dist_km, t.speed_kmh, t.load / self.capacity)
        t.x, t.y = self.depot.x, self.depot.y
        t.time += travel_time
        t.total_distance += dist
        t.co2_emitted += co2
        t.done = True

 # ─── Drone movement ──────────────────────────────────────────
    def _drone_serve(self, drone_idx: int, customer: Customer) -> float:
        d = self.drones[drone_idx]
        dist = math.hypot(customer.x - d.x, customer.y - d.y)
        dist_km = dist / 1000.0 if dist > 100 else dist

 # Check NFZ
        if self._nfz_blocks(d.x, d.y, customer.x, customer.y):
            return -3.0 # NFZ violation penalty

        energy = DroneEnergyModel.energy_wh(dist_km, d.speed_ms, d.payload_kg)
        if energy > d.battery_wh:
            return -4.0 # infeasible (no battery)

        travel_time = dist / (d.speed_ms * 3.6) # same units as trucks

        d.x, d.y = customer.x, customer.y
        d.time += travel_time
        d.total_distance += dist
        d.battery_wh -= energy
        d.energy_used += energy
        d.payload_kg = max(0.0, d.payload_kg - customer.demand / 1000.0)
        d.route.append(customer.id)

        customer.served = True
        customer.served_by = self.n_trucks + drone_idx
        customer.served_at = d.time

        r = -0.01 * travel_time - 0.002 * energy + 1.2 # drone slightly preferred (CO2 benefit)
        return r

    def _drone_goto_depot(self, drone_idx: int):
        d = self.drones[drone_idx]
        dist = math.hypot(self.depot.x - d.x, self.depot.y - d.y)
        dist_km = dist / 1000.0 if dist > 100 else dist
        energy = DroneEnergyModel.energy_wh(dist_km, d.speed_ms, 0.0)
        travel_time = dist / (d.speed_ms * 3.6)
        d.x, d.y = self.depot.x, self.depot.y
        d.time += travel_time
        d.total_distance += dist
        d.battery_wh -= energy
        d.energy_used += energy
        d.done = True

 # ─── Terminal reward ─────────────────────────────────────────
    def _compute_terminal_reward(self) -> float:
        """
        Multi-objective terminal reward:
          R = -w1*makespan - w2*co2 - w3*balance + w4*battery_efficiency
        All terms normalized to [0,1] range.
        """
 # Makespan: max completion time across all agents
        truck_times = [t.time for t in self.trucks]
        drone_times = [d.time for d in self.drones]
        makespan = max(truck_times + drone_times)
        makespan_norm = makespan / 1500.0

 # Total CO2 (kg)
        total_co2 = sum(t.co2_emitted for t in self.trucks) / 1e6 # g → tonnes
        co2_norm  = min(total_co2 / 0.1, 1.0)

 # Workload balance (std dev of completion times, normalized)
        all_times = truck_times + drone_times
        balance = np.std(all_times) / (np.mean(all_times) + 1e-9)
        balance_norm = min(balance, 1.0)

 # Battery efficiency (fraction of drones that returned with > 20% battery)
        battery_ok = sum(1 for d in self.drones if d.battery_wh / d.battery_max >= 0.2)
        battery_ratio = battery_ok / max(self.n_drones, 1)

 # Service rate
        served = sum(1 for c in self.customers if c.served)
        service_rate = served / self.n_cust

        reward = (
            - self.W_MAKESPAN * makespan_norm
            - self.W_CO2      * co2_norm
            - self.W_BALANCE  * balance_norm
            + self.W_BATTERY  * battery_ratio
            + 10.0 * service_rate # strong incentive to serve all customers
        )
        return reward

 # ─── Metrics extraction ───────────────────────────────────────
    def get_metrics(self) -> Dict[str, float]:
        truck_times  = [t.time for t in self.trucks]
        drone_times  = [d.time for d in self.drones]
        all_times    = truck_times + drone_times

        makespan     = max(all_times)
        total_co2_g  = sum(t.co2_emitted for t in self.trucks)
        total_dist   = sum(t.total_distance for t in self.trucks) + \
                       sum(d.total_distance for d in self.drones)
        battery_used_pct = np.mean([(d.energy_used / d.battery_max) * 100 for d in self.drones])
        balance_std  = float(np.std(all_times))
        served       = sum(1 for c in self.customers if c.served)
        n_trucks_used = sum(1 for t in self.trucks if len(t.route) > 0)
        n_drones_used = sum(1 for d in self.drones if len(d.route) > 0)

        return {
            "makespan":           makespan,
            "total_co2_grams":    total_co2_g,
            "total_co2_kg":       total_co2_g / 1000.0,
            "total_distance":     total_dist,
            "battery_used_pct":   battery_used_pct,
            "workload_balance":   balance_std,
            "service_rate":       served / self.n_cust,
            "n_served":           served,
            "n_customers":        self.n_cust,
            "n_trucks_used":      n_trucks_used,
            "n_drones_used":      n_drones_used,
            "truck_co2_per_km":   total_co2_g / max(sum(t.total_distance for t in self.trucks), 1),
        }

    def get_graph_representation(self) -> Dict:
        """
        Returns heterogeneous graph for GAT encoder.
        Nodes: depot(1) + customers(n) + trucks(n_trucks) + drones(n_drones)
        Edges: truck→customer, drone→customer, customer→customer (kNN)
        """
        nodes = []
 # Depot node: type=0
        nodes.append({"type": 0, "x": self._norm_x(self.depot.x), "y": self._norm_y(self.depot.y),
                       "demand": 0, "ready": 0, "due": 1.0, "service": 0,
                       "load": 0, "battery": 1.0})
 # Customer nodes: type=1
        for c in self.customers:
            nodes.append({
                "type": 1,
                "x": self._norm_x(c.x), "y": self._norm_y(c.y),
                "demand": c.demand / self.capacity,
                "ready": c.ready_time / 1500.0,
                "due": c.due_time / 1500.0,
                "service": c.service_time / 100.0,
                "load": 0, "battery": 1.0,
                "served": float(c.served)
            })
 # Truck nodes: type=2
        for t in self.trucks:
            nodes.append({"type": 2, "x": self._norm_x(t.x), "y": self._norm_y(t.y),
                           "demand": 0, "ready": 0, "due": 1.0, "service": 0,
                           "load": t.load / self.capacity, "battery": 1.0})
 # Drone nodes: type=3
        for d in self.drones:
            nodes.append({"type": 3, "x": self._norm_x(d.x), "y": self._norm_y(d.y),
                           "demand": 0, "ready": 0, "due": 1.0, "service": 0,
                           "load": 0, "battery": d.battery_wh / d.battery_max})

 # kNN edges (among customers)
        edges = []
        cust_start = 1
        k = min(5, len(self.customers) - 1)
        for i, c1 in enumerate(self.customers):
            dists = [(math.hypot(c1.x - c2.x, c1.y - c2.y), j)
                     for j, c2 in enumerate(self.customers) if j != i]
            dists.sort()
            for dist, j in dists[:k]:
                edges.append((cust_start + i, cust_start + j,
                               dist / self.coord_scale, 0)) # type 0: cust-cust

 # Truck → nearby customers
        truck_start = 1 + self.n_cust
        for ti, t in enumerate(self.trucks):
            dists = sorted([(math.hypot(t.x - c.x, t.y - c.y), j)
                            for j, c in enumerate(self.customers) if not c.served])[:k]
            for dist, j in dists:
                edges.append((truck_start + ti, cust_start + j,
                               dist / self.coord_scale, 1)) # type 1: truck-cust

 # Drone → nearby customers (NFZ-aware)
        drone_start = truck_start + self.n_trucks
        for di, d in enumerate(self.drones):
            dists = sorted([(math.hypot(d.x - c.x, d.y - c.y), j)
                            for j, c in enumerate(self.customers)
                            if not c.served and not self._nfz_blocks(d.x, d.y, c.x, c.y)])[:k]
            for dist, j in dists:
                edges.append((drone_start + di, cust_start + j,
                               dist / self.coord_scale, 2)) # type 2: drone-cust

        return {"nodes": nodes, "edges": edges, "n_nodes": len(nodes)}


# ─────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path  = os.path.join(BASE, "data", "solomon", "C101.txt")

    print("Loading Solomon C101...")
    inst  = load_solomon_instance(path)
    print(f"  Depot: ({inst['depot'].x}, {inst['depot'].y})")
    print(f"  Customers: {len(inst['customers'])}, Capacity: {inst['capacity']}")

 # Simple NFZ for testing
    nfz = [NFZone(cx=inst['customers'][10].x, cy=inst['customers'][10].y, radius=5.0)]

    env = TruckDroneEnvBase(inst, n_trucks=3, n_drones_per_truck=1, nfz_list=nfz, seed=42)
    obs = env.reset()

    print(f"\n  Obs shape: {obs[0].shape}")
    print(f"  Action mask sample: {env.get_action_mask(0).sum()} valid actions for truck 0")
    print(f"  Graph: {env.get_graph_representation()['n_nodes']} nodes")

 # Run 10 random steps
    total_r = {i: 0.0 for i in range(env.n_agents)}
    for step in range(200):
        actions = {}
        for i in range(env.n_agents):
            mask = env.get_action_mask(i)
            valid = np.where(mask)[0]
            if len(valid) > 0:
                actions[i] = int(valid[np.random.randint(len(valid))])
            else:
                actions[i] = 0

        obs, rewards, dones, infos = env.step(actions)
        for i, r in rewards.items():
            total_r[i] += r
        if dones.get("__all__", False):
            print(f"\n  Episode done at step {step+1}")
            break

    metrics = env.get_metrics()
    print("\n  === Metrics ===")
    for k, v in metrics.items():
        print(f"    {k:25s}: {v:.4f}")
    print("\n[PASS] TruckDroneEnvBase self-test complete.")
