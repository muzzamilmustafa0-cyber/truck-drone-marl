# Cooperative Truck–Drone Last-Mile Delivery via Graph-Attention Multi-Agent Reinforcement Learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20682276.svg)](https://doi.org/10.5281/zenodo.20682276)

A heterogeneous truck–drone routing system for the Vehicle Routing Problem with
Time Windows (VRPTW). A graph-attention multi-agent reinforcement learning policy
(**GAT-MARL**) decides the time-window-feasible served set and the truck/drone
allocation; a classical consolidation + 2-opt post-processor optimises the truck
routes (**GAT-MARL-LS**). Evaluated on all 56 Solomon VRPTW instances against
learning-based (Attention Model, POMO, TD-Split) and heuristic/metaheuristic
(NNH, Clarke–Wright Savings, ALNS) baselines.

## Key results (56 Solomon instances)

Service rate is reported as **time-window feasible** (a customer counts as served
only if reached within its time window). CO₂ uses one consistent accounting for
all methods: truck COPERT emissions + drone electricity (Italian grid,
0.258 kg CO₂/kWh).

| Method        | Feasible service | CO₂ (kg) | Makespan | Distance / BKS |
|---------------|:---------------:|:--------:|:--------:|:--------------:|
| **GAT-MARL-LS** | **99.9 %**     | **128.2**| 1039     | 1.45×          |
| GAT-MARL      | 99.9 %          | 280.7    | 967      | 4.45×          |
| TD-Split      | 95.5 %          | 217.0    | 965      | 3.33×          |
| NNH           | 94.8 %          | 231.3    | 967      | 2.88×          |
| ALNS          | 94.8 %          | 91.7     | 1017     | 0.95×          |
| AM            | 94.3 %          | 230.6    | 967      | 2.88×          |
| POMO          | 93.5 %          | 224.9    | 967      | 2.81×          |
| CWS           | 39.3 %          | 78.3     | 1732     | 0.83×          |

Findings:
- **GAT-MARL-LS attains the highest feasible service rate (99.9 %)** and
  **Pareto-dominates all learning-based baselines and NNH** on both feasible
  service and CO₂. The non-dominated set is {GAT-MARL-LS, ALNS, CWS}.
- **Clarke–Wright Savings (CWS)** appears to reach 100 % coverage but only 39.3 %
  of deliveries are within their time windows — its solutions are largely
  infeasible.
- **Route consolidation + 2-opt** cuts the learned policy's CO₂ by 54 %
  (280.7 → 128.2 kg) and fleet size from 24 → 9 trucks with **no loss of feasible
  service** (0 customers dropped across 56 instances).
- An **ablation** (`results/ablation_*`) shows entropy annealing is the
  load-bearing training component (its removal causes stochastic policy collapse);
  the graph-attention encoder improves training efficiency rather than final
  service rate.
- A **carbon-preference sweep** (`results/carbon_sweep.csv`) shows that reward
  scalarisation alone does not yield a controllable emission front at convergence
  (Spearman ρ ≥ 0, non-significant) — motivating the consolidation post-processor.

## Repository layout

```
src/        model, environment, baselines, experiment + figure scripts
results/    final CSV outputs (one row per instance/method/seed)
figures/    manuscript figures (regenerated from results/)
logs/       training / experiment logs
data/solomon/   56 Solomon VRPTW benchmark instances
```

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce the figures (fast — from saved results)

```bash
cd src
python make_concept_figure.py
python make_architecture_figure.py
python make_route_figure.py
python make_feasible_service_fig.py
python make_pareto_fig.py
python make_training_figure.py
python make_ablation_fig.py
python make_carbon_sweep_fig.py
python make_heatmap_figure.py
```

## Reproduce the experiments (slow — requires training)

```bash
cd src
python run_main_experiment.py      # train GAT-MARL + run baselines on 56 instances
python run_ablation.py             # robust ablation (5 variants × 4 instances × 3 seeds)
python run_carbon_sweep.py         # carbon-preference weight sweep
python compute_feasible_service.py # time-window feasibility of all methods
python run_hybrid.py               # consolidation + 2-opt post-processing (GAT-MARL-LS)
python compute_harmonized_co2.py   # one consistent CO2 accounting across methods
```

## Trained models

The trained model weights (one per Solomon instance, ~725 MB) are released as a
versioned archive rather than committed to the repository:

- **Archived (DOI):** [10.5281/zenodo.20682276](https://doi.org/10.5281/zenodo.20682276)
- **Download:** [`trained_models.zip` (Releases)](https://github.com/muzzamilmustafa0-cyber/truck-drone-marl/releases/tag/v1.0.0)
- Extract into `data/models/` to run the inference and hybrid scripts without retraining.
- Alternatively, `run_main_experiment.py` regenerates them from scratch.

The Solomon instances in `data/solomon/` are the standard public benchmark. Every
figure reproduces directly from the CSVs in `results/` and does not require the models.

## Figures

| File | Content |
|------|---------|
| `fig_concept.png`          | Operational concept: depot, truck routes, drone sorties, customer time windows |
| `fig_architecture.png`     | System pipeline: GAT encoder → actor-critic policy → Dec-POMDP env → consolidation |
| `fig_routes.png`           | Truck-route geometry before/after consolidation on a Solomon instance (RC101) |
| `fig_feasible_service.png` | Raw vs time-window-feasible service rate; CWS feasibility by class |
| `fig_pareto.png`           | Consolidation effect; feasible-service vs CO₂ Pareto front |
| `fig_training.png`         | Training dynamics (service, reward, entropy + annealing, critic loss) over episodes |
| `fig_ablation.png`         | Ablation convergence (service rate and CO₂ over training) |
| `fig_carbon_sweep.png`     | CO₂ vs carbon-preference weight λ per instance |
| `fig_heatmap.png`          | Feasible service rate per method × all 56 instances |

## License

Released under the MIT License (see `LICENSE`).
