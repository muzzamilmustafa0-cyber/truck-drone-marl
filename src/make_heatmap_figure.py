"""Figure: time-window-feasible service rate per method x instance (56 instances)."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
feas = pd.read_csv(ROOT / "results" / "feasible_service.csv")
hyb  = pd.read_csv(ROOT / "results" / "hybrid_results.csv")
FIG  = ROOT / "figures"; FIG.mkdir(exist_ok=True)

# GAT-MARL-LS per-instance feasible service from hybrid results
gls = hyb[["instance", "service_rate"]].copy(); gls["method"] = "GAT-MARL-LS"
base = feas[["instance", "method", "service_rate_feasible"]].rename(
    columns={"service_rate_feasible": "service_rate"})
allm = pd.concat([gls[["instance", "method", "service_rate"]], base], ignore_index=True)

ORDER = ["GAT-MARL-LS", "GAT-MARL", "TD-Split", "ALNS", "NNH", "AM", "POMO", "CWS"]
insts = sorted(allm.instance.unique(), key=lambda s: (s[:2] if s[1].isalpha() else s[0], s))
M = allm.pivot_table(index="method", columns="instance", values="service_rate").reindex(ORDER)[insts] * 100

fig, ax = plt.subplots(figsize=(16, 3.8))
im = ax.imshow(M.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER, fontsize=9)
ax.set_xticks(range(len(insts))); ax.set_xticklabels(insts, rotation=90, fontsize=5.5)
ax.set_xlabel("Solomon instance")
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
cbar.set_label("Feasible service rate (%)", fontsize=9)
out = FIG / "fig_heatmap.png"
plt.savefig(out, bbox_inches="tight", dpi=200); plt.close()
print("saved", out)
