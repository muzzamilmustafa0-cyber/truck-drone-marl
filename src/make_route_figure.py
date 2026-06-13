"""Figure: truck-route geometry before vs after consolidation (one instance)."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
d = json.loads((ROOT / "results" / "route_geometry.json").read_text())
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

depot = d["depot"]
cust = {int(k): v for k, v in d["customers"].items()}
drone_ids = set(d["drone_ids"])

def draw(ax, routes, label):
    # drone deliveries: dashed gray depot<->customer
    for cid in drone_ids:
        x, y = cust[cid]
        ax.plot([depot[0], x], [depot[1], y], color="#bbbbbb", lw=0.5, ls=":", zorder=1)
    # truck routes
    colors = cm.tab20(np.linspace(0, 1, max(len(routes), 1)))
    for r, col in zip(routes, colors):
        pts = [depot] + [cust[c] for c in r] + [depot]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-", color=col, lw=1.3, zorder=2)
        ax.scatter([cust[c][0] for c in r], [cust[c][1] for c in r],
                   s=14, color=col, zorder=3, edgecolor="white", linewidth=0.3)
    # drone customers
    ax.scatter([cust[c][0] for c in drone_ids], [cust[c][1] for c in drone_ids],
               s=12, marker="^", color="#888888", zorder=3, label="drone delivery")
    ax.scatter([depot[0]], [depot[1]], s=180, marker="s", color="black", zorder=5, label="depot")
    ax.text(0.03, 0.97, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values(): s.set_edgecolor("#cccccc")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 6.4))
fig.subplots_adjust(wspace=0.05, bottom=0.10)
draw(axA, d["raw_routes"],          f"(a) GAT-MARL  —  {d['n_trucks_raw']} truck routes")
draw(axB, d["consolidated_routes"], f"(b) GAT-MARL-LS  —  {d['n_trucks_cons']} truck routes")
h, l = axB.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0))

out = FIG / "fig_routes.png"
plt.savefig(out, bbox_inches="tight", dpi=200); plt.close()
print("saved", out)
