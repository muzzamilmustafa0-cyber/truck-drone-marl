"""Figure: operational concept of heterogeneous truck-drone delivery with time windows."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

ROOT = Path(__file__).parent.parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

depot = (0.0, 0.0)
# two truck routes (sequences of customer coordinates)
route1 = [(-3.2, 1.6), (-3.8, 3.0), (-2.4, 3.6), (-1.4, 2.4)]
route2 = [(3.0, 1.2), (3.9, 2.6), (2.6, 3.4)]
# drone-served customers (served by short sorties)
drones = [(-1.0, 4.2), (1.2, 3.9), (2.0, -1.6), (-2.0, -1.4), (3.6, -0.4)]

fig, ax = plt.subplots(figsize=(9, 6.2))

TRUCK = "#2c7bb6"; DRONE = "#888888"; CUST = "#1a9641"
# truck routes
for rt, col in [(route1, TRUCK), (route2, TRUCK)]:
    pts = [depot] + rt + [depot]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=col, lw=2.0, zorder=2)
    ax.scatter([p[0] for p in rt], [p[1] for p in rt], s=70, color=CUST,
               edgecolor="white", linewidth=0.6, zorder=3)
# drone sorties (launched from depot here, dashed) + customers
for d in drones:
    ax.plot([depot[0], d[0]], [depot[1], d[1]], "--", color=DRONE, lw=1.3, zorder=1)
    ax.scatter(*d, s=60, marker="^", color=DRONE, edgecolor="white", linewidth=0.5, zorder=3)
# depot
ax.scatter(*depot, s=320, marker="s", color="black", zorder=5)
ax.annotate("Depot", depot, textcoords="offset points", xytext=(10, -16), fontsize=11, fontweight="bold")

# time-window annotation on one truck customer and one drone customer
ax.annotate(r"$[a_i, b_i]$", route1[1], textcoords="offset points", xytext=(-46, 6),
            fontsize=10, color=CUST)
ax.annotate(r"$[a_j, b_j]$", drones[1], textcoords="offset points", xytext=(8, 6),
            fontsize=10, color="#555")

ax.set_xlim(-5.5, 5.5); ax.set_ylim(-3, 5.2)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_edgecolor("#cccccc")

legend = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor="black", markersize=12, label="Depot"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=CUST, markersize=10, label="Customer (time window $[a,b]$)"),
    Line2D([0], [0], color=TRUCK, lw=2, label="Truck route"),
    Line2D([0], [0], color=DRONE, lw=1.3, ls="--", marker="^", markerfacecolor=DRONE, label="Drone sortie"),
]
ax.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.13))

out = FIG / "fig_concept.png"
plt.savefig(out, bbox_inches="tight", dpi=200); plt.close()
print("saved", out)
