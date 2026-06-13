"""Figure: system architecture / method pipeline (schematic)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

ROOT = Path(__file__).parent.parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(15, 5.5))
ax.set_xlim(0, 15); ax.set_ylim(0, 5.5); ax.axis("off")

LEARN = "#cde3f0"; ENVC = "#dbeedd"; POST = "#fbe3d6"; IO = "#eeeeee"
boxes = [
    (0.3, 2.1, 2.4, 1.3, IO,    "VRPTW instance\n(depot, customers,\ntime windows, demands)"),
    (3.1, 2.1, 2.4, 1.3, LEARN, "Heterogeneous Graph\nAttention Encoder\n(128-dim embeddings)"),
    (5.9, 2.1, 2.4, 1.3, LEARN, "Actor–Critic Policy\n(REINFORCE + value\nbaseline, entropy\nannealing)"),
    (8.7, 2.1, 2.6, 1.3, ENVC,  "Truck–Drone\nDec-POMDP rollout\n(late customers rejected\n→ feasible served set)"),
    (11.7, 2.1, 2.9, 1.3, POST, "Route consolidation\n+ 2-opt / Or-opt\n(fewer, fuller routes)"),
]
for (x, y, w, h, col, txt) in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                facecolor=col, edgecolor="#555", linewidth=1.2))
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=9.5)

def arrow(x0, x1, y=2.75, **kw):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=16,
                                 color="#444", lw=1.6, **kw))
arrow(2.7, 3.1); arrow(5.5, 5.9); arrow(8.3, 8.7); arrow(11.3, 11.7)

# training feedback loop (env -> policy)
ax.add_patch(FancyArrowPatch((9.9, 2.1), (7.1, 2.1), connectionstyle="arc3,rad=0.45",
                             arrowstyle="-|>", mutation_scale=14, color="#1a7", lw=1.4, ls="--"))
ax.text(8.4, 0.95, "reward / policy gradient (training)", ha="center", fontsize=8.5, color="#1a7", style="italic")

# stage labels above
ax.text(1.5, 3.75, "input", ha="center", fontsize=9, color="#666", style="italic")
ax.text(7.1, 4.45, "learned policy  (GAT-MARL)", ha="center", fontsize=10, color="#2c6", fontweight="bold")
ax.add_patch(FancyBboxPatch((3.0, 1.95), 5.5, 1.65, boxstyle="round,pad=0.05",
                            fill=False, edgecolor="#9cf", linewidth=1.0, linestyle=(0,(4,3))))
ax.text(13.15, 3.75, "classical refinement\n(GAT-MARL-LS)", ha="center", fontsize=9, color="#c63", style="italic")

out = FIG / "fig_architecture.png"
plt.savefig(out, bbox_inches="tight", dpi=200); plt.close()
print("saved", out)
