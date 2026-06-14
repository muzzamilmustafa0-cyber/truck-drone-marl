"""Figure: detailed GAT-MARL-LS architecture (heterogeneous graph -> encoder ->
actor-critic policy -> Dec-POMDP rollout -> consolidation/2-opt refinement)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 6.6))
ax.set_xlim(0, 16); ax.set_ylim(0, 6.6); ax.axis("off")

GRAPHC = "#eef3f8"; ENC = "#cde3f0"; POL = "#d7e9d8"; ENVC = "#fde9d6"; POST = "#fbe0e0"
def box(x, y, w, h, c, title, lines, ec="#555"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.10",
                                facecolor=c, edgecolor=ec, linewidth=1.3, zorder=2))
    ax.text(x+w/2, y+h-0.28, title, ha="center", va="top", fontsize=10.5, fontweight="bold", zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x+w/2, y+h-0.62-0.34*i, ln, ha="center", va="top", fontsize=8.6, zorder=3)

# (1) heterogeneous graph illustration
gx, gy, gw, gh = 0.3, 1.7, 2.7, 3.2
ax.add_patch(FancyBboxPatch((gx, gy), gw, gh, boxstyle="round,pad=0.04,rounding_size=0.10",
                            facecolor=GRAPHC, edgecolor="#557", linewidth=1.3, zorder=2))
ax.text(gx+gw/2, gy+gh-0.25, "Heterogeneous graph $G$", ha="center", va="top", fontsize=10, fontweight="bold")
np.random.seed(3)
depot = (gx+gw/2, gy+1.5)
custs = [(gx+0.6, gy+2.4), (gx+2.1, gy+2.5), (gx+0.7, gy+0.7), (gx+2.0, gy+0.8), (gx+1.35, gy+2.7)]
for c in custs:
    ax.plot([depot[0], c[0]], [depot[1], c[1]], color="#aab", lw=0.7, zorder=2)
    ax.add_patch(Circle(c, 0.12, facecolor="#1a9641", edgecolor="white", lw=0.6, zorder=3))
ax.add_patch(Circle((gx+0.95, gy+1.55), 0.12, facecolor="#888", edgecolor="white", lw=0.6, zorder=3))  # drone
ax.add_patch(Rectangle((depot[0]-0.13, depot[1]-0.13), 0.26, 0.26, facecolor="black", zorder=4))  # depot
ax.text(gx+gw/2, gy+0.18, "depot / customer / truck / drone", ha="center", fontsize=7.3, color="#555")

# (2) encoder
box(3.5, 1.7, 2.9, 3.2, ENC, "HeteroGAT encoder",
    ["Type-specific", "projections (4 types)", "", "Multi-head attention", "$H{=}4$ heads, $L{=}2$ layers", "", "$\\rightarrow$ graph emb. $g\\in\\mathbb{R}^{128}$"])
# (3) policy
box(6.9, 3.05, 2.7, 1.85, POL, "Actor--critic policy",
    ["Actor $\\pi_\\theta(a\\,|\\,o,g,m)$", "Critic $V_\\phi(g)$", "REINFORCE + baseline"])
# (4) environment
box(6.9, 1.0, 2.7, 1.75, ENVC, "Dec-POMDP rollout",
    ["masked actions (cap.\\ + TW)", "late customers rejected", "$\\rightarrow$ feasible served set"])
# (5) consolidation
box(10.2, 1.7, 2.7, 3.2, POST, "Route consolidation",
    ["Fill-first", "nearest-neighbour", "re-packing", "", "+ TW-aware", "2-opt / Or-opt"])
# (6) output
box(13.3, 2.3, 2.4, 2.0, "#eeeeee", "Solution",
    ["high feasible service", "low-CO$_2$ routes", "(GAT-MARL-LS)"])

def arrow(x0, y0, x1, y1, c="#444", style="-|>", rad=0.0, ls="-", lw=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=15,
                                 color=c, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))
arrow(3.0, 3.3, 3.5, 3.3)            # graph -> encoder
arrow(6.4, 3.7, 6.9, 3.9)            # encoder -> policy
arrow(6.4, 2.9, 6.9, 1.9)            # encoder -> env
arrow(8.25, 3.05, 8.25, 2.75)       # policy -> env (act)
arrow(9.6, 1.9, 10.2, 3.0)           # env -> consolidation (served set)
arrow(12.9, 3.3, 13.3, 3.3)          # consolidation -> solution
# training feedback (env -> policy)
arrow(9.0, 2.75, 9.0, 3.05, c="#1a7", style="-|>", ls=(0,(4,3)), lw=1.3)
ax.text(9.62, 2.86, "reward", fontsize=8, color="#1a7", style="italic")

# grouping brackets
ax.add_patch(FancyBboxPatch((3.4, 0.82), 6.35, 4.3, boxstyle="round,pad=0.05",
                            fill=False, edgecolor="#3a7", lw=1.2, ls=(0,(5,3)), zorder=1))
ax.text(6.55, 5.28, "Learned policy (GAT-MARL)", ha="center", fontsize=9.5, color="#2a7", fontweight="bold")
ax.add_patch(FancyBboxPatch((10.1, 1.55), 2.9, 3.5, boxstyle="round,pad=0.05",
                            fill=False, edgecolor="#c55", lw=1.2, ls=(0,(5,3)), zorder=1))
ax.text(11.55, 5.28, "Classical refinement", ha="center", fontsize=9.5, color="#c44", fontweight="bold")

out = FIG / "fig_architecture.png"
plt.savefig(out, bbox_inches="tight", dpi=200); plt.close()
print("saved", out)
