"""Figure: ablation convergence (service rate + CO2 over training episodes)."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
df = pd.read_csv(ROOT / "results" / "ablation_curves.csv")
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

INSTANCES = ["C101", "C201", "R101", "RC101"]
VARIANTS  = ["Full", "A1-NoGAT", "A2-FixedEnt", "A3-NoCritic", "A4-NoMasks"]
COLORS = {"Full":"#2c7bb6","A1-NoGAT":"#7b3294","A2-FixedEnt":"#d7191c",
          "A3-NoCritic":"#fdae61","A4-NoMasks":"#1a9641"}
LW = {"Full":2.4,"A1-NoGAT":1.6,"A2-FixedEnt":2.0,"A3-NoCritic":1.6,"A4-NoMasks":1.6}

def band(sub, col):
    g = sub.groupby("episode")[col]; ep = sorted(sub.episode.unique())
    return np.array(ep), g.mean().reindex(ep).values, g.min().reindex(ep).values, g.max().reindex(ep).values

fig, axes = plt.subplots(2, 4, figsize=(17, 8))
fig.subplots_adjust(hspace=0.30, wspace=0.26, bottom=0.12)
for j, inst in enumerate(INSTANCES):
    ax = axes[0, j]
    for v in VARIANTS:
        sub = df[(df.instance==inst)&(df.variant==v)]
        if sub.empty: continue
        ep, mu, lo, hi = band(sub, "service_rate")
        ax.plot(ep, mu*100, color=COLORS[v], lw=LW[v], label=v)
        ax.fill_between(ep, lo*100, hi*100, color=COLORS[v], alpha=0.12)
    ax.set_ylim(-5, 108); ax.axhline(100, color="gray", ls="--", lw=0.7, alpha=0.5)
    ax.set_xlabel("Episode"); ax.grid(alpha=0.2)
    if j == 0: ax.set_ylabel("Service rate (%)")
    ax.text(0.04, 0.06, inst, transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax2 = axes[1, j]
    for v in VARIANTS:
        sub = df[(df.instance==inst)&(df.variant==v)]
        if sub.empty: continue
        ep, mu, lo, hi = band(sub, "co2_kg")
        ax2.plot(ep, mu, color=COLORS[v], lw=LW[v])
        ax2.fill_between(ep, lo, hi, color=COLORS[v], alpha=0.10)
    ax2.set_xlabel("Episode"); ax2.grid(alpha=0.2)
    if j == 0: ax2.set_ylabel("Total CO₂ (kg)")
    ax2.text(0.04, 0.92, inst, transform=ax2.transAxes, fontsize=11, fontweight="bold", va="top")
    ax2.spines[["top","right"]].set_visible(False)

h, l = axes[0,0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.0))
out = FIG / "fig_ablation.png"
plt.savefig(out, bbox_inches="tight", dpi=180); plt.close()
print("saved", out)
