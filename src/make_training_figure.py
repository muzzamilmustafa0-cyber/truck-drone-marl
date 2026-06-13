"""Figure: training dynamics aggregated over instances (mean +/- std vs episode)."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
df = pd.read_csv(ROOT / "results" / "training_curves.csv")
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

def band(col):
    g = df.groupby("episode")[col]
    ep = sorted(df.episode.unique())
    return np.array(ep), g.mean().reindex(ep).values, g.std().reindex(ep).values

fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
fig.subplots_adjust(hspace=0.28, wspace=0.24)

panels = [
    ("served",   "Service rate (%)", "#2c7bb6", "(a)"),
    ("reward",   "Episode reward",   "#1a9641", "(b)"),
    ("entropy",  "Policy entropy",   "#d7191c", "(c)"),
    ("critic_loss", "Critic loss",   "#7b3294", "(d)"),
]
for ax, (col, ylab, c, tag) in zip(axes.ravel(), panels):
    ep, mu, sd = band(col)
    ax.plot(ep, mu, color=c, lw=2)
    ax.fill_between(ep, mu - sd, mu + sd, color=c, alpha=0.15)
    ax.set_xlabel("Episode"); ax.set_ylabel(ylab)
    ax.grid(alpha=0.2); ax.spines[["top","right"]].set_visible(False)
    ax.text(0.02, 0.97, tag, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

# overlay entropy-coefficient schedule on panel (c)
ax_c = axes.ravel()[2]
ep, mu_ec, _ = band("entropy_coef")
ax2 = ax_c.twinx()
ax2.plot(ep, mu_ec, color="#999", lw=1.5, ls="--")
ax2.set_ylabel("Entropy coefficient (annealed)", color="#777", fontsize=9)
ax2.tick_params(axis="y", labelcolor="#777")
ax2.spines["top"].set_visible(False)

out = FIG / "fig_training.png"
plt.savefig(out, bbox_inches="tight", dpi=190); plt.close()
print("saved", out)
