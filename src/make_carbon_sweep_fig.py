"""Figure: carbon-preference sweep — CO2 vs weight lambda per instance."""
import pandas as pd, numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
df = pd.read_csv(ROOT / "results" / "carbon_sweep.csv")
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

INSTANCES = ["C101", "C201", "R101", "RC101"]
COLORS = {"C101":"#2c7bb6","C201":"#1a9641","R101":"#d7191c","RC101":"#7b3294"}

fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
fig.subplots_adjust(wspace=0.30, bottom=0.18)
for ax, inst in zip(axes, INSTANCES):
    sub = df[df.instance == inst]
    lams = sorted(sub["lambda"].unique())
    means = [sub[sub["lambda"]==l]["total_co2_kg"].mean() for l in lams]
    stds  = [sub[sub["lambda"]==l]["total_co2_kg"].std()  for l in lams]
    svc   = [sub[sub["lambda"]==l]["service_rate"].mean()*100 for l in lams]
    rho, p = stats.spearmanr(sub["lambda"], sub["total_co2_kg"])
    ax.scatter(sub["lambda"], sub["total_co2_kg"], color=COLORS[inst], alpha=0.35, s=38)
    ax.errorbar(lams, means, yerr=stds, color=COLORS[inst], lw=2.2, marker="o", ms=7, capsize=4)
    ax.set_xlabel("Carbon weight  λ"); ax.set_xticks(lams)
    if inst == "C101": ax.set_ylabel("Total CO₂ (kg)")
    ax.grid(alpha=0.25)
    ax.set_title(f"{inst}   (service ≈ {np.mean(svc):.0f}%)\nρ={rho:+.2f}, p={p:.2f}",
                 fontsize=9.5, fontweight="bold")
    ax.margins(y=0.12)
    ax.spines[["top","right"]].set_visible(False)

out = FIG / "fig_carbon_sweep.png"
plt.savefig(out, bbox_inches="tight", dpi=190); plt.close()
print("saved", out)
