"""Figure: raw vs time-window-feasible service rate (56 Solomon instances)."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
df = pd.read_csv(ROOT / "results" / "feasible_service.csv")
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)

ORDER = ["GAT-MARL", "TD-Split", "ALNS", "NNH", "AM", "POMO", "CWS"]
agg = (df.groupby("method").agg(raw=("service_rate_raw","mean"),
                                feas=("service_rate_feasible","mean")).reindex(ORDER) * 100)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8),
                              gridspec_kw={"width_ratios": [2.1, 1]})
fig.subplots_adjust(wspace=0.30, bottom=0.20)

x = np.arange(len(ORDER)); w = 0.38
ax.bar(x - w/2, agg["raw"],  w, color="#bdbdbd", label="Raw (counts late visits)")
b2 = ax.bar(x + w/2, agg["feas"], w, color="#2c7bb6", label="Time-window feasible")
b2[ORDER.index("CWS")].set_color("#d7191c")
for i, mth in enumerate(ORDER):
    ax.text(x[i]-w/2, agg["raw"].iloc[i]+1, f"{agg['raw'].iloc[i]:.0f}", ha="center", fontsize=8, color="#555")
    col = "#d7191c" if mth == "CWS" else "#2c7bb6"
    ax.text(x[i]+w/2, agg["feas"].iloc[i]+1, f"{agg['feas'].iloc[i]:.0f}", ha="center", fontsize=8, color=col, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(ORDER, rotation=18, ha="right", fontsize=10)
ax.set_ylabel("Service rate (%)"); ax.set_ylim(0, 112)
ax.axhline(100, color="gray", ls="--", lw=0.7, alpha=0.5)
ax.legend(fontsize=9, loc="lower left", frameon=False)
ax.text(0.012, 0.97, "(a)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
ax.spines[["top","right"]].set_visible(False)

cws = df[df.method == "CWS"]
bc = cws.groupby("class").agg(raw=("service_rate_raw","mean"), feas=("service_rate_feasible","mean")) * 100
xc = np.arange(len(bc))
ax2.bar(xc - w/2, bc["raw"], w, color="#bdbdbd")
ax2.bar(xc + w/2, bc["feas"], w, color="#d7191c")
for i in range(len(bc)):
    ax2.text(xc[i]+w/2, bc["feas"].iloc[i]+1, f"{bc['feas'].iloc[i]:.0f}", ha="center", fontsize=8, color="#d7191c", fontweight="bold")
ax2.set_xticks(xc); ax2.set_xticklabels(list(bc.index), fontsize=10)
ax2.set_ylabel("CWS service rate (%)"); ax2.set_ylim(0, 112)
ax2.set_xlabel("Solomon class")
ax2.text(0.03, 0.97, "(b)", transform=ax2.transAxes, fontsize=12, fontweight="bold", va="top")
ax2.spines[["top","right"]].set_visible(False)

out = FIG / "fig_feasible_service.png"
plt.savefig(out, bbox_inches="tight", dpi=200); plt.close()
print("saved", out)
