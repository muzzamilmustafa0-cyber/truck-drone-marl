"""Figure: (a) route-consolidation effect, (b) feasible-service vs CO2 Pareto."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
harm = pd.read_csv(ROOT / "results" / "harmonized_pareto.csv")
hyb  = pd.read_csv(ROOT / "results" / "hybrid_results.csv")
FIG  = ROOT / "figures"; FIG.mkdir(exist_ok=True)

COL = {"GAT-MARL-LS":"#d7191c","GAT-MARL":"#fdae61","TD-Split":"#2c7bb6","AM":"#1a9641",
       "POMO":"#abdda4","ALNS":"#7b3294","NNH":"#08519c","CWS":"#888888"}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2))
fig.subplots_adjust(wspace=0.28, bottom=0.15)

# (a) consolidation before/after (relative bars)
co2_pre, co2_post = hyb.co2_pre_kg.mean(), hyb.total_co2_kg.mean()
trk_pre, trk_post = hyb.trucks_pre.mean(), hyb.n_trucks_used.mean()
dbk_pre = (hyb.dist_pre/hyb.bks_distance).mean(); dbk_post = (hyb.total_distance/hyb.bks_distance).mean()
svc_pre = hyb.n_served_pre.sum()/hyb.n_customers.sum()*100
svc_post = hyb.n_served.sum()/hyb.n_customers.sum()*100
groups = ["CO₂ (kg)", "Trucks", "Dist / BKS", "Feasible svc (%)"]
pre  = [co2_pre, trk_pre, dbk_pre, svc_pre]
post = [co2_post, trk_post, dbk_post, svc_post]
x = np.arange(len(groups)); w = 0.36
axA.bar(x-w/2, [1]*4, w, color="#cccccc", label="Before (GAT-MARL)")
axA.bar(x+w/2, [post[i]/pre[i] for i in range(4)], w, color="#d7191c", label="After (GAT-MARL-LS)")
axA.axhline(1.0, color="gray", lw=0.7, ls="--")
for i in range(4):
    fmt = (lambda v: f"{v:.0f}") if i < 2 else ((lambda v: f"{v:.2f}") if i == 2 else (lambda v: f"{v:.1f}"))
    axA.text(x[i]-w/2, 1.02, fmt(pre[i]), ha="center", fontsize=8, color="#555")
    axA.text(x[i]+w/2, post[i]/pre[i]+0.02, fmt(post[i]), ha="center", fontsize=8, color="#d7191c", fontweight="bold")
axA.set_xticks(x); axA.set_xticklabels(groups, fontsize=9.5)
axA.set_ylabel("Relative to 'before' (= 1.0)"); axA.set_ylim(0, 1.25)
axA.legend(fontsize=9, loc="upper right", frameon=False)
axA.text(0.02, 0.97, "(a)", transform=axA.transAxes, fontsize=12, fontweight="bold", va="top")
axA.spines[["top","right"]].set_visible(False)

# (b) Pareto
agg = {r["method"]: (r["feas_service_%"], r["co2_harmonized_kg"]) for _, r in harm.iterrows()}
def nd_set(a):
    nd=[]
    for m,(s,c) in a.items():
        if not any(m2!=m and s2>=s and c2<=c and (s2>s or c2<c) for m2,(s2,c2) in a.items()): nd.append(m)
    return nd
nd = nd_set(agg)
for m in sorted(nd, key=lambda m: agg[m][1]):
    pass
xs=[agg[m][1] for m in sorted(nd,key=lambda m:agg[m][1])]; ys=[agg[m][0] for m in sorted(nd,key=lambda m:agg[m][1])]
axB.plot(xs, ys, "--", color="#999", lw=1.2, zorder=1)
for m,(s,c) in agg.items():
    isnd = m in nd
    axB.scatter(c, s, s=260 if m=="GAT-MARL-LS" else (190 if isnd else 120),
                color=COL.get(m,"#333"), edgecolor="black" if isnd else "white",
                linewidth=2.0 if isnd else 0.8, marker="*" if m=="GAT-MARL-LS" else "o", zorder=3)
    axB.annotate(m, (c, s), textcoords="offset points", xytext=(8,5), fontsize=8.5,
                 fontweight="bold" if m=="GAT-MARL-LS" else "normal", color=COL.get(m,"#333"))
axB.set_xlabel("Total CO₂ (kg)"); axB.set_ylabel("Feasible service rate (%)")
axB.grid(alpha=0.22)
axB.text(0.02, 0.97, "(b)", transform=axB.transAxes, fontsize=12, fontweight="bold", va="top")
axB.spines[["top","right"]].set_visible(False)

out = FIG / "fig_pareto.png"
plt.savefig(out, bbox_inches="tight", dpi=200); plt.close()
print("saved", out)
