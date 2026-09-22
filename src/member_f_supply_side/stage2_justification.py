"""Member F, step 9: empirical justification for the second stage.

Objection: stage 2 assumes that an absent audience response leads to exit. So far
that is an assumption. This script tests it directly:
    do creators whose content received no clicks in the early window stop
    publishing more often during the label window?
It additionally holds effort constant (number of upload days), so the finding is
not simply "whoever posts little receives little and quits".

    python -m src.member_f_supply_side.stage2_justification
"""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config

OUT = config.DERIVED_DIR / "member_f"
OUT.mkdir(parents=True, exist_ok=True)
R = {}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
ACC, ORA, GREY = "#0D6E6E", "#C77B30", "#8899A0"

F = pd.read_parquet(OUT / "creator_features.parquet")
# Creators who published in the early window AND whose content actually circulated
pop = F[(F.e_pub_days > 0) & (F.e_impr > 0)].copy()
R["n"] = int(len(pop))
print(f"Creators with an upload and circulating content in days 1-7: {len(pop):,}")

# ------------------------------------------------------------ 1. Coarse split
noresp = pop[pop.e_clicks == 0]
resp = pop[pop.e_clicks > 0]
R["no_response"] = {"n": int(len(noresp)), "churn": round(float(noresp.churn.mean()), 4)}
R["response"] = {"n": int(len(resp)), "churn": round(float(resp.churn.mean()), 4)}
R["ratio"] = round(float(noresp.churn.mean() / resp.churn.mean()), 3)
print("\n" + "=" * 70)
print("DO CREATORS WITHOUT A RESPONSE QUIT MORE OFTEN?")
print("=" * 70)
print(f"  no clicks in days 1-7 : {len(noresp):>6,} creators | churn {noresp.churn.mean():.1%}")
print(f"  received clicks       : {len(resp):>6,} creators | churn {resp.churn.mean():.1%}")
print(f"  ratio: {R['ratio']:.2f}x")

# ---------------------------------------------------------------- 2. Gradient
bins = [-1, 0, 2, 10, 50, np.inf]
labs = ["0", "1-2", "3-10", "11-50", "51+"]
pop["click_bin"] = pd.cut(pop.e_clicks, bins, labels=labs)
grad = pop.groupby("click_bin", observed=True).agg(
    n=("creatorId", "size"), churn=("churn", "mean")).reset_index()
grad["churn"] = grad.churn.round(4)
R["gradient"] = grad.to_dict("records")
print("\n" + "=" * 70)
print("GRADIENT BY NUMBER OF EARLY CLICKS")
print("=" * 70)
for r in grad.itertuples():
    print(f"  {str(r.click_bin):>6} clicks : {r.n:>6,} creators | churn {r.churn:.1%}")

# --------------------------------------------- 3. Holding effort constant
print("\n" + "=" * 70)
print("HOLDING EFFORT CONSTANT - by number of upload days in days 1-7")
print("=" * 70)
ctrl = []
for d_, g in pop.groupby("e_pub_days"):
    if len(g) < 300 or g[g.e_clicks == 0].shape[0] < 50:
        continue
    a = float(g[g.e_clicks == 0].churn.mean())
    b = float(g[g.e_clicks > 0].churn.mean())
    ctrl.append({"upload_days": int(d_), "n": int(len(g)),
                 "churn_no_clicks": round(a, 4), "churn_with_clicks": round(b, 4),
                 "gap_pp": round((a - b) * 100, 1)})
R["controlled"] = ctrl
cd_ = pd.DataFrame(ctrl)
print(f"  {'Upload days':<13}{'Creators':>10}{'no clicks':>14}{'with clicks':>13}{'difference':>12}")
for r in cd_.itertuples():
    print(f"  {r.upload_days:<13}{r.n:>10,}{r.churn_no_clicks:>13.1%}{r.churn_with_clicks:>13.1%}"
          f"{r.gap_pp:>+11.1f} pp")
R["mean_gap_pp"] = round(float(cd_.gap_pp.mean()), 1)
print(f"\n  Mean difference across all effort levels: {cd_.gap_pp.mean():+.1f} percentage points")

# --------------------------------------------------------------------- Figure
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))
g = grad.copy()
ax[0].bar(range(len(g)), g.churn * 100,
          color=[ORA if i < 2 else ACC for i in range(len(g))], width=.7)
ax[0].set_xticks(range(len(g)))
ax[0].set_xticklabels([f"{l}\n(n={n:,})" for l, n in zip(g.click_bin, g.n)], fontsize=7.5)
ax[0].set_xlabel("Clicks received in the early window")
ax[0].set_ylabel("Share who stop publishing (%)")
ax[0].set_title("(a) No audience response, higher exit", loc="left", fontsize=9.5)
for i, v in enumerate(g.churn * 100):
    ax[0].text(i, v + 1.2, f"{v:.0f}%", ha="center", fontsize=7.5, color="#333")
ax[0].set_ylim(0, 68)

x = np.arange(len(cd_))
w = .38
ax[1].bar(x - w / 2, cd_.churn_no_clicks * 100, w, color=ORA, label="no clicks")
ax[1].bar(x + w / 2, cd_.churn_with_clicks * 100, w, color=ACC, label="clicks received")
ax[1].set_xticks(x)
ax[1].set_xticklabels(cd_.upload_days, fontsize=8)
ax[1].set_xlabel("Days with an upload in the early window")
ax[1].set_ylabel("Share who stop publishing (%)")
ax[1].set_title("(b) Holding effort constant", loc="left", fontsize=9.5)
ax[1].legend(frameon=False, fontsize=7.5)
plt.savefig(OUT / "fig10_stage2_justification.png")
plt.close()

json.dump(R, open(OUT / "10_stage2_justification.json", "w"), indent=2)
print("\n-> fig10_stage2_justification.png, 10_stage2_justification.json")
