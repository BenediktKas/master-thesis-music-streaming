"""Member F, step 8: heterogeneity on the preferred specification, plus summary.

Preferred specification (from robustness.py):
    7-day blocks, binary outcome, feedback restricted to legacy content,
    creator and block fixed effects, control for production in the preceding block.
    Likes effect +0.04004 (t=3.28), placebo -0.01445 (t=-1.04) -> clean.

This script asks for whom the effect holds: by follower count, by creator segment
and across all feedback types.

    python -m src.member_f_supply_side.heterogeneity
"""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config

RAW = config.RAW_DIR
OUT = config.DERIVED_DIR / "member_f"
OUT.mkdir(parents=True, exist_ok=True)
R = {}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
ACC, ORA, GREY = "#0D6E6E", "#C77B30", "#8899A0"

md = pd.read_csv(RAW / "mlog_demographics.csv", usecols=["mlogId", "creatorId", "publishTime"])
LEG = md[md.publishTime > 30]
ms = pd.read_csv(RAW / "mlog_stats.csv")
fb = (ms.merge(LEG[["mlogId", "creatorId"]], on="mlogId", how="inner")
      .groupby(["creatorId", "dt"])
      .agg(impr=("userImprssionCount", "sum"), likes=("userLikeCount", "sum"),
           comments=("userCommentCount", "sum"), shares=("userShareCount", "sum"),
           follows=("userFollowCreatorCount", "sum")).reset_index())
FB = ["impr", "likes", "comments", "shares", "follows"]

base = pd.read_parquet(OUT / "creator_panel.parquet")[
    ["creatorId", "dt", "pub", "n_pub", "c_followers", "c_tenure"]]
seg = pd.read_parquet(OUT / "creator_segments.parquet")
v = base.groupby("creatorId").pub.agg(["sum", "size"])
keep = set(v[(v["sum"] > 0) & (v["sum"] < v["size"])].index) & set(LEG.creatorId.unique())

p = base[base.creatorId.isin(keep)].merge(fb, on=["creatorId", "dt"], how="left")
p[FB] = p[FB].fillna(0)
p = p[p.dt <= 28].copy()
p["blk"] = ((p.dt - 1) // 7) + 1
w = p.groupby(["creatorId", "blk"]).agg(
    pub=("pub", "max"), c_followers=("c_followers", "first"),
    c_tenure=("c_tenure", "first"), **{c: (c, "sum") for c in FB}
).reset_index().sort_values(["creatorId", "blk"])
gw = w.groupby("creatorId", sort=False)
for c in FB:
    w["L_" + c] = np.log1p(gw[c].shift(1).clip(lower=0))
    w["F_" + c] = np.log1p(gw[c].shift(-1).clip(lower=0))
w["L_pub"] = gw["pub"].shift(1)
w = w.dropna(subset=["L_likes", "F_likes", "L_pub"]).merge(seg, on="creatorId", how="left")
print(f"Preferred specification: {len(w):,} observations, {w.creatorId.nunique():,} creators")


def ols_cluster(y, X, cl, k_abs=0):
    X = np.column_stack([X, np.ones(len(X))]).astype(float)
    y = np.asarray(y, float)
    Xi = np.linalg.pinv(X.T @ X)
    b = Xi @ (X.T @ y)
    e = y - X @ b
    A = pd.DataFrame(X * e[:, None])
    A["g"] = cl.values
    S = A.groupby("g").sum().values
    V = Xi @ (S.T @ S) @ Xi
    G, n = len(np.unique(cl)), len(y)
    V *= (G / (G - 1)) * ((n - 1) / max(n - X.shape[1] - k_abs, 1))
    return b[:-1], np.sqrt(np.abs(np.diag(V)))[:-1]


CTRL = ["L_comments", "L_shares", "L_follows", "L_impr", "L_pub"]


def fit(df, treat="L_likes", ycol="pub"):
    cols = [treat] + [c for c in CTRL if c != treat]
    dm = pd.DataFrame(index=df.index)
    for c in cols + [ycol]:
        dm[c] = (df[c] - df.groupby("creatorId")[c].transform("mean")
                 - df.groupby("blk")[c].transform("mean") + df[c].mean())
    b, se = ols_cluster(dm[ycol].values, dm[cols].values, df.creatorId,
                        df.creatorId.nunique() + df.blk.nunique())
    t = b[0] / se[0] if se[0] > 0 else 0.0
    return round(float(b[0]), 5), round(float(se[0]), 5), round(float(t), 2)


# ------------------------------------------------------------- Overall effect
bb, bse, bt = fit(w)
R["preferred"] = {"coef": bb, "se": bse, "t": bt, "base_rate": round(float(w.pub.mean()), 4),
                  "n_obs": int(len(w)), "n_creators": int(w.creatorId.nunique())}
R["preferred"]["relative_effect"] = round(bb / float(w.pub.mean()), 4)
print(f"\nOverall likes effect: {bb:+.5f} (t={bt:.2f}) | base rate {w.pub.mean():.4f} "
      f"| relative {bb / w.pub.mean():+.1%}")

# ----------------------------------------------------------- By follower count
w["fq"] = pd.qcut(w.c_followers.rank(method="first"), 4,
                  labels=["Q1 (fewest followers)", "Q2", "Q3", "Q4 (most followers)"])
het = []
for q, g in w.groupby("fq", observed=True):
    b, se, t = fit(g)
    het.append({"group": str(q), "n_obs": len(g), "n_creators": g.creatorId.nunique(),
                "median_followers": float(g.c_followers.median()),
                "base_rate": round(float(g.pub.mean()), 4),
                "coef": b, "se": se, "t": t,
                "relative": round(b / float(g.pub.mean()), 4)})
R["by_followers"] = het
print("\n" + "=" * 92)
print("HETEROGENEITY BY FOLLOWER COUNT")
print("=" * 92)
print(pd.DataFrame(het).to_string(index=False))

# --------------------------------------------------------------- By segment
segres = []
names = {0: "Professionals", 1: "Occasional creators"}
for s, g in w.dropna(subset=["segment"]).groupby("segment"):
    if len(g) < 2000:
        continue
    b, se, t = fit(g)
    segres.append({"segment": names.get(int(s), str(s)), "n_obs": len(g),
                   "n_creators": g.creatorId.nunique(),
                   "base_rate": round(float(g.pub.mean()), 4),
                   "coef": b, "se": se, "t": t,
                   "relative": round(b / float(g.pub.mean()), 4)})
R["by_segment"] = segres
print("\n" + "=" * 92)
print("HETEROGENEITY BY CREATOR SEGMENT")
print("=" * 92)
print(pd.DataFrame(segres).to_string(index=False) if segres else "too few observations")

# -------------------------------------------------- Every feedback type on its own
allf = {}
for tr, lb in [("L_likes", "Likes"), ("L_comments", "Comments"),
               ("L_shares", "Shares"), ("L_follows", "New followers"),
               ("L_impr", "Impressions (exposure)")]:
    b, se, t = fit(w, treat=tr)
    allf[lb] = {"coef": b, "se": se, "t": t,
                "sig": "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96
                       else "*" if abs(t) > 1.64 else ""}
R["all_feedback_types"] = allf
print("\n" + "=" * 60)
print("ALL FEEDBACK TYPES (preferred specification)")
print("=" * 60)
print(f"{'Type':<26}{'Coefficient':>13}{'t':>8}")
print("-" * 50)
for k, v_ in allf.items():
    print(f"{k:<26}{v_['coef']:>+13.5f}{v_['t']:>8.2f} {v_['sig']}")

# --------------------------------------------------------------------- Figure
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.5))
h = pd.DataFrame(het)
xs = np.arange(len(h))
ax[0].errorbar(xs, h.coef, yerr=1.96 * h.se, fmt="s", color=ACC, ms=6, lw=0,
               elinewidth=1.2, capsize=3)
ax[0].axhline(0, color="#333", lw=.9)
ax[0].set_xticks(xs)
ax[0].set_xticklabels(["Q1\n(fewest)", "Q2", "Q3", "Q4\n(most)"], fontsize=7.5)
ax[0].set_xlabel("Follower quartile")
ax[0].set_ylabel("Effect of likes on P(production)")
ax[0].set_title("(a) Effect by creator size", loc="left", fontsize=9.5)

lbs = list(allf.keys())
co = [allf[k]["coef"] for k in lbs]
se = [1.96 * allf[k]["se"] for k in lbs]
ys = np.arange(len(lbs))[::-1]
ax[1].errorbar(co, ys, xerr=se, fmt="s", color=ACC, ms=6, lw=0, elinewidth=1.2, capsize=3)
ax[1].axvline(0, color="#333", lw=.9)
ax[1].set_yticks(ys)
ax[1].set_yticklabels(lbs, fontsize=8)
ax[1].set_xlabel("Effect on P(production in following block)")
ax[1].set_title("(b) Only likes matter", loc="left", fontsize=9.5)
plt.savefig(OUT / "fig9_heterogeneity.png")
plt.close()

json.dump(R, open(OUT / "08_heterogeneity.json", "w"), indent=2)
print("\n-> fig9_heterogeneity.png, 08_heterogeneity.json")
