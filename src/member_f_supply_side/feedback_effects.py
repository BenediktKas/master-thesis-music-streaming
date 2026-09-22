"""Member F, step 4: does feedback affect subsequent production?

From prediction to lever. Specifications in increasing strictness:
    (1) pooled OLS                  - naive, all creators in one pot
    (2) + creator fixed effects     - each creator compared only with himself
    (3) + day fixed effects         - absorbs weekends and platform-wide events
    (4) + control for prior posting - against autocorrelation / reverse causality
    (5) PLACEBO: feedback from TOMORROW - must show no effect

Two-way demeaning on a balanced panel; standard errors clustered by creator.

NOTE: this specification FAILS its placebo test. The result is reported in the
chapter as the first of two failed attempts; see identification_fix.py and
longer_lags.py for the repairs.

    python -m src.member_f_supply_side.feedback_effects
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

p = pd.read_parquet(OUT / "creator_panel.parquet")

# --------------------------------------------------------- Estimation sample
# Only creators whose posting varies within the month identify a fixed-effects
# estimate; creators who always post or never post contribute nothing.
v = p.groupby("creatorId").pub.agg(["sum", "size"])
varying = v[(v["sum"] > 0) & (v["sum"] < v["size"])].index
R["creators_with_within_variation"] = int(len(varying))
print(f"Creators with within-variation in posting: {len(varying):,}")

d = p[p.creatorId.isin(varying)].dropna(subset=["L1_likes", "L2_pub", "F1_likes"]).copy()
R["n_obs"] = int(len(d))
R["n_creators"] = int(d.creatorId.nunique())
R["base_publish_rate"] = round(float(d.pub.mean()), 4)
print(f"Observations: {len(d):,} | creators: {d.creatorId.nunique():,} "
      f"| base publish rate: {d.pub.mean():.4f}")

# ------------------------------------------------------------------ Variables
for c in ["likes", "comments", "shares", "follows", "impr", "clicks"]:
    d["l_" + c] = np.log1p(d["L1_" + c])          # feedback on the previous day
    d["f_" + c] = np.log1p(d["F1_" + c])          # feedback TOMORROW (placebo)

TREAT = ["l_likes", "l_comments", "l_shares", "l_follows"]
EXPOSURE = ["l_impr"]
DYN = ["L1_pub", "L2_pub"]


def demean(df, cols, by):
    """Within transformation: subtract the group mean."""
    out = df[cols].copy()
    for c in cols:
        out[c] = df[c] - df.groupby(by)[c].transform("mean")
    return out


def twoway(df, cols):
    """Two-way demeaning (creator and day) on a balanced panel."""
    out = df[cols].copy()
    for c in cols:
        gm = df[c].mean()
        out[c] = (df[c] - df.groupby("creatorId")[c].transform("mean")
                  - df.groupby("dt")[c].transform("mean") + gm)
    return out


def ols_cluster(y, X, cluster, k_absorbed=0):
    """OLS with cluster-robust standard errors (clustered by creator)."""
    X = np.column_stack([X, np.ones(len(X))])
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    # Meat matrix, summed within clusters
    df_c = pd.DataFrame(X * e[:, None])
    df_c["g"] = cluster.values
    S = df_c.groupby("g").sum().values
    meat = S.T @ S
    V = XtX_inv @ meat @ XtX_inv
    G = len(np.unique(cluster))
    n, kk = len(y), X.shape[1] + k_absorbed
    V *= (G / (G - 1)) * ((n - 1) / max(n - kk, 1))
    se = np.sqrt(np.diag(V))
    return b[:-1], se[:-1]


def run(name, treat, extra, mode):
    cols = treat + EXPOSURE + extra
    if mode == "pooled":
        Xd, yd, kabs = d[cols].values, d.pub.values, 0
    elif mode == "creator":
        dm = demean(d, cols + ["pub"], "creatorId")
        Xd, yd, kabs = dm[cols].values, dm.pub.values, d.creatorId.nunique()
    else:  # twoway
        dm = twoway(d, cols + ["pub"])
        Xd, yd, kabs = dm[cols].values, dm.pub.values, d.creatorId.nunique() + 30
    b, se = ols_cluster(yd, Xd, d.creatorId, kabs)
    res = {}
    for i, c in enumerate(cols):
        t = b[i] / se[i] if se[i] > 0 else 0.0
        res[c] = {"coef": round(float(b[i]), 6), "se": round(float(se[i]), 6),
                  "t": round(float(t), 2),
                  "sig": "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96
                         else "*" if abs(t) > 1.64 else ""}
    R[name] = res
    return res


print("\n" + "=" * 78)
print("EFFECT OF FEEDBACK AT t-1 ON PUBLISHING ON DAY t")
print("=" * 78)

specs = [
    ("(1) Pooled OLS",                 TREAT, [],  "pooled"),
    ("(2) + creator FE",               TREAT, [],  "creator"),
    ("(3) + creator & day FE",         TREAT, [],  "twoway"),
    ("(4) + control for prior posting", TREAT, DYN, "twoway"),
]
results = {}
for nm, tr, ex, md in specs:
    results[nm] = run(nm, tr, ex, md)

hdr = f"{'Variable':<16}" + "".join(f"{n.split(')')[0] + ')':>15}" for n, *_ in specs)
print(hdr)
print("-" * len(hdr))
for v in TREAT + EXPOSURE + DYN:
    row = f"{v:<16}"
    for nm, *_ in specs:
        r = results[nm].get(v)
        cell = f"{r['coef']:+.5f}{r['sig']}" if r else "-"
        row += f"{cell:>15}"
    print(row)
print(f"\nBase publish rate: {d.pub.mean():.4f}   (*** p<0.01, ** p<0.05, * p<0.10)")
print("Standard errors clustered by creator.")

# --------------------------------------------------------------------- PLACEBO
print("\n" + "=" * 78)
print("PLACEBO - feedback from TOMORROW (t+1) on publishing TODAY (t)")
print("=" * 78)
FTREAT = ["f_likes", "f_comments", "f_shares", "f_follows"]
pl = run("(5) Placebo (lead)", FTREAT, DYN, "twoway")
print(f"{'Variable':<16}{'Coefficient':>14}{'t':>10}")
print("-" * 40)
for v in FTREAT:
    print(f"{v:<16}{pl[v]['coef']:>+14.5f}{pl[v]['t']:>10.2f}{pl[v]['sig']}")
actual = results["(4) + control for prior posting"]
print("\nfor comparison, the actual lag effects from column (4):")
for v in TREAT:
    print(f"{v:<16}{actual[v]['coef']:>+14.5f}{actual[v]['t']:>10.2f}{actual[v]['sig']}")

# -------------------------------------------------------------- Heterogeneity
print("\n" + "=" * 78)
print("HETEROGENEITY - by follower quartile (specification 4)")
print("=" * 78)
d["fq"] = pd.qcut(d.c_followers.rank(method="first"), 4,
                  labels=["Q1 (fewest)", "Q2", "Q3", "Q4 (most)"])
het = []
for q, g in d.groupby("fq", observed=True):
    if len(g) < 5000:
        continue
    cols = TREAT + EXPOSURE + DYN
    dm = g[cols + ["pub"]].copy()
    for c in cols + ["pub"]:
        gm = g[c].mean()
        dm[c] = (g[c] - g.groupby("creatorId")[c].transform("mean")
                 - g.groupby("dt")[c].transform("mean") + gm)
    b, se = ols_cluster(dm.pub.values, dm[cols].values, g.creatorId,
                        g.creatorId.nunique() + 30)
    het.append({"quartile": str(q), "n": len(g), "base_rate": round(float(g.pub.mean()), 4),
                **{c: round(float(b[i]), 5) for i, c in enumerate(TREAT)}})
R["heterogeneity_followers"] = het
print(pd.DataFrame(het).to_string(index=False))

# --------------------------------------------------------------------- Figure
lbl = {"l_likes": "Likes", "l_comments": "Comments",
       "l_shares": "Shares", "l_follows": "New followers"}
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
xs = np.arange(len(TREAT))
for j, (nm, c, mk) in enumerate([("(2) + creator FE", GREY, "o"),
                                 ("(4) + control for prior posting", ACC, "s")]):
    co = [results[nm][v]["coef"] for v in TREAT]
    se = [results[nm][v]["se"] for v in TREAT]
    ax[0].errorbar(xs + (j - .5) * .16, co, yerr=[1.96 * s for s in se], fmt=mk,
                   color=c, ms=5, lw=0, elinewidth=1.2, capsize=3, label=nm)
ax[0].axhline(0, color="#333", lw=.8)
ax[0].set_xticks(xs)
ax[0].set_xticklabels([lbl[v] for v in TREAT], fontsize=8)
ax[0].set_ylabel("Effect on P(upload) the following day")
ax[0].set_title("(a) Feedback effects, 95% intervals", loc="left", fontsize=9.5)
ax[0].legend(frameon=False, fontsize=7.5)

co_r = [actual[v]["coef"] for v in TREAT]
co_p = [pl[v.replace("l_", "f_")]["coef"] for v in TREAT]
w = .36
ax[1].bar(xs - w / 2, co_r, w, color=ACC, label="Actual: feedback yesterday")
ax[1].bar(xs + w / 2, co_p, w, color=ORA, label="Placebo: feedback tomorrow")
ax[1].axhline(0, color="#333", lw=.8)
ax[1].set_xticks(xs)
ax[1].set_xticklabels([lbl[v] for v in TREAT], fontsize=8)
ax[1].set_ylabel("Effect on P(upload) today")
ax[1].set_title("(b) Placebo test", loc="left", fontsize=9.5)
ax[1].legend(frameon=False, fontsize=7.5)
plt.savefig(OUT / "fig5_feedback_effects.png")
plt.close()

json.dump(R, open(OUT / "04_feedback_effects.json", "w"), indent=2)
print("\n-> fig5_feedback_effects.png, 04_feedback_effects.json")
