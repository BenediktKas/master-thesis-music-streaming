"""Member F, step 6: second repair - identification over longer horizons.

Idea: the placebo fails on day-to-day comparisons because creators have "active
spells" in which they post and receive responses at the same time. Those spells
presumably last a few days, so a longer horizon should average them out.

Two designs:
  A) weekly block panel  feedback in week w-1 -> production in week w
  B) distributed lag with a gap: feedback over t-7..t-3 -> posting on day t
     (the immediately preceding days t-2 and t-1 are deliberately left out)
Both carry the same placebo in reversed time direction.

Design A passes the placebo test and becomes the preferred specification.

    python -m src.member_f_supply_side.longer_lags
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

# ------------------------------------- Feedback on legacy content (as in step 5)
md = pd.read_csv(RAW / "mlog_demographics.csv", usecols=["mlogId", "creatorId", "publishTime"])
LEG = md[md.publishTime > 30]
ms = pd.read_csv(RAW / "mlog_stats.csv")
mL = ms.merge(LEG[["mlogId", "creatorId"]], on="mlogId", how="inner")
fb = mL.groupby(["creatorId", "dt"]).agg(
    impr=("userImprssionCount", "sum"), likes=("userLikeCount", "sum"),
    comments=("userCommentCount", "sum"), shares=("userShareCount", "sum"),
    follows=("userFollowCreatorCount", "sum")).reset_index()

p = pd.read_parquet(OUT / "creator_panel.parquet")[["creatorId", "dt", "pub", "n_pub"]]
p = p.merge(fb, on=["creatorId", "dt"], how="left")
FB = ["impr", "likes", "comments", "shares", "follows"]
p[FB] = p[FB].fillna(0)

has_leg = set(LEG.creatorId.unique())
v = p.groupby("creatorId").pub.agg(["sum", "size"])
vary = set(v[(v["sum"] > 0) & (v["sum"] < v["size"])].index)
p = p[p.creatorId.isin(has_leg & vary)].copy()
print(f"Creators in the analysis: {p.creatorId.nunique():,}")


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


def twoway_fit(df, ycol, cols, unit="creatorId", time="t"):
    dm = pd.DataFrame(index=df.index)
    for c in cols + [ycol]:
        dm[c] = (df[c] - df.groupby(unit)[c].transform("mean")
                 - df.groupby(time)[c].transform("mean") + df[c].mean())
    b, se = ols_cluster(dm[ycol].values, dm[cols].values, df[unit],
                        df[unit].nunique() + df[time].nunique())
    out = {}
    for i, c in enumerate(cols):
        t = b[i] / se[i] if se[i] > 0 else 0.0
        out[c] = {"coef": round(float(b[i]), 6), "se": round(float(se[i]), 6),
                  "t": round(float(t), 2),
                  "sig": "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96
                         else "*" if abs(t) > 1.64 else ""}
    return out


def report(title, real, plac, keys, fkeys, labels):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"{'Variable':<20}{'actual':>13}{'placebo':>15}{'ratio':>13}")
    print("-" * 62)
    rat = {}
    for k, fk, lb in zip(keys, fkeys, labels):
        rc, pc = real[k]["coef"], plac[fk]["coef"]
        rat[lb] = round(float(pc / rc), 2) if abs(rc) > 1e-9 else None
        print(f"{lb:<20}{rc:>+12.5f}{real[k]['sig']:<3}"
              f"{pc:>+12.5f}{plac[fk]['sig']:<3}"
              f"{(f'{rat[lb]:.2f}' if rat[lb] is not None else '-'):>10}")
    return rat


# =============================================== DESIGN A - weekly block panel
p["week"] = ((p.dt - 1) // 7) + 1          # weeks 1..5 (days 29-30 are the remainder)
w = p[p.week <= 4].groupby(["creatorId", "week"]).agg(
    pub=("pub", "max"), n_pub=("n_pub", "sum"),
    **{c: (c, "sum") for c in FB}).reset_index()
w = w.sort_values(["creatorId", "week"])
gw = w.groupby("creatorId", sort=False)
for c in FB:
    w["L_" + c] = np.log1p(gw[c].shift(1).clip(lower=0))
    w["F_" + c] = np.log1p(gw[c].shift(-1).clip(lower=0))
w["L_pub"] = gw["pub"].shift(1)
wA = w.dropna(subset=["L_likes", "F_likes", "L_pub"]).copy()
wA["t"] = wA.week
R["designA_obs"] = int(len(wA))
R["designA_creators"] = int(wA.creatorId.nunique())
print(f"\nDesign A (weekly panel): {len(wA):,} observations, "
      f"{wA.creatorId.nunique():,} creators, base rate {wA.pub.mean():.3f}")

TA = ["L_likes", "L_comments", "L_shares", "L_follows"]
FA = ["F_likes", "F_comments", "F_shares", "F_follows"]
LAB = ["Likes", "Comments", "Shares", "New followers"]
rA = twoway_fit(wA, "pub", TA + ["L_impr", "L_pub"])
pA = twoway_fit(wA, "pub", FA + ["L_impr", "L_pub"])
R["designA_real"], R["designA_placebo"] = rA, pA
R["designA_ratio"] = report("DESIGN A - feedback in week w-1  ->  upload in week w",
                            rA, pA, TA, FA, LAB)

# ========================================= DESIGN B - distributed lag with a gap
p = p.sort_values(["creatorId", "dt"])
g = p.groupby("creatorId", sort=False)
for c in FB:
    # mean over t-7..t-3 (the immediately preceding days deliberately left out)
    p["LL_" + c] = sum(g[c].shift(k) for k in range(3, 8)) / 5
    p["FF_" + c] = sum(g[c].shift(-k) for k in range(3, 8)) / 5
p["L1_pub"] = g["pub"].shift(1)
b = p.dropna(subset=["LL_likes", "FF_likes", "L1_pub"]).copy()
for c in FB:
    b["LL_" + c] = np.log1p(b["LL_" + c].clip(lower=0))
    b["FF_" + c] = np.log1p(b["FF_" + c].clip(lower=0))
b["t"] = b.dt
R["designB_obs"] = int(len(b))
R["designB_creators"] = int(b.creatorId.nunique())
print(f"\nDesign B (lag t-7..t-3): {len(b):,} observations, "
      f"{b.creatorId.nunique():,} creators, base rate {b.pub.mean():.3f}")

TB = ["LL_likes", "LL_comments", "LL_shares", "LL_follows"]
FB2 = ["FF_likes", "FF_comments", "FF_shares", "FF_follows"]
rB = twoway_fit(b, "pub", TB + ["LL_impr", "L1_pub"])
pB = twoway_fit(b, "pub", FB2 + ["FF_impr", "L1_pub"])
R["designB_real"], R["designB_placebo"] = rB, pB
R["designB_ratio"] = report("DESIGN B - feedback t-7..t-3  ->  upload on day t",
                            rB, pB, TB, FB2, LAB)

# --------------------------------------------------------------------- Figure
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6))
for k, (title, real, plac, keys, fkeys) in enumerate(
        [("(a) Weekly block panel", rA, pA, TA, FA),
         ("(b) Daily, lag t-7 to t-3", rB, pB, TB, FB2)]):
    xs = np.arange(4)
    wd = .36
    rc = [real[v]["coef"] for v in keys]
    rs = [1.96 * real[v]["se"] for v in keys]
    pc = [plac[v]["coef"] for v in fkeys]
    ps = [1.96 * plac[v]["se"] for v in fkeys]
    ax[k].bar(xs - wd / 2, rc, wd, yerr=rs, color=ACC, label="Actual (preceding period)",
              error_kw=dict(lw=.8, ecolor="#333", capsize=2.5))
    ax[k].bar(xs + wd / 2, pc, wd, yerr=ps, color=ORA, label="Placebo (following period)",
              error_kw=dict(lw=.8, ecolor="#333", capsize=2.5))
    ax[k].axhline(0, color="#333", lw=.8)
    ax[k].set_xticks(xs)
    ax[k].set_xticklabels(LAB, fontsize=7.5, rotation=12)
    ax[k].set_title(title, loc="left", fontsize=9.5)
    ax[k].set_ylabel("Effect on P(upload)" if k == 0 else "")
ax[0].legend(frameon=False, fontsize=7.5)
plt.savefig(OUT / "fig7_longer_lags.png")
plt.close()

json.dump(R, open(OUT / "06_longer_lags.json", "w"), indent=2)
print("\n-> fig7_longer_lags.png, 06_longer_lags.json")
