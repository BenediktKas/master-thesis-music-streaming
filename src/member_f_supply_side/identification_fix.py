"""Member F, step 5: first repair of the identification strategy.

PROBLEM (step 4): the placebo test fails. Feedback at t+1 predicts posting at t
just as strongly as feedback at t-1. The suspected cause is a mechanical channel:
a creator who uploads on day t creates a new card, which then collects feedback on
days t and t+1. So posting -> feedback, not feedback -> posting.

FIX: count feedback only on LEGACY content, i.e. cards published before the
observation window (publishTime > 30). Their response cannot have been produced
mechanically by an upload made today.

Decoding of publishTime (undocumented in the data description):
    age of the card in days; publication day within the index = 31 - publishTime.
    Verified: no card records statistics before its publication day (100%).

NOTE: this repair helps but is NOT sufficient - the placebo still fails.
See longer_lags.py for the second repair, which passes.

    python -m src.member_f_supply_side.identification_fix
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

# ---------------------------------------------- Feedback on legacy content only
md = pd.read_csv(RAW / "mlog_demographics.csv",
                 usecols=["mlogId", "creatorId", "publishTime"])
md["pub_day"] = 31 - md.publishTime
LEGACY = md[md.publishTime > 30]                    # published before November
R["legacy_cards"] = int(len(LEGACY))
R["legacy_share"] = round(float(len(LEGACY) / len(md)), 4)
R["creators_with_legacy"] = int(LEGACY.creatorId.nunique())
print(f"Legacy cards: {len(LEGACY):,} ({len(LEGACY) / len(md):.1%}) "
      f"| creators: {LEGACY.creatorId.nunique():,}")

ms = pd.read_csv(RAW / "mlog_stats.csv")
mL = ms.merge(LEGACY[["mlogId", "creatorId"]], on="mlogId", how="inner")
print(f"Card-day rows on legacy content: {len(mL):,}")

fbL = mL.groupby(["creatorId", "dt"]).agg(
    Limpr=("userImprssionCount", "sum"), Lclicks=("userClickCount", "sum"),
    Llikes=("userLikeCount", "sum"), Lcomments=("userCommentCount", "sum"),
    Lshares=("userShareCount", "sum"), Lfollows=("userFollowCreatorCount", "sum"),
).reset_index()

# ------------------------------------------------------------ Rebuild the panel
p = pd.read_parquet(OUT / "creator_panel.parquet")
keep = ["creatorId", "dt", "pub", "n_pub", "weekend", "c_followers", "c_tenure", "split"]
p = p[keep].merge(fbL, on=["creatorId", "dt"], how="left")
FBL = ["Limpr", "Lclicks", "Llikes", "Lcomments", "Lshares", "Lfollows"]
p[FBL] = p[FBL].fillna(0)
p = p.sort_values(["creatorId", "dt"]).reset_index(drop=True)
g = p.groupby("creatorId", sort=False)
for c in FBL + ["pub"]:
    p["L1_" + c] = g[c].shift(1)
    p["F1_" + c] = g[c].shift(-1)
p["L2_pub"] = g["pub"].shift(2)

# only creators with legacy content AND within-variation in posting
has_legacy = set(LEGACY.creatorId.unique())
v = p.groupby("creatorId").pub.agg(["sum", "size"])
varying = set(v[(v["sum"] > 0) & (v["sum"] < v["size"])].index)
sel = has_legacy & varying
d = p[p.creatorId.isin(sel)].dropna(subset=["L1_Llikes", "L2_pub", "F1_Llikes"]).copy()
R["n_obs"] = int(len(d))
R["n_creators"] = int(d.creatorId.nunique())
R["base_publish_rate"] = round(float(d.pub.mean()), 4)
print(f"Estimation sample: {len(d):,} observations | {d.creatorId.nunique():,} creators "
      f"| base rate {d.pub.mean():.4f}")

for c in ["Llikes", "Lcomments", "Lshares", "Lfollows", "Limpr"]:
    d["l_" + c] = np.log1p(d["L1_" + c].clip(lower=0))
    d["f_" + c] = np.log1p(d["F1_" + c].clip(lower=0))

TREAT = ["l_Llikes", "l_Lcomments", "l_Lshares", "l_Lfollows"]
FTREAT = ["f_Llikes", "f_Lcomments", "f_Lshares", "f_Lfollows"]
EXPO, DYN = ["l_Limpr"], ["L1_pub", "L2_pub"]


def twoway(df, cols):
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = (df[c] - df.groupby("creatorId")[c].transform("mean")
                  - df.groupby("dt")[c].transform("mean") + df[c].mean())
    return out


def ols_cluster(y, X, cluster, k_abs=0):
    X = np.column_stack([X, np.ones(len(X))]).astype(np.float64)
    y = np.asarray(y, dtype=np.float64)
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    A = pd.DataFrame(X * e[:, None])
    A["g"] = cluster.values
    S = A.groupby("g").sum().values
    V = XtX_inv @ (S.T @ S) @ XtX_inv
    G = len(np.unique(cluster))
    n = len(y)
    V *= (G / (G - 1)) * ((n - 1) / max(n - X.shape[1] - k_abs, 1))
    return b[:-1], np.sqrt(np.abs(np.diag(V)))[:-1]


def run(name, treat):
    cols = treat + EXPO + DYN
    dm = twoway(d, cols + ["pub"])
    b, se = ols_cluster(dm.pub.values, dm[cols].values, d.creatorId,
                        d.creatorId.nunique() + 30)
    res = {}
    for i, c in enumerate(cols):
        t = b[i] / se[i] if se[i] > 0 else 0.0
        res[c] = {"coef": float(b[i]), "se": float(se[i]), "t": float(t),
                  "sig": "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96
                         else "*" if abs(t) > 1.64 else ""}
    R[name] = {k: {kk: round(vv, 6) if isinstance(vv, float) else vv
                   for kk, vv in val.items()} for k, val in res.items()}
    return res


print("\n" + "=" * 74)
print("FEEDBACK ON LEGACY CONTENT (t-1)  ->  PUBLISHING ON DAY t")
print("Creator and day fixed effects, control for prior posting")
print("=" * 74)
real = run("main_legacy", TREAT)
plac = run("placebo_legacy", FTREAT)

lbl = {"Llikes": "Likes", "Lcomments": "Comments", "Lshares": "Shares",
       "Lfollows": "New followers", "Limpr": "Impressions (exposure)"}
print(f"{'Variable':<26}{'Coefficient':>14}{'Std. error':>13}{'t':>8}")
print("-" * 62)
for v in TREAT + EXPO:
    r = real[v]
    print(f"{lbl[v[2:]]:<26}{r['coef']:>+14.5f}{r['se']:>13.5f}{r['t']:>8.2f} {r['sig']}")
for v in DYN:
    r = real[v]
    label = "Uploaded yesterday" if v == "L1_pub" else "Uploaded two days ago"
    print(f"{label:<26}{r['coef']:>+14.5f}{r['se']:>13.5f}{r['t']:>8.2f} {r['sig']}")

print("\n" + "=" * 74)
print("PLACEBO - legacy feedback from TOMORROW (t+1) on posting TODAY")
print("=" * 74)
print(f"{'Variable':<26}{'actual (t-1)':>14}{'placebo (t+1)':>16}{'ratio':>13}")
print("-" * 70)
ratios = {}
for v, fv in zip(TREAT, FTREAT):
    rc, pc = real[v]["coef"], plac[fv]["coef"]
    ratios[v] = pc / rc if abs(rc) > 1e-9 else np.nan
    print(f"{lbl[v[2:]]:<26}{rc:>+13.5f}{real[v]['sig']:<3}"
          f"{pc:>+13.5f}{plac[fv]['sig']:<3}{ratios[v]:>11.2f}")
R["placebo_ratio"] = {k: round(float(v), 3) for k, v in ratios.items()}
print("\nRatio near 0 = placebo is clean. Near 1 = the effect is an artefact.")

# --------------------------------------------------------------------- Figure
fig, ax = plt.subplots(figsize=(6.2, 3.6))
xs = np.arange(len(TREAT))
w = .36
rc = [real[v]["coef"] for v in TREAT]
rs = [1.96 * real[v]["se"] for v in TREAT]
pc = [plac[v]["coef"] for v in FTREAT]
ps = [1.96 * plac[v]["se"] for v in FTREAT]
ax.bar(xs - w / 2, rc, w, yerr=rs, color=ACC, label="Actual: feedback at t-1",
       error_kw=dict(lw=.8, ecolor="#333", capsize=2.5))
ax.bar(xs + w / 2, pc, w, yerr=ps, color=ORA, label="Placebo: feedback at t+1",
       error_kw=dict(lw=.8, ecolor="#333", capsize=2.5))
ax.axhline(0, color="#333", lw=.8)
ax.set_xticks(xs)
ax.set_xticklabels([lbl[v[2:]] for v in TREAT], fontsize=8)
ax.set_ylabel("Effect on P(upload)")
ax.set_title("Placebo test after restricting to legacy content", loc="left", fontsize=9.5)
ax.legend(frameon=False, fontsize=7.5)
plt.savefig(OUT / "fig6_placebo_legacy.png")
plt.close()

json.dump(R, open(OUT / "05_identification_fix.json", "w"), indent=2)
print("\n-> fig6_placebo_legacy.png, 05_identification_fix.json")
