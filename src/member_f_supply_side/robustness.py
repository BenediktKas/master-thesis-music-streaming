"""Member F, step 7: robustness of the weekly-block finding.

Design A showed that likes in block w-1 raise the probability of producing in
block w, with a clean placebo. Before that counts as a finding it has to survive
the arbitrary choices the design makes:

  1. block length   5 / 6 / 7 / 10 days
  2. outcome        binary (publishes at all) vs. number of uploads
  3. sample         legacy-only creators vs. all creators
  4. placebo        carried along in every variant

That is 4 x 2 x 2 = 16 specifications. The four ten-day variants are degenerate:
differencing leaves a single usable period, the fixed effects absorb all remaining
variation, and every coefficient comes out exactly zero. They are reported for
completeness; the discussion refers to the twelve informative specifications.

    python -m src.member_f_supply_side.robustness
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

fb_leg = (ms.merge(LEG[["mlogId", "creatorId"]], on="mlogId", how="inner")
          .groupby(["creatorId", "dt"])
          .agg(impr=("userImprssionCount", "sum"), likes=("userLikeCount", "sum"),
               comments=("userCommentCount", "sum"), shares=("userShareCount", "sum"),
               follows=("userFollowCreatorCount", "sum")).reset_index())
fb_all = (ms.merge(md[["mlogId", "creatorId"]], on="mlogId", how="inner")
          .groupby(["creatorId", "dt"])
          .agg(impr=("userImprssionCount", "sum"), likes=("userLikeCount", "sum"),
               comments=("userCommentCount", "sum"), shares=("userShareCount", "sum"),
               follows=("userFollowCreatorCount", "sum")).reset_index())

base = pd.read_parquet(OUT / "creator_panel.parquet")[["creatorId", "dt", "pub", "n_pub"]]
FB = ["impr", "likes", "comments", "shares", "follows"]
v = base.groupby("creatorId").pub.agg(["sum", "size"])
VARY = set(v[(v["sum"] > 0) & (v["sum"] < v["size"])].index)
LEGC = set(LEG.creatorId.unique())


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


def fit(df, ycol, cols):
    dm = pd.DataFrame(index=df.index)
    for c in cols + [ycol]:
        dm[c] = (df[c] - df.groupby("creatorId")[c].transform("mean")
                 - df.groupby("blk")[c].transform("mean") + df[c].mean())
    b, se = ols_cluster(dm[ycol].values, dm[cols].values, df.creatorId,
                        df.creatorId.nunique() + df.blk.nunique())
    t = b[0] / se[0] if se[0] > 0 else 0.0
    return float(b[0]), float(se[0]), float(t)


def spec(block, outcome, sample):
    fb = fb_leg if sample == "legacy" else fb_all
    keep = (VARY & LEGC) if sample == "legacy" else VARY
    p = base[base.creatorId.isin(keep)].merge(fb, on=["creatorId", "dt"], how="left")
    p[FB] = p[FB].fillna(0)
    nb = 30 // block
    p = p[p.dt <= nb * block].copy()
    p["blk"] = ((p.dt - 1) // block) + 1
    agg = {c: (c, "sum") for c in FB}
    agg["pub"] = ("pub", "max")
    agg["n_pub"] = ("n_pub", "sum")
    w = p.groupby(["creatorId", "blk"]).agg(**agg).reset_index().sort_values(["creatorId", "blk"])
    gw = w.groupby("creatorId", sort=False)
    for c in FB:
        w["L_" + c] = np.log1p(gw[c].shift(1).clip(lower=0))
        w["F_" + c] = np.log1p(gw[c].shift(-1).clip(lower=0))
    w["L_pub"] = gw["pub"].shift(1)
    w = w.dropna(subset=["L_likes", "F_likes", "L_pub"])
    if len(w) < 2000:
        return None
    y = "pub" if outcome == "binary" else "n_pub"
    if outcome == "count":
        w = w.assign(n_pub=np.log1p(w.n_pub))
    ctrl = ["L_comments", "L_shares", "L_follows", "L_impr", "L_pub"]
    rb, rse, rt = fit(w, y, ["L_likes"] + ctrl)
    fctrl = ["F_comments", "F_shares", "F_follows", "F_impr", "L_pub"]
    pb, pse, pt = fit(w, y, ["F_likes"] + fctrl)
    return {"block_days": block, "outcome": outcome, "sample": sample,
            "n_obs": int(len(w)), "n_creators": int(w.creatorId.nunique()),
            "periods": int(w.blk.nunique()), "base_rate": round(float(w[y].mean()), 4),
            "likes_coef": round(rb, 5), "likes_se": round(rse, 5), "likes_t": round(rt, 2),
            "placebo_coef": round(pb, 5), "placebo_t": round(pt, 2),
            "ratio": round(pb / rb, 2) if abs(rb) > 1e-9 else None}


rows = []
print("Estimating specifications ...")
for blk in [5, 6, 7, 10]:
    for out_ in ["binary", "count"]:
        for smp in ["legacy", "all"]:
            r = spec(blk, out_, smp)
            if r:
                rows.append(r)
res = pd.DataFrame(rows)
R["specifications"] = rows


def stars(t):
    return "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96 else "*" if abs(t) > 1.64 else ""


print("\n" + "=" * 104)
print("ROBUSTNESS - effect of likes in the preceding block on production in the next")
print("=" * 104)
show = res.copy()
show["Likes (t)"] = show.apply(
    lambda r: f"{r.likes_coef:+.5f} ({r.likes_t:+.2f})" + stars(r.likes_t), axis=1)
show["Placebo (t)"] = show.apply(
    lambda r: f"{r.placebo_coef:+.5f} ({r.placebo_t:+.2f})" + stars(r.placebo_t), axis=1)
print(show[["block_days", "outcome", "sample", "periods", "n_obs", "n_creators",
            "base_rate", "Likes (t)", "Placebo (t)"]].to_string(index=False))

informative = res[res.block_days != 10]
sig = informative[informative.likes_t.abs() > 1.96]
clean = sig[sig.placebo_t.abs() < 1.96]
R["n_specs"] = len(res)
R["n_informative"] = int(len(informative))
R["n_significant"] = int(len(sig))
R["n_significant_and_clean_placebo"] = int(len(clean))
R["median_effect_informative"] = round(float(informative.likes_coef.median()), 5)
print(f"\nSpecifications in total                  : {len(res)}")
print(f"of which informative (excl. 10-day blocks): {len(informative)}")
print(f"likes effect significant at 5%            : {len(sig)}")
print(f"of those WITH a clean placebo             : {len(clean)}")
print(f"Median effect over the informative specs  : {informative.likes_coef.median():+.5f}")

# --------------------------------------------------------------------- Figure
fig, ax = plt.subplots(figsize=(7.4, 4))
res2 = res.sort_values(["outcome", "sample", "block_days"]).reset_index(drop=True)
ys = np.arange(len(res2))
ax.errorbar(res2.likes_coef, ys - .13, xerr=1.96 * res2.likes_se, fmt="s",
            color=ACC, ms=4.5, lw=0, elinewidth=1.1, capsize=2.5,
            label="Likes in preceding block")
ax.plot(res2.placebo_coef, ys + .13, "o", color=ORA, ms=4.5,
        label="Placebo (following block)")
ax.axvline(0, color="#333", lw=.9)
ax.set_yticks(ys)
ax.set_yticklabels([f"{r.block_days}-day blocks · {r.outcome} · "
                    f"{'legacy only' if r.sample == 'legacy' else 'all content'}"
                    for r in res2.itertuples()], fontsize=7.5)
ax.set_xlabel("Effect on production in the following block")
ax.set_title("Robustness across block length, outcome and sample", loc="left", fontsize=9.5)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
ax.invert_yaxis()
plt.savefig(OUT / "fig8_robustness.png")
plt.close()

res.to_csv(OUT / "07_robustness.csv", index=False)
json.dump(R, open(OUT / "07_robustness.json", "w"), indent=2)
print("\n-> fig8_robustness.png, 07_robustness.csv")
