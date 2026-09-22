"""Member F, step 1: build the creator-day panel.

Mirrors the demand-side study design on the supply side:
    early window   days 1-7    -> features
    label window   days 8-30   -> churn label
    fixed split    70/15/15 by hash of creatorId

    python -m src.member_f_supply_side.build_panel

Writes to data/derived/member_f/:
    creator_panel.parquet     (creator x day, 90,534 x 30)
    creator_features.parquet  (one row per creator, used by the models)
    01_panel_log.json
"""
import hashlib
import json
import time

import numpy as np
import pandas as pd

from src import config

RAW = config.RAW_DIR
OUT = config.DERIVED_DIR / "member_f"
OUT.mkdir(parents=True, exist_ok=True)
LOG = {}
t0 = time.time()

# ------------------------------------------------------------------- 1. Spine
cs = pd.read_csv(RAW / "creator_stats.csv").rename(columns={"PushlishMlogCnt": "n_pub"})
LOG["creator_stats_rows"] = len(cs)
LOG["creators"] = int(cs.creatorId.nunique())
print(f"creator_stats: {cs.shape}  creators={cs.creatorId.nunique():,}  dt={cs.dt.min()}-{cs.dt.max()}")

# Fill the panel out to all 30 days (not every creator has 30 rows on file).
allc = cs.creatorId.unique()
full = pd.MultiIndex.from_product([allc, range(1, 31)],
                                  names=["creatorId", "dt"]).to_frame(index=False)
cs = full.merge(cs, on=["creatorId", "dt"], how="left")
cs["n_pub"] = cs.n_pub.fillna(0).astype(int)
LOG["panel_rows_after_fill"] = len(cs)
LOG["rows_added_by_fill"] = len(cs) - LOG["creator_stats_rows"]
print(f"Panel filled out: {cs.shape}  (+{LOG['rows_added_by_fill']:,} rows)")

# -------------------------------------------------- 2. Feedback per creator-day
md = pd.read_csv(RAW / "mlog_demographics.csv",
                 usecols=["mlogId", "creatorId", "publishTime", "type", "songId", "artistId"])
ms = pd.read_csv(RAW / "mlog_stats.csv")
m = ms.merge(md[["mlogId", "creatorId"]], on="mlogId", how="inner")
LOG["mlog_join_rate"] = round(len(m) / len(ms), 5)
print(f"mlog_stats x mlog_demographics: match rate {LOG['mlog_join_rate']:.5f}")

fb = m.groupby(["creatorId", "dt"]).agg(
    impr=("userImprssionCount", "sum"),
    clicks=("userClickCount", "sum"),
    likes=("userLikeCount", "sum"),
    comments=("userCommentCount", "sum"),
    shares=("userShareCount", "sum"),
    viewcomments=("userViewCommentCount", "sum"),
    homepage=("userIntoPersonalHomepageCount", "sum"),
    follows=("userFollowCreatorCount", "sum"),
    cards_live=("mlogId", "nunique"),
).reset_index()
print(f"Feedback rows: {len(fb):,}  creators with feedback: {fb.creatorId.nunique():,}")

p = cs.merge(fb, on=["creatorId", "dt"], how="left")
FB = ["impr", "clicks", "likes", "comments", "shares", "viewcomments",
      "homepage", "follows", "cards_live"]
p[FB] = p[FB].fillna(0)

# ------------------------------------------------------- 3. Creator attributes
cd = pd.read_csv(RAW / "creator_demographics.csv").rename(columns={
    "follows": "c_follows_out",      # accounts the creator follows
    "followeds": "c_followers",      # accounts following the creator
    "level": "c_level",
    "gender": "c_gender",
    "registeredMonthCnt": "c_tenure",
    "creatorType": "c_type"})
p = p.merge(cd, on="creatorId", how="left")

# ---------------------------------------------------- 4. Outcome, lags, leads
p = p.sort_values(["creatorId", "dt"]).reset_index(drop=True)
p["pub"] = (p.n_pub > 0).astype(int)
g = p.groupby("creatorId", sort=False)
for c in ["likes", "comments", "shares", "follows", "impr", "clicks", "pub", "n_pub"]:
    p["L1_" + c] = g[c].shift(1)          # previous day -> treatment
    p["F1_" + c] = g[c].shift(-1)         # following day -> PLACEBO
for c in ["likes", "follows", "pub"]:
    p["L2_" + c] = g[c].shift(2)

# Calendar (1 Nov 2019 was a Friday)
p["weekday"] = (p.dt + 3) % 7             # 0 = Monday
p["weekend"] = p.weekday.isin([5, 6]).astype(int)

# ------------------------------------------- 5. Fixed split by creatorId hash
def part(cid: str) -> str:
    h = int(hashlib.md5(cid.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


split = pd.DataFrame({"creatorId": allc, "split": [part(c) for c in allc]})
p = p.merge(split, on="creatorId", how="left")
LOG["split"] = split.split.value_counts().to_dict()
print(f"Split (creators): {LOG['split']}")

p.to_parquet(OUT / "creator_panel.parquet", index=False)
print(f"\n-> creator_panel.parquet  {p.shape}")

# ============================================================ 6. Feature set
# early window days 1-7   ->  features       (mirrors the demand side)
# label window days 8-30  ->  churn label
early = p[p.dt <= 7]
late = p[p.dt >= 8]

E = early.groupby("creatorId").agg(
    e_pub_days=("pub", "sum"),
    e_n_pub=("n_pub", "sum"),
    e_impr=("impr", "sum"),
    e_clicks=("clicks", "sum"),
    e_likes=("likes", "sum"),
    e_comments=("comments", "sum"),
    e_shares=("shares", "sum"),
    e_follows=("follows", "sum"),
    e_viewcomments=("viewcomments", "sum"),
    e_homepage=("homepage", "sum"),
    e_cards_live=("cards_live", "max"),
).reset_index()

L = late.groupby("creatorId").agg(
    l_pub_days=("pub", "sum"),
    l_n_pub=("n_pub", "sum"),
    l_impr=("impr", "sum"),
    l_clicks=("clicks", "sum"),
).reset_index()

F = E.merge(L, on="creatorId", how="left").merge(cd, on="creatorId", how="left")
F = F.merge(split, on="creatorId", how="left")

# --- rates and derived features
F["e_ctr"] = np.where(F.e_impr > 0, F.e_clicks / F.e_impr, np.nan)
F["e_like_rate"] = np.where(F.e_impr > 0, F.e_likes / F.e_impr, np.nan)
F["e_engagement"] = F.e_likes + F.e_comments + F.e_shares + F.e_follows
F["e_engagement_per_impr"] = np.where(F.e_impr > 0, F.e_engagement / F.e_impr, np.nan)
F["e_active"] = (F.e_pub_days > 0).astype(int)     # published at all in the early window

# --- TARGETS (mirroring the demand side)
# Stage 1: churn = no publication at all during the label window.
F["churn"] = (F.l_pub_days.fillna(0) == 0).astype(int)
# Stage 2: "publishing into the void" = keeps publishing, content DOES circulate,
# but receives no clicks at all. Conditioning on l_impr > 0 separates a genuine
# absence of response from content that was never distributed.
F["void"] = np.where((F.churn == 0) & (F.l_impr.fillna(0) > 0),
                     ((F.l_clicks.fillna(0)) == 0).astype(int), np.nan)

F.to_parquet(OUT / "creator_features.parquet", index=False)

# ------------------------------------------------------------- 7. Headline numbers
pop = F[F.e_active == 1]                 # study population: active in the early window
LOG["creators_total"] = int(len(F))
LOG["creators_active_early"] = int(len(pop))
LOG["churn_rate_all"] = round(float(F.churn.mean()), 4)
LOG["churn_rate_studypop"] = round(float(pop.churn.mean()), 4)
LOG["void_rate_studypop"] = round(float(pop.void.mean(skipna=True)), 4)
LOG["void_n"] = int(pop.void.notna().sum())
LOG["excluded_never_circulated"] = int(((pop.churn == 0) & (pop.l_impr.fillna(0) == 0)).sum())
LOG["never_published_month"] = round(float(((F.e_pub_days + F.l_pub_days.fillna(0)) == 0).mean()), 4)
LOG["runtime_s"] = round(time.time() - t0)

print("\n" + "=" * 62)
print("STUDY POPULATION AND BASE RATES")
print("=" * 62)
print(f"Creators in total                        : {LOG['creators_total']:,}")
print(f"of which active in the early window      : {LOG['creators_active_early']:,}")
print(f"publish nothing during the whole month   : {LOG['never_published_month']:.1%}")
print(f"\nStage 1 - churn rate (study population)  : {LOG['churn_rate_studypop']:.1%}")
print(f"Stage 2 - publishing into the void       : {LOG['void_rate_studypop']:.1%}")
print(f"\nRuntime: {LOG['runtime_s']}s")

json.dump(LOG, open(OUT / "01_panel_log.json", "w"), indent=2)
print("-> creator_features.parquet, 01_panel_log.json")
