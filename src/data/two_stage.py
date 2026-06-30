"""Two-stage early-prediction model.

  Stage 1 (retention): among all week-1 users, predict whether the user RETURNS
                       to the feed (vs. churns) in days 8-30.
  Stage 2 (activeness): among returning users, predict INACTIVE (zero clicks).

Reporting the two separately avoids the survivorship bias of modelling activeness
alone, and decomposes the overall outcome as
  P(engaged) = P(return) * P(click | return).
Both stages use early-window features only and the shared hash split.

    python -m src.data.two_stage
"""
import json, polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from src import config

DER = config.DERIVED_DIR
BEHAV = ["e_impr", "e_clicks", "e_likes", "e_shares", "e_comments", "e_viewcomment",
         "e_homepage", "e_active_days", "e_avg_pos", "e_click_rate", "e_like_rate", "e_avg_view_time"]


def fit_report(df, feats, target):
    df = df.with_columns([pl.col(c).fill_null(0.0) for c in feats])
    assert not (set(feats) & config.FORBIDDEN_FEATURES), "leakage guard"
    def XY(s):
        d = df.filter(pl.col("split") == s)
        return d.select(feats).to_numpy(), d[target].to_numpy().astype(int)
    Xtr, ytr = XY("train"); Xte, yte = XY("test")
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.07, max_depth=6,
                                         l2_regularization=1.0, random_state=42).fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return {"n_train": int(len(ytr)), "n_test": int(len(yte)),
            "base_rate": round(float(yte.mean()), 4),
            "roc_auc": round(roc_auc_score(yte, p), 4),
            "pr_auc": round(average_precision_score(yte, p), 4)}


def _add_tenure(df):
    """Join non-leaky platform tenure (registeredMonthCnt) if available."""
    ud = config.RAW_DIR.parent / "user_demographics.csv"
    if ud.exists() and "registeredMonthCnt" not in df.columns:
        u = pl.read_csv(ud).select(["userId", "registeredMonthCnt"])
        df = df.join(u, on="userId", how="left")
    return df


def main():
    # tenure is a non-leaky covariate (fixed before the window); include when present
    TEN = ["registeredMonthCnt"]

    # Stage 1 — retention (churn) over all early-window users
    ret = _add_tenure(pl.read_parquet(DER / "user_retention_table.parquet"))
    feats1 = BEHAV + [c for c in ["ct_video_share_seen", "ct_seen_n_content"] if c in ret.columns]
    feats1 += [c for c in TEN if c in ret.columns]
    s1 = fit_report(ret, feats1, "churned")

    # Stage 2 — activeness among returning users (existing study population)
    mt = _add_tenure(pl.read_parquet(DER / "user_modeling_table.parquet"))
    feats2 = BEHAV + [c for c in TEN if c in mt.columns]
    s2 = fit_report(mt, feats2, "is_inactive")

    out = {"stage1_retention": s1, "stage2_activeness": s2}
    json.dump(out, open(DER / "two_stage_metrics.json", "w"), indent=2)
    print("STAGE 1 — retention (predict churn among week-1 users)")
    print(f"  users: train {s1['n_train']:,} / test {s1['n_test']:,} | churn base rate {s1['base_rate']}")
    print(f"  ROC-AUC {s1['roc_auc']} | PR-AUC {s1['pr_auc']}")
    print("STAGE 2 — activeness (predict inactive among returning users)")
    print(f"  users: train {s2['n_train']:,} / test {s2['n_test']:,} | inactive base rate {s2['base_rate']}")
    print(f"  ROC-AUC {s2['roc_auc']} | PR-AUC {s2['pr_auc']}")


if __name__ == "__main__":
    main()
