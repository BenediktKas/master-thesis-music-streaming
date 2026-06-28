"""Step 3 (VALIDATED): reference baseline — gradient boosting predicting
`is_inactive` from EARLY-WINDOW features only (no leakage; never uses `level`
or any label-window field). Every fancier model (Members B/C) should beat this.
"""
import json, polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from src import config

DER = config.DERIVED_DIR
FEATS = ["e_impr", "e_clicks", "e_likes", "e_shares", "e_comments", "e_viewcomment",
         "e_homepage", "e_active_days", "e_avg_pos", "e_click_rate", "e_like_rate",
         "e_avg_view_time"]


def main():
    df = pl.read_parquet(DER / "user_modeling_table.parquet").with_columns(
        [pl.col(c).fill_null(0.0) for c in FEATS])
    assert not (set(FEATS) & config.FORBIDDEN_FEATURES), "leakage: forbidden feature in set"

    def XY(s):
        d = df.filter(pl.col("split") == s)
        return d.select(FEATS).to_numpy(), d["is_inactive"].to_numpy().astype(int)

    Xtr, ytr = XY("train"); Xte, yte = XY("test")
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.07, max_depth=6,
                                         l2_regularization=1.0, random_state=42)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    res = {"roc_auc": round(roc_auc_score(yte, p), 4),
           "pr_auc": round(average_precision_score(yte, p), 4),
           "test_base_rate": round(float(yte.mean()), 4)}
    json.dump(res, open(DER / "baseline_metrics.json", "w"), indent=2)
    print(res)


if __name__ == "__main__":
    main()
