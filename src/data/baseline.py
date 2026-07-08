"""Step 3 (VALIDATED): reference baseline — gradient boosting predicting
`is_inactive` from EARLY-WINDOW features only (no leakage; never uses `level`
or any label-window field). Every fancier model (Members B/C) should beat this.
"""
import argparse, json, numpy as np, polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from src import config

DER = config.DERIVED_DIR
FEATS = ["e_impr", "e_clicks", "e_likes", "e_shares", "e_comments", "e_viewcomment",
         "e_homepage", "e_active_days", "e_avg_pos", "e_click_rate", "e_like_rate",
         "e_avg_view_time"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=1, help="repeat with N seeds and report mean +/- std")
    args = ap.parse_args()

    df = pl.read_parquet(DER / "user_modeling_table.parquet").with_columns(
        [pl.col(c).fill_null(0.0) for c in FEATS])
    assert not (set(FEATS) & config.FORBIDDEN_FEATURES), "leakage: forbidden feature in set"

    def XY(s):
        d = df.filter(pl.col("split") == s)
        return d.select(FEATS).to_numpy(), d["is_inactive"].to_numpy().astype(int)

    Xtr, ytr = XY("train"); Xte, yte = XY("test")
    aucs, prs = [], []
    for seed in range(args.n_seeds):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.07, max_depth=6,
                                             l2_regularization=1.0, random_state=seed)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        a, pr = roc_auc_score(yte, p), average_precision_score(yte, p)
        aucs.append(a); prs.append(pr)
        print(f"seed {seed}: ROC-AUC={a:.4f}  PR-AUC={pr:.4f}")
    aucs, prs = np.array(aucs), np.array(prs)
    if args.n_seeds > 1:
        print(f"\n=== tabular baseline over {args.n_seeds} seeds ===")
        print(f"ROC-AUC: mean {aucs.mean():.4f}  std {aucs.std():.4f}  range [{aucs.min():.4f}, {aucs.max():.4f}]")
        print(f"PR-AUC : mean {prs.mean():.4f}  std {prs.std():.4f}  range [{prs.min():.4f}, {prs.max():.4f}]")
    res = {"roc_auc": round(float(aucs.mean()), 4), "pr_auc": round(float(prs.mean()), 4),
           "test_base_rate": round(float(yte.mean()), 4)}
    json.dump(res, open(DER / "baseline_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
