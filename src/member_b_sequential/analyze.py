"""Member B analysis: WHERE does the sequence model beat the tabular baseline?
Trains the full GRU once, scores the same test users with the baseline, and
breaks the comparison down by (a) early-sequence length and (b) activity segment
from Member A, plus an overall calibration check. Writes a JSON summary.

    python -m src.member_b_sequential.analyze --epochs 20
"""
import argparse, json, numpy as np, polars as pl
from src import config
from src.member_b_sequential.sequence_model import build_arrays, fit_predict, N_ACTION

DER = config.DERIVED_DIR
BEHAV = ["e_impr", "e_clicks", "e_likes", "e_shares", "e_comments", "e_viewcomment",
         "e_homepage", "e_active_days", "e_avg_pos", "e_click_rate", "e_like_rate", "e_avg_view_time"]


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier

    Xn, Xt, Xc, y, sp, users, nnum = build_arrays(None)
    te = sp == "test"
    y_te = y[te]
    seq_len = (Xn[te][:, :, N_ACTION] > 0).sum(1)          # pos>0 marks a real event
    test_users = users[te]

    # --- sequence GRU predictions on the test set ---
    _, p_gru = fit_predict(Xn, Xt, Xc, y, sp, "gru", args.epochs, seed=0)

    # --- tabular baseline on the SAME test users ---
    mt = pl.read_parquet(DER / "user_modeling_table.parquet").with_columns(
        [pl.col(c).fill_null(0.0) for c in BEHAV])
    def XY(s):
        d = mt.filter(pl.col("split") == s); return d, d.select(BEHAV).to_numpy(), d["is_inactive"].to_numpy().astype(int)
    _, Xtr, ytr = XY("train")
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.07, max_depth=6,
                                         l2_regularization=1.0, random_state=0).fit(Xtr, ytr)
    base_map = dict(zip(mt.filter(pl.col("split") == "test")["userId"].to_list(),
                        clf.predict_proba(mt.filter(pl.col("split") == "test").select(BEHAV).to_numpy())[:, 1]))
    p_base = np.array([base_map[u] for u in test_users])

    out = {"overall": {"gru_auc": round(auc(y_te, p_gru), 4), "base_auc": round(auc(y_te, p_base), 4)}}

    # --- by sequence length quartile ---
    qs = np.quantile(seq_len, [0.25, 0.5, 0.75])
    bins = np.digitize(seq_len, qs)
    out["by_length"] = []
    for b in range(4):
        m = bins == b
        out["by_length"].append({"bin": b, "n": int(m.sum()),
                                 "len_range": [int(seq_len[m].min()), int(seq_len[m].max())],
                                 "gru_auc": round(auc(y_te[m], p_gru[m]), 4),
                                 "base_auc": round(auc(y_te[m], p_base[m]), 4)})

    # --- by activity segment (Member A) ---
    seg_path = DER / "member_a" / "user_segments.parquet"
    if seg_path.exists():
        seg = dict(zip(*[pl.read_parquet(seg_path)[c].to_list() for c in ("userId", "segment")]))
        s_arr = np.array([seg.get(u, -1) for u in test_users])
        out["by_segment"] = []
        for s in sorted(set(s_arr) - {-1}):
            m = s_arr == s
            out["by_segment"].append({"segment": int(s), "n": int(m.sum()),
                                      "gru_auc": round(auc(y_te[m], p_gru[m]), 4),
                                      "base_auc": round(auc(y_te[m], p_base[m]), 4)})

    # --- calibration of the GRU (deciles of predicted prob) ---
    dec = np.clip((p_gru * 10).astype(int), 0, 9)
    out["calibration"] = [{"decile": d, "pred": round(float(p_gru[dec == d].mean()), 3),
                           "obs": round(float(y_te[dec == d].mean()), 3), "n": int((dec == d).sum())}
                          for d in range(10) if (dec == d).sum() > 0]

    json.dump(out, open(DER / "member_b_analysis.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
