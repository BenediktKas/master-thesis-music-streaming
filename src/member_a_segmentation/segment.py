"""Member A — behavioural + content-taste segmentation.

Clusters study-population users on early-window behaviour and content-taste
features (NOT the label), picks k by silhouette, profiles each segment by size,
activeness rate, and mean features, and saves assignments + a profile table +
figures. The resulting segments feed Members D and E.

Run after build_labels.py AND build_content_taste.py:
    python -m src.member_a_segmentation.segment
"""
import numpy as np, polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src import config

DER = config.DERIVED_DIR
OUTDIR = DER / "member_a"; OUTDIR.mkdir(parents=True, exist_ok=True)

# count-like features get a log1p; rates/shares used as-is
LOG_FEATS = ["e_impr", "e_clicks", "e_likes", "e_shares", "e_comments",
             "e_active_days", "ct_click_n_content", "ct_click_n_creators"]
RAW_FEATS = ["e_avg_pos", "e_click_rate", "e_avg_view_time",
             "ct_video_share_seen", "ct_video_share_click"]
FEATS = LOG_FEATS + RAW_FEATS


def load():
    mt = pl.read_parquet(DER / "user_modeling_table.parquet")
    ct = pl.read_parquet(DER / "user_content_taste.parquet")
    df = mt.join(ct, on="userId", how="inner")
    assert "level" not in FEATS, "leakage guard"
    return df.with_columns([pl.col(c).fill_null(0.0) for c in FEATS])


def matrix(df):
    X = df.select(FEATS).to_numpy().astype(float)
    for j, f in enumerate(FEATS):
        if f in LOG_FEATS:
            X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    return StandardScaler().fit_transform(X)


def choose_k(Xs, ks=range(3, 8), sample=10000, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xs), size=min(sample, len(Xs)), replace=False)
    scores = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(Xs[idx])
        scores[k] = silhouette_score(Xs[idx], km.labels_)
    best = max(scores, key=scores.get)
    print("silhouette by k:", {k: round(v, 3) for k, v in scores.items()}, "-> k =", best)
    return best, scores


def main():
    df = load()
    print(f"users (study ∩ taste): {df.height:,}")
    Xs = matrix(df)
    k, scores = choose_k(Xs)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    df = df.with_columns(pl.Series("segment", km.labels_))

    # profile: size, activeness, key feature means
    prof = (df.group_by("segment")
              .agg([pl.len().alias("n"),
                    pl.col("is_inactive").mean().round(3).alias("inactive_rate"),
                    *[pl.col(f).mean().round(2).alias(f) for f in
                      ["e_impr", "e_clicks", "e_active_days", "ct_video_share_click",
                       "ct_click_n_content"]]])
              .sort("segment"))
    prof.write_csv(OUTDIR / "segment_profiles.csv")
    df.select(["userId", "segment"]).write_parquet(OUTDIR / "user_segments.parquet")

    print("\nSegment profiles:")
    print(prof)

    # figures (polars -> lists; no pandas/pyarrow dependency)
    segs = [str(s) for s in prof["segment"].to_list()]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(segs, prof["n"].to_list(), color="#2E5C8A"); ax[0].set_title("Segment size"); ax[0].set_xlabel("segment")
    ax[1].bar(segs, prof["inactive_rate"].to_list(), color="#C0392B"); ax[1].set_title("Inactive rate by segment"); ax[1].set_xlabel("segment"); ax[1].set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(OUTDIR / "segments_overview.png", dpi=120); plt.close()
    print(f"\nwrote {OUTDIR}/ (user_segments.parquet, segment_profiles.csv, segments_overview.png)")


if __name__ == "__main__":
    main()
