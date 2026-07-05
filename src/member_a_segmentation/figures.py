"""Member A figure: PCA scatter of the study population coloured by segment.
Writes data/derived/member_a/segments_pca.png.

    python -m src.member_a_segmentation.figures
"""
import numpy as np, polars as pl
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src import config
from src.member_a_segmentation.segment import load, matrix

OUT = config.DERIVED_DIR / "member_a"


def main(sample=15000, seed=42):
    df = load().join(pl.read_parquet(OUT / "user_segments.parquet"), on="userId", how="inner")
    Xs = matrix(df)
    pca = PCA(n_components=2, random_state=seed).fit(Xs)
    Z = pca.transform(Xs); ev = pca.explained_variance_ratio_ * 100
    seg = df["segment"].to_numpy()
    rng = np.random.default_rng(seed); idx = rng.choice(len(Z), min(sample, len(Z)), replace=False)
    Zs, ss = Z[idx], seg[idx]
    names = {1: "Dormant majority", 2: "Engaged middle", 0: "Power users"}
    cols = {1: "#B0B7BF", 2: "#2E5C8A", 0: "#C0392B"}
    plt.figure(figsize=(7, 5.2))
    for s in [1, 2, 0]:
        m = ss == s
        plt.scatter(Zs[m, 0], Zs[m, 1], s=8, alpha=0.35 if s == 1 else 0.6, c=cols[s], label=names[s], edgecolors="none")
    plt.xlabel(f"PC1 ({ev[0]:.0f}% variance)"); plt.ylabel(f"PC2 ({ev[1]:.0f}% variance)")
    plt.title("User segments in the first two principal components")
    plt.legend(markerscale=2, framealpha=0.9, loc="upper right")
    plt.tight_layout(); plt.savefig(OUT / "segments_pca.png", dpi=150); plt.close()
    print(f"wrote {OUT}/segments_pca.png  (PC1={ev[0]:.0f}%, PC2={ev[1]:.0f}%)")


if __name__ == "__main__":
    main()
