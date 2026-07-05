"""Member A robustness: silhouette vs k, GMM and HDBSCAN agreement, and
bootstrap stability of the k=3 k-means segmentation. Runs on a sample of the
study population for tractability. Prints a short report.

    python -m src.member_a_segmentation.robustness
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from src import config
from src.member_a_segmentation.segment import LOG_FEATS, RAW_FEATS, FEATS, load, matrix

DER = config.DERIVED_DIR


def main(sample=50000, seed=42):
    Xs = matrix(load())
    rng = np.random.default_rng(seed)
    S = Xs[rng.choice(len(Xs), min(sample, len(Xs)), replace=False)]

    print("Silhouette by k:")
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(S)
        print(f"  k={k}: {silhouette_score(S[:15000], km.labels_[:15000]):.3f}")

    k3 = KMeans(n_clusters=3, n_init=10, random_state=42).fit(S).labels_
    gmm = GaussianMixture(n_components=3, n_init=3, random_state=42).fit(S).predict(S)
    hdb = HDBSCAN(min_cluster_size=2000, min_samples=50).fit_predict(S)
    print(f"\nGMM(3) vs k-means(3): ARI={adjusted_rand_score(k3, gmm):.3f}")
    print(f"HDBSCAN: {len(set(hdb)) - (1 if -1 in hdb else 0)} clusters, "
          f"{(hdb == -1).mean():.1%} noise, ARI vs k-means={adjusted_rand_score(k3, hdb):.3f}")

    labs = [KMeans(n_clusters=3, n_init=3, random_state=b)
            .fit(S[rng.choice(len(S), len(S), replace=True)]).predict(S) for b in range(8)]
    aris = [adjusted_rand_score(labs[i], labs[j]) for i in range(len(labs)) for j in range(i + 1, len(labs))]
    print(f"\nk=3 bootstrap stability: mean pairwise ARI={np.mean(aris):.3f} (min {np.min(aris):.3f})")


if __name__ == "__main__":
    main()
