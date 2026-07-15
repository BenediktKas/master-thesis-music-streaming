"""Member A robustness: silhouette vs k, GMM and HDBSCAN agreement, and
bootstrap stability of the k=3 k-means segmentation. Runs on a sample of the
study population for tractability. Prints a short report AND saves it to
data/derived/member_a_robustness.json.

    python -m src.member_a_segmentation.robustness
"""
import json, warnings; warnings.filterwarnings("ignore")
import numpy as np, polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from src import config
from src.member_a_segmentation.segment import LOG_FEATS, RAW_FEATS, FEATS, load, matrix

DER = config.DERIVED_DIR


def main(sample=50000, seed=42):
    Xs = matrix(load())
    rng = np.random.default_rng(seed)
    S = Xs[rng.choice(len(Xs), min(sample, len(Xs)), replace=False)]

    print("Silhouette by k:")
    sil = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(S)
        sil[str(k)] = round(float(silhouette_score(S[:15000], km.labels_[:15000])), 3)
        print(f"  k={k}: {sil[str(k)]:.3f}")

    k3 = KMeans(n_clusters=3, n_init=10, random_state=42).fit(S).labels_
    gmm = GaussianMixture(n_components=3, n_init=3, random_state=42).fit(S).predict(S)
    hdb = HDBSCAN(min_cluster_size=2000, min_samples=50).fit_predict(S)
    gmm_ari = round(float(adjusted_rand_score(k3, gmm)), 3)
    hdb_clusters = len(set(hdb)) - (1 if -1 in hdb else 0)
    hdb_noise = round(float((hdb == -1).mean()) * 100, 1)
    hdb_ari = round(float(adjusted_rand_score(k3, hdb)), 3)
    print(f"\nGMM(3) vs k-means(3): ARI={gmm_ari:.3f}")
    print(f"HDBSCAN: {hdb_clusters} clusters, {hdb_noise:.1f}% noise, ARI vs k-means={hdb_ari:.3f}")

    labs = [KMeans(n_clusters=3, n_init=3, random_state=b)
            .fit(S[rng.choice(len(S), len(S), replace=True)]).predict(S) for b in range(8)]
    aris = [adjusted_rand_score(labs[i], labs[j]) for i in range(len(labs)) for j in range(i + 1, len(labs))]
    boot_mean, boot_min = round(float(np.mean(aris)), 3), round(float(np.min(aris)), 3)
    print(f"\nk=3 bootstrap stability: mean pairwise ARI={boot_mean:.3f} (min {boot_min:.3f})")

    pca = PCA(n_components=2).fit(Xs)
    report = {"sample_size": int(len(S)), "silhouette_by_k": sil,
              "gmm3_vs_kmeans3_ari": gmm_ari,
              "hdbscan": {"n_clusters": int(hdb_clusters), "pct_noise": hdb_noise, "ari_vs_kmeans": hdb_ari},
              "kmeans3_bootstrap_mean_ari": boot_mean, "kmeans3_bootstrap_min_ari": boot_min,
              "pca_explained_variance_pct": {"PC1": round(float(pca.explained_variance_ratio_[0]) * 100, 1),
                                             "PC2": round(float(pca.explained_variance_ratio_[1]) * 100, 1)}}
    json.dump(report, open(DER / "member_a_robustness.json", "w"), indent=2)
    print(f"\nwrote {DER / 'member_a_robustness.json'}")


if __name__ == "__main__":
    main()
