"""Member F, step 2: descriptive analysis and creator segmentation.

Mirrors the demand-side user segmentation:
    log-transform count features -> standardise -> k-means
    number of clusters chosen by mean silhouette coefficient
    the churn label plays NO role in forming the clusters (only in describing them)

    python -m src.member_f_supply_side.descriptives_segments
"""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src import config

OUT = config.DERIVED_DIR / "member_f"
OUT.mkdir(parents=True, exist_ok=True)
R = {}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
ACC, GREY = "#0D6E6E", "#8899A0"

F = pd.read_parquet(OUT / "creator_features.parquet")
panel = pd.read_parquet(OUT / "creator_panel.parquet")

# ========================================================= 1. Descriptive stats
tot = F.e_n_pub + F.l_n_pub.fillna(0)
R["prod_total"] = int(tot.sum())
R["never_published"] = round(float((tot == 0).mean()), 4)
for q in [1, 5, 10, 20]:
    R[f"share_top{q}pct"] = round(float(tot.nlargest(int(len(F) * q / 100)).sum() / tot.sum()), 4)
# Gini coefficient of production
x = np.sort(tot.values)
n = len(x)
R["gini_production"] = round(float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())), 4)

pubday = panel.groupby("dt").pub.mean()
R["publish_rate_overall"] = round(float(panel.pub.mean()), 4)
R["publish_rate_weekday"] = round(float(panel[panel.weekend == 0].pub.mean()), 4)
R["publish_rate_weekend"] = round(float(panel[panel.weekend == 1].pub.mean()), 4)

print("=" * 66)
print("DESCRIPTIVES - CONTENT SUPPLY")
print("=" * 66)
print(f"Publications during the month, total : {R['prod_total']:,}")
print(f"Creators who never publish           : {R['never_published']:.1%}")
print(f"Output share of top 1% / 10%         : {R['share_top1pct']:.1%} / {R['share_top10pct']:.1%}")
print(f"Gini coefficient of production       : {R['gini_production']:.3f}")
print(f"Publish rate weekday / weekend       : {R['publish_rate_weekday']:.2%} / {R['publish_rate_weekend']:.2%}")

# ----------------------------------------------------- Figure 1: concentration
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
cum = np.cumsum(np.sort(tot.values)[::-1]) / tot.sum()
ax[0].plot(np.arange(1, len(cum) + 1) / len(cum) * 100, cum * 100, color=ACC, lw=1.8)
ax[0].axvline(10, color=GREY, ls=":", lw=1)
ax[0].axhline(R["share_top10pct"] * 100, color=GREY, ls=":", lw=1)
ax[0].set_xlabel("Share of creators (%, ranked by output)")
ax[0].set_ylabel("Cumulative share of content (%)")
ax[0].set_title("(a) Concentration of content production", loc="left", fontsize=9.5)
ax[0].set_xlim(0, 100)
ax[0].set_ylim(0, 101)
ax[0].annotate(f"Top 10% = {R['share_top10pct']:.0%}", xy=(10, R["share_top10pct"] * 100),
               xytext=(28, 55), fontsize=8, color="#333",
               arrowprops=dict(arrowstyle="->", color=GREY, lw=.8))
ax[1].bar(pubday.index, pubday.values * 100,
          color=[ACC if w == 0 else GREY for w in
                 panel.groupby("dt").weekend.first().values], width=.75)
ax[1].set_xlabel("Day in November 2019")
ax[1].set_ylabel("Share of creator-days with an upload (%)")
ax[1].set_title("(b) Publishing rate over time", loc="left", fontsize=9.5)
ax[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ACC),
                      plt.Rectangle((0, 0), 1, 1, color=GREY)],
             labels=["Weekday", "Weekend"], frameon=False, fontsize=7.5)
plt.savefig(OUT / "fig1_concentration.png")
plt.close()

# ============================================================ 2. Segmentation
pop = F[F.e_pub_days > 0].copy()                      # study population
R["studypop"] = int(len(pop))
R["studypop_churn"] = round(float(pop.churn.mean()), 4)

SEG = ["e_n_pub", "e_pub_days", "e_impr", "e_clicks", "e_likes",
       "e_comments", "e_follows", "c_followers"]
X = pop[SEG].fillna(0).copy()
X = np.log1p(X.clip(lower=0))                          # right-skewed -> log
Xs = StandardScaler().fit_transform(X)

sil = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    idx = np.random.RandomState(0).choice(len(Xs), min(10000, len(Xs)), replace=False)
    sil[k] = round(float(silhouette_score(Xs[idx], km.labels_[idx])), 4)
R["silhouette_by_k"] = sil
K = max(sil, key=sil.get)
print(f"\nSilhouette by k: {sil}  ->  chosen k={K}")

km = KMeans(n_clusters=K, n_init=25, random_state=42).fit(Xs)
pop["segment"] = km.labels_

prof = pop.groupby("segment").agg(
    n=("creatorId", "size"),
    uploads=("e_n_pub", "mean"),
    upload_days=("e_pub_days", "mean"),
    impressions=("e_impr", "mean"),
    clicks=("e_clicks", "mean"),
    likes=("e_likes", "mean"),
    followers=("c_followers", "median"),
    tenure=("c_tenure", "median"),
    churn=("churn", "mean")).round(2)
prof["share"] = (prof.n / len(pop) * 100).round(1)
prof = prof.sort_values("uploads", ascending=False)
print("\n" + "=" * 66)
print("CREATOR SEGMENTS (the label played no role in forming them)")
print("=" * 66)
print(prof.to_string())

# Robustness: Gaussian mixture
gm = GaussianMixture(n_components=K, random_state=42).fit(Xs)
R["ari_kmeans_vs_gmm"] = round(float(adjusted_rand_score(km.labels_, gm.predict(Xs))), 4)
# Bootstrap stability
aris = []
rs = np.random.RandomState(1)
for _ in range(5):
    idx = rs.choice(len(Xs), len(Xs), replace=True)
    lab = KMeans(n_clusters=K, n_init=10, random_state=7).fit_predict(Xs[idx])
    aris.append(adjusted_rand_score(km.labels_[idx], lab))
R["ari_bootstrap_mean"] = round(float(np.mean(aris)), 4)
print(f"\nRobustness: ARI k-means vs. GMM = {R['ari_kmeans_vs_gmm']:.2f} | "
      f"bootstrap ARI = {R['ari_bootstrap_mean']:.2f}")

# -------------------------------------------------- Figure 2: PCA projection
pcs = PCA(n_components=2).fit(Xs)
P = pcs.transform(Xs)
samp = np.random.RandomState(0).choice(len(P), min(8000, len(P)), replace=False)
fig, ax = plt.subplots(figsize=(5.2, 4))
cols = ["#0D6E6E", "#C77B30", "#5A3D8C", "#3B7BBF", "#A03D2F"]
order = list(prof.index)
for i, s in enumerate(order):
    m = km.labels_[samp] == s
    ax.scatter(P[samp][m, 0], P[samp][m, 1], s=4, alpha=.4, color=cols[i % len(cols)],
               label=f"Segment {s} ({prof.loc[s, 'share']:.0f}%)", linewidths=0)
ax.set_xlabel(f"PC1 ({pcs.explained_variance_ratio_[0]:.0%} of variance)")
ax.set_ylabel(f"PC2 ({pcs.explained_variance_ratio_[1]:.0%})")
ax.set_title("Creator segments in principal component space", loc="left", fontsize=9.5)
ax.legend(frameon=False, fontsize=7.5, markerscale=2.5)
plt.savefig(OUT / "fig2_segments_pca.png")
plt.close()

# Robustness check: k=3, even though the silhouette selects k=2
km3 = KMeans(n_clusters=3, n_init=25, random_state=42).fit(Xs)
pop["seg3"] = km3.labels_
prof3 = pop.groupby("seg3").agg(
    n=("creatorId", "size"), uploads=("e_n_pub", "mean"),
    impressions=("e_impr", "mean"), followers=("c_followers", "median"),
    churn=("churn", "mean")).round(2).sort_values("uploads", ascending=False)
prof3["share"] = (prof3.n / len(pop) * 100).round(1)
R["ari_k2_vs_k3"] = round(float(adjusted_rand_score(km.labels_, km3.labels_)), 4)
R["segment_profile_k3"] = prof3.reset_index().to_dict("records")
print("\n" + "=" * 66)
print(f"ROBUSTNESS k=3 (silhouette {sil[3]:.3f} against {sil[2]:.3f} at k=2; "
      f"ARI to k=2: {R['ari_k2_vs_k3']:.2f})")
print("=" * 66)
print(prof3.to_string())
prof3.to_csv(OUT / "02_segment_profile_k3.csv")

pop[["creatorId", "segment"]].to_parquet(OUT / "creator_segments.parquet", index=False)
prof.to_csv(OUT / "02_segment_profile.csv")
R["segment_profile"] = prof.reset_index().to_dict("records")
json.dump(R, open(OUT / "02_descriptives.json", "w"), indent=2)
print("\n-> fig1_concentration.png, fig2_segments_pca.png, 02_segment_profile.csv")
