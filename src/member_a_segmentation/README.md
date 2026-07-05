# Member A — Behavioural Segmentation & Content-Taste Profiling

**Answers:** Part (a) — who active users are and what they prefer.

Clusters study-population users on early-window **behaviour** + **content-taste**
features (not the label), then profiles each segment by size, activeness rate,
and characteristics. Produces the segments that Members D and E consume.

See `docs/Thesis_Blueprint_Q2.docx` (Section 6.1) for the full work order.

## Pipeline (run from repo root)

```bash
# 1. shared foundation (if not done already)
bash src/data/run_user_window_agg.sh "../Dataset/Raw_Data.zip"
python3 -m src.data.build_labels

# 2. content-taste features — joins impressions to card metadata (full file)
bash src/data/run_content_taste.sh "../Dataset/Raw_Data.zip"

# 3. segmentation
python3 -m src.member_a_segmentation.segment
```

Outputs land in `data/derived/member_a/`:
- `user_segments.parquet` — userId -> segment (hand to Members D/E)
- `segment_profiles.csv` — size, inactive rate, key feature means per segment
- `segments_overview.png` — size and inactive-rate bars

## What the validation run showed (sample data)
k=3 separated cleanly into a low-intent dormant majority (~77% inactive), a small
power-user cluster (~17% inactive, ~25 early clicks, broad taste), and an engaged
middle. Re-run on full data for final segments.


## Full analysis (run in order, from repo root)

```bash
python3 -m src.member_a_segmentation.segment       # k-means segments + profiles + overview figure
python3 -m src.member_a_segmentation.figures       # PCA scatter (Figure 2 in the chapter)
python3 -m src.member_a_segmentation.robustness    # silhouette-vs-k, GMM/HDBSCAN agreement, bootstrap stability
python3 -m src.member_a_segmentation.preferences   # descriptive taste by activeness/segment (note: breadth is volume-confounded)
bash src/data/run_taste_analysis.sh "../Dataset/Raw_Data.zip"   # volume-controlled test: is there taste structure? (no)
```

Findings (full data): k=3 is highly stable (bootstrap ARI 0.98) and agrees with a Gaussian mixture (ARI 0.74); HDBSCAN fragments, because users form an engagement gradient. The segmentation is ACTIVITY-based, not taste-based: 'broader taste' is a volume artifact (corr up to 0.996), and a volume-controlled composition test finds no separable taste structure (silhouette <0.18). True genre was dropped from the data. See taste_analysis.py.

## Ideas to extend (your individual contribution)
- Content-taste entropy (Shannon over `contentId`) as a diversity axis.
- Try GMM / HDBSCAN and compare cluster stability across resamples.
- Name the archetypes and tie them to retention recommendations.
- Item2vec / NMF embeddings over engaged cards for a richer taste space.

## Conventions
- Import shared paths/labels/splits from `src.config`; never redefine them.
- Never use `level` as a feature (leakage). Segmentation is unsupervised — the
  label is only used to *profile* segments afterward, never to fit them.
