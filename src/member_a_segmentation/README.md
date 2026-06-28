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

## Ideas to extend (your individual contribution)
- Content-taste entropy (Shannon over `contentId`) as a diversity axis.
- Try GMM / HDBSCAN and compare cluster stability across resamples.
- Name the archetypes and tie them to retention recommendations.
- Item2vec / NMF embeddings over engaged cards for a richer taste space.

## Conventions
- Import shared paths/labels/splits from `src.config`; never redefine them.
- Never use `level` as a feature (leakage). Segmentation is unsupervised — the
  label is only used to *profile* segments afterward, never to fit them.
