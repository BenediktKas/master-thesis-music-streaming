# Groundwork Findings — Shared Foundation

Results from running the shared pipeline (Section 5 of the blueprint), now on the
**full dataset** (all 57.75M impressions, run on a Mac).

## Headline results (full data, confirmed)

| Metric | Value |
|---|---|
| Total users | 2,085,602 (matches the documented 2,085,533) |
| Study users (impressions in both windows) | 542,842 (26%) |
| Inactive rate (click rate ≤ 0) | 0.7167 |
| Baseline ROC-AUC / PR-AUC (test) | 0.7055 / 0.8322 |

These full-data figures matched an earlier 20M-row validation sample (721k users)
to within 0.1% on every headline metric, confirming the sample was representative
and the pipeline is correct end-to-end.

> The detailed distribution tables below are on **full data**
> (regenerate anytime with `python -m src.data.eda`).

## Data-quality issue found (important)

`impression_data.csv` column 1, `detailMlogInfoList`, is **unquoted JSON containing
commas**. ~1.2% of rows therefore have a variable, inflated column count (one row
had 6,222 fields). Naive CSV parsing silently corrupts these rows. The 12 fields we
need are always the **last 12 columns**, so the pipeline strips the leading JSON
with `awk` before parsing. Any code reading this file must do the same.

Also confirmed: `dt` runs **1–30** (not 0–29), and `mlogViewTime` is `NA` on
non-click rows (dwell is only defined after a click).

## Decisions the data forces on the team

### 1. Activeness threshold is consequential
Inactive = label-window click rate ≤ threshold. The rate is highly sensitive:

| Threshold (click rate ≤) | Inactive share |
|---|---|
| 0.00 ("zero clicks") | 71.7% |
| 0.01 | 73.0% |
| 0.05 | 83.4% |
| 0.10 | 91.4% |

Even the strictest definition labels ~72% of users inactive — a heavy class
imbalance every predictive model must handle (class weights / PR-AUC focus). The
team should consciously pick the primary threshold and justify it.

### 2. Survivorship / window overlap is large
Of the sampled users, only **26%** have impressions in **both** windows
(early dt 1–7 and label dt 8–30):

| Group | Users |
|---|---|
| Both windows (study population) | 542,842 |
| Early window only | 298,828 |
| Label window only | 1,243,932 |
| **Study population** | **542,842** |

Many users appear only later in the month (1.24M are label-window-only). The
study population (both windows) must be defined explicitly in the thesis, and the
bias acknowledged.

### 3. Active vs. inactive users differ clearly on early behaviour
Means over early-window features (study population):

| | Active | Inactive |
|---|---|---|
| Early impressions | 40.4 | 11.8 |
| Early clicks | 2.92 | 0.33 |
| Early active days | 2.25 | 1.64 |
| Early like rate | 0.0025 | 0.0008 |

Volume and breadth of early exposure separate the classes strongly — consistent
with the prior thesis's "intensity/persistence" finding, and a good sign that
Part (b) is learnable.

## Baseline (the number to beat)

Gradient boosting (`HistGradientBoostingClassifier`) on **early-window features
only** (leakage-safe; `level` excluded), shared hash split:

| Metric | Test |
|---|---|
| ROC-AUC | 0.705 |
| PR-AUC | 0.832 |
| Base rate (inactive) | 0.719 |

Top features by permutation importance: early impression count, average feed
position, early clicks, average view time. Members B (sequential) and C
(survival) should report against this exact baseline on the same split.

## Artifacts produced (in `data/derived/`, gitignored)

- `user_window_agg.parquet` — per-user early + label window aggregates
- `user_modeling_table.parquet` — study population + label + split + features
- `baseline_metrics.json`, `eda_summary.json`

## How to reproduce / scale to full data

```bash
# 1. put Raw_Data.zip at data/raw/Raw_Data.zip
bash src/data/run_user_window_agg.sh        # full-file streaming aggregate
python -m src.data.build_labels             # label + study filter + split
python -m src.data.baseline                 # reference metrics
```
