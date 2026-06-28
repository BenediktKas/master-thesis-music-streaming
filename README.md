# Master Arbeit — Music Streaming (NetEase Cloud Music), Research Question 2

Predicting, understanding, and growing **active users** on NCM's "cloud village"
feed, using one month of impression-level data (57.75M impressions, ~2.1M users).

Full plan: **`docs/Thesis_Blueprint_Q2.docx`**.

## Research question (Q2), three parts
- **(a)** What characterizes active users and what do they prefer?
- **(b)** Can we predict active vs. inactive from early actions?
- **(c)** How do we design a recommender to maximize the number of active users?

## Team layout (one algorithm per member)
| Member | Package | Focus |
|---|---|---|
| A | `src/member_a_segmentation` | Segmentation + content-taste (a) |
| B | `src/member_b_sequential`   | Sequential deep model (b) |
| C | `src/member_c_survival`     | Time-to-disengagement (b) |
| D | `src/member_d_uplift`       | Causal uplift targeting (c) |
| E | `src/member_e_recommender`  | Activeness-optimized recommender (c, optional) |

## Getting started
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Put Raw_Data.zip at data/raw/Raw_Data.zip, then build the shared foundation:
bash src/data/run_user_window_agg.sh   # stream zip -> per-user window aggregates
python -m src.data.build_labels        # activeness label + study filter + split
python -m src.data.baseline            # reference metrics (number to beat)

# Optional (Members B/E only): full row-level Parquet for sequences/recommender
bash src/data/run_convert_to_parquet.sh
```
See **`docs/GROUNDWORK_FINDINGS.md`** for results from the first run and the
decisions the data forces (threshold choice, survivorship, the data-quality
quirk in `detailMlogInfoList`).

## Ground rules
- Shared label/window/split live in `src/config.py` — agree them once, never fork.
- **Never** commit data (it's gitignored) or use `level` as a predictor (leakage).
- Every model trains on the same split so AUC/PR numbers are comparable.
- The impression CSV has unquoted JSON (commas) in `detailMlogInfoList`; always
  keep only the **last 12 columns** when parsing (the pipeline does this).

## Layout
```
src/config.py        shared decisions (labels, windows, split, leakage guard)
src/data/            CSV -> Parquet -> features -> labels pipeline
src/member_*/        one package per team member
notebooks/           EDA and results
docs/                the thesis blueprint
data/                local only, gitignored
```
