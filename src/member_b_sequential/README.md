# Member B — Sequential Deep Model for Activity Prediction

**Answers:** Part (b) — predict activeness from early actions, using their **order
and timing**, which the tabular baseline discards.

A GRU reads each user's ordered early-window event stream — one step per
impression, carrying the action bits (click/like/share/comment/view-comment/
into-homepage), the feed position, dwell time, and the time gap since the previous
event — and predicts `is_inactive`. It is benchmarked against the tabular baseline
(ROC-AUC 0.707 / PR-AUC 0.836) on the identical shared split.

See `docs/Thesis_Blueprint_Q2.docx` (Section 6.2).

## Pipeline (run from repo root)

```bash
# 1. shared foundation + labels (if not already done)
bash src/data/run_user_window_agg.sh "../Dataset/Raw_Data.zip"
python3 -m src.data.build_labels

# 2. build ordered early-window event sequences (now include card type + content category)
bash src/data/run_sequences.sh "../Dataset/Raw_Data.zip"     # -> data/derived/early_events.parquet

# 3. train (needs PyTorch; CPU is fine). Two encoders:
pip3 install torch
python3 -m src.member_b_sequential.sequence_model --model gru --epochs 10
python3 -m src.member_b_sequential.sequence_model --model transformer --epochs 10
# quick smoke test on a subset:
python3 -m src.member_b_sequential.sequence_model --model gru --epochs 5 --sample_users 40000
```

The script prints test ROC-AUC / PR-AUC per epoch and the baseline to beat.

## Results (full data, 5 seeds)
| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Tabular baseline (aggregates) | 0.705 ± 0.0001 | 0.832 ± 0.0001 |
| GRU, actions only (v1)        | 0.699 (1 run)  | 0.826 |
| **GRU, + card content (v2)**  | **0.710 ± 0.0005** | **0.839 ± 0.0004** |

**Finding:** the sequence GRU beats the tabular baseline by a small but robust margin
(seed distributions do not overlap; gap ~0.005 ROC / ~0.007 PR, ~10x the seed spread).
The win comes from adding *what card* each event was — order/timing alone (v1) did NOT
beat the baseline. Short sequences (median 6 events) leave little order to exploit.
Written up in `Thesis_Chapter_Foundation.docx`, Section 4.

Reproduce the seed check: `python3 -m src.member_b_sequential.sequence_model --model gru --epochs 20 --n_seeds 5`

## Design notes / ideas to extend
- Event features: 6 action bits + normalised feed position + log dwell + log time-gap.
- Sequences truncated/padded to 50 events (covers the 90th percentile).
- Try a small Transformer (self-attention) and attention-weight interpretation as a
  contrast to the baseline's feature importances.
- Add card content category / type as an embedded token per event.
- Report whether order/timing actually beats the static aggregates — a negative
  result here is still a finding.

## Conventions
- Import shared labels/split from `src.config`; never redefine them.
- Never use `level` (leakage). Features are early-window only.
