# Member F — Supply Side: Creator Churn and the Feedback Lever

**Answers:** the supply-side condition for part (c). A recommender can only
recommend content that exists, so the chapter asks who stops producing it,
whether that can be predicted from one early week, and whether audience feedback
causally affects continued production.

Uses the three tables the demand side does not touch — `creator_stats`,
`creator_demographics`, `mlog_demographics` — plus `mlog_stats` for the feedback
each creator receives. Same window definitions, same model class and same
evaluation protocol as the demand side, so the two sides can be read against one
another; the split is a fixed hash on `creatorId` rather than `userId`.

## Headline results

| | |
|---|---|
| Creators in the panel | 90,534 over 30 days |
| Study population (published in days 1–7) | 15,498, of whom 40.8% stop |
| Stage 1 — churn | ROC-AUC 0.809, PR-AUC 0.725 against a base rate of 0.417 |
| Stage 2 — publishes but gets no clicks | PR-AUC 0.412 against a base rate of 0.145 (lift 2.84) |
| Preferred causal estimate | likes in week w−1 raise P(publish in week w) by +0.0400 (t = 3.28) |
| Placebo on that specification | −0.0145 (t = −1.04), indistinguishable from zero |
| Heterogeneity | +24.7% for the smallest creators, +7.5% and insignificant for the largest |

## Pipeline (run from repo root)

Each step depends on the ones before it. Everything is written to
`data/derived/member_f/` (gitignored).

```bash
python -m src.member_f_supply_side.build_panel             # creator-day panel + features
python -m src.member_f_supply_side.descriptives_segments   # concentration, k-means segments
python -m src.member_f_supply_side.churn_model             # two-stage churn model
python -m src.member_f_supply_side.stage2_justification    # does no response really cause exit?

# from prediction to lever — the identification sequence, in order
python -m src.member_f_supply_side.feedback_effects        # naive daily spec  -> placebo FAILS
python -m src.member_f_supply_side.identification_fix      # legacy content    -> placebo STILL FAILS
python -m src.member_f_supply_side.longer_lags             # weekly blocks     -> placebo PASSES
python -m src.member_f_supply_side.robustness              # 16 specifications, each with its placebo
python -m src.member_f_supply_side.heterogeneity           # for whom does it hold
```

Only `pandas`, `numpy`, `scikit-learn` and `matplotlib` are needed — all already
in `requirements.txt`. Fixed-effects estimation and cluster-robust standard errors
are implemented directly rather than through an econometrics package.

## The identification problem, and why three scripts

This is the part of the chapter that is reported as a sequence rather than a
result, because the first two attempts fail:

1. **`feedback_effects.py`** — daily linear probability model, creator and day
   fixed effects. Likes at t−1 look like they raise publishing at t. Then the
   placebo: feedback at t+1 predicts publishing at t **twice as strongly**.
   Tomorrow cannot cause today, so the specification measures co-movement, not a
   direction. The estimate is discarded.

2. **`identification_fix.py`** — the mechanical channel is that uploading creates
   a card which then collects feedback. `mlog_demographics.publishTime` is
   undocumented; it decodes as the age of the card in days, so publication day
   = `31 - publishTime`, verified for 100% of the 91,002 cards published inside
   the window. Restricting feedback to cards with `publishTime > 30` means the
   response cannot have been produced by an upload made during the window. The
   effect shrinks and the placebo shrinks — but stays larger than the estimate.
   Still failing.

3. **`longer_lags.py`** — what remains is a time-varying confounder: creators have
   active spells of several days in which they are present, their catalogue
   circulates, and they upload. A spell correlates in both time directions, which
   is exactly what a placebo detects. Collapsing the panel into 7-day blocks
   averages the spell out. The placebo comes back clean and this becomes the
   preferred specification.

`robustness.py` then re-estimates across block length, outcome definition and
sample, each with its own placebo. The placebo separates the specifications
cleanly: it is quiet in the legacy-restricted samples and strongly significant
wherever content published inside the window is included. The legacy restriction
is therefore not a robustness check but a condition for the estimate to mean
anything.

## Notes for whoever merges this

- `c_level` is banned as a predictor for the same reason `level` is banned on the
  demand side — it is measured at the end of the period and encodes the outcome.
  See `FORBIDDEN_FEATURES` in `src/config.py`.
- The `creator_stats` column is spelled `PushlishMlogCnt` in the raw file. That
  typo is in the source data, not in this code.
- Churn here means ceasing to publish for the remaining 23 days, not permanent
  departure from the platform. One month is all the data allows.
- The dataset citation is Zhang, D. J., M. Hu, X. Liu, Y. Wu, and Y. Li. 2022.
  "NetEase Cloud Music Data." *Manufacturing & Service Operations Management*
  24 (1): 275-284.
