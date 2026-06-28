# Member E — Activeness-Optimized Ranking / Bandit Recommender

**Answers:** Part (c) — the recommender itself (optional 5th member)

Learning-to-rank / contextual bandit with an activeness-weighted objective. Offline off-policy evaluation vs a CTR-only baseline. Fold into future work if the team is only four.

See `docs/Thesis_Blueprint_Q2.docx` (Section 6) for the full work order:
goal, method, inputs/outputs, evaluation metric, and the main risk.

## Conventions
- Import shared paths/labels/splits from `src.config` — never redefine them.
- Train on the shared train/val/test split so results compare across members.
- Never use `level` as a predictor (leakage). See `config.FORBIDDEN_FEATURES`.
