# Member D — Uplift / Causal Targeting Model

**Answers:** Part (c) — maximize the number of active users

Heterogeneous treatment effects (X-learner / causal forest) using impression position as the exogeneity source. Identifies persuadable users.

See `docs/Thesis_Blueprint_Q2.docx` (Section 6) for the full work order:
goal, method, inputs/outputs, evaluation metric, and the main risk.

## Conventions
- Import shared paths/labels/splits from `src.config` — never redefine them.
- Train on the shared train/val/test split so results compare across members.
- Never use `level` as a predictor (leakage). See `config.FORBIDDEN_FEATURES`.
