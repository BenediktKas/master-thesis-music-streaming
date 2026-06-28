# Member C — Survival / Time-to-Disengagement Model

**Answers:** Part (b) reframed — when does a user go inactive

Cox / discrete-time survival with right-censoring at day 30. Reports C-index and time-dependent AUC.

See `docs/Thesis_Blueprint_Q2.docx` (Section 6) for the full work order:
goal, method, inputs/outputs, evaluation metric, and the main risk.

## Conventions
- Import shared paths/labels/splits from `src.config` — never redefine them.
- Train on the shared train/val/test split so results compare across members.
- Never use `level` as a predictor (leakage). See `config.FORBIDDEN_FEATURES`.
