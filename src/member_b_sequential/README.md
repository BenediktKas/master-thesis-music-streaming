# Member B — Sequential Deep Model for Activity Prediction

**Answers:** Part (b) — predict active/inactive from early actions

GRU / Transformer over the ordered early-action stream. Benchmark explicitly against a static XGBoost baseline on the shared split.

See `docs/Thesis_Blueprint_Q2.docx` (Section 6) for the full work order:
goal, method, inputs/outputs, evaluation metric, and the main risk.

## Conventions
- Import shared paths/labels/splits from `src.config` — never redefine them.
- Train on the shared train/val/test split so results compare across members.
- Never use `level` as a predictor (leakage). See `config.FORBIDDEN_FEATURES`.
