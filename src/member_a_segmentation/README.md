# Member A — Behavioural Segmentation & Content-Taste Profiling

**Answers:** Part (a) — who active users are and what they prefer

Cluster users into behavioural archetypes and build content-taste vectors from engaged cards (contentId / talkId / artistId / card type). Produces segments consumed by Members D and E.

See `docs/Thesis_Blueprint_Q2.docx` (Section 6) for the full work order:
goal, method, inputs/outputs, evaluation metric, and the main risk.

## Conventions
- Import shared paths/labels/splits from `src.config` — never redefine them.
- Train on the shared train/val/test split so results compare across members.
- Never use `level` as a predictor (leakage). See `config.FORBIDDEN_FEATURES`.
