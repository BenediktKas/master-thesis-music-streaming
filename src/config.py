"""Shared configuration. EDIT THESE TOGETHER, ONCE, AS A TEAM.

Every sub-project imports from here so all models use identical
labels, windows, and splits. Do not redefine these locally.
"""
from pathlib import Path

# ---- Paths ----------------------------------------------------------------
# Raw CSVs are NOT in the repo (see .gitignore / data/README.md).
# Point RAW_DIR at wherever you unzipped Raw_Data.zip locally.
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "csv_data"     # gitignored
PARQUET_DIR = REPO_ROOT / "data" / "parquet"          # gitignored
DERIVED_DIR = REPO_ROOT / "data" / "derived"          # gitignored

# ---- Activeness label (NCM definition: low average click probability) -----
# A user is INACTIVE if her mean click rate in the label window is <= threshold.
ACTIVE_CLICK_THRESHOLD = 0.0          # primary: "zero or very low"
ROBUSTNESS_THRESHOLDS = [0.01, 0.05]  # pre-registered alternatives

# ---- Temporal split (prevents leakage) ------------------------------------
EARLY_WINDOW_DAYS = (1, 7)    # build features here (dt runs 1..30)
LABEL_WINDOW_DAYS = (8, 30)   # define activeness here
# Only users with >=1 impression in the early window enter the study.

# ---- Train/val/test split (fixed by userId hash, reused everywhere) -------
SPLIT_SEED = 42
TEST_FRAC = 0.15
VAL_FRAC = 0.15

# ---- LEAKAGE GUARD --------------------------------------------------------
# user_demographics.level is measured up to Dec 1, 2019 and can encode the
# outcome. NEVER use it as a predictor. Validation/alt-label only.
FORBIDDEN_FEATURES = {"level"}
