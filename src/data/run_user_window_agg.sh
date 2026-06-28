#!/bin/bash
# Full-file streaming aggregation. Run on a machine with no time limit (your Mac).
# Keeps only the last 12 (stable) columns before parsing — see the .py docstring.
set -euo pipefail
ZIP="${1:-data/raw/Raw_Data.zip}"
unzip -p "$ZIP" csv_data/impression_data.csv \
  | awk -F',' 'NF>=12{out=$(NF-11); for(i=NF-10;i<=NF;i++) out=out","$i; print out}' \
  | python -m src.data.build_user_window_agg
