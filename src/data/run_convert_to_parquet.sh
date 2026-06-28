#!/bin/bash
set -euo pipefail
ZIP="${1:-data/raw/Raw_Data.zip}"
unzip -p "$ZIP" csv_data/impression_data.csv \
  | awk -F',' 'NF>=12{out=$(NF-11); for(i=NF-10;i<=NF;i++) out=out","$i; print out}' \
  | python -m src.data.convert_to_parquet
