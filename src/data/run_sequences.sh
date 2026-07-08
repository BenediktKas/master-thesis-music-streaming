#!/bin/bash
set -euo pipefail
ZIP="${1:-data/raw/Raw_Data.zip}"
mkdir -p data/raw; unzip -o -j "$ZIP" csv_data/mlog_demographics.csv -d data/raw/ >/dev/null   # card metadata
unzip -p "$ZIP" csv_data/impression_data.csv \
  | awk -F',' 'NF>=12{out=$(NF-11); for(i=NF-10;i<=NF;i++) out=out","$i; print out}' \
  | python3 -m src.data.build_sequences
