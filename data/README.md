# data/

**Nothing in here is committed** (see repo `.gitignore`). The raw NetEase
files are far over GitHub's 100 MB per-file limit.

Expected local layout (create it yourself after unzipping Raw_Data.zip):

```
data/
  raw/csv_data/        # the 6 raw CSVs (impression_data.csv is 6.3 GB)
  parquet/             # output of src/data/convert_to_parquet.py
  derived/             # user feature table, sequences, labels
```

Run the pipeline in order:
1. `python -m src.data.convert_to_parquet`
2. `python -m src.data.build_user_features`
3. `python -m src.data.build_labels`
