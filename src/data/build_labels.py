"""Step 2 (VALIDATED): build the modeling table from user_window_agg.parquet —
activeness label, study-population filter, derived early features, fixed split.

Study population: users with impressions in BOTH the early and label windows.
Label: inactive if label-window click rate <= ACTIVE_CLICK_THRESHOLD (NCM's
"zero or very low average click probability"). Split is fixed by userId hash.
"""
import hashlib, polars as pl
from src import config

DER = config.DERIVED_DIR


def split(u: str) -> str:
    h = int(hashlib.md5(f"{u}-{config.SPLIT_SEED}".encode()).hexdigest(), 16)
    r = (h % 10_000) / 10_000
    if r < config.TEST_FRAC:
        return "test"
    if r < config.TEST_FRAC + config.VAL_FRAC:
        return "val"
    return "train"


def main():
    df = pl.read_parquet(DER / "user_window_agg.parquet")
    n0 = df.height
    df = df.filter((pl.col("e_impr") > 0) & (pl.col("l_impr") > 0))
    df = df.with_columns((pl.col("l_clicks") / pl.col("l_impr")).alias("l_click_rate"))
    df = df.with_columns([
        (pl.col("l_click_rate") <= config.ACTIVE_CLICK_THRESHOLD).alias("is_inactive"),
        pl.col("userId").map_elements(split, return_dtype=pl.Utf8).alias("split"),
        (pl.col("e_clicks") / pl.col("e_impr")).alias("e_click_rate"),
        (pl.col("e_likes") / pl.col("e_impr")).alias("e_like_rate"),
        (pl.col("e_view_time_sum") / pl.col("e_impr")).alias("e_avg_view_time"),
    ])
    df.write_parquet(DER / "user_modeling_table.parquet")
    print(f"study_users={df.height} (from {n0}); inactive_rate={df['is_inactive'].mean():.4f}")


if __name__ == "__main__":
    main()
