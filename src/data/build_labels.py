"""Step 3: activeness label (LABEL window) + fixed train/val/test split.

Label and split definitions come from config so every model is comparable.
"""
import hashlib
import duckdb
import polars as pl
from src import config

config.DERIVED_DIR.mkdir(parents=True, exist_ok=True)


def _split(user_id: str) -> str:
    h = int(hashlib.md5(f"{user_id}-{config.SPLIT_SEED}".encode()).hexdigest(), 16)
    r = (h % 10_000) / 10_000
    if r < config.TEST_FRAC:
        return "test"
    if r < config.TEST_FRAC + config.VAL_FRAC:
        return "val"
    return "train"


def main():
    con = duckdb.connect()
    lo, hi = config.LABEL_WINDOW_DAYS
    glob = str(config.PARQUET_DIR / "impressions" / "**" / "*.parquet")
    df = con.execute(f"""
        SELECT userId, AVG(isClick) AS label_click_rate
        FROM read_parquet('{glob}', hive_partitioning=true)
        WHERE dt BETWEEN {lo} AND {hi}
        GROUP BY userId
    """).pl()
    df = df.with_columns([
        (pl.col("label_click_rate") <= config.ACTIVE_CLICK_THRESHOLD)
            .alias("is_inactive"),
        pl.col("userId").map_elements(_split, return_dtype=pl.Utf8).alias("split"),
    ])
    out = config.DERIVED_DIR / "labels.parquet"
    df.write_parquet(out)
    print(f"Wrote {out}: {df.height} users")
    print(df.group_by("split").len())


if __name__ == "__main__":
    main()
