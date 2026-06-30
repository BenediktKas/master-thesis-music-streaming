"""Stage-1 (retention) modeling table. Population = all users with early-window
activity (e_impr > 0); target `churned` = the user is NOT observed in the label
window (l_impr == 0), i.e. she stops appearing in the discovery feed after week 1.
This deliberately uses the users that the activeness study (stage 2) drops, so the
survivorship selection is modelled rather than discarded.

Run after build_user_window_agg + build_content_taste:
    python -m src.data.build_retention
"""
import hashlib, polars as pl
from src import config

DER = config.DERIVED_DIR


def split(u: str) -> str:
    h = int(hashlib.md5(f"{u}-{config.SPLIT_SEED}".encode()).hexdigest(), 16)
    r = (h % 10_000) / 10_000
    return "test" if r < config.TEST_FRAC else ("val" if r < config.TEST_FRAC + config.VAL_FRAC else "train")


def main():
    agg = pl.read_parquet(DER / "user_window_agg.parquet").filter(pl.col("e_impr") > 0)
    df = agg.with_columns([
        (pl.col("l_impr") == 0).alias("churned"),
        (pl.col("e_clicks") / pl.col("e_impr")).alias("e_click_rate"),
        (pl.col("e_likes") / pl.col("e_impr")).alias("e_like_rate"),
        (pl.col("e_view_time_sum") / pl.col("e_impr")).alias("e_avg_view_time"),
        pl.col("userId").map_elements(split, return_dtype=pl.Utf8).alias("split"),
    ])
    ct = config.DERIVED_DIR / "user_content_taste.parquet"
    if ct.exists():
        taste = pl.read_parquet(ct).select(["userId", "ct_video_share_seen", "ct_seen_n_content"])
        df = df.join(taste, on="userId", how="left")
    df.write_parquet(DER / "user_retention_table.parquet")
    print(f"early-window users={df.height}; churn rate={df['churned'].mean():.4f}")
    print(df.group_by("split").agg(pl.len().alias("n"), pl.col("churned").mean().round(4).alias("churn_rate")).sort("split"))


if __name__ == "__main__":
    main()
