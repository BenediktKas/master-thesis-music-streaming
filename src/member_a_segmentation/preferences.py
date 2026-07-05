"""Member A content preferences: what active vs inactive users, and each
segment, engage with (video share, breadth of content/creators/artists).
Revealed taste is measured among early-window clickers.

    python -m src.member_a_segmentation.preferences
"""
import polars as pl
from src import config

DER = config.DERIVED_DIR


def main():
    mt = pl.read_parquet(DER / "user_modeling_table.parquet")
    ct = pl.read_parquet(DER / "user_content_taste.parquet")
    seg = pl.read_parquet(DER / "member_a" / "user_segments.parquet")
    df = mt.join(ct, on="userId", how="inner").join(seg, on="userId", how="inner")
    clk = df.filter(pl.col("ct_clicked") > 0)

    print(f"Revealed preferences among early-window clickers (n={clk.height}):")
    print(clk.group_by("is_inactive").agg([
        pl.len().alias("n"),
        pl.col("ct_video_share_click").mean().round(3).alias("video_share"),
        pl.col("ct_click_n_content").mean().round(2).alias("content_cats"),
        pl.col("ct_click_n_creators").mean().round(2).alias("creators"),
        pl.col("ct_click_n_artists").mean().round(2).alias("artists"),
    ]).sort("is_inactive"))
    print("\nBy segment:")
    print(clk.group_by("segment").agg([
        pl.len().alias("n"),
        pl.col("ct_video_share_click").mean().round(3).alias("video_share"),
        pl.col("ct_click_n_content").mean().round(2).alias("content_cats"),
        pl.col("ct_click_n_creators").mean().round(2).alias("creators"),
    ]).sort("segment"))


if __name__ == "__main__":
    main()
