"""Step 2: aggregate impressions to one row per user over the EARLY window.

Output (~2M rows) fits in memory and feeds the tabular/segmentation work.
Do NOT include any FORBIDDEN_FEATURES (see config). Extend as needed.
"""
import duckdb
from src import config

config.DERIVED_DIR.mkdir(parents=True, exist_ok=True)


def main():
    con = duckdb.connect()
    lo, hi = config.EARLY_WINDOW_DAYS
    glob = str(config.PARQUET_DIR / "impressions" / "**" / "*.parquet")
    print("Building early-window user features...")
    con.execute(f"""
        COPY (
            SELECT
                userId,
                COUNT(*)                         AS n_impressions,
                AVG(isClick)                     AS click_rate,
                SUM(isLike)                      AS n_likes,
                SUM(isShare)                     AS n_shares,
                SUM(isComment)                   AS n_comments,
                SUM(isIntoPersonalHomepage)      AS n_into_homepage,
                AVG(TRY_CAST(mlogViewTime AS DOUBLE)) AS avg_view_time,
                COUNT(DISTINCT dt)               AS active_days,
                AVG(impressPosition)             AS avg_position
            FROM read_parquet('{glob}', hive_partitioning=true)
            WHERE dt BETWEEN {lo} AND {hi}
            GROUP BY userId
        ) TO '{config.DERIVED_DIR / "user_features_early.parquet"}' (FORMAT PARQUET);
    """)
    print("Wrote user_features_early.parquet")


if __name__ == "__main__":
    main()
