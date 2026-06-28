"""Step 1 (VALIDATED): per-user, per-window aggregates from the impression
stream, computed in a SINGLE pass. Reads the cleaned CSV stream on stdin and
writes data/derived/user_window_agg.parquet.

Why a stream + awk upstream: `detailMlogInfoList` (column 1) is UNQUOTED JSON
with embedded commas, so ~1.2% of rows have an inflated, variable column count.
The 12 fields we need are ALWAYS the last 12 columns. The shell wrapper keeps
those with awk before this script parses, so naive parsing never corrupts rows.

Usage (full file, run on your Mac — no time limit there):
    bash src/data/run_user_window_agg.sh
"""
import sys, duckdb
from src import config

OUT = config.DERIVED_DIR / "user_window_agg.parquet"
config.DERIVED_DIR.mkdir(parents=True, exist_ok=True)
COLS = ("{'dt':'INT','impressPosition':'INT','impressTime':'BIGINT','isClick':'INT',"
        "'isComment':'INT','isIntoPersonalHomepage':'INT','isShare':'INT',"
        "'isViewComment':'INT','isLike':'INT','mlogId':'VARCHAR',"
        "'mlogViewTime':'VARCHAR','userId':'VARCHAR'}")
e0, e1 = config.EARLY_WINDOW_DAYS
l0, l1 = config.LABEL_WINDOW_DAYS


def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute(f"PRAGMA temp_directory='{config.DERIVED_DIR / 'duckdb_tmp'}'")
    con.execute(f"""
    COPY (
      SELECT userId,
        COUNT(*)                    FILTER (WHERE dt BETWEEN {e0} AND {e1}) AS e_impr,
        SUM(isClick)                FILTER (WHERE dt BETWEEN {e0} AND {e1}) AS e_clicks,
        SUM(isLike)                 FILTER (WHERE dt BETWEEN {e0} AND {e1}) AS e_likes,
        SUM(isShare)                FILTER (WHERE dt BETWEEN {e0} AND {e1}) AS e_shares,
        SUM(isComment)              FILTER (WHERE dt BETWEEN {e0} AND {e1}) AS e_comments,
        SUM(isViewComment)          FILTER (WHERE dt BETWEEN {e0} AND {e1}) AS e_viewcomment,
        SUM(isIntoPersonalHomepage) FILTER (WHERE dt BETWEEN {e0} AND {e1}) AS e_homepage,
        SUM(TRY_CAST(mlogViewTime AS DOUBLE)) FILTER (WHERE dt BETWEEN {e0} AND {e1}) AS e_view_time_sum,
        COUNT(DISTINCT dt)          FILTER (WHERE dt BETWEEN {e0} AND {e1}) AS e_active_days,
        AVG(impressPosition)        FILTER (WHERE dt BETWEEN {e0} AND {e1}) AS e_avg_pos,
        COUNT(*)                    FILTER (WHERE dt BETWEEN {l0} AND {l1}) AS l_impr,
        SUM(isClick)                FILTER (WHERE dt BETWEEN {l0} AND {l1}) AS l_clicks
      FROM read_csv('/dev/stdin', header=true, columns={COLS})
      GROUP BY userId
    ) TO '{OUT}' (FORMAT PARQUET)
    """)
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT}')").fetchone()[0]
    print(f"wrote {OUT}  ({n} users)", file=sys.stderr)


if __name__ == "__main__":
    main()
