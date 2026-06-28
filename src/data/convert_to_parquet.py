"""OPTIONAL: materialize the FULL impression table as partitioned Parquet.

Only needed by Members B (sequences) and E (recommender) who require row-level
events, not just user aggregates. Members A/C/D should NOT run this — use the
much cheaper build_user_window_agg.sh instead.

Robust parsing note: `detailMlogInfoList` is unquoted JSON with commas, so we
keep only the stable last-12 columns via awk before writing Parquet. Large
output (~1-2 GB); run on your Mac.

Usage:
    bash src/data/run_convert_to_parquet.sh
"""
import sys, duckdb
from src import config

OUT = config.PARQUET_DIR / "impressions"
config.PARQUET_DIR.mkdir(parents=True, exist_ok=True)
COLS = ("{'dt':'INT','impressPosition':'INT','impressTime':'BIGINT','isClick':'INT',"
        "'isComment':'INT','isIntoPersonalHomepage':'INT','isShare':'INT',"
        "'isViewComment':'INT','isLike':'INT','mlogId':'VARCHAR',"
        "'mlogViewTime':'VARCHAR','userId':'VARCHAR'}")


def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute(f"PRAGMA temp_directory='{config.PARQUET_DIR / 'duckdb_tmp'}'")
    con.execute(f"""
        COPY (SELECT * FROM read_csv('/dev/stdin', header=true, columns={COLS}))
        TO '{OUT}' (FORMAT PARQUET, PARTITION_BY (dt), OVERWRITE_OR_IGNORE)
    """)
    print(f"wrote partitioned Parquet to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
