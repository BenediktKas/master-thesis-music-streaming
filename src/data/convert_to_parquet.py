"""Step 1: stream the 6.3 GB impression CSV into partitioned Parquet.

One-time cost, run by ONE person. Uses DuckDB so it never loads the
whole file into memory. The small tables are converted too for convenience.
"""
import duckdb
from src import config

config.PARQUET_DIR.mkdir(parents=True, exist_ok=True)


def main():
    con = duckdb.connect()
    imp = config.RAW_DIR / "impression_data.csv"
    print(f"Converting {imp} -> partitioned Parquet (by dt)...")
    con.execute(f"""
        COPY (SELECT * FROM read_csv_auto('{imp}', header=true))
        TO '{config.PARQUET_DIR / "impressions"}'
        (FORMAT PARQUET, PARTITION_BY (dt), OVERWRITE_OR_IGNORE);
    """)
    for name in ["user_demographics", "mlog_demographics",
                 "mlog_stats", "creator_demographics", "creator_stats"]:
        src = config.RAW_DIR / f"{name}.csv"
        if src.exists():
            con.execute(f"""
                COPY (SELECT * FROM read_csv_auto('{src}', header=true))
                TO '{config.PARQUET_DIR / (name + ".parquet")}' (FORMAT PARQUET);
            """)
            print(f"  wrote {name}.parquet")
    print("Done.")


if __name__ == "__main__":
    main()
