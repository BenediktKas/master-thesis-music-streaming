"""Per-user early-window EVENT SEQUENCES for the sequential model (Member B).
Streams the impression CSV on stdin, keeps early-window rows, and writes one row
per impression event (ordered later by impressTime) with the fields a sequence
model needs. Unlike the aggregate pipeline, this preserves event-level order.

Output: data/derived/early_events.parquet  (userId, impressTime + action/context fields)

    bash src/data/run_sequences.sh "../Dataset/Raw_Data.zip"
"""
import sys, duckdb
from src import config

OUT = config.DERIVED_DIR / "early_events.parquet"
CARDS = config.RAW_DIR.parent / "mlog_demographics.csv"   # for card type + content category
config.DERIVED_DIR.mkdir(parents=True, exist_ok=True)
COLS = ("{'dt':'INT','impressPosition':'INT','impressTime':'BIGINT','isClick':'INT',"
        "'isComment':'INT','isIntoPersonalHomepage':'INT','isShare':'INT',"
        "'isViewComment':'INT','isLike':'INT','mlogId':'VARCHAR',"
        "'mlogViewTime':'VARCHAR','userId':'VARCHAR'}")
e0, e1 = config.EARLY_WINDOW_DAYS


def main():
    con = duckdb.connect(); con.execute("PRAGMA threads=4"); con.execute("PRAGMA disable_progress_bar")
    con.execute(f"PRAGMA temp_directory='{config.DERIVED_DIR / 'duckdb_tmp'}'")
    # card metadata: type (1 image / 2 video) and primary content category (first of contentId)
    con.execute(f"""CREATE TABLE cards AS SELECT mlogId, type AS card_type,
        NULLIF(TRIM(split_part(contentId, ',', 1)), '') AS content_cat
        FROM read_csv_auto('{CARDS}', header=true)""")
    con.execute(f"""
      COPY (
        SELECT i.userId, i.impressTime, i.impressPosition,
               i.isClick, i.isComment, i.isIntoPersonalHomepage, i.isShare, i.isViewComment, i.isLike,
               TRY_CAST(i.mlogViewTime AS DOUBLE) AS viewTime,
               c.card_type, c.content_cat
        FROM read_csv('/dev/stdin', header=true, columns={COLS}) i
        LEFT JOIN cards c ON i.mlogId = c.mlogId
        WHERE i.dt BETWEEN {e0} AND {e1}
      ) TO '{OUT}' (FORMAT PARQUET)
    """)
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT}')").fetchone()[0]
    print(f"wrote {OUT}  ({n} early-window events)", file=sys.stderr)


if __name__ == "__main__":
    main()
