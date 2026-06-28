"""Content-taste features (Member A input). Streams the impression CSV on stdin,
joins each impression to its card's metadata (mlog_demographics), and aggregates
per-user EARLY-WINDOW taste signals: what kind of content the user is exposed to
and engages with (video vs image, breadth of content/creators/artists).

Card metadata is read from data/raw/mlog_demographics.csv (extracted by the
shell wrapper). Writes data/derived/user_content_taste.parquet.

Usage (full file, on your Mac):  bash src/data/run_content_taste.sh "../Dataset/Raw_Data.zip"
"""
import sys, duckdb
from src import config

CARDS = config.RAW_DIR.parent / "mlog_demographics.csv"   # data/raw/mlog_demographics.csv
OUT = config.DERIVED_DIR / "user_content_taste.parquet"
config.DERIVED_DIR.mkdir(parents=True, exist_ok=True)
COLS = ("{'dt':'INT','impressPosition':'INT','impressTime':'BIGINT','isClick':'INT',"
        "'isComment':'INT','isIntoPersonalHomepage':'INT','isShare':'INT',"
        "'isViewComment':'INT','isLike':'INT','mlogId':'VARCHAR',"
        "'mlogViewTime':'VARCHAR','userId':'VARCHAR'}")
e0, e1 = config.EARLY_WINDOW_DAYS


def main():
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA disable_progress_bar")
    con.execute(f"CREATE TABLE cards AS SELECT mlogId, type, contentId, creatorId, artistId "
                f"FROM read_csv_auto('{CARDS}', header=true)")
    con.execute(f"""
    COPY (
      SELECT i.userId,
        COUNT(*)                                                        AS ct_seen,
        AVG(CASE WHEN c.type = 2 THEN 1.0 ELSE 0.0 END)                 AS ct_video_share_seen,
        COUNT(DISTINCT c.contentId)                                     AS ct_seen_n_content,
        COUNT(*) FILTER (WHERE i.isClick = 1)                           AS ct_clicked,
        AVG(CASE WHEN c.type = 2 THEN 1.0 ELSE 0.0 END) FILTER (WHERE i.isClick = 1) AS ct_video_share_click,
        COUNT(DISTINCT c.contentId) FILTER (WHERE i.isClick = 1)        AS ct_click_n_content,
        COUNT(DISTINCT c.creatorId) FILTER (WHERE i.isClick = 1)        AS ct_click_n_creators,
        COUNT(DISTINCT c.artistId)  FILTER (WHERE i.isClick = 1)        AS ct_click_n_artists
      FROM read_csv('/dev/stdin', header=true, columns={COLS}) i
      LEFT JOIN cards c ON i.mlogId = c.mlogId
      WHERE i.dt BETWEEN {e0} AND {e1}
      GROUP BY i.userId
    ) TO '{OUT}' (FORMAT PARQUET)
    """)
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{OUT}')").fetchone()[0]
    print(f"wrote {OUT}  ({n} users)", file=sys.stderr)


if __name__ == "__main__":
    main()
