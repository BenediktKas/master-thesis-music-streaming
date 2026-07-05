"""Volume-controlled taste test (Member A). Reads the cleaned impression stream on
stdin, joins to card content categories, and asks whether users differ in WHAT
they click once volume is removed. Reports category concentration, per-user
specialisation, and the silhouette of clusters formed on click-composition
vectors. A low silhouette means there is no separable taste structure.

Run:  bash src/data/... | python -m src.member_a_segmentation.taste_analysis
or use run_taste_analysis.sh.
"""
import numpy as np, duckdb
from src import config

CARDS = config.RAW_DIR.parent / "mlog_demographics.csv"
COLS = ("{'dt':'INT','impressPosition':'INT','impressTime':'BIGINT','isClick':'INT',"
        "'isComment':'INT','isIntoPersonalHomepage':'INT','isShare':'INT',"
        "'isViewComment':'INT','isLike':'INT','mlogId':'VARCHAR',"
        "'mlogViewTime':'VARCHAR','userId':'VARCHAR'}")
e0, e1 = config.EARLY_WINDOW_DAYS


def main():
    con = duckdb.connect(); con.execute("PRAGMA disable_progress_bar"); con.execute("PRAGMA threads=4")
    con.execute(f"CREATE TABLE cc AS SELECT mlogId, TRIM(unnest(string_split(contentId, ','))) AS cat "
                f"FROM read_csv_auto('{CARDS}', header=true) WHERE contentId IS NOT NULL AND contentId <> ''")
    con.execute(f"""CREATE TABLE uac AS
        SELECT i.userId, cc.cat, COUNT(*) impr, SUM(i.isClick) clk
        FROM read_csv('/dev/stdin', header=true, columns={COLS}) i JOIN cc ON i.mlogId = cc.mlogId
        WHERE i.dt BETWEEN {e0} AND {e1} GROUP BY i.userId, cc.cat""")
    g = con.execute("SELECT cat, SUM(clk) c FROM uac GROUP BY cat ORDER BY c DESC").df()
    cum = g["c"].cumsum() / g["c"].sum()
    print(f"content categories: {len(g)}; top-1/5/10 click share: "
          f"{cum.iloc[0]:.2f}/{cum.iloc[4]:.2f}/{cum.iloc[9]:.2f}; "
          f"categories covering 80% of clicks: {int((cum < 0.8).sum()) + 1}")
    topcats = list(g["cat"].head(40))
    rows = con.execute(f"SELECT userId, cat, clk FROM uac WHERE cat IN "
                       f"({','.join(chr(39)+c+chr(39) for c in topcats)})").df()
    users = con.execute("SELECT userId FROM uac GROUP BY userId HAVING SUM(clk) >= 5").df()["userId"]
    uidx = {u: i for i, u in enumerate(users)}; cidx = {c: j for j, c in enumerate(topcats)}
    M = np.zeros((len(users), len(topcats)))
    for _, r in rows.iterrows():
        if r["userId"] in uidx:
            M[uidx[r["userId"]], cidx[r["cat"]]] = r["clk"]
    M = M[M.sum(1) > 0]; S = M / M.sum(1, keepdims=True)
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    print(f"users with >=5 early clicks: {len(S)}; mean top-category share: {np.max(S, 1).mean():.3f}")
    sub = np.random.default_rng(0).choice(len(S), min(8000, len(S)), replace=False)
    for k in (2, 3, 4, 5):
        lab = KMeans(n_clusters=k, n_init=5, random_state=0).fit(S).labels_
        print(f"  composition clusters k={k}: silhouette={silhouette_score(S[sub], lab[sub]):.3f}")
    print("Low silhouette (<0.2) => no separable taste structure; users are activity-differentiated, not taste-differentiated.")


if __name__ == "__main__":
    main()
