"""Optional EDA: window-overlap breakdown, threshold sensitivity, and the
active-vs-inactive early-behaviour contrast. Writes data/derived/eda_summary.json
and prints the tables. Run after build_labels.
"""
import json, polars as pl
from src import config

DER = config.DERIVED_DIR


def main():
    agg = pl.read_parquet(DER / "user_window_agg.parquet")
    mt = pl.read_parquet(DER / "user_modeling_table.parquet")
    out = {"users_in_sample": agg.height, "study_users_both_windows": mt.height}

    both = ((agg["e_impr"] > 0) & (agg["l_impr"] > 0)).sum()
    early_only = ((agg["e_impr"] > 0) & (agg["l_impr"] == 0)).sum()
    label_only = ((agg["e_impr"] == 0) & (agg["l_impr"] > 0)).sum()
    out["window_overlap"] = {"both": int(both), "early_only": int(early_only),
                             "label_only": int(label_only)}

    for th in [0.0, 0.01, 0.05, 0.10]:
        out[f"inactive_rate_thresh_{th}"] = round((mt["l_click_rate"] <= th).mean(), 4)

    contrast = mt.group_by("is_inactive").agg([
        pl.col("e_impr").mean().round(2).alias("e_impr"),
        pl.col("e_clicks").mean().round(2).alias("e_clicks"),
        pl.col("e_active_days").mean().round(2).alias("e_active_days"),
        pl.col("e_avg_view_time").mean().round(2).alias("e_avg_view_time"),
        pl.col("e_like_rate").mean().round(4).alias("e_like_rate"),
    ]).sort("is_inactive")

    print("window overlap:", out["window_overlap"])
    print("\ninactive rate vs threshold:")
    for th in [0.0, 0.01, 0.05, 0.10]:
        print(f"  click_rate <= {th}: {out[f'inactive_rate_thresh_{th}']}")
    print("\nactive(false) vs inactive(true) early-window means:")
    print(contrast)
    json.dump(out, open(DER / "eda_summary.json", "w"), indent=2)
    print("\nwrote eda_summary.json")


if __name__ == "__main__":
    main()
