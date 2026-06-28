"""DEPRECATED — merged into build_user_window_agg.py.

Early-window AND label-window aggregates are now computed together in a single
streaming pass (avoids reading the 6.3 GB file twice). Use:
    bash src/data/run_user_window_agg.sh
then build_labels.py. Kept as a pointer to avoid confusion.
"""
raise SystemExit(__doc__)
