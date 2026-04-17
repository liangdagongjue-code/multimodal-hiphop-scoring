"""
Stage 6 of the pipeline: merge the three per-dimension feature tables
into a single 10-column feature matrix.

Reads the three CSVs produced by stages 3, 4, and 5 and outer-joins them
on the video_id column so that a missing sub-feature in one block does
not silently drop the whole row. Missing values are filled with zero,
and rows are sorted by numeric video_id for downstream stability.
"""

import pandas as pd
import os

# ================= Configuration =================
technique_csv = r"E:\lzt\liulei\Technique_Features.csv".strip()
musicality_csv = r"E:\lzt\liulei\Musicality_Features.csv".strip()
space_control_csv = r"E:\lzt\liulei\Space_Control_Features.csv".strip()

# Master feature table (inputs to the machine-learning stage).
output_master_csv = r"E:\lzt\liulei\Master_Dance_Features.csv".strip()
# ==================================================


def safe_read_csv(filepath):
    """Read CSV with UTF-8-SIG fallback to GBK (handles files re-saved by Excel)."""
    try:
        return pd.read_csv(filepath, encoding='utf-8-sig')
    except UnicodeDecodeError:
        return pd.read_csv(filepath, encoding='gbk')


def merge_all_features():
    print(f"\n{'='*50}")
    print("Merging Technique + Musicality + Space-Control feature tables")
    print(f"{'='*50}\n")

    try:
        print("Loading feature tables...")
        df_tech = safe_read_csv(technique_csv)
        df_music = safe_read_csv(musicality_csv)
        df_space = safe_read_csv(space_control_csv)

        # Cast the join key to string so that numeric / stringified IDs match.
        df_tech['video_id'] = df_tech['video_id'].astype(str)
        df_music['video_id'] = df_music['video_id'].astype(str)
        df_space['video_id'] = df_space['video_id'].astype(str)

        # Outer join preserves rows even when one block is missing.
        print("Joining Technique + Musicality ...")
        df_master = pd.merge(df_tech, df_music, on='video_id', how='outer')

        print("Joining Space-Control ...")
        df_master = pd.merge(df_master, df_space, on='video_id', how='outer')

        # Numeric-aware sort by video_id (1, 2, 3, ... rather than 1, 10, 2).
        df_master['sort_key'] = pd.to_numeric(df_master['video_id'], errors='coerce')
        df_master = df_master.sort_values('sort_key').drop('sort_key', axis=1)

        # Replace NaN with 0 so that downstream ML code does not crash.
        df_master = df_master.fillna(0)

        df_master.to_csv(output_master_csv, index=False, encoding='utf-8-sig')

        print(f"\nMerge successful. {len(df_master)} videos with complete features.")
        print(f"Master feature matrix saved to: {output_master_csv}")
        print(f"{'='*50}\n")

    except Exception as e:
        print(f"[ERROR] merge failed: {e}")


if __name__ == "__main__":
    merge_all_features()
