"""
Stage 8 of the pipeline: data cleaning and outlier handling.

- IQR-based outlier detection per AI feature (report only).
- Winsorization: clip each of the 10 AI features to its 1st-99th
  percentile range rather than removing outlier rows, so that the
  n=100 sample size is preserved.
- Expert Likert scores (columns 11-17) and the weighted total label
  (column 18) are NOT clipped: rater decisions should not be adjusted.
- Near-zero-variance feature report.
"""
import pandas as pd
import numpy as np

# ================= Configuration =================
input_csv = r"E:\lzt\liulei\Final_Training_Dataset_V2.csv"
output_csv = r"E:\lzt\liulei\Final_Training_Dataset_Cleaned.csv"
# ==================================================

# Column iloc layout:
#   [0]         video_id
#   [1..10]     10 AI features (3 technique + 3 musicality + 4 space-control)
#   [11..17]    7 expert Likert scores
#   [18]        weighted total_score (reference label)
FEATURE_ILOC = list(range(1, 11))


def detect_outliers_iqr(series, k=1.5):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - k * IQR, Q3 + k * IQR
    return (series < lower) | (series > upper)


def clean_data():
    df = pd.read_csv(input_csv, encoding='utf-8-sig')
    print(f"Original shape: {df.shape}")
    # The weighted total score is the last column.
    total_score = df.iloc[:, -1]
    print(f"Score range: [{total_score.min():.1f}, {total_score.max():.1f}]\n")

    # --- Outlier report (AI features only; expert scores are not touched). ---
    print("=" * 60)
    print("Outlier Detection Report (IQR method, k=1.5)")
    print("=" * 60)

    total_outliers = 0
    for idx in FEATURE_ILOC:
        col_name = df.columns[idx]
        series = df.iloc[:, idx]
        mask = detect_outliers_iqr(series)
        n_outliers = mask.sum()
        if n_outliers > 0:
            total_outliers += n_outliers
            outlier_vals = series[mask].values
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            print(f"  Col[{idx}] {col_name}: {n_outliers} outliers")
            print(f"    Values: {outlier_vals}")
            print(f"    Normal range: [{Q1 - 1.5*IQR:.2f}, {Q3 + 1.5*IQR:.2f}]")

    print(f"\nTotal outlier cells: {total_outliers}")

    # --- Winsorisation: clip the 10 AI feature columns to [p01, p99]. ---
    print("\nApplying Winsorisation (clip to 1st-99th percentile)...")
    df_clean = df.copy()
    for idx in FEATURE_ILOC:
        col_name = df.columns[idx]
        lo = df.iloc[:, idx].quantile(0.01)
        hi = df.iloc[:, idx].quantile(0.99)
        before = df_clean.iloc[:, idx].copy()
        df_clean.iloc[:, idx] = df_clean.iloc[:, idx].clip(lo, hi)
        n_clipped = (before != df_clean.iloc[:, idx]).sum()
        if n_clipped > 0:
            print(f"  Col[{idx}] {col_name}: clipped {n_clipped} values to [{lo:.4f}, {hi:.4f}]")

    # --- Near-zero-variance feature warning. ---
    print("\n" + "=" * 60)
    print("Near-Zero-Variance Feature Check")
    print("=" * 60)
    for idx in FEATURE_ILOC:
        col_name = df.columns[idx]
        nonzero_ratio = (df_clean.iloc[:, idx] != 0).mean()
        if nonzero_ratio < 0.1:
            print(f"  WARNING: Col[{idx}] '{col_name}' has {nonzero_ratio*100:.0f}% non-zero values")
            print(f"    -> This feature may have low discriminative power")

    df_clean.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nCleaned data saved to: {output_csv}")


if __name__ == "__main__":
    clean_data()
