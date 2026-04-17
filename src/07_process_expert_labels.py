"""
Stage 7 of the pipeline: merge the seven-dimension expert Likert scores
with the ten-dimension AI feature matrix, and compute the weighted
100-point reference label.

Weighting (a priori, set by the rater):
    Technique      (mean of 2 sub-scores)  -> weight 30/70
    Musicality     (mean of 3 sub-scores)  -> weight 25/70
    Space/Control  (mean of 2 sub-scores)  -> weight 15/70
    total_score = sum of the three weighted components, rescaled to 0-100.

The seven Chinese Likert column headings are renamed to English for
downstream tooling and for the manuscript.
"""

import pandas as pd

# ================= Configuration =================
# Raw single-rater seven-dimension Likert scores.
teacher_scores_csv = r"E:\lzt\liulei\clean_teacher_scores.csv".strip()
# AI-derived 10-dimension feature matrix (stage 6 output).
ai_features_csv = r"E:\lzt\liulei\Master_Dance_Features.csv".strip()
# Combined training table: 10 features + 7 scores + total.
output_final_csv = r"E:\lzt\liulei\Final_Training_Dataset_V2.csv".strip()
# ==================================================


def process_and_merge_labels():
    print(f"\n{'='*50}")
    print("Merging expert Likert scores with AI feature matrix")
    print(f"{'='*50}\n")

    # 1. Load the cleaned expert score sheet.
    df_teacher = pd.read_csv(teacher_scores_csv, encoding='utf-8-sig')

    # Normalise the video_id column to a plain string ("12.0" -> "12").
    df_teacher['video_id'] = df_teacher.iloc[:, 0].astype(str).str.replace('.0', '', regex=False)

    # 2. Weighted-sum aggregation (see module docstring).
    #    Columns referenced positionally to avoid header-encoding issues.
    tech_score = df_teacher.iloc[:, 1:3].mean(axis=1) / 10.0 * (100 * 30 / 70)
    music_score = df_teacher.iloc[:, 3:6].mean(axis=1) / 10.0 * (100 * 25 / 70)
    space_score = df_teacher.iloc[:, 6:8].mean(axis=1) / 10.0 * (100 * 15 / 70)

    df_teacher['total_score'] = (tech_score + music_score + space_score).round(2)

    # 3. Load the AI feature matrix and align the key types.
    df_features = pd.read_csv(ai_features_csv, encoding='utf-8')
    df_features['video_id'] = df_features['video_id'].astype(str).str.replace('.0', '', regex=False)

    # Rename the seven raw Likert columns to English.
    score_cols_map = {
        df_teacher.columns[1]: 'Score_CleanExecution',
        df_teacher.columns[2]: 'Score_HighDifficulty',
        df_teacher.columns[3]: 'Score_BeatPrecision',
        df_teacher.columns[4]: 'Score_MusicSync',
        df_teacher.columns[5]: 'Score_RhythmRichness',
        df_teacher.columns[6]: 'Score_BodyControl',
        df_teacher.columns[7]: 'Score_StageSpace',
    }
    df_teacher = df_teacher.rename(columns=score_cols_map)
    score_cols = list(score_cols_map.values())

    # Inner join: AI features + 7 Likert dimensions + weighted total.
    merge_cols = ['video_id'] + score_cols + ['total_score']
    df_final = pd.merge(df_features, df_teacher[merge_cols], on='video_id', how='inner')

    df_final.to_csv(output_final_csv, index=False, encoding='utf-8-sig')

    print(f"[OK] merged. First video total_score = {df_teacher['total_score'][0]}")
    print(f"Final training set saved to: {output_final_csv}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    process_and_merge_labels()
