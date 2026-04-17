"""
Stage 9 of the pipeline: machine-learning evaluation framework.

Runs the complete experimental evaluation reported in the manuscript:
  - Repeated 5-fold cross-validation (10 repeats = 50 folds) across seven
    regularised regressors (XGBoost, LightGBM, Random Forest, Gradient
    Boosting, SVR(RBF), Ridge, ElasticNet).
  - Per-model mean / std / 95% CI for MAE, RMSE, R^2, MAPE.
  - Friedman omnibus test followed by paired t-tests against the
    best-MAE model.
  - SHAP feature importance on the Random Forest refit to the full
    n=100 sample.
  - Leave-one-dimension-out ablation (Technique / Musicality /
    Space-Control), reporting MAE increase rather than R^2 ratio to
    avoid division artefacts when the baseline R^2 is negative.
  - Saves all intermediate artefacts to disk so that 10_plot_figures.py
    can regenerate paper figures without rerunning the experiments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, RepeatedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import lightgbm as lgb
import shap
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ================= Configuration =================
input_csv = r"E:\lzt\liulei\Final_Training_Dataset_Cleaned.csv"
output_dir = r"E:\lzt\liulei\Evaluation_Results"

# Column iloc layout in the input CSV:
#   [0]        video_id
#   [1..10]    10 AI features
#   [11..17]   7 expert Likert scores
#   [18]       weighted total_score (reference label)
FEATURE_ILOC = list(range(1, 11))
EXPERT_SCORE_ILOC = list(range(11, 18))
TARGET_ILOC = 18

# English labels for the 7 expert dimensions (correlation analysis).
EXPERT_SCORE_NAMES = [
    'Clean Execution',
    'High Difficulty',
    'Beat Precision',
    'Music Sync',
    'Rhythm Richness',
    'Body Control',
    'Stage Space',
]

# English names for the 10 AI features (SHAP plots / correlation heatmap).
FEATURE_NAMES = [
    'Limb Explosiveness',
    'Max Opening Angle',
    'Handstand Floor Ratio',
    'Beat Hit Rate',
    'Beat Avg Error (s)',
    'Beat Variance',
    'Peak-Trough Control',
    'Kinetic Energy CV',
    'CoG Total Distance',
    'Convex Hull Coverage',
]

# Feature grouping for the leave-one-dimension-out ablation.
# Indices are 0-based positions within FEATURE_ILOC (i.e. columns of X).
DIMENSION_MAP = {
    'Technique':     [0, 1, 2],
    'Musicality':    [3, 4, 5],
    'Space Control': [6, 7, 8, 9],
}

N_SPLITS = 5
N_REPEATS = 10   # 10 repeats x 5 folds = 50 total folds
RANDOM_SEED = 42

# ==================================================

import os
os.makedirs(output_dir, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# =====================================================
# SECTION 1: Data Preparation and Model Definitions
# =====================================================

def load_and_prepare_data():
    """Load the cleaned training CSV by iloc position (column names may differ)."""
    df = pd.read_csv(input_csv, encoding='utf-8-sig')
    X = df.iloc[:, FEATURE_ILOC].values
    y = df.iloc[:, TARGET_ILOC].values
    print(f"  Features shape: {X.shape}, Target range: [{y.min():.1f}, {y.max():.1f}]")
    return X, y, df


def create_models():
    """Instantiate all seven regressors with small-n regularised hyperparameters."""
    models = {
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=5.0,
            random_state=RANDOM_SEED, verbosity=0
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            num_leaves=8, min_child_samples=10,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=5.0,
            random_state=RANDOM_SEED, verbose=-1
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=3, min_samples_leaf=10,
            min_samples_split=10, max_features='sqrt',
            random_state=RANDOM_SEED, n_jobs=-1
        ),
        'GBR': GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            min_samples_leaf=10, subsample=0.8,
            random_state=RANDOM_SEED
        ),
        'SVR (RBF)': SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.5),
        'Ridge': Ridge(alpha=10.0),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000),
    }
    return models

# =====================================================
# SECTION 2: Repeated k-fold cross-validation
# =====================================================

def evaluate_with_kfold(X, y):
    """Run Repeated 5-Fold CV across all seven regressors.

    Returns a dict keyed by model name with per-fold lists of MAE, RMSE,
    R^2, MAPE plus per-fold y_pred / y_true for scatter-plot use.
    """
    models = create_models()
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)
    total_folds = N_SPLITS * N_REPEATS

    results = {model_name: {'MAE': [], 'RMSE': [], 'R2': [], 'MAPE': [],
                            'y_pred': [], 'y_true': []}
               for model_name in models}
    scaler = StandardScaler()

    print(f"\n{'='*70}")
    print(f" Repeated {N_SPLITS}-Fold CV ({N_REPEATS} repeats = {total_folds} folds)")
    print(f"{'='*70}\n")

    fold_idx = 1
    for train_idx, test_idx in rkf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # In-fold standardisation (no leakage from test fold).
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        repeat_num = (fold_idx - 1) // N_SPLITS + 1
        fold_in_repeat = (fold_idx - 1) % N_SPLITS + 1
        if fold_in_repeat == 1:
            print(f"Repeat {repeat_num}/{N_REPEATS}...", end=" ", flush=True)
        if fold_in_repeat == N_SPLITS:
            print("done")

        for model_name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)

            results[model_name]['MAE'].append(mae)
            results[model_name]['RMSE'].append(rmse)
            results[model_name]['R2'].append(r2)
            results[model_name]['MAPE'].append(mape)
            results[model_name]['y_pred'].append(y_pred.tolist())
            results[model_name]['y_true'].append(y_test.tolist())

        fold_idx += 1

    return results


def compute_cv_statistics(results):
    """Aggregate per-fold metrics into mean, std and 95% CI per model."""
    stats_summary = {}

    for model_name in results.keys():
        stats_summary[model_name] = {}
        for metric in ['MAE', 'RMSE', 'R2', 'MAPE']:
            values = np.array(results[model_name][metric])
            mean_val = values.mean()
            std_val = values.std()
            n_folds = N_SPLITS * N_REPEATS
            ci_lower = mean_val - 1.96 * std_val / np.sqrt(n_folds)
            ci_upper = mean_val + 1.96 * std_val / np.sqrt(n_folds)

            stats_summary[model_name][metric] = {
                'mean': mean_val,
                'std': std_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }

    return stats_summary


def print_cv_results(stats_summary):
    """Pretty-print the CV summary table."""
    print(f"\n{'='*70}")
    print(" Cross-Validation Results Summary (Mean +/- SD)")
    print(f"{'='*70}\n")

    for metric in ['MAE', 'RMSE', 'R2', 'MAPE']:
        print(f"\n{metric}:")
        print("-" * 70)
        for model_name, stats_dict in stats_summary.items():
            s = stats_dict[metric]
            if metric in ['MAE', 'RMSE', 'MAPE']:
                print(f"  {model_name:20s}: {s['mean']:6.3f} +/- {s['std']:5.3f}  "
                      f"[95% CI: {s['ci_lower']:6.3f}, {s['ci_upper']:6.3f}]")
            else:  # R^2
                print(f"  {model_name:20s}: {s['mean']:6.4f} +/- {s['std']:5.4f}  "
                      f"[95% CI: {s['ci_lower']:6.4f}, {s['ci_upper']:6.4f}]")

# =====================================================
# SECTION 3: Statistical Significance Tests
# =====================================================

def perform_statistical_tests(results):
    """Friedman omnibus + paired t-tests of the best-MAE model against others."""
    print(f"\n{'='*70}")
    print(" Statistical Significance Tests")
    print(f"{'='*70}\n")

    model_names = list(results.keys())

    # Friedman test: non-parametric alternative to repeated-measures ANOVA.
    print("Friedman Test (H0: all models have equal MAE distributions):\n")
    mae_arrays = [np.array(results[m]['MAE']) for m in model_names]
    stat, p_value = stats.friedmanchisquare(*mae_arrays)
    print(f"  Friedman chi-square statistic: {stat:.4f}")
    print(f"  p-value: {p_value:.2e}")
    if p_value < 0.05:
        print("  SIGNIFICANT (p < 0.05): models differ significantly.\n")
    else:
        print("  NOT significant (p >= 0.05).\n")

    # Best-MAE model as baseline for pairwise tests.
    mean_maes = {m: np.mean(results[m]['MAE']) for m in model_names}
    best_model = min(mean_maes, key=mean_maes.get)
    best_mae = np.array(results[best_model]['MAE'])

    print(f"Pairwise paired t-tests (best = '{best_model}') on per-fold MAE:\n")

    for model_name in model_names:
        if model_name == best_model:
            continue
        other_mae = np.array(results[model_name]['MAE'])
        t_stat, p_val = stats.ttest_rel(best_mae, other_mae)
        mean_diff = (best_mae - other_mae).mean()
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else "ns"
        print(f"  {best_model} vs {model_name:20s}: delta MAE = {mean_diff:+6.3f}, "
              f"t={t_stat:+7.3f}, p={p_val:.4f} {sig}")
    print()

# =====================================================
# SECTION 4: SHAP Feature Importance
# =====================================================

def compute_shap_importance(X, y):
    """Compute SHAP values on the Random Forest refit to the full sample."""
    print(f"\n{'='*70}")
    print(" SHAP Feature Importance Analysis (Random Forest)")
    print(f"{'='*70}\n")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(
        n_estimators=200, max_depth=3, min_samples_leaf=10,
        min_samples_split=10, max_features='sqrt',
        random_state=RANDOM_SEED, n_jobs=-1
    )
    model.fit(X_scaled, y)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_scaled)
    shap_values.feature_names = FEATURE_NAMES

    # Feature importance = mean absolute SHAP value per feature.
    importance = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'Feature': FEATURE_NAMES,
        'SHAP_Importance': importance
    }).sort_values('SHAP_Importance', ascending=False)

    print("Top Features (by mean |SHAP value|):")
    for idx, row in feature_importance_df.iterrows():
        print(f"  {row['Feature']:30s}: {row['SHAP_Importance']:.4f}")

    return shap_values, feature_importance_df

# =====================================================
# SECTION 5: Leave-One-Dimension-Out Ablation
# =====================================================

def ablation_study(X, y):
    """Leave-one-dimension-out ablation, reported as MAE increase.

    Using MAE delta rather than R^2 ratio avoids the numerical blow-up
    that occurs when the baseline R^2 is near zero or negative.
    """
    print(f"\n{'='*70}")
    print(" Ablation Study: Dimension Contribution Analysis")
    print(f"{'='*70}\n")

    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)
    scaler = StandardScaler()

    def _make_model():
        return RandomForestRegressor(
            n_estimators=200, max_depth=3, min_samples_leaf=10,
            min_samples_split=10, max_features='sqrt',
            random_state=RANDOM_SEED, n_jobs=-1
        )

    # Baseline: all 10 features.
    full_mae_scores = []
    full_r2_scores = []
    for train_idx, test_idx in rkf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = _make_model()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        full_mae_scores.append(mean_absolute_error(y_test, y_pred))
        full_r2_scores.append(r2_score(y_test, y_pred))

    baseline_mae = np.mean(full_mae_scores)
    baseline_r2 = np.mean(full_r2_scores)

    print(f"Baseline (all features): MAE = {baseline_mae:.3f}, R2 = {baseline_r2:.4f}\n")
    print("Leave-One-Dimension-Out Performance:\n")

    ablation_results = {}
    for dim_name, dim_indices in DIMENSION_MAP.items():
        # Drop this dimension; keep everything else.
        keep_indices = [i for i in range(X.shape[1]) if i not in dim_indices]
        X_ablated = X[:, keep_indices]

        ablated_mae_scores = []
        ablated_r2_scores = []
        for train_idx, test_idx in rkf.split(X_ablated):
            X_train, X_test = X_ablated[train_idx], X_ablated[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model = _make_model()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            ablated_mae_scores.append(mean_absolute_error(y_test, y_pred))
            ablated_r2_scores.append(r2_score(y_test, y_pred))

        ablated_mae = np.mean(ablated_mae_scores)
        ablated_r2 = np.mean(ablated_r2_scores)
        # Positive delta => removing this dimension hurts => dimension is useful.
        mae_increase = ablated_mae - baseline_mae

        ablation_results[dim_name] = {
            'ablated_mae': ablated_mae,
            'ablated_r2': ablated_r2,
            'mae_increase': mae_increase,
        }

        print(f"  Without '{dim_name}':")
        print(f"    MAE = {ablated_mae:.3f} (delta MAE = {mae_increase:+.3f}), R2 = {ablated_r2:.4f}")

    return ablation_results

# =====================================================
# MAIN
# =====================================================

def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION FRAMEWORK")
    print("   Repeated k-fold CV + multi-model comparison + statistical tests")
    print("="*70)

    # Step 1: load data.
    X, y, df = load_and_prepare_data()
    print(f"\nData loaded: {X.shape}")

    # Step 2: Repeated k-fold CV across all seven regressors.
    results = evaluate_with_kfold(X, y)
    stats_summary = compute_cv_statistics(results)
    print_cv_results(stats_summary)

    # Step 3: Statistical significance tests.
    perform_statistical_tests(results)

    # Step 4: SHAP feature importance (Random Forest refit to full n=100).
    shap_values, feature_importance_df = compute_shap_importance(X, y)

    # Step 5: Leave-one-dimension-out ablation.
    ablation_results = ablation_study(X, y)

    # Step 6: Persist intermediate artefacts for 10_plot_figures.py.
    import json

    # 6a: per-fold metrics (feeds the R^2 boxplot and stability analyses).
    cv_results_export = {}
    for model_name in results:
        cv_results_export[model_name] = {
            'MAE': results[model_name]['MAE'],
            'RMSE': results[model_name]['RMSE'],
            'R2': results[model_name]['R2'],
            'MAPE': results[model_name]['MAPE'],
        }
    with open(f'{output_dir}/cv_results.json', 'w') as f:
        json.dump(cv_results_export, f, indent=2)

    # 6b: per-model mean / std / 95% CI.
    with open(f'{output_dir}/stats_summary.json', 'w') as f:
        json.dump(stats_summary, f, indent=2)

    # 6c: SHAP feature importance + raw SHAP values for the beeswarm plot.
    feature_importance_df.to_csv(f'{output_dir}/shap_importance.csv', index=False)
    # shap_values.values has shape (100, 10): save with feature names.
    shap_raw_df = pd.DataFrame(shap_values.values, columns=FEATURE_NAMES)
    shap_raw_df.to_csv(f'{output_dir}/shap_values_raw.csv', index=False)
    # Scaled feature values used for beeswarm colouring.
    scaler_save = StandardScaler()
    X_scaled_save = scaler_save.fit_transform(X)
    feat_vals_df = pd.DataFrame(X_scaled_save, columns=FEATURE_NAMES)
    feat_vals_df.to_csv(f'{output_dir}/shap_feature_values.csv', index=False)

    # 6d: ablation results.
    with open(f'{output_dir}/ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)

    # 6e: best-model predictions for the scatter plot (first repeat only,
    # so each video contributes exactly one out-of-fold prediction).
    best_model_name = 'Random Forest'
    y_true_all = []
    y_pred_all = []
    for fold_i in range(N_SPLITS):
        y_true_all.extend(results[best_model_name]['y_true'][fold_i])
        y_pred_all.extend(results[best_model_name]['y_pred'][fold_i])
    pred_df = pd.DataFrame({'y_true': y_true_all, 'y_pred': y_pred_all})
    pred_df.to_csv(f'{output_dir}/best_model_predictions.csv', index=False)

    print(f"\n Intermediate data saved to {output_dir}/")
    print("   cv_results.json, stats_summary.json, shap_importance.csv,")
    print("   ablation_results.json, best_model_predictions.csv")
    print("   Run 10_plot_figures.py to regenerate paper figures without re-running experiments.")

    # Step 7: plain-text summary report.
    with open(f'{output_dir}/Evaluation_Report.txt', 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write("Dataset: 100 videos, 10 features\n")
        f.write(f"CV Strategy: Repeated {N_SPLITS}-Fold CV ({N_REPEATS} repeats "
                f"= {N_SPLITS*N_REPEATS} folds)\n")
        f.write("Models: XGBoost, LightGBM, Random Forest, GBR, SVR, Ridge, ElasticNet\n\n")

        f.write("RESULTS SUMMARY\n")
        f.write("-" * 70 + "\n")
        for model_name, stats_dict in stats_summary.items():
            f.write(f"\n{model_name}:\n")
            for metric in ['MAE', 'RMSE', 'R2', 'MAPE']:
                s = stats_dict[metric]
                f.write(f"  {metric}: {s['mean']:.4f} +/- {s['std']:.4f}\n")

    print(f"\n Report saved: Evaluation_Report.txt")
    print(f"\n All evaluations complete. See {output_dir}/ for outputs.")


if __name__ == "__main__":
    main()
