"""
Publication-Grade Figure Generator
===================================
Reads saved experiment results and generates figures.
Run comprehensive_evaluation.py first to produce the data files.

Figures:
  01 - Model Comparison (MAE + R², horizontal bars, 95% CI)
  02 - SHAP Feature Importance (colored by dimension)
  03 - Ablation Study (MAE increase with error context)
  04 - Model Stability (boxplot of 50-fold R² distribution)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ================= Configuration =================
output_dir = r"E:\lzt\liulei\Evaluation_Results"

N_SPLITS = 5
N_REPEATS = 10
N_FOLDS = N_SPLITS * N_REPEATS

FEATURE_NAMES = [
    'Limb Explosiveness', 'Max Opening Angle', 'Handstand Floor Ratio',
    'Beat Hit Rate', 'Beat Avg Error (s)', 'Beat Variance',
    'Peak-Trough Control', 'Kinetic Energy CV', 'CoG Total Distance', 'Convex Hull Coverage',
]

DIMENSION_MAP = {
    'Technique': [0, 1, 2],
    'Musicality': [3, 4, 5],
    'Space Control': [6, 7, 8, 9],
}

# Unified color palette (Tableau-inspired, colorblind-friendly)
PALETTE = {
    'primary':    '#4C72B0',   # steel blue — best model / main accent
    'secondary':  '#C44E52',   # muted red — fit lines / highlights
    'muted':      '#CCCCCC',   # light gray — non-best models
    'technique':  '#C44E52',   # muted red
    'musicality': '#DD8452',   # warm orange
    'space':      '#4C72B0',   # steel blue
    'positive':   '#C44E52',   # bar positive
    'negative':   '#55A868',   # muted green — bar negative
    'scatter':    '#4C72B0',   # scatter dots
    'annotation': '#F5F0E1',   # warm cream — annotation box
}

DIM_COLORS = {
    'Technique': PALETTE['technique'],
    'Musicality': PALETTE['musicality'],
    'Space Control': PALETTE['space'],
}

BEST_COLOR = PALETTE['primary']
OTHER_COLOR = PALETTE['muted']

# ================= Load Data =================

def load_results():
    with open(f'{output_dir}/cv_results.json', 'r') as f:
        cv_results = json.load(f)
    with open(f'{output_dir}/stats_summary.json', 'r') as f:
        stats_summary = json.load(f)
    shap_df = pd.read_csv(f'{output_dir}/shap_importance.csv')
    with open(f'{output_dir}/ablation_results.json', 'r') as f:
        ablation_results = json.load(f)
    return cv_results, stats_summary, shap_df, ablation_results

# ================= Style Setup =================

def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11,
        'axes.unicode_minus': False,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

# ================= Figure 1: Model Comparison =================

def plot_model_comparison(stats_summary):
    model_names = list(stats_summary.keys())
    n_models = len(model_names)

    # Sort by MAE (best first)
    sorted_mae = sorted(model_names, key=lambda m: stats_summary[m]['MAE']['mean'])
    best_mae = sorted_mae[0]

    # Sort by R² (best first)
    sorted_r2 = sorted(model_names, key=lambda m: stats_summary[m]['R2']['mean'], reverse=True)
    best_r2 = sorted_r2[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left: MAE ---
    mae_means = [stats_summary[m]['MAE']['mean'] for m in sorted_mae]
    mae_cis = [1.96 * stats_summary[m]['MAE']['std'] / np.sqrt(N_FOLDS) for m in sorted_mae]
    colors = [BEST_COLOR if m == best_mae else OTHER_COLOR for m in sorted_mae]

    ax1.barh(range(n_models), mae_means, xerr=mae_cis, capsize=4,
             color=colors, edgecolor='white', linewidth=0.8, height=0.6)
    ax1.set_yticks(range(n_models))
    ax1.set_yticklabels(sorted_mae, fontsize=11)
    ax1.set_xlabel('MAE (Mean Absolute Error)', fontsize=12)
    ax1.set_title('(a) MAE Comparison', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    for i, (val, ci) in enumerate(zip(mae_means, mae_cis)):
        ax1.text(val + ci + 0.03, i, f'{val:.3f}', va='center', fontsize=10,
                 fontweight='bold' if sorted_mae[i] == best_mae else 'normal')

    # --- Right: R² ---
    r2_means = [stats_summary[m]['R2']['mean'] for m in sorted_r2]
    r2_cis = [1.96 * stats_summary[m]['R2']['std'] / np.sqrt(N_FOLDS) for m in sorted_r2]
    colors = [BEST_COLOR if m == best_r2 else OTHER_COLOR for m in sorted_r2]

    ax2.barh(range(n_models), r2_means, xerr=r2_cis, capsize=4,
             color=colors, edgecolor='white', linewidth=0.8, height=0.6)
    ax2.set_yticks(range(n_models))
    ax2.set_yticklabels(sorted_r2, fontsize=11)
    ax2.set_xlabel('R² Score', fontsize=12)
    ax2.set_title('(b) R² Comparison', fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    ax2.axvline(x=0, color='black', linewidth=0.8)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    # Expand left margin so negative labels don't overlap with model names
    x_min = min(r2_means) - max(r2_cis) - 0.06
    x_max = max(r2_means) + max(r2_cis) + 0.06
    ax2.set_xlim(x_min, x_max)

    for i, (val, ci) in enumerate(zip(r2_means, r2_cis)):
        offset = ci + 0.01 if val >= 0 else -(ci + 0.01)
        ha = 'left' if val >= 0 else 'right'
        ax2.text(val + offset, i, f'{val:.3f}', va='center', ha=ha, fontsize=10,
                 fontweight='bold' if sorted_r2[i] == best_r2 else 'normal')

    fig.suptitle(f'Model Performance (Repeated {N_SPLITS}-Fold CV, {N_REPEATS} repeats, 95% CI)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f'{output_dir}/01_Model_Comparison.png', dpi=300, bbox_inches='tight')
    print(" Saved: 01_Model_Comparison.png")
    plt.close()

# ================= Figure 2: SHAP Beeswarm Plot =================

def plot_shap_beeswarm():
    import shap

    # Load raw SHAP values and feature values
    shap_raw = pd.read_csv(f'{output_dir}/shap_values_raw.csv')
    feat_vals = pd.read_csv(f'{output_dir}/shap_feature_values.csv')

    # Reconstruct shap.Explanation object for the beeswarm plot
    explanation = shap.Explanation(
        values=shap_raw.values,
        data=feat_vals.values,
        feature_names=list(shap_raw.columns)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.sca(ax)
    shap.plots.beeswarm(explanation, show=False, max_display=10)
    ax.set_title('SHAP Beeswarm Plot (Random Forest)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('SHAP value (impact on prediction)', fontsize=11)

    plt.tight_layout()
    fig.savefig(f'{output_dir}/02_SHAP_Beeswarm.png', dpi=300, bbox_inches='tight')
    print(" Saved: 02_SHAP_Beeswarm.png")
    plt.close()

# ================= Figure 3: Ablation Study =================

def plot_ablation(ablation_results):
    fig, ax = plt.subplots(figsize=(8, 5))

    dims = list(ablation_results.keys())
    mae_increases = [ablation_results[d]['mae_increase'] for d in dims]
    colors = [DIM_COLORS.get(d, '#888888') for d in dims]

    bars = ax.bar(dims, mae_increases, color=colors, edgecolor='white', linewidth=1.5, width=0.55)
    ax.set_ylabel('$\Delta$ MAE When Removed', fontsize=12)
    ax.set_title('Dimension Contribution (Leave-One-Dimension-Out)', fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Set y-axis limits with extra room below for negative labels
    y_min = min(mae_increases) - 0.06
    y_max = max(mae_increases) + 0.05
    ax.set_ylim(y_min, y_max)

    for bar, val in zip(bars, mae_increases):
        height = bar.get_height()
        if height >= 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:+.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height - 0.01,
                    f'{val:+.3f}', ha='center', va='top', fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig.savefig(f'{output_dir}/03_Ablation_Study.png', dpi=300, bbox_inches='tight')
    print(" Saved: 03_Ablation_Study.png")
    plt.close()

# ================= Figure 4: R² Boxplot =================

def plot_r2_boxplot(cv_results):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Sort models by median R²
    model_names = sorted(cv_results.keys(),
                         key=lambda m: np.median(cv_results[m]['R2']), reverse=True)
    data = [cv_results[m]['R2'] for m in model_names]
    best_model = model_names[0]

    bp = ax.boxplot(data, patch_artist=True, vert=True, widths=0.5,
                    medianprops=dict(color=PALETTE['secondary'], linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5))

    for i, (patch, name) in enumerate(zip(bp['boxes'], model_names)):
        color = BEST_COLOR if name == best_model else OTHER_COLOR
        patch.set_facecolor(color)
        patch.set_edgecolor('white')
        patch.set_linewidth(1.2)
        patch.set_alpha(0.85)

    ax.set_xticklabels(model_names, fontsize=11, rotation=30, ha='right')
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title(f'R² Distribution Across {N_FOLDS} Folds (Repeated {N_SPLITS}-Fold CV)',
                 fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Annotate median values
    for i, name in enumerate(model_names):
        median_val = np.median(cv_results[name]['R2'])
        ax.text(i + 1, median_val + 0.02, f'{median_val:.3f}',
                ha='center', fontsize=9.5, fontweight='bold' if name == best_model else 'normal')

    plt.tight_layout()
    fig.savefig(f'{output_dir}/04_Model_R2_Boxplot.png', dpi=300, bbox_inches='tight')
    print(" Saved: 04_Model_R2_Boxplot.png")
    plt.close()

# ================= Figure 5: Feature vs Expert Score Correlation Heatmap =================

def plot_correlation_heatmap():
    import seaborn as sns

    input_csv = r"E:\lzt\liulei\Final_Training_Dataset_Cleaned.csv"
    df = pd.read_csv(input_csv, encoding='utf-8-sig')

    # Extract AI features (cols 1-10) and expert scores (cols 11-17)
    X_features = df.iloc[:, 1:11]
    X_features.columns = FEATURE_NAMES

    EXPERT_SCORE_NAMES = [
        'Clean Execution', 'High Difficulty', 'Beat Precision',
        'Music Sync', 'Rhythm Richness', 'Body Control', 'Stage Space',
    ]
    Y_scores = df.iloc[:, 11:18]
    Y_scores.columns = EXPERT_SCORE_NAMES

    # 10x7 correlation matrix
    corr_matrix = pd.DataFrame(
        np.zeros((len(FEATURE_NAMES), len(EXPERT_SCORE_NAMES))),
        index=FEATURE_NAMES, columns=EXPERT_SCORE_NAMES
    )
    for feat in FEATURE_NAMES:
        for score in EXPERT_SCORE_NAMES:
            corr_matrix.loc[feat, score] = X_features[feat].corr(Y_scores[score])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix.astype(float), annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-0.5, vmax=0.5,
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Pearson r', 'shrink': 0.8},
                ax=ax)
    ax.set_title('AI Feature vs Expert Score Correlation', fontsize=13, fontweight='bold')
    ax.set_xlabel('Expert Scoring Dimension', fontsize=12)
    ax.set_ylabel('AI-Extracted Feature', fontsize=12)
    ax.tick_params(axis='x', rotation=35)
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    fig.savefig(f'{output_dir}/05_Feature_Expert_Correlation.png', dpi=300, bbox_inches='tight')
    print(" Saved: 05_Feature_Expert_Correlation.png")
    plt.close()

# ================= Figure 6: Predicted vs Actual Scatter =================

def plot_pred_vs_actual():
    from scipy import stats as sp_stats

    pred_df = pd.read_csv(f'{output_dir}/best_model_predictions.csv')
    y_true = pred_df['y_true'].values
    y_pred = pred_df['y_pred'].values

    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    mae = np.mean(np.abs(y_true - y_pred))

    # Linear fit
    slope, intercept, r_val, p_val, std_err = sp_stats.linregress(y_true, y_pred)
    fit_x = np.linspace(y_true.min() - 1, y_true.max() + 1, 100)
    fit_y = slope * fit_x + intercept

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_true, y_pred, c=PALETTE['scatter'], alpha=0.55, edgecolors='white',
               linewidth=0.6, s=55, zorder=3)

    # Perfect prediction line
    lim_min = min(y_true.min(), y_pred.min()) - 2
    lim_max = max(y_true.max(), y_pred.max()) + 2
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color='#888888', linestyle='--',
            linewidth=1, alpha=0.6, label='Perfect prediction')

    # Regression fit line
    ax.plot(fit_x, fit_y, color=PALETTE['secondary'], linewidth=2.2, label=f'Fit (r={r_val:.3f})')

    ax.set_xlabel('Actual Score', fontsize=12)
    ax.set_ylabel('Predicted Score', fontsize=12)
    ax.set_title('Random Forest: Predicted vs Actual', fontsize=13, fontweight='bold')
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10)

    # Annotation box
    textstr = f'R² = {r2:.3f}\nMAE = {mae:.2f}\nn = {len(y_true)}'
    props = dict(boxstyle='round,pad=0.5', facecolor=PALETTE['annotation'], alpha=0.85)
    ax.text(0.97, 0.05, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    fig.savefig(f'{output_dir}/06_Pred_vs_Actual.png', dpi=300, bbox_inches='tight')
    print(" Saved: 06_Pred_vs_Actual.png")
    plt.close()

# ================= Main =================

def main():
    setup_style()
    print("Loading experiment results...")
    cv_results, stats_summary, shap_df, ablation_results = load_results()
    print(f"  Models: {list(cv_results.keys())}")
    print(f"  Folds per model: {len(cv_results[list(cv_results.keys())[0]]['MAE'])}")
    print()

    plot_model_comparison(stats_summary)
    plot_shap_beeswarm()
    plot_ablation(ablation_results)
    plot_r2_boxplot(cv_results)
    plot_correlation_heatmap()
    plot_pred_vs_actual()

    print(f"\nAll 6 figures saved to {output_dir}/")

if __name__ == "__main__":
    main()
