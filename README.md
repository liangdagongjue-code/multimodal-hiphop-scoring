# A Multimodal Interpretable Pipeline for Semi-Automated Scoring of Competitive Hip-Hop Dance

**A Single-Rater Pilot Study on Feasibility, Feature Importance, and the Dominance of Spatial–Kinematic Signal**

This repository accompanies the manuscript submitted to *Sports Medicine – Open* (Springer) and contains the complete pipeline for tracking a single target performer through cluttered multi-person competition footage, extracting ten interpretable multimodal features per routine, and evaluating seven regularised regressors under Repeated 5-Fold cross-validation with SHAP and leave-one-dimension-out ablation analyses.

---

## Abstract

Competitive hip-hop dance is contested at major international championships but lacks a unified international scoring standard and suffers from documented inter-judge reliability problems on subjective sub-criteria. Preliminary auditions, battle heats, and showcase rounds also pose a computer-vision challenge that off-the-shelf multi-person pose pipelines cannot address: a cluttered, multi-person background from which a single target performer must be isolated for reliable skeletal tracking.

This pilot study develops and evaluates a semi-automated, multimodal, hand-crafted pipeline that approximates the scores of one experienced Chinese hip-hop judge across 100 Bilibili competition clips. A cascaded YOLO11–SAM2 vision stack (initialised with a single manual click per video) produces per-frame skeletal keypoints robust to the cluttered multi-person background; `librosa` onset detection extracts beat timestamps from the synchronised audio. Ten interpretable features are engineered across three dimensions (Technique × 3, Musicality × 3, Space Control × 4) and evaluated under Repeated 5-Fold cross-validation (10 repeats × 5 folds = 50 total folds) across seven regularised regressors. Model differences are assessed by Friedman and paired t-tests, feature importance by SHAP, and subdimension contributions by leave-one-dimension-out ablation.

**Results.** Random Forest achieves the lowest mean absolute error (MAE = 5.06 ± 0.58) and highest mean R² (+0.10), corresponding to less than one label standard deviation on the observed label range (58–86, σ = 6.7) and within the magnitude of inter-rater disagreement reported in adjacent aesthetic-sport literature. SHAP attribution and ablation converge on the same dimension-level ranking: Space Control features occupy three of the top four SHAP positions and, when removed, increase MAE by +0.295 points. Hand-crafted onset-matching musicality features, on the present single-rater sample with unprocessed broadcast audio, behave as noise (removing them *improves* MAE by 0.061 points).

---

## Repository layout

```
.
├── src/                                        Pipeline (stages 1–10, ordered by prefix)
│   ├── 01_video_tracking_yolo_sam.py           YOLO11 + SAM2 cascaded target isolation
│   ├── 02_audio_skeleton_alignment.py          Audio / skeleton time-axis alignment
│   ├── 03_extract_technique.py                 Technique features (F1–F3)
│   ├── 04_extract_musicality.py                Musicality features (F4–F6)
│   ├── 05_extract_space_control.py             Space-control features (F7–F10)
│   ├── 06_merge_features.py                    Merge per-dimension CSVs
│   ├── 07_process_expert_labels.py             Merge expert Likert scores + weighted total
│   ├── 08_clean_outliers.py                    Winsorise the 10 AI features
│   ├── 09_comprehensive_evaluation.py          7-model CV + Friedman + SHAP + ablation
│   └── 10_plot_figures.py                      Regenerate the six paper figures
├── data/
│   ├── expert_scores_raw.csv                   Raw single-rater 7-dimension Likert scores
│   ├── final_training_dataset.csv              10 features + 7 Likert + weighted total
│   └── final_training_dataset_cleaned.csv      Winsorised inputs to stages 9–10
├── results/
│   ├── figures/                                The six paper figures (fig01–fig06, 300 dpi)
│   └── cache/                                  Saved CV / SHAP / ablation artefacts
├── requirements.txt                            Python dependencies
├── LICENSE                                     MIT
└── README.md                                   This file
```

---

## Data

**Primary video corpus (not included in this repository).** One hundred competitive hip-hop performances (1080p, 30 fps, 40–60 s) were self-collected from publicly available footage on Bilibili (https://www.bilibili.com) covering preliminary audition rounds and related multi-person hip-hop competition events. Because the videos are third-party broadcast material released by the respective event organisers, and because the dataset is too large to version-control on GitHub, the raw videos are not redistributed here. Please contact the corresponding author for access arrangements if needed for strict replication.

**Provided in `data/`.**

| File | Shape | Description |
| --- | --- | --- |
| `expert_scores_raw.csv` | 100 × 8 | Single-rater seven-dimension Likert scores (1–10) per video: Clean Execution, High Difficulty, Beat Precision, Music Sync, Rhythm Richness, Body Control, Stage Space |
| `final_training_dataset.csv` | 100 × 19 | video_id + 10 AI features + 7 Likert scores + weighted 100-point `total_score` (main inputs to stages 8–9) |
| `final_training_dataset_cleaned.csv` | 100 × 19 | Same as above after Winsorisation (1%/99%) of the 10 AI feature columns; expert scores and the weighted total are untouched |

**Column layout of the training CSVs (iloc positions):**

```
[ 0]  video_id
[ 1]  limb_explosiveness            (F1, technique)
[ 2]  max_opening_angle             (F2, technique)
[ 3]  inversion_ratio               (F3, technique)
[ 4]  beat_hit_rate                 (F4, musicality)
[ 5]  beat_avg_error_sec            (F5, musicality)
[ 6]  beat_variance                 (F6, musicality)
[ 7]  peak_trough_control_ratio    (F7, space control)
[ 8]  kinetic_energy_cv             (F8, space control)
[ 9]  cog_total_distance            (F9, space control)
[10]  convex_hull_coverage          (F10, space control)
[11]  Score_CleanExecution          (Likert 1–10)
[12]  Score_HighDifficulty          (Likert 1–10)
[13]  Score_BeatPrecision           (Likert 1–10)
[14]  Score_MusicSync               (Likert 1–10)
[15]  Score_RhythmRichness          (Likert 1–10)
[16]  Score_BodyControl             (Likert 1–10)
[17]  Score_StageSpace              (Likert 1–10)
[18]  total_score                   (weighted 0–100 reference label)
```

---

## Installation

```bash
# 1. Clone the repository.
git clone https://github.com/<user>/<repo>.git
cd <repo>

# 2. Create an isolated environment (conda or venv, Python 3.10+).
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 3. Install pinned Python dependencies.
pip install -r requirements.txt

# 4. Install SAM2 separately (not on PyPI).
pip install git+https://github.com/facebookresearch/sam2.git

# 5. Download the pretrained weights into a local `checkpoints/` folder:
#      - SAM2 Hiera-Small:  https://github.com/facebookresearch/sam2
#      - YOLO11s-pose, YOLO11n, YOLOv8n: https://github.com/ultralytics/ultralytics
```

---

## Reproducing the results

The pipeline is split into ten numbered stages (`src/01_*.py` through `src/10_*.py`). Each stage reads from and writes to absolute paths defined in its own `Configuration` block at the top of the file — edit those paths once to point to your local layout before running. Stages 1–2 are required only if you start from raw video; stages 3–7 regenerate `final_training_dataset.csv`; stages 8–10 are sufficient to reproduce the manuscript's numbers from the provided CSVs.

### Full pipeline (starting from 100 raw videos)

```bash
python src/01_video_tracking_yolo_sam.py        # interactive: one click per video
python src/02_audio_skeleton_alignment.py
python src/03_extract_technique.py
python src/04_extract_musicality.py
python src/05_extract_space_control.py
python src/06_merge_features.py
python src/07_process_expert_labels.py
python src/08_clean_outliers.py
python src/09_comprehensive_evaluation.py       # ~5-10 min CPU-bound
python src/10_plot_figures.py                   # regenerates the six paper figures
```

### Quick replication (using the provided CSVs)

If you only want to verify the modelling results, skip stages 1–7 and run:

```bash
python src/08_clean_outliers.py                 # writes final_training_dataset_cleaned.csv
python src/09_comprehensive_evaluation.py
python src/10_plot_figures.py
```

To regenerate only the figures from the cached experiment artefacts in `results/cache/` (no re-training):

```bash
python src/10_plot_figures.py
```

---

## Key design decisions

- **Single manual click per video (stage 1).** In cluttered multi-person competition frames (5–20 persons), purely automatic multi-object trackers (e.g. ByteTrack, BoT-SORT) drift between identities, which directly contaminates every downstream feature. One human click amortised over 40–60 s is a negligible annotation cost relative to the stability it guarantees.
- **SAM2 for identity, YOLO-pose for joints (stage 1).** SAM2's mask propagation is robust to extreme articulation and brief occlusion; YOLO11s-pose provides a strong backbone for keypoint regression. An IoU ≥ 0.1 gate between SAM2's silhouette-tight box and YOLO's looser person box accepts modest mismatch while rejecting non-target skeletons.
- **`librosa.onset.onset_detect` over `beat_track` (stage 4).** Hip-hop track percussion frequently violates the steady-tempo assumption that periodic beat trackers exploit.
- **Winsorisation rather than outlier removal (stage 8).** Clipping the 10 AI features to their 1%/99% percentiles preserves the n = 100 sample size. Expert Likert scores and the weighted total label are *not* clipped.
- **Strong regularisation and no grid search (stage 9).** On a 100-sample dataset an exhaustive hyperparameter search is dominated by partitioning noise; we fix a single small-n-safe configuration per model.
- **MAE as the primary metric.** On a noisy subjective target, `R²` ceilings are dictated by label noise; mean absolute error in the same units as the 100-point reference score is the more appropriate yardstick.

---

## Main results at a glance (Random Forest, 50-fold CV)

| Metric | Value |
| --- | --- |
| MAE | 5.055 ± 0.584 |
| RMSE | 6.427 ± 0.704 |
| R² | 0.096 |
| MAPE | 6.96% |
| Friedman test across 7 models | p < 0.001 |
| Top SHAP features | Convex Hull Coverage, CoG Total Distance, Max Opening Angle, Peak-Trough Control |
| Remove Space Control | ΔMAE = **+0.295** (most damaging) |
| Remove Musicality | ΔMAE = −0.061 (improves) |

---

## Limitations

This is a **single-rater pilot study**. Labels are provided by one experienced Chinese hip-hop judge; inter-rater reliability (intraclass correlation coefficient, ICC) is therefore not estimable on the present data, and all results should be read as conditional on one rater's scoring behaviour. Multi-rater replication with at least three independent judges and reported ICC values is identified as the primary follow-up this pilot enables. The pipeline also uses 2D pose only (Z-axis information is discarded by the monocular camera), raw broadcast audio without source separation, and a 30 fps sampling rate that marginally undersamples pops / hits / isolations. See the Limitations section of the manuscript for a full discussion.

---

## Citation

```
Liu L, Li Z, Ma Y, Alsulaimi A. A multimodal interpretable pipeline for semi-automated
scoring of competitive hip-hop dance in cluttered multi-person scenes: a single-rater
pilot study on feasibility, feature importance, and the dominance of spatial–kinematic
signal. Sports Medicine – Open (submitted, 2026).
```

---

## License

This code is released under the MIT License (see [LICENSE](LICENSE)).
Pretrained model weights (SAM2, YOLO) are governed by their respective upstream licenses.

---

## Acknowledgements

We thank the hip-hop judge who contributed their time and expertise to the scoring protocol, and the Bilibili community for making the competition footage publicly accessible. This study was supported by the Scientific Research Program Funded by the Education Department of Shaanxi Provincial Government (Program No. 25JK0277).
