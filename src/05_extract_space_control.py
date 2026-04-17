"""
Stage 5 of the pipeline: Space-Control-dimension feature extraction.

Computes four scalar features per video that quantify how the dancer
occupies and traverses the stage:

  F7  Peak-to-trough control ratio
      Ratio of mean peak to mean trough on the whole-body kinetic-energy
      envelope (sum of squared joint velocities across 17 joints).
      Captures the dancer's ability to alternate between explosive and
      controlled segments within a routine.
  F8  Kinetic-energy coefficient of variation (CV = std / mean)
      Larger CV -> wider dynamic range (more explosive peaks separated by
      calm passages); smaller CV -> smoother, more sustained motion.
  F9  Total centre-of-gravity travel distance (pixels)
      Sum of frame-to-frame Euclidean displacements of the hip midpoint.
  F10 Convex-hull floor coverage
      Area of the convex hull over the sequence of hip-midpoint positions,
      normalised by the 1920x1080 frame area. Serves as a proxy for how
      much of the performance area the dancer claims across the routine.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull

# ================= Configuration =================
# Aligned, trimmed keypoint tensors produced by stage 2.
input_npy_folder = r"E:\lzt\liulei\Aligned_Keypoints".strip()
# Per-video space-control feature table.
output_csv_path = r"E:\lzt\liulei\Space_Control_Features.csv".strip()

# Assumed capture resolution for Convex-Hull normalisation (1080p).
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_AREA = FRAME_WIDTH * FRAME_HEIGHT
# ==================================================

L_HIP, R_HIP = 11, 12


def process_space_control_features():
    print(f"\n{'='*50}")
    print("Extracting Space-Control-dimension features")
    print(f"{'='*50}\n")

    features_list = []
    npy_files = [f for f in os.listdir(input_npy_folder) if f.endswith('.npy')]

    for npy_file in npy_files:
        video_id = npy_file.replace('_aligned.npy', '')
        filepath = os.path.join(input_npy_folder, npy_file)

        try:
            data = np.load(filepath)  # shape: (frames, 17, 3)
            frames = data.shape[0]
            if frames < 3:
                continue

            coords = data[:, :, :2]  # shape: (frames, 17, 2)

            # ---------------------------------------------------------
            # F7 / F8: kinetic-energy-based control features.
            # ---------------------------------------------------------
            # Per-joint frame-to-frame speed.
            velocity = np.linalg.norm(np.diff(coords, axis=0), axis=2)  # (frames-1, 17)
            # Whole-body kinetic energy (up to a constant mass factor).
            kinetic_energy = np.sum(velocity ** 2, axis=1)  # (frames-1,)

            # Peaks and troughs on the kinetic-energy envelope.
            peaks, _ = find_peaks(kinetic_energy, distance=5)
            valleys, _ = find_peaks(-kinetic_energy, distance=5)

            mean_peak_energy = np.mean(kinetic_energy[peaks]) if len(peaks) > 0 else 0
            mean_valley_energy = np.mean(kinetic_energy[valleys]) if len(valleys) > 0 else 1e-6

            # Peak-to-trough ratio (+1 stabiliser in the denominator).
            control_ratio = mean_peak_energy / (mean_valley_energy + 1.0)

            # Coefficient of variation of kinetic energy.
            ke_mean = np.mean(kinetic_energy)
            ke_cv = np.std(kinetic_energy) / ke_mean if ke_mean > 0 else 0

            # ---------------------------------------------------------
            # F9 / F10: centre-of-gravity trajectory features.
            # ---------------------------------------------------------
            # Hip midpoint (proxy for the dancer's centre of gravity).
            hip_center = (coords[:, L_HIP, :] + coords[:, R_HIP, :]) / 2  # (frames, 2)

            # Total travel distance = sum of frame-to-frame Euclidean displacements.
            total_distance = np.sum(np.linalg.norm(np.diff(hip_center, axis=0), axis=1))

            # Convex hull area over the CoG trajectory.
            unique_points = np.unique(hip_center, axis=0)  # drop duplicates (stationary dancer)
            convex_area = 0.0

            if len(unique_points) >= 3:  # ConvexHull requires at least 3 points
                try:
                    hull = ConvexHull(unique_points)
                    # In SciPy 2D, ConvexHull.volume is the enclosed area.
                    convex_area = hull.volume
                except Exception:
                    # Degenerate case: all points on a single line. Skip.
                    pass

            # Stage coverage = hull area / frame area (dimensionless).
            coverage_ratio = convex_area / FRAME_AREA

            features_list.append({
                "video_id": video_id,
                "peak_trough_control_ratio": round(control_ratio, 2),
                "kinetic_energy_cv": round(ke_cv, 2),
                "cog_total_distance": round(total_distance, 2),
                "convex_hull_coverage": round(coverage_ratio, 4)
            })

            print(f"[OK] {video_id} | control ratio: {control_ratio:.1f} | "
                  f"travel: {total_distance:.0f} px | coverage: {coverage_ratio*100:.2f}%")

        except Exception as e:
            print(f"[ERROR] {npy_file}: {e}")

    df = pd.DataFrame(features_list)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    print(f"\n{'='*50}")
    print(f"Done. Space-Control features saved to: {output_csv_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    process_space_control_features()
