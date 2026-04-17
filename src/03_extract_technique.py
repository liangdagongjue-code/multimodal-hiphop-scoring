"""
Stage 3 of the pipeline: Technique-dimension feature extraction.

Reads aligned (T, 17, 3) keypoint tensors in COCO format and computes three
scalar features per video that quantify the Technique supra-dimension:

  F1  Limb explosiveness
      Literature reports that "cleanliness" in hip-hop is driven by
      explosive hits followed by abrupt holds. We take the 95th percentile
      of the per-frame wrist/ankle acceleration magnitude as a proxy for
      the peak explosive power of the routine.

  F2  Maximum leg opening angle (flexibility)
      Vector from left hip to left ankle (v_L) and right hip to right
      ankle (v_R); the angle between v_L and v_R via arccos on the
      normalised dot product. The per-video maximum reflects the dancer's
      achieved flexibility range (e.g. splits, high kicks).

  F3  Inversion / floorwork ratio
      In image coordinates y increases downward, so upright posture implies
      hip_y > shoulder_y. Frames in which shoulder_y > hip_y correspond to
      handstands, power moves, or supine floorwork. We report the fraction
      of frames in that state, with a torso-length threshold to reject
      crouching false positives.

COCO-17 keypoint indices used:
  shoulders  5 (L), 6 (R)
  wrists     9 (L), 10 (R)
  hips      11 (L), 12 (R)
  ankles    15 (L), 16 (R)
"""

import os
import numpy as np
import pandas as pd
import math

# ================= Configuration =================
# Aligned, trimmed keypoint tensors produced by stage 2.
input_npy_folder = r"E:\lzt\liulei\Aligned_Keypoints".strip()
# Per-video technique feature table.
output_csv_path = r"E:\lzt\liulei\Technique_Features.csv".strip()
# ==================================================

# COCO-17 keypoint indices
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16


def calculate_angle(v1, v2):
    """Angle (degrees) between two 2D vectors via arccos of the normalised dot product."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    # Clip to avoid arccos domain errors from floating-point drift.
    cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def process_technique_features():
    print(f"\n{'='*50}")
    print("Extracting Technique-dimension features")
    print(f"{'='*50}\n")

    features_list = []
    npy_files = [f for f in os.listdir(input_npy_folder) if f.endswith('.npy')]

    for npy_file in npy_files:
        video_id = npy_file.split('_')[0]
        filepath = os.path.join(input_npy_folder, npy_file)

        try:
            data = np.load(filepath)  # shape: (frames, 17, 3)
            frames = data.shape[0]
            if frames < 3:
                continue

            coords = data[:, :, :2]  # keep only (x, y)

            # ---------------------------------------------------------
            # F1: Limb explosiveness (95th-percentile extremity acceleration).
            # ---------------------------------------------------------
            # Wrists and ankles: shape (frames, 4, 2)
            extremities = coords[:, [L_WRIST, R_WRIST, L_ANKLE, R_ANKLE], :]

            # Per-frame velocity magnitude via frame-to-frame coordinate differences.
            velocity = np.linalg.norm(np.diff(extremities, axis=0), axis=2)  # (frames-1, 4)
            # Acceleration as first difference of velocity.
            acceleration = np.diff(velocity, axis=0)  # (frames-2, 4)

            # 95th percentile as the "peak explosiveness" proxy for the routine.
            if len(acceleration) > 0:
                explosive_power = np.percentile(np.abs(acceleration), 95)
            else:
                explosive_power = 0.0

            # ---------------------------------------------------------
            # F2: Maximum leg opening angle (flexibility).
            # ---------------------------------------------------------
            max_leg_angle = 0.0
            for i in range(frames):
                # Left leg vector: hip to ankle.
                vec_left = coords[i, L_ANKLE] - coords[i, L_HIP]
                # Right leg vector: hip to ankle.
                vec_right = coords[i, R_ANKLE] - coords[i, R_HIP]

                angle = calculate_angle(vec_left, vec_right)
                if angle > max_leg_angle:
                    max_leg_angle = angle

            # ---------------------------------------------------------
            # F3: Inversion / floorwork ratio.
            # ---------------------------------------------------------
            # Image coordinates: y grows downward. When upright, hip_y > shoulder_y.
            # If shoulder_y > hip_y in a given frame, the dancer is inverted
            # (handstand, freeze, power move) or supine on the floor.
            shoulder_y = (coords[:, L_SHOULDER, 1] + coords[:, R_SHOULDER, 1]) / 2
            hip_y = (coords[:, L_HIP, 1] + coords[:, R_HIP, 1]) / 2

            # Shoulder-to-hip distance used as a threshold against crouch-induced false positives.
            torso_length = np.abs(hip_y - shoulder_y)

            # Inverted iff shoulders are above hips AND torso is not fully collapsed.
            inverted_frames = np.sum((shoulder_y > hip_y) & (torso_length > 20))
            inversion_ratio = inverted_frames / frames if frames > 0 else 0.0

            features_list.append({
                "video_id": video_id,
                "limb_explosiveness": round(explosive_power, 2),
                "max_opening_angle": round(max_leg_angle, 2),
                "inversion_ratio": round(inversion_ratio, 4)
            })

            print(f"[OK] {video_id} | explosive: {explosive_power:.1f} | "
                  f"opening angle: {max_leg_angle:.0f} deg | "
                  f"inversion: {inversion_ratio*100:.1f}%")

        except Exception as e:
            print(f"[ERROR] {npy_file}: {e}")

    df = pd.DataFrame(features_list)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    print(f"\n{'='*50}")
    print(f"Done. Technique features saved to: {output_csv_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    process_technique_features()
