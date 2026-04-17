"""
Stage 4 of the pipeline: Musicality-dimension feature extraction.

Aligns motion-onset timestamps (peaks in whole-body acceleration) against
audio-onset timestamps from librosa.onset.onset_detect and reports three
beat-alignment features per video:

  F4  Beat hit rate
      Fraction of motion onsets that fall within MATCH_THRESHOLD_SEC of
      the nearest audio onset.
  F5  Mean absolute timing error (seconds)
      Average temporal offset between each matched motion onset and its
      nearest audio onset.
  F6  Timing variance (seconds^2)
      Variance of the matched timing errors, used as a stability proxy.

Notes:
  - librosa.onset.onset_detect is preferred over beat_track because hip-hop
    track percussion frequently violates the steady-tempo assumption that
    periodic beat trackers exploit.
  - Motion onsets are extracted as local maxima of the mean wrist/ankle
    acceleration magnitude via scipy.signal.find_peaks.
"""

import os
import numpy as np
import pandas as pd
import librosa
from scipy.signal import find_peaks

# ================= Configuration =================
# Aligned audio (stage 2) and aligned skeletons (stage 2).
audio_folder = r"E:\lzt\liulei\Aligned_Audio".strip()
npy_folder = r"E:\lzt\liulei\Aligned_Keypoints".strip()
output_csv_path = r"E:\lzt\liulei\Musicality_Features.csv".strip()

# Source video frame rate, used to convert frame indices into seconds.
VIDEO_FPS = 30.0
# A motion onset within this tolerance of an audio onset is counted as "on beat".
MATCH_THRESHOLD_SEC = 0.15
# ==================================================

L_WRIST, R_WRIST = 9, 10
L_ANKLE, R_ANKLE = 15, 16


def process_musicality_features():
    print(f"\n{'='*50}")
    print("Extracting Musicality-dimension features")
    print(f"{'='*50}\n")

    features_list = []
    npy_files = [f for f in os.listdir(npy_folder) if f.endswith('.npy')]

    for npy_file in npy_files:
        base_name = npy_file.replace('_aligned.npy', '')
        npy_path = os.path.join(npy_folder, npy_file)
        audio_path = os.path.join(audio_folder, f"{base_name}.wav")

        if not os.path.exists(audio_path):
            print(f"[SKIP] {base_name}: matching audio file not found.")
            continue

        try:
            # ---------------------------------------------------------
            # Step 1: extract audio onsets (librosa).
            # ---------------------------------------------------------
            y, sr = librosa.load(audio_path, sr=None)
            # Returns onset timestamps in seconds.
            audio_onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')

            # ---------------------------------------------------------
            # Step 2: extract motion onsets from skeleton kinematics.
            # ---------------------------------------------------------
            data = np.load(npy_path)  # (frames, 17, 3)
            frames = data.shape[0]
            if frames < 3:
                continue

            # Wrist and ankle (x, y) coordinates only.
            extremities = data[:, [L_WRIST, R_WRIST, L_ANKLE, R_ANKLE], :2]
            velocity = np.linalg.norm(np.diff(extremities, axis=0), axis=2)
            acceleration = np.abs(np.diff(velocity, axis=0))  # (frames-2, 4)

            # Mean acceleration across the four extremities as a whole-body
            # instantaneous "hit" signal.
            mean_accel = np.mean(acceleration, axis=1)

            # Peak-picking: minimum separation of 10 frames (~1/3 s at 30 fps)
            # collapses adjacent sub-peaks that belong to the same hit.
            peaks, _ = find_peaks(mean_accel, height=np.mean(mean_accel), distance=10)

            # Convert peak frame indices back to seconds. The +2 offset
            # accounts for the two successive np.diff calls above.
            motion_timestamps = (peaks + 2) / VIDEO_FPS

            # ---------------------------------------------------------
            # Step 3: nearest-neighbour matching between motion and audio onsets.
            # ---------------------------------------------------------
            hit_count = 0
            time_errors = []

            for m_time in motion_timestamps:
                if len(audio_onsets) == 0:
                    break
                closest_audio_idx = np.argmin(np.abs(audio_onsets - m_time))
                closest_audio_time = audio_onsets[closest_audio_idx]

                error = abs(m_time - closest_audio_time)

                # Count as a hit iff the offset is within the tolerance window.
                if error <= MATCH_THRESHOLD_SEC:
                    hit_count += 1
                    time_errors.append(error)

            total_motion_peaks = len(motion_timestamps)
            # F4: fraction of motion onsets that land on a beat.
            hit_ratio = hit_count / total_motion_peaks if total_motion_peaks > 0 else 0.0
            # F5: mean absolute timing error over matched onsets.
            mean_error = np.mean(time_errors) if len(time_errors) > 0 else MATCH_THRESHOLD_SEC
            # F6: timing variance over matched onsets.
            variance_error = np.var(time_errors) if len(time_errors) > 0 else 0.0

            features_list.append({
                "video_id": base_name,
                "beat_hit_rate": round(hit_ratio, 3),
                "beat_avg_error_sec": round(mean_error, 3),
                "beat_variance": round(variance_error, 5)
            })

            print(f"[OK] {base_name} | motion onsets: {total_motion_peaks} | "
                  f"matched: {hit_count} | hit rate: {hit_ratio*100:.1f}%")

        except Exception as e:
            print(f"[ERROR] {base_name}: {e}")

    df = pd.DataFrame(features_list)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    print(f"\n{'='*50}")
    print(f"Done. Musicality features saved to: {output_csv_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    process_musicality_features()
