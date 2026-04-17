"""
Stage 2 of the pipeline: audio-skeleton temporal alignment.

Raw broadcast footage often contains a leading segment in which the target
dancer has not yet entered the frame, so the keypoint tensor begins with a
block of all-zero frames (no detection). This script finds the first
non-zero keypoint frame t_start, crops the skeleton tensor from t_start
onwards, and cuts the accompanying audio at t_start / fps seconds so that
the audio stream and the skeleton stream start at the same moment.
"""

import os
import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip

# ================= Configuration =================
# Raw inputs
video_folder = r"E:\lzt\liulei\dataset".strip()
npy_folder = r"E:\lzt\liulei\StreetDance_Keypoints".strip()

# Aligned outputs (perfectly time-matched audio and skeletons)
output_audio_folder = r"E:\lzt\liulei\Aligned_Audio".strip()
output_npy_folder = r"E:\lzt\liulei\Aligned_Keypoints".strip()
# ==================================================

os.makedirs(output_audio_folder, exist_ok=True)
os.makedirs(output_npy_folder, exist_ok=True)


def process_alignment():
    print(f"\n{'='*50}")
    print("Audio-skeleton alignment and audio extraction")
    print(f"{'='*50}\n")

    video_files = [f for f in os.listdir(video_folder)
                   if f.lower().endswith(('.mp4', '.mov', '.avi'))]

    try:
        video_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    except Exception:
        video_files.sort()

    success_count = 0

    for video_name in video_files:
        base_name = os.path.splitext(video_name)[0]
        video_path = os.path.join(video_folder, video_name)
        npy_path = os.path.join(npy_folder, f"{base_name}_keypoints.npy")

        out_audio_path = os.path.join(output_audio_folder, f"{base_name}.wav")
        out_npy_path = os.path.join(output_npy_folder, f"{base_name}_aligned.npy")

        if not os.path.exists(npy_path):
            print(f"[SKIP] skeleton file for {base_name} not found.")
            continue

        try:
            # 1. Locate the first valid frame (drop leading all-zero frames).
            kpts_data = np.load(npy_path)  # shape: (frames, 17, 3)

            # Per-frame sum over the 17x3 keypoint matrix.
            frame_sums = np.sum(np.abs(kpts_data), axis=(1, 2))

            # First frame with any detected keypoint.
            valid_indices = np.where(frame_sums > 0)[0]

            if len(valid_indices) == 0:
                print(f"[FAIL] {base_name}: skeleton is entirely zero, no tracking signal.")
                continue

            start_idx = valid_indices[0]

            # 2. Trim the skeleton tensor.
            aligned_kpts = kpts_data[start_idx:]
            np.save(out_npy_path, aligned_kpts)

            # 3. Read the video FPS and cut audio at the matching timestamp.
            with VideoFileClip(video_path) as clip:
                fps = clip.fps
                start_time_sec = start_idx / fps  # millisecond precision

                if clip.audio is None:
                    print(f"[WARN] {base_name}: original video has no audio track.")
                else:
                    # Crop the audio starting from start_time_sec.
                    audio_subclip = clip.audio.subclip(start_time_sec)
                    # Export at 44.1 kHz (suitable for downstream spectral analysis).
                    audio_subclip.write_audiofile(out_audio_path, fps=44100, logger=None)

            print(f"[OK] {base_name}: trimmed {start_idx} frames "
                  f"(~{start_time_sec:.2f} s). Audio and skeleton aligned.")
            success_count += 1

        except Exception as e:
            print(f"[ERROR] while processing {base_name}: {e}")

    print(f"\n{'='*50}")
    print(f"Done. Aligned {success_count} files.")
    print(f"Audio output: {output_audio_folder}")
    print(f"Skeleton output: {output_npy_folder}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    process_alignment()
