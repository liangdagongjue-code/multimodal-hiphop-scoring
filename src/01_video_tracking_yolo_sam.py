"""
Stage 1 of the pipeline: YOLO11-SAM2 cascaded single-target tracking.

For each input video, the user manually clicks the judged performer on one
reference frame to initialise a bounding box. SAM2 then propagates the box
through the entire video as a tight silhouette mask, and YOLO11s-pose is
matched to the SAM2 mask (via IoU gating) to obtain a stable 17-keypoint
COCO skeleton for the target dancer only.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import math
import sys
import multiprocessing
import shutil
import torch
from moviepy.video.io.VideoFileClip import VideoFileClip

# ================= Configuration =================
input_folder = r"E:\lzt\liulei\dataset".strip()
output_folder = r"E:\lzt\liulei\StreetDance_Cleaned".strip()
keypoints_folder = r"E:\lzt\liulei\StreetDance_Keypoints".strip()

SKIP_FILES = [f"{i}.mp4" for i in range(41, 100)]

MAX_WORKERS = 4

SAM2_MODEL_CFG = "sam2/sam2_hiera_s.yaml"
SAM2_CHECKPOINT = r"E:\lzt\liulei\checkpoints\sam2_hiera_small.pt"
# ===================================================

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(keypoints_folder):
    os.makedirs(keypoints_folder)

click_point = None


def init_worker(l):
    global lock
    lock = l


def calculate_iou(box1, box2):
    """Intersection-over-union of two axis-aligned bounding boxes."""
    if box1 is None or box2 is None:
        return 0.0
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(box1Area + box2Area - interArea + 1e-6)


def mouse_callback(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)


# ==============================================================
# Phase 1: manual target selection (YOLO-assisted)
# ==============================================================
def phase_one_selection(video_files):
    """Interactive per-video target-box collection.

    For each video, the operator advances frames with 'D', clicks once on
    the judged performer, and confirms with SPACE. The chosen frame index
    and bounding box are stored together so that SAM2 can be initialised
    at the exact frame on which the box was drawn.
    """
    global click_point
    print(f"\n{'='*50}")
    print("Phase 1: collecting target-box anchors (YOLO-assisted)")
    print("Usage: click the target dancer -> press SPACE to confirm -> auto advance")
    print(f"{'='*50}\n")

    select_model = YOLO('yolov8n.pt')
    tasks = []

    cv2.namedWindow('Select_Dancer', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Select_Dancer', mouse_callback)

    for f in video_files:
        input_path = os.path.join(input_folder, f)
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            continue
        ret, current_frame = cap.read()
        if not ret:
            continue

        current_frame_idx = 0

        results = select_model.predict(current_frame, conf=0.1, verbose=False, classes=[0])
        click_point = None
        target_box = None
        confirmed = False

        while not confirmed:
            display_img = current_frame.copy()
            boxes = []
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

            if click_point is not None:
                cx, cy = click_point
                selected_idx, best_dist = -1, float('inf')
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    dist = math.hypot((x1 + x2) / 2 - cx, (y1 + y2) / 2 - cy)
                    if x1 < cx < x2 and y1 < cy < y2:
                        selected_idx = i
                        break
                    if dist < best_dist and dist < 100:
                        best_dist = dist
                        selected_idx = i

                if selected_idx != -1:
                    target_box = boxes[selected_idx]
                    bx1, by1, bx2, by2 = map(int, target_box)
                    cv2.rectangle(display_img, (bx1, by1), (bx2, by2), (0, 0, 255), 4)
                    cv2.putText(display_img, "Ready", (bx1, by1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    click_point = None

            cv2.putText(display_img, f"[{f}] Frame: {current_frame_idx} | Click & SPACE",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.imshow('Select_Dancer', display_img)

            key = cv2.waitKey(30) & 0xFF
            if key == 32 and target_box is not None:
                # Record the exact frame index at which the box was drawn,
                # not only the box itself, so SAM2 can be seeded correctly.
                tasks.append({
                    "filename": f,
                    "initial_box": [float(v) for v in target_box],
                    "start_frame_idx": current_frame_idx
                })
                print(f"[OK] {f} registered (starts at frame {current_frame_idx})")
                confirmed = True
            elif key == ord('d') or key == ord('D'):
                ret, current_frame = cap.read()
                if not ret:
                    break
                current_frame_idx += 1
                results = select_model.predict(current_frame, conf=0.1, verbose=False, classes=[0])
                click_point, target_box = None, None
            elif key == 27:
                cap.release()
                cv2.destroyAllWindows()
                sys.exit()

        cap.release()
    cv2.destroyAllWindows()
    del select_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tasks


# ==============================================================
# Phase 2: SAM2 propagation + YOLO-pose keypoint extraction
# ==============================================================
def process_video_worker(task):
    """Per-video worker run inside a multiprocessing pool.

    Dumps every frame to disk, seeds SAM2 at the user-selected frame and
    box, propagates the mask through the full clip, then runs YOLO11s-pose
    on every frame and matches candidate pose boxes to the SAM2 mask via
    IoU >= 0.1. The final output is (i) a visualisation video with the
    mask box and accepted skeleton overlaid, and (ii) a (T, 17, 3) NumPy
    array of keypoints for the target dancer.
    """
    global lock
    filename = task['filename']
    initial_box = task['initial_box']
    start_frame_idx = task['start_frame_idx']

    base_name = os.path.splitext(filename)[0]
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    temp_output_path = os.path.join(output_folder, base_name + "_temp.mp4")
    npy_output_path = os.path.join(keypoints_folder, base_name + "_keypoints.npy")

    temp_frame_dir = os.path.join(output_folder, f"temp_frames_{base_name}")
    os.makedirs(temp_frame_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(temp_frame_dir, f"{frame_idx:05d}.jpg"), frame)
        frame_idx += 1
    cap.release()

    device_id = "cuda:0" if torch.cuda.is_available() else "cpu"

    with lock:
        from sam2.build_sam import build_sam2_video_predictor
        import hydra
        from hydra.core.global_hydra import GlobalHydra

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        hydra.initialize_config_dir(
            config_dir=r"E:\lzt\liulei\.venv\Lib\site-packages\sam2\configs",
            version_base="1.2"
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            predictor = build_sam2_video_predictor(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=device_id)
            inference_state = predictor.init_state(video_path=temp_frame_dir)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        # Seed SAM2 at the frame on which the operator actually drew the
        # box, not a hard-coded frame 0, otherwise the mask drifts.
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=start_frame_idx,
            obj_id=1,
            box=np.array(initial_box, dtype=np.float32)
        )

        sam2_tracked_boxes = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
            if mask.sum() > 0:
                y_indices, x_indices = np.where(mask)
                sam2_tracked_boxes[out_frame_idx] = [
                    x_indices.min(), y_indices.min(),
                    x_indices.max(), y_indices.max()
                ]
            else:
                sam2_tracked_boxes[out_frame_idx] = None

    del predictor
    del inference_state
    torch.cuda.empty_cache()

    pose_model = YOLO('yolo11s-pose.pt')
    out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    video_keypoints = []

    for idx in range(frame_idx):
        img_path = os.path.join(temp_frame_dir, f"{idx:05d}.jpg")
        frame = cv2.imread(img_path)

        sam_box = sam2_tracked_boxes.get(idx, None)
        best_kpts = None

        if sam_box is not None:
            results = pose_model.predict(frame, conf=0.15, verbose=False, classes=[0], device=0, half=True)
            best_iou = 0
            best_match_idx = -1

            if len(results) > 0 and len(results[0].boxes) > 0:
                yolo_boxes = results[0].boxes.xyxy.cpu().numpy()
                for i, ybox in enumerate(yolo_boxes):
                    iou = calculate_iou(ybox, sam_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = i

            if best_iou > 0.1 and hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                best_kpts = results[0].keypoints.data[best_match_idx].cpu().numpy()
                video_keypoints.append(best_kpts)

                x1, y1, x2, y2 = map(int, sam_box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                for kpt in best_kpts:
                    px, py, p_conf = kpt
                    if p_conf > 0.4:
                        cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 0), -1)
            else:
                video_keypoints.append(np.zeros((17, 3)))
                x1, y1, x2, y2 = map(int, sam_box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        else:
            video_keypoints.append(np.zeros((17, 3)))

        out.write(frame)

    out.release()

    shutil.rmtree(temp_frame_dir, ignore_errors=True)
    try:
        np.save(npy_output_path, np.array(video_keypoints))
    except Exception:
        pass

    try:
        if os.path.exists(output_path):
            os.remove(output_path)
        original_clip = VideoFileClip(input_path)
        processed_clip = VideoFileClip(temp_output_path)
        if original_clip.audio:
            final_clip = processed_clip.set_audio(original_clip.audio)
            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', logger=None)
        else:
            processed_clip.close()
            original_clip.close()
            shutil.copy2(temp_output_path, output_path)
        original_clip.close()
        processed_clip.close()
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
    except Exception:
        pass

    print(f"[DONE] {filename}: skeleton data extracted.")


if __name__ == '__main__':
    multiprocessing.freeze_support()

    all_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
    try:
        all_files.sort(key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else x)
    except Exception:
        pass

    pending_files = [f for f in all_files if f not in SKIP_FILES]
    if not pending_files:
        print("No pending files to process.")
        sys.exit()

    tasks = phase_one_selection(pending_files)
    if not tasks:
        sys.exit()

    print(f"\n{'='*50}")
    print("Phase 2: SAM2 + YOLO-pose running in parallel")
    print(f"Workers: {MAX_WORKERS} (watch GPU memory)")
    print(f"{'='*50}\n")

    lock = multiprocessing.Lock()
    with multiprocessing.Pool(processes=MAX_WORKERS, initializer=init_worker, initargs=(lock,)) as pool:
        pool.map(process_video_worker, tasks)

    print("-" * 30)
    print("All videos processed: skeleton extraction complete.")
