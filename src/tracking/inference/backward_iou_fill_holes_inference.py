"""
Backward IoU linking followed by linear filling of confident internal gaps.

GT-free version of backward_iou/postprocess_fill_holes.py. See this folder's README for differences.
Set IMAGE_ROOT, CHECKPOINT_PATH, RESULTS_DIR, and RESOLUTION for your model.
Set DETECTION_MIN_CONF for detection; OUTPUT_MIN_CONF
filters saved detections after tracking. Edit tracking constants below to tune
association, memory, gap filling, score boosts, or cleanup. No COCO is read.

Each image directory is one naturally sorted sequence. Empty detection frames
remain in the timeline; missing files are not invented. Track IDs are local to
sequence and class. Output: per-sequence annotated MP4 and CSV, plus tracks.json
with all frames, boxes, scores, IDs, and interpolation flags. No accuracy metrics.
Untracked detections have null/blank IDs. Existing result directories are refused.
Sources are unchanged and import does not execute inference. Requires the adjacent
_shared.py and src/model/inference.py. Uses raw RGB inputs with no ASM.
"""
from collections import defaultdict
from pathlib import Path
import copy
import sys
import numpy as np
from _shared import load_detections, export_results, link_complete_frames

IMAGE_ROOT = Path(r"D:\lensfree_timeseries")
CHECKPOINT_PATH = Path(r"D:\cw_rf_detr_dataset_raw\output\RAW_20x20\checkpoint_best_ema.pth")
RESULTS_DIR = Path(r"D:\cw_rf_detr_dataset_raw\tracking_inference\backward_iou_fill_holes")
RESOLUTION = 512
DETECTION_MIN_CONF = 0.4
OUTPUT_MIN_CONF = 0.4
VIDEO_FPS = 5
TRACK_IOU_THR = 0.2
TRACK_MAX_AGE = 60
TRACK_MIN_CONF = 0.4
MIN_TRACK_LEN_FOR_BONUS = 2
SCORE_BONUS_PER_FRAME = 0.0
MAX_SCORE = 1.0
MAX_HOLE_LEN = 10
INTERP_SCORE_PENALTY = 0.0
HOLE_MIN_EDGE_CONF = 0.6
TRACK_MERGE_DIST_PX = 0


def iou_xyxy(a, b):
    (ax1, ay1, ax2, ay2) = a
    (bx1, by1, bx2, by2) = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-09
    return inter / union

def box_center_xy(b):
    (x1, y1, x2, y2) = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def track(pred_store, seq_to_img_ids, CLASSES, cfg=None):
    """Apply the original association/postprocessing to supplied detections."""
    all_tracks_by_seq_cls = defaultdict(lambda: defaultdict(list))  # seq -> cls -> [track_dict]

    for seq_name, seq_img_ids in seq_to_img_ids.items():
        if not seq_img_ids:
            continue

        frame_idx_to_img_id = {fi: img_id for fi, img_id in enumerate(seq_img_ids)}
        img_id_to_frame_idx = {img_id: fi for fi, img_id in frame_idx_to_img_id.items()}

        tracks_by_cls = {cls: [] for cls in CLASSES}
        next_track_id_by_cls = {cls: 1 for cls in CLASSES}

        for cls in CLASSES:
            for fi in reversed(range(len(seq_img_ids))):
                img_id = frame_idx_to_img_id[fi]
                dets = pred_store[img_id][cls]

                cand_dets = [d for d in dets if d["base_score"] >= TRACK_MIN_CONF]

                for det in cand_dets:
                    box = det["box"]
                    best_trk = None
                    best_iou = 0.0
                    for trk in tracks_by_cls[cls]:
                        frame_gap = trk["last_frame"] - fi  # since we're going backwards
                        if frame_gap <= 0 or frame_gap > TRACK_MAX_AGE:
                            continue
                        iou = iou_xyxy(box, trk["last_box"])
                        if iou > best_iou:
                            best_iou = iou
                            best_trk = trk

                    if best_trk is not None and best_iou >= TRACK_IOU_THR:
                        det["track_id"] = best_trk["id"]
                        best_trk["detections"].append((fi, img_id, det))
                        best_trk["last_box"] = box
                        best_trk["last_frame"] = fi
                    else:
                        new_id = next_track_id_by_cls[cls]
                        next_track_id_by_cls[cls] += 1
                        det["track_id"] = new_id
                        trk = {
                            "id": new_id,
                            "cls": cls,
                            "detections": [(fi, img_id, det)],
                            "last_box": box,
                            "last_frame": fi,
                        }
                        tracks_by_cls[cls].append(trk)

        for cls in CLASSES:
            all_tracks_by_seq_cls[seq_name][cls] = tracks_by_cls[cls]


    for seq_name, cls_tracks in all_tracks_by_seq_cls.items():
        for cls, tracks in cls_tracks.items():
            for trk in tracks:
                track_len = len(trk["detections"])
                if track_len >= MIN_TRACK_LEN_FOR_BONUS:
                    bonus = SCORE_BONUS_PER_FRAME * (track_len - 1)
                    if bonus < 0.0:
                        bonus = 0.0
                else:
                    bonus = 0.0

                for (fi, img_id, det) in trk["detections"]:
                    new_score = min(MAX_SCORE, det["score"] + bonus)
                    det["score"] = new_score


    interpolated_count = 0

    for seq_name, cls_tracks in all_tracks_by_seq_cls.items():
        seq_img_ids = seq_to_img_ids[seq_name]
        if not seq_img_ids:
            continue
        frame_idx_to_img_id = {fi: img_id for fi, img_id in enumerate(seq_img_ids)}

        for cls, tracks in cls_tracks.items():
            for trk in tracks:
                det_list = sorted(trk["detections"], key=lambda x: x[0])
                n = len(det_list)
                if n < 2:
                    continue

                for idx_d in range(n - 1):
                    fi, img_id_i, det_i = det_list[idx_d]
                    fj, img_id_j, det_j = det_list[idx_d + 1]
                    gap = fj - fi - 1
                    if gap <= 0 or gap > MAX_HOLE_LEN:
                        continue

                    if det_i["score"] < HOLE_MIN_EDGE_CONF or det_j["score"] < HOLE_MIN_EDGE_CONF:
                        continue

                    box_i = det_i["box"]
                    box_j = det_j["box"]
                    score_i = det_i["score"]
                    score_j = det_j["score"]

                    for k in range(1, gap + 1):
                        alpha = k / (gap + 1)  # 0<alpha<1
                        new_box = (1 - alpha) * box_i + alpha * box_j
                        raw_score = (1 - alpha) * score_i + alpha * score_j

                        new_score = max(TRACK_MIN_CONF, raw_score - INTERP_SCORE_PENALTY)
                        new_score = min(MAX_SCORE, new_score)

                        mid_frame_idx = fi + k
                        if mid_frame_idx not in frame_idx_to_img_id:
                            continue
                        mid_img_id = frame_idx_to_img_id[mid_frame_idx]

                        new_det = {
                            "box": new_box.astype(np.float32),
                            "score": float(new_score),
                            "base_score": float(raw_score),
                            "track_id": trk["id"],
                            "interp": True,
                        }

                        pred_store[mid_img_id][cls].append(new_det)
                        trk["detections"].append((mid_frame_idx, mid_img_id, new_det))
                        interpolated_count += 1

    print(f"[DEBUG] Hole filling DONE: interpolated_detections={interpolated_count}")
    sys.stdout.flush()


    postproc_preds_total = 0
    for img_id, per_cls in pred_store.items():
        for cls in CLASSES:
            postproc_preds_total += len(per_cls[cls])

    print(
        f"[DEBUG] Tracking + score boosting + hole filling DONE: "
        f"postproc_preds_total={postproc_preds_total}"
    )
    sys.stdout.flush()

    return pred_store

def main():
    if RESULTS_DIR.exists():
        raise FileExistsError(f'Choose a new RESULTS_DIR: {RESULTS_DIR}')
    store, sequences, images, classes, names = load_detections(
        IMAGE_ROOT, CHECKPOINT_PATH, RESOLUTION, DETECTION_MIN_CONF)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in globals().items()
                if k.isupper()}
    processed = track(store, sequences, classes)
    export_results(processed, sequences, images, names, IMAGE_ROOT, RESULTS_DIR, settings)


if __name__ == "__main__":
    main()
