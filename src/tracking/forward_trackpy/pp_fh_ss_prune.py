"""
Run forward Trackpy tracking with gap filling and short-segment pruning.

Configure BASE_DIR, CHECKPOINT_NAME (folder containing checkpoint_best_ema.pth),
and GT_ANNOTATION (filename without .json). Override IMG_DIR, ANN_FILE, and
checkpoint for other layouts; set RESULTS_DIR for reports and videos.
CONF_THR_LIST sets evaluation cutoffs AND its first value filters inference.
TRACK_MIN_CONF filters the surviving detections used for linking.
IOU_THR and CENTROID_TOL_PX control evaluation; VIDEO_FPS controls playback.

Use prepared RGB images and a COCO file listing sequence-relative filenames,
classes, boxes, and track IDs for identity evaluation. Sequences are grouped
by the first filename directory, ordered by frame_id then filename. Frame
counts use positions in that list. Ignore dummy and crowd annotations.
Ground-truth identities are not used to build predicted tracks.

Link box centres forwards per sequence and class using Trackpy's
NearestVelocityPredict, then optionally boost scores and interpolate internal
gaps in linked tracks. Interpolate box coordinates and endpoint scores only
when both endpoint scores meet HOLE_MIN_EDGE_CONF; never extrapolate ends.

Set TRACKPY_SEARCH_RANGE (pixels around predicted positions), TRACKPY_MEMORY
(linking memory), MAX_HOLE_LEN (missing positions to fill), HOLE_MIN_EDGE_CONF,
and INTERP_SCORE_PENALTY. TRACK_IOU_THR and TRACK_MAX_AGE are legacy logging
settings, not active association controls. MIN_TRACK_LEN_FOR_BONUS,
SCORE_BONUS_PER_FRAME, and MAX_SCORE control optional score boosts/capping.

Defaults: search range 30, memory 10, hole limit 5, endpoint confidence 0.6,
and no score bonus. Edit the outer TRACK_MERGE_DIST_PX list for nearby-ID
continuation cleanup; 0 permits only exact-position matches. Each value
reruns the full pipeline.

After gap filling and ID merging, remove contiguous tracked segments shorter
than MIN_FINAL_SEGMENT_LEN (defined near the pruning code; default 2).
Retain only the highest-score detection per sequence/class/track ID/frame
in the prediction store. Pruning can remove real brief appearances as well
as false positives. Interpolated detections contribute to segment length.
UPSCALE_VIDEO and UPSCALED_SIZE change video output size only.
Report detection F1/AP, custom MOTA/MOTP, approximate IDF1, and track lengths.

Write text/JSON metrics and annotated MP4 videos to RESULTS_DIR, overwriting
matching outputs. Individual detections and tracks are not exported. No
training or reconstruction occurs; importing this file starts execution.
See ../README.md for script selection, shared limitations, and tradeoffs.
"""

import os
import json
from collections import defaultdict
from PIL import Image
import numpy as np
from rfdetr import RFDETRMedium
import sys
import traceback
import cv2  # for drawing and video writing
import pandas as pd
import trackpy as tp
from trackpy.linking.utils import SubnetOversizeException
import trackpy.predict as tpp

# =========================
# --- CONFIG -------------
# =========================
for TRACK_MERGE_DIST_PX in [0]:
    CHECKPOINT_NAME = "ASM_perf_gt20"  #  "ASM_old_gt"
    GT_ANNOTATION = "_perfect_gt_annotations_20x20.coco"

    BASE_DIR = "D:/rf_detr_dataset"
    checkpoint = f"{BASE_DIR}/output/{CHECKPOINT_NAME}/checkpoint_best_ema.pth"
    IMG_DIR = f"{BASE_DIR}/valid"  # folder with the images
    ANN_FILE = f"{BASE_DIR}/valid/{GT_ANNOTATION}.json"  # COCO-style GT

    # where to save results
    RESULTS_DIR = os.path.join(
        BASE_DIR,
        "results_rfdetr_eval",
        "trackpy_fill_holes_velocity_static_switch",
        "prune_short_tracks",
        CHECKPOINT_NAME,
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f" checkpoint: {checkpoint}")
    print(f" ANN_FILE: {ANN_FILE}")
    print(f" IMG_DIR: {IMG_DIR}")
    print(f" RESULTS_DIR: {RESULTS_DIR}")
    print()

    # evaluation hyperparams
    CONF_THR_LIST = [0.4]
    IOU_THR = 0.5           # IoU threshold for detection eval and tracking metrics
    CENTROID_TOL_PX = 10    # centroid matching tolerance in pixels

    # tracking hyperparams (used for filtering + metrics; linking done by trackpy)
    TRACK_IOU_THR = 0.2     # (no longer used for association, kept for logging)
    TRACK_MAX_AGE = 30      # used as a "memory"-like scale for trackpy
    TRACK_MIN_CONF = 0.4    # min detection score used for building tracks

    MIN_TRACK_LEN_FOR_BONUS = 2   # track length from which we start boosting scores
    SCORE_BONUS_PER_FRAME = 0.0   # how much score bonus per extra frame in the track
    MAX_SCORE = 1.0               # cap on boosted scores

    # trackpy-specific hyperparams
    TRACKPY_SEARCH_RANGE = 30  # pixels: max displacement between frames
    TRACKPY_MEMORY = 10        # frames: how many frames a particle can vanish

    # hole filling hyperparams
    MAX_HOLE_LEN = 5                # maximum number of consecutive missing frames to fill
    INTERP_SCORE_PENALTY = 0.0      # score penalty applied to interpolated detections
    # minimum confidence required for BOTH ends of a hole to be interpolated
    HOLE_MIN_EDGE_CONF = 0.6

    # video config
    VIDEO_FPS = 5
        # upscaling config
    UPSCALE_VIDEO = True          # set False to keep original size
    UPSCALED_SIZE = (2048, 2048)  # (width, height)

    # =====================================================
    # --- helpers -----------------------------------------
    # =====================================================

    def xywh_to_xyxy(b):
        x, y, w, h = b
        return np.array([x, y, x + w, y + h], dtype=np.float32)

    def remove_det_from_pred_store(pred_store, img_id, cls, det):
        """Remove a detection dict from pred_store by object identity, not equality."""
        det_list = pred_store[img_id][cls]
        for i, d in enumerate(det_list):
            if d is det:        # identity check, no NumPy equality involved
                del det_list[i]
                return True
        return False

    def iou_xyxy(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        iw = max(0.0, inter_x2 - inter_x1)
        ih = max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = area_a + area_b - inter + 1e-9
        return inter / union

    # centroid helpers
    def box_center_xy(b):
        x1, y1, x2, y2 = b
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def center_dist_px(a, b):
        ax, ay = box_center_xy(a)
        bx, by = box_center_xy(b)
        dx = ax - bx
        dy = ay - by
        return float(np.hypot(dx, dy))

    # --- AP/mAP helpers ---
    def compute_ap_from_pr(recalls, precisions):
        """
        Compute AP using the VOC2010-style integral of the precision envelope.
        recalls, precisions: 1D numpy arrays of same length.
        Returns a scalar AP in [0,1].
        """
        if len(recalls) == 0:
            return 0.0

        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        # Make precision a monotonically non-increasing envelope from right to left
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        # Compute area under curve at points where recall changes
        idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
        ap = float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))
        return ap

    def compute_ap_for_class(pred_store, gt_by_img_cls, class_name, iou_thr):
        """
        Flatten detections for a given class across the dataset,
        sort by score, and perform 1-to-1 greedy matching against
        ground-truths within each image.
        """
        # Collect all detections of this class: list of (img_id, score, box)
        dets = []
        for img_id, per_cls in pred_store.items():
            for det in per_cls.get(class_name, []):
                dets.append(
                    (img_id, float(det["score"]), np.asarray(det["box"], dtype=np.float32))
                )

        if len(dets) == 0:
            return 0.0, np.array([]), np.array([])

        # Sort detections by confidence descending
        dets.sort(key=lambda x: x[1], reverse=True)

        # Count total positives (GT boxes) for this class
        npos = 0
        for img_id, cls_map in gt_by_img_cls.items():
            npos += len(cls_map.get(class_name, []))

        if npos == 0:
            # No ground-truths for this class: define AP=0
            return 0.0, np.array([]), np.array([])

        # For each image, track which GT boxes have been matched already
        matched_gt = {
            img_id: np.zeros(len(cls_map.get(class_name, [])), dtype=bool)
            for img_id, cls_map in gt_by_img_cls.items()
        }

        tp = np.zeros(len(dets), dtype=np.float32)
        fp = np.zeros(len(dets), dtype=np.float32)

        for i, (img_id, score, dbox) in enumerate(dets):
            gts = gt_by_img_cls.get(img_id, {}).get(class_name, [])
            if len(gts) == 0:
                fp[i] = 1.0
                continue

            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if matched_gt[img_id][j]:
                    continue
                iou = iou_xyxy(dbox, g)
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou >= iou_thr and best_j != -1:
                matched_gt[img_id][best_j] = True
                tp[i] = 1.0
            else:
                fp[i] = 1.0

        # Precision-Recall curve
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / float(npos)
        precisions = np.divide(cum_tp, (cum_tp + cum_fp + 1e-9))

        ap = compute_ap_from_pr(recalls, precisions)
        return ap, recalls, precisions

    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    # =====================================================
    # --- load GT COCO ------------------------------------
    # =====================================================
    with open(ANN_FILE, "r") as f:
        coco = json.load(f)

    # maps
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    name_to_cat_id = {v: k for k, v in cat_id_to_name.items()}
    imgid_to_info = {im["id"]: im for im in coco["images"]}

    # group images into sequences based on directory name in file_name
    # e.g. "20230323_livedead_A549xy3_0_0/cycle_0001.png" -> "20230323_livedead_A549xy3_0_0"
    seq_to_img_ids = defaultdict(list)
    imgid_to_seq_name = {}

    for img_id, im in imgid_to_info.items():
        fname = im["file_name"]
        seq_name = fname.split("/")[0] if "/" in fname else fname
        seq_to_img_ids[seq_name].append(img_id)
        imgid_to_seq_name[img_id] = seq_name

    # sort frames within each sequence by frame_id (if available) then by file_name
    for seq_name, img_ids in seq_to_img_ids.items():
        seq_to_img_ids[seq_name] = sorted(
            img_ids,
            key=lambda i: (
                imgid_to_info[i].get("frame_id", 0),
                imgid_to_info[i]["file_name"],
            ),
        )

    # derive CLASSES from categories actually used in annotations (excluding "dummy")
    IGNORE_CLASSES = {"dummy"}
    used_cat_ids = set(
        ann["category_id"]
        for ann in coco["annotations"]
        if ann.get("iscrowd", 0) == 0
    )
    CLASSES = sorted(
        cat_id_to_name[cat_id]
        for cat_id in used_cat_ids
        if cat_id_to_name[cat_id] not in IGNORE_CLASSES
    )

    print("[DEBUG] Categories in annotations:")
    print(f" cat_id_to_name: {cat_id_to_name}")
    print(f" name_to_cat_id: {name_to_cat_id}")
    print(f" #images in GT: {len(imgid_to_info)}")
    print(f"[DEBUG] Using CLASSES from COCO (ignoring {IGNORE_CLASSES}): {CLASSES}")
    print(f" CONF_THR_LIST: {CONF_THR_LIST}")
    print(f" IOU_THR: {IOU_THR}")
    print(f" CENTROID_TOL_PX: {CENTROID_TOL_PX}")
    print(f" TRACK_IOU_THR: {TRACK_IOU_THR}")
    print(f" TRACK_MAX_AGE: {TRACK_MAX_AGE}")
    print(f" TRACK_MIN_CONF: {TRACK_MIN_CONF}")
    print(f" MIN_TRACK_LEN_FOR_BONUS: {MIN_TRACK_LEN_FOR_BONUS}")
    print(f" SCORE_BONUS_PER_FRAME: {SCORE_BONUS_PER_FRAME}")
    print(f" MAX_SCORE: {MAX_SCORE}")
    print(f" MAX_HOLE_LEN: {MAX_HOLE_LEN}")
    print(f" INTERP_SCORE_PENALTY: {INTERP_SCORE_PENALTY}")
    print(f" HOLE_MIN_EDGE_CONF: {HOLE_MIN_EDGE_CONF}")
    print(f" TRACKPY_SEARCH_RANGE: {TRACKPY_SEARCH_RANGE}")
    print(f" TRACKPY_MEMORY: {TRACKPY_MEMORY}")
    sys.stdout.flush()

    # build GT per image per class (xyxy boxes) and GT tracks (for tracking metrics)
    gt_by_img_cls = defaultdict(lambda: defaultdict(list))       # img_id -> cls -> [box]
    gt_tracks_by_img_cls = defaultdict(lambda: defaultdict(list))  # img_id -> cls -> [ {box, track_id} ]

    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        cls_name = cat_id_to_name[cat_id]
        if cls_name not in CLASSES:
            continue

        box_xyxy = xywh_to_xyxy(ann["bbox"])
        gt_by_img_cls[img_id][cls_name].append(box_xyxy)

        track_id = ann.get("track_id", None)
        gt_tracks_by_img_cls[img_id][cls_name].append(
            {"box": box_xyxy, "track_id": track_id}
        )

    # --- GT track length stats (group by seq + cls + track_id) ---
    gt_track_lengths = []
    gt_track_lengths_by_cls = {cls: [] for cls in CLASSES}
    gt_tracks_seen = set()  # to count unique tracks

    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        cls_name = cat_id_to_name[cat_id]
        if cls_name not in CLASSES:
            continue
        tid = ann.get("track_id", None)
        if tid is None:
            continue
        seq_name = imgid_to_seq_name[img_id]
        key = (seq_name, cls_name, tid)
        gt_tracks_seen.add(key)

    # count length per track
    gt_track_count = defaultdict(int)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        cls_name = cat_id_to_name[cat_id]
        if cls_name not in CLASSES:
            continue
        tid = ann.get("track_id", None)
        if tid is None:
            continue
        seq_name = imgid_to_seq_name[img_id]
        key = (seq_name, cls_name, tid)
        gt_track_count[key] += 1

    for key, length in gt_track_count.items():
        _, cls_name, _ = key
        gt_track_lengths.append(length)
        gt_track_lengths_by_cls[cls_name].append(length)

    total_gt_dets_with_id = sum(gt_track_count.values())
    total_gt_tracks = len(gt_track_count)

    # =====================================================
    # --- model (RFDETR native) ---------------------------
    # =====================================================
    model = RFDETRMedium(pretrain_weights=checkpoint)
    model_class_names = getattr(model, "class_names", None)

    print("[DEBUG] Model loaded.")
    print(f" Has class_names: {model_class_names is not None}")
    if model_class_names is not None:
        print(f" model.class_names (len={len(model_class_names)}): {model_class_names}")
    sys.stdout.flush()

    # =====================================================
    # --- inference: collect raw detections ---------------
    # =====================================================
    # store raw predictions per image (no thresholding yet)
    # pred_store[img_id][cls] -> list of det dict:
    # {"box": np.array(4,), "score": float, "base_score": float, "track_id": None, "interp": bool}
    pred_store = defaultdict(lambda: defaultdict(list))

    images_processed = 0
    raw_preds_total = 0
    preds_dropped_by_conf = 0

    for img_id, im_meta in imgid_to_info.items():
        try:
            img_path = os.path.join(IMG_DIR, im_meta["file_name"])
            if not os.path.exists(img_path):
                print(f"[WARN] Missing image: {img_path}")
                sys.stdout.flush()
                continue

            image = Image.open(img_path).convert("RGB")

            # GT summary for this image
            gt_counts = {
                c: len(gt_by_img_cls[img_id].get(c, []))
                for c in CLASSES
            }

            if images_processed < 3:
                print(
                    f"[DEBUG] Image {images_processed+1}: "
                    f"id={img_id} path={img_path}"
                )
                print(f" GT counts: {gt_counts}")
                sys.stdout.flush()

            # run inference (use lowest conf thr just to prune complete garbage)
            preds = model.predict(image, threshold=CONF_THR_LIST[0])
            pred0 = preds[0] if isinstance(preds, (list, tuple)) else preds

            # Inspect raw prediction structure
            if images_processed < 3:
                print(f" pred type: {type(preds)} -> first elem type: {type(pred0)}")
                if isinstance(pred0, dict):
                    print(f" pred0 keys: {list(pred0.keys())}")
                elif hasattr(pred0, "predictions"):
                    print(" pred0 has attribute .predictions")
                sys.stdout.flush()

            def label_to_name(lbl):
                if isinstance(lbl, str):
                    return lbl
                if isinstance(lbl, (int, np.integer)):
                    if (
                        hasattr(model, "class_names")
                        and model.class_names is not None
                        and 0 <= lbl < len(model.class_names)
                    ):
                        return model.class_names[lbl]
                    if lbl in cat_id_to_name:
                        return cat_id_to_name[lbl]
                return None

            used_branch = None

            # Branch A: dict with 'scores', 'labels', 'boxes'
            if isinstance(pred0, dict) and all(
                k in pred0 for k in ("scores", "labels", "boxes")
            ):
                used_branch = "dict"
                scores = np.array(pred0["scores"]).astype(float)
                labels = pred0["labels"]
                boxes = np.array(pred0["boxes"], dtype=np.float32)
                raw_preds_total += len(scores)

                for s, lbl, box in zip(scores, labels, boxes):
                    cls = label_to_name(lbl)
                    if cls in CLASSES:
                        # ensure xyxy ordering if needed
                        if box[2] < box[0] or box[3] < box[1]:
                            box = xywh_to_xyxy(box)
                        det = {
                            "box": box.astype(np.float32),
                            "score": float(s),
                            "base_score": float(s),
                            "track_id": None,
                            "interp": False,
                        }
                        pred_store[img_id][cls].append(det)

            # Branch B: supervision.Detections style
            elif hasattr(pred0, "xyxy") and hasattr(pred0, "class_id"):
                used_branch = "supervision.Detections"
                boxes = np.array(pred0.xyxy, dtype=np.float32)
                class_ids = np.array(getattr(pred0, "class_id"))
                if class_ids is None:
                    class_ids = np.zeros((len(boxes),), dtype=int)
                scores = getattr(pred0, "confidence", None)
                if scores is None:
                    scores = np.ones((len(boxes),), dtype=float)
                else:
                    scores = np.array(scores, dtype=float)
                raw_preds_total += len(scores)

                def id_to_name(cid):
                    cid = int(cid)
                    if isinstance(model_class_names, dict):
                        name = model_class_names.get(cid)
                        if name is not None:
                            return name
                    if isinstance(model_class_names, (list, tuple)) and 0 <= cid < len(
                        model_class_names
                    ):
                        return model_class_names[cid]
                    if cid in cat_id_to_name:
                        return cat_id_to_name[cid]
                    if 0 <= cid < len(CLASSES):
                        return CLASSES[cid]
                    return None

                for s, cid, box in zip(scores, class_ids, boxes):
                    cls = id_to_name(cid)
                    if cls in CLASSES:
                        det = {
                            "box": box.astype(np.float32),
                            "score": float(s),
                            "base_score": float(s),
                            "track_id": None,
                            "interp": False,
                        }
                        pred_store[img_id][cls].append(det)

            # Branch C: object with .predictions
            elif hasattr(pred0, "predictions"):
                used_branch = "obj.predictions"
                plist = list(getattr(pred0, "predictions"))
                raw_preds_total += len(plist)

                for p in plist:
                    cls = label_to_name(
                        getattr(p, "class_name", getattr(p, "label", None))
                    )
                    conf = float(
                        getattr(p, "confidence", getattr(p, "score", 1.0))
                    )
                    if cls in CLASSES:
                        if (
                            hasattr(p, "x")
                            and hasattr(p, "y")
                            and hasattr(p, "width")
                            and hasattr(p, "height")
                        ):
                            cx, cy, w, h = p.x, p.y, p.width, p.height
                            x1 = cx - w / 2
                            y1 = cy - h / 2
                            x2 = cx + w / 2
                            y2 = cy + h / 2
                            box = np.array(
                                [x1, y1, x2, y2], dtype=np.float32
                            )
                        elif hasattr(p, "bbox"):
                            box = xywh_to_xyxy(p.bbox)
                        else:
                            continue

                        det = {
                            "box": box.astype(np.float32),
                            "score": float(conf),
                            "base_score": float(conf),
                            "track_id": None,
                            "interp": False,
                        }
                        pred_store[img_id][cls].append(det)
                    else:
                        preds_dropped_by_conf += 1
            else:
                used_branch = "unrecognized"

            if images_processed < 3:
                print(f" used_branch: {used_branch}")
                if used_branch == "supervision.Detections":
                    cid_raw = getattr(pred0, "class_id", None)
                    if cid_raw is not None and len(cid_raw) > 0:
                        cid_preview = list(
                            map(int, np.array(cid_raw)[:2])
                        )
                    else:
                        cid_preview = []
                    print(f" raw class_id preview: {cid_preview}")
                # quick preview of per-class counts
                debug_counts = {
                    c: len(pred_store[img_id][c])
                    for c in CLASSES
                }
                print(f" pred counts by class: {debug_counts}")
                for c in CLASSES:
                    if pred_store[img_id][c]:
                        d0 = pred_store[img_id][c][0]
                        print(
                            f" sample {c} pred[0]: "
                            f"box={d0['box']}, score={d0['score']:.3f}"
                        )
                sys.stdout.flush()

            images_processed += 1
            if images_processed % 100 == 0:
                print(
                    f"[DEBUG] Processed {images_processed} images. "
                    f"raw_preds_total={raw_preds_total}, "
                    f"dropped_by_conf={preds_dropped_by_conf}"
                )
                sys.stdout.flush()

            image.close()

        except Exception as e:
            print(
                f"[ERROR] Exception on image id={img_id} "
                f"path={im_meta.get('file_name','?')}: {e}"
            )
            traceback.print_exc()
            sys.stdout.flush()
            continue

    print(
        f"[DEBUG] Inference DONE: images_processed={images_processed}, "
        f"raw_preds_total={raw_preds_total}, "
        f"dropped_by_conf={preds_dropped_by_conf}"
    )
    sys.stdout.flush()

    # =====================================================
    # --- tracking with trackpy (per sequence & class) ----
    # =====================================================
    # all_tracks_by_seq_cls[seq][cls] -> list of track dicts:
    # {"id": int, "cls": str, "detections": [(frame_idx, img_id, det_dict)],
    #  "last_box": np.array, "last_frame": int}
    all_tracks_by_seq_cls = defaultdict(lambda: defaultdict(list))

    # ------ Velocity based
    for seq_name, seq_img_ids in seq_to_img_ids.items():
        if not seq_img_ids:
            continue

        frame_idx_to_img_id = {fi: img_id for fi, img_id in enumerate(seq_img_ids)}
        img_id_to_frame_idx = {img_id: fi for fi, img_id in frame_idx_to_img_id.items()}

        for cls in CLASSES:
            rows = []
            for img_id in seq_img_ids:
                fi = img_id_to_frame_idx[img_id]
                dets = pred_store[img_id][cls]
                for det_idx, det in enumerate(dets):
                    if det["base_score"] < TRACK_MIN_CONF:
                        continue
                    cx, cy = box_center_xy(det["box"])
                    rows.append(
                        {
                            "x": cx,
                            "y": cy,
                            "frame": fi,
                            "img_id": img_id,
                            "det_idx": det_idx,
                        }
                    )

            if not rows:
                all_tracks_by_seq_cls[seq_name][cls] = []
                continue

            df = pd.DataFrame(rows)

            # --- use a dynamic velocity predictor ---
            pred = tpp.NearestVelocityPredict()  # learns velocities as it links
            try:
                linked = pred.link_df(
                    df,
                    search_range=TRACKPY_SEARCH_RANGE,
                    memory=TRACKPY_MEMORY,
                    adaptive_stop=1,
                    adaptive_step=0.95,
                )
            except SubnetOversizeException:
                smaller_range = max(5, int(TRACKPY_SEARCH_RANGE / 2))
                print(
                    f"[WARN] SubnetOversizeException in seq='{seq_name}', "
                    f"cls='{cls}'. Retrying with smaller search_range={smaller_range}."
                )
                linked = pred.link_df(
                    df,
                    search_range=smaller_range,
                    memory=TRACKPY_MEMORY,
                    adaptive_stop=1,
                    adaptive_step=0.95,
                )

            linked = linked.dropna(subset=["particle"])
            if linked.empty:
                all_tracks_by_seq_cls[seq_name][cls] = []
                continue

            tracks_dict = defaultdict(list)
            for _, row in linked.iterrows():
                pid = int(row["particle"])
                img_id = int(row["img_id"])
                det_idx = int(row["det_idx"])
                fi = int(row["frame"])
                det = pred_store[img_id][cls][det_idx]
                det["track_id"] = pid
                tracks_dict[pid].append((fi, img_id, det))

            tracks = []
            for pid, det_list in tracks_dict.items():
                det_list_sorted = sorted(det_list, key=lambda x: x[0])
                last_frame = det_list_sorted[-1][0]
                last_box = det_list_sorted[-1][2]["box"]
                trk = {
                    "id": pid,
                    "cls": cls,
                    "detections": det_list_sorted,
                    "last_box": last_box,
                    "last_frame": last_frame,
                }
                tracks.append(trk)

            all_tracks_by_seq_cls[seq_name][cls] = tracks

    print("[DEBUG] Trackpy tracking DONE.")
    sys.stdout.flush()

    # =====================================================
    # --- score boosting from tracks ----------------------
    # =====================================================
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

    # =====================================================
    # --- hole filling (temporal interpolation) -----------
    # =====================================================
    interpolated_count = 0

    for seq_name, cls_tracks in all_tracks_by_seq_cls.items():
        seq_img_ids = seq_to_img_ids[seq_name]
        if not seq_img_ids:
            continue

        frame_idx_to_img_id = {fi: img_id for fi, img_id in enumerate(seq_img_ids)}

        for cls, tracks in cls_tracks.items():
            for trk in tracks:
                # sort detections along this track by frame index (ascending)
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

                    # only fill this hole if both ends are high-confidence
                    if (
                        det_i["score"] < HOLE_MIN_EDGE_CONF
                        or det_j["score"] < HOLE_MIN_EDGE_CONF
                    ):
                        continue

                    box_i = det_i["box"]
                    box_j = det_j["box"]
                    score_i = det_i["score"]
                    score_j = det_j["score"]

                    # linearly interpolate boxes and scores for missing frames
                    for k in range(1, gap + 1):
                        alpha = k / (gap + 1)  # 0<alpha<1
                        new_box = (1 - alpha) * box_i + alpha * box_j
                        raw_score = (1 - alpha) * score_i + alpha * score_j

                        # clamp with TRACK_MIN_CONF & penalty if you like
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

    # =====================================================
    # --- track ID switch cleanup (merge near-static IDs) -
    # =====================================================

    def merge_track_ids_near_static(
        all_tracks_by_seq_cls, seq_to_img_ids, merge_dist_px=TRACK_MERGE_DIST_PX
    ):
        """
        Heuristic cleanup to reduce ID switches when cells barely move.

        Strategy (per seq & class):
        - Build frame -> detections mapping (using existing track IDs).
        - For each detection in frame f from an "old" track A (present before f),
          find the closest detection in frame f+1 within merge_dist_px.
        - If that neighbour at frame f+1 belongs to a "new" track B whose first
          appearance is exactly at f+1, treat B as a continuation of A and merge
          B into A (rename all B's detections to track_id A).
        """
        merge_count = 0

        for seq_name, cls_tracks in all_tracks_by_seq_cls.items():
            seq_img_ids = seq_to_img_ids[seq_name]
            if not seq_img_ids:
                continue

            frame_idx_to_img_id = {fi: img_id for fi, img_id in enumerate(seq_img_ids)}
            img_id_to_frame_idx = {img_id: fi for fi, img_id in frame_idx_to_img_id.items()}

            for cls, tracks in cls_tracks.items():
                if not tracks:
                    continue

                # --- build track -> frames and frame -> dets ---
                track_frames = defaultdict(set)
                frame_to_dets = defaultdict(list)

                for trk in tracks:
                    tid = trk["id"]
                    for fi, img_id, det in trk["detections"]:
                        track_frames[tid].add(fi)
                        frame_to_dets[fi].append(det)

                changed = True
                removed_track_ids = set()

                while changed:
                    changed = False

                    track_frames.clear()
                    frame_to_dets.clear()

                    for trk in tracks:
                        if trk["id"] in removed_track_ids:
                            continue
                        tid = trk["id"]
                        trk["detections"].sort(key=lambda x: x[0])  # ensure time-ordered
                        for fi, img_id, det in trk["detections"]:
                            track_frames[tid].add(fi)
                            frame_to_dets[fi].append(det)

                    num_frames = len(seq_img_ids)
                    for fi in range(num_frames - 1):
                        dets_f = frame_to_dets.get(fi, [])
                        dets_next = frame_to_dets.get(fi + 1, [])
                        if not dets_f or not dets_next:
                            continue

                        for det in dets_f:
                            pid = det.get("track_id", None)
                            if pid is None or pid in removed_track_ids:
                                continue

                            frames_pid = track_frames.get(pid, set())
                            if not frames_pid or min(frames_pid) >= fi:
                                continue

                            best_det2 = None
                            best_dist = float("inf")
                            for det2 in dets_next:
                                qid = det2.get("track_id", None)
                                if qid is None or qid == pid or qid in removed_track_ids:
                                    continue
                                d = center_dist_px(det["box"], det2["box"])
                                if d < best_dist:
                                    best_dist = d
                                    best_det2 = det2

                            if best_det2 is None or best_dist > merge_dist_px:
                                continue

                            qid = best_det2.get("track_id", None)
                            if qid is None or qid in removed_track_ids or qid == pid:
                                continue

                            frames_qid = track_frames.get(qid, set())
                            if not frames_qid:
                                continue

                            if min(frames_qid) != fi + 1:
                                continue

                            trk_pid = None
                            trk_qid = None
                            for trk in tracks:
                                if trk["id"] == pid:
                                    trk_pid = trk
                                elif trk["id"] == qid:
                                    trk_qid = trk

                            if trk_pid is None or trk_qid is None:
                                continue

                            for fi2, img2, det2 in trk_qid["detections"]:
                                det2["track_id"] = pid
                                trk_pid["detections"].append((fi2, img2, det2))

                            removed_track_ids.add(qid)
                            merge_count += 1
                            changed = True
                            break

                        if changed:
                            break

                if removed_track_ids:
                    new_tracks = []
                    for trk in tracks:
                        if trk["id"] in removed_track_ids:
                            continue
                        if trk["detections"]:
                            trk["detections"].sort(key=lambda x: x[0])
                            trk["last_frame"] = trk["detections"][-1][0]
                            trk["last_box"] = trk["detections"][-1][2]["box"]
                        new_tracks.append(trk)
                    cls_tracks[cls] = new_tracks

        return merge_count

    merged_pairs = merge_track_ids_near_static(
        all_tracks_by_seq_cls, seq_to_img_ids, merge_dist_px=TRACK_MERGE_DIST_PX
    )
    print(
        f"[DEBUG] Track ID switch cleanup DONE: merged_track_pairs={merged_pairs}"
    )
    sys.stdout.flush()

    # =====================================================
    # --- final cleanup: drop tiny track segments --------
    # =====================================================
    MIN_FINAL_SEGMENT_LEN = 2  # or 3 if you want to be stricter

    def prune_short_track_segments(
        all_tracks_by_seq_cls, pred_store, min_seg_len=MIN_FINAL_SEGMENT_LEN
    ):
        """
        For each track, split detections into contiguous segments by frame index.
        Any segment shorter than min_seg_len is removed from the track and from pred_store.
        Tracks that become empty are removed entirely.
        """
        removed_dets = 0
        removed_tracks = 0

        for seq_name, cls_tracks in all_tracks_by_seq_cls.items():
            for cls, tracks in list(cls_tracks.items()):
                new_tracks = []
                for trk in tracks:
                    dets_sorted = sorted(trk["detections"], key=lambda x: x[0])
                    if not dets_sorted:
                        removed_tracks += 1
                        continue

                    segments = []
                    current_seg = [dets_sorted[0]]
                    for prev, cur in zip(dets_sorted, dets_sorted[1:]):
                        prev_fi = prev[0]
                        cur_fi = cur[0]
                        if cur_fi == prev_fi + 1:
                            current_seg.append(cur)
                        else:
                            segments.append(current_seg)
                            current_seg = [cur]
                    segments.append(current_seg)

                    kept_dets = []
                    for seg in segments:
                        if len(seg) >= min_seg_len:
                            kept_dets.extend(seg)
                        else:
                            for fi, img_id, det in seg:
                                if remove_det_from_pred_store(pred_store, img_id, cls, det):
                                    removed_dets += 1

                    if kept_dets:
                        kept_dets.sort(key=lambda x: x[0])
                        trk["detections"] = kept_dets
                        trk["last_frame"] = kept_dets[-1][0]
                        trk["last_box"] = kept_dets[-1][2]["box"]
                        new_tracks.append(trk)
                    else:
                        removed_tracks += 1

                cls_tracks[cls] = new_tracks

        print(
            f"[DEBUG] prune_short_track_segments: "
            f"removed_detections={removed_dets}, removed_tracks={removed_tracks}"
        )
        sys.stdout.flush()

    prune_short_track_segments(all_tracks_by_seq_cls, pred_store, MIN_FINAL_SEGMENT_LEN)

    # =====================================================
    # --- final cleanup: enforce one ID per frame --------
    # =====================================================
    def enforce_unique_id_per_frame(all_tracks_by_seq_cls, pred_store):
        """
        For each sequence, class, and frame:
        If multiple detections share the same track_id, keep the highest-score one.
        """
        removed_dets = 0

        for seq_name, cls_tracks in all_tracks_by_seq_cls.items():
            for cls, tracks in cls_tracks.items():
                frame_id_map = defaultdict(lambda: defaultdict(list))
                for trk in tracks:
                    tid = trk["id"]
                    for fi, img_id, det in trk["detections"]:
                        tid_eff = det.get("track_id", tid)
                        frame_id_map[fi][tid_eff].append((img_id, det))

                for fi, tid_map in frame_id_map.items():
                    for tid_eff, det_list in tid_map.items():
                        if len(det_list) <= 1:
                            continue
                        det_list_sorted = sorted(
                            det_list, key=lambda x: x[1]["score"], reverse=True
                        )
                        keep_img, keep_det = det_list_sorted[0]
                        for img_id, det in det_list_sorted[1:]:
                            if remove_det_from_pred_store(pred_store, img_id, cls, det):
                                removed_dets += 1
                            det["track_id"] = None

        print(
            f"[DEBUG] enforce_unique_id_per_frame: "
            f"removed_detections={removed_dets}"
        )
        sys.stdout.flush()

    enforce_unique_id_per_frame(all_tracks_by_seq_cls, pred_store)

    # =====================================================
    # --- count detections after postproc ----------------
    # =====================================================
    postproc_preds_total = 0
    for img_id, per_cls in pred_store.items():
        for cls in CLASSES:
            postproc_preds_total += len(per_cls[cls])

    print(
        f"[DEBUG] Tracking + score boosting + hole filling DONE: "
        f"postproc_preds_total={postproc_preds_total}"
    )
    sys.stdout.flush()

    # =====================================================
    # --- evaluate detections across thresholds -----------
    # =====================================================
    results_by_thr = {}

    for THR in CONF_THR_LIST:
        TP_thr = {c: 0 for c in CLASSES}
        FP_thr = {c: 0 for c in CLASSES}
        FN_thr = {c: 0 for c in CLASSES}

        TPc_thr = {c: 0 for c in CLASSES}
        FPc_thr = {c: 0 for c in CLASSES}
        FNc_thr = {c: 0 for c in CLASSES}

        for img_id, im_meta in imgid_to_info.items():
            for cls in CLASSES:
                gts = [b.copy() for b in gt_by_img_cls[img_id].get(cls, [])]
                dets_scored = pred_store.get(img_id, {}).get(cls, [])
                dets = [d["box"].copy() for d in dets_scored if d["score"] >= THR]

                matched_gt = set()
                for d in dets:
                    best_iou, best_j = 0.0, -1
                    for j, g in enumerate(gts):
                        if j in matched_gt:
                            continue
                        iou = iou_xyxy(d, g)
                        if iou > best_iou:
                            best_iou, best_j = iou, j

                    if best_iou >= IOU_THR and best_j != -1:
                        TP_thr[cls] += 1
                        matched_gt.add(best_j)
                    else:
                        FP_thr[cls] += 1

                FN_thr[cls] += (len(gts) - len(matched_gt))

                matched_gt_centroid = set()
                for d in dets:
                    best_d, best_j = 1e9, -1
                    for j, g in enumerate(gts):
                        if j in matched_gt_centroid:
                            continue
                        dist = center_dist_px(d, g)
                        if dist < best_d:
                            best_d, best_j = dist, j

                    if best_d <= CENTROID_TOL_PX and best_j != -1:
                        TPc_thr[cls] += 1
                        matched_gt_centroid.add(best_j)
                    else:
                        FPc_thr[cls] += 1

                FNc_thr[cls] += (len(gts) - len(matched_gt_centroid))

        results_by_thr[THR] = (
            TP_thr,
            FP_thr,
            FN_thr,
            TPc_thr,
            FPc_thr,
            FNc_thr,
        )

    print(
        f"[DEBUG] Detection evaluation DONE: images_processed={images_processed}, "
        f"raw_preds_total={raw_preds_total}, "
        f"postproc_preds_total={postproc_preds_total}"
    )
    sys.stdout.flush()

    # =====================================================
    # --- tracking metrics (CLEAR MOT + global IDF1) ------
    # =====================================================

    def compute_clear_mot_metrics(pred_store, gt_tracks_by_img_cls, THR):
        """
        Compute CLEAR MOT-like metrics: MOTA, MOTP, ID switches, TP, FP, FN
        using greedy IoU matching per frame and tracking GT->pred ID mapping.
        Also tags detections involved in ID switches with det["id_switch"] = True.
        """
        TP = 0
        FP = 0
        FN = 0
        IDS = 0
        iou_sum = 0.0
        match_cnt = 0
        GT_total = total_gt_dets_with_id

        for seq_name, seq_img_ids in seq_to_img_ids.items():
            if not seq_img_ids:
                continue
            for cls in CLASSES:
                prev_match_for_gt = {}

                for img_id in seq_img_ids:
                    gt_objs = [
                        g
                        for g in gt_tracks_by_img_cls[img_id][cls]
                        if g["track_id"] is not None
                    ]
                    pred_objs = [
                        d
                        for d in pred_store[img_id][cls]
                        if d["score"] >= THR and d["track_id"] is not None
                    ]

                    if not gt_objs and not pred_objs:
                        continue

                    matches = []
                    for gi, g in enumerate(gt_objs):
                        for pi, p in enumerate(pred_objs):
                            iou = iou_xyxy(g["box"], p["box"])
                            if iou >= IOU_THR:
                                matches.append((iou, gi, pi))
                    matches.sort(reverse=True, key=lambda x: x[0])

                    used_gt = set()
                    used_pred = set()
                    for iou, gi, pi in matches:
                        if gi in used_gt or pi in used_pred:
                            continue
                        used_gt.add(gi)
                        used_pred.add(pi)

                        g = gt_objs[gi]
                        p = pred_objs[pi]

                        TP += 1
                        iou_sum += iou
                        match_cnt += 1

                        gt_key = (seq_name, cls, g["track_id"])
                        pred_key = (seq_name, cls, p["track_id"])

                        prev_pred_key = prev_match_for_gt.get(gt_key, None)
                        if prev_pred_key is not None and prev_pred_key != pred_key:
                            IDS += 1
                            # mark this detection as ID switch
                            p["id_switch"] = True

                        prev_match_for_gt[gt_key] = pred_key

                    FP += (len(pred_objs) - len(used_pred))
                    FN += (len(gt_objs) - len(used_gt))

        MOTA = 1.0 - safe_div(FN + FP + IDS, GT_total)
        MOTP = safe_div(iou_sum, match_cnt)
        return {
            "MOTA": MOTA,
            "MOTP": MOTP,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "IDS": IDS,
            "GT_total": GT_total,
            "matches": match_cnt,
        }

    def compute_global_id_metrics(pred_store, gt_tracks_by_img_cls, THR):
        """
        Approximate IDF1-style metrics by:
        - building global counts of (GT track, Pred track) co-occurrences via IoU matches
        - greedily assigning GT tracks to Pred tracks to maximize matched detections
        """
        edge_weights = defaultdict(int)
        total_gt_det = 0
        total_pred_det = 0
        gt_track_keys = set()
        pred_track_keys = set()

        for seq_name, seq_img_ids in seq_to_img_ids.items():
            if not seq_img_ids:
                continue
            for cls in CLASSES:
                for img_id in seq_img_ids:
                    gt_objs = [
                        g
                        for g in gt_tracks_by_img_cls[img_id][cls]
                        if g["track_id"] is not None
                    ]
                    pred_objs = [
                        d
                        for d in pred_store[img_id][cls]
                        if d["score"] >= THR and d["track_id"] is not None
                    ]

                    total_gt_det += len(gt_objs)
                    total_pred_det += len(pred_objs)

                    if not gt_objs or not pred_objs:
                        continue

                    matches = []
                    for gi, g in enumerate(gt_objs):
                        for pi, p in enumerate(pred_objs):
                            iou = iou_xyxy(g["box"], p["box"])
                            if iou >= IOU_THR:
                                matches.append((iou, gi, pi))
                    matches.sort(reverse=True, key=lambda x: x[0])

                    used_gt = set()
                    used_pred = set()
                    for iou, gi, pi in matches:
                        if gi in used_gt or pi in used_pred:
                            continue
                        used_gt.add(gi)
                        used_pred.add(pi)

                        g = gt_objs[gi]
                        p = pred_objs[pi]

                        gt_key = (seq_name, cls, g["track_id"])
                        pred_key = (seq_name, cls, p["track_id"])

                        gt_track_keys.add(gt_key)
                        pred_track_keys.add(pred_key)
                        edge_weights[(gt_key, pred_key)] += 1

        edges_sorted = sorted(
            edge_weights.items(), key=lambda kv: kv[1], reverse=True
        )
        used_gt_tracks = set()
        used_pred_tracks = set()
        IDTP = 0
        for (gt_key, pred_key), w in edges_sorted:
            if gt_key in used_gt_tracks or pred_key in used_pred_tracks:
                continue
            used_gt_tracks.add(gt_key)
            used_pred_tracks.add(pred_key)
            IDTP += w

        IDFP = total_pred_det - IDTP
        IDFN = total_gt_det - IDTP
        IDP = safe_div(IDTP, IDTP + IDFP)
        IDR = safe_div(IDTP, IDTP + IDFN)
        IDF1 = safe_div(2 * IDP * IDR, IDP + IDR) if (IDP + IDR) > 0 else 0.0

        gt_tracks_with_match = len(used_gt_tracks)
        pred_tracks_with_match = len(used_pred_tracks)
        total_gt_tracks_local = len(set(k[2] for k in gt_track_keys))
        total_pred_tracks_local = len(set(k[2] for k in pred_track_keys))

        return {
            "IDP": IDP,
            "IDR": IDR,
            "IDF1": IDF1,
            "IDTP": IDTP,
            "IDFP": IDFP,
            "IDFN": IDFN,
            "total_gt_det": total_gt_det,
            "total_pred_det": total_pred_det,
            "gt_tracks_with_match": gt_tracks_with_match,
            "pred_tracks_with_match": pred_tracks_with_match,
            "approx_total_gt_tracks": total_gt_tracks_local,
            "approx_total_pred_tracks": total_pred_tracks_local,
        }

    def compute_pred_track_length_stats(all_tracks_by_seq_cls, THR):
        """
        Compute statistics of predicted track lengths (in frames / detections).
        Returns dict with overall stats.
        """
        lengths_all = []
        lengths_thr = []

        for seq_name, cls_tracks in all_tracks_by_seq_cls.items():
            for cls, tracks in cls_tracks.items():
                for trk in tracks:
                    L_all = len(trk["detections"])
                    L_thr = sum(
                        1
                        for (_, _, d) in trk["detections"]
                        if d["score"] >= THR
                    )
                    lengths_all.append(L_all)
                    lengths_thr.append(L_thr)

        def stats_from_list(x):
            if not x:
                return {
                    "mean": 0.0,
                    "median": 0.0,
                    "min": 0,
                    "max": 0,
                    "num": 0,
                }
            arr = np.array(x, dtype=float)
            return {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "min": int(arr.min()),
                "max": int(arr.max()),
                "num": int(len(arr)),
            }

        return {
            "all": stats_from_list(lengths_all),
            "thr": stats_from_list(lengths_thr),
        }

    # =====================================================
    # --- metrics + logging -------------------------------
    # =====================================================
    summary_lines = []

    summary_lines.append(
        "==== RF-DETR EVAL + TRACKING (trackpy + hole-filled, per-sequence) ====\n"
    )
    summary_lines.append(f"checkpoint: {checkpoint}")
    summary_lines.append(f"ann_file: {ANN_FILE}")
    summary_lines.append(f"img_dir: {IMG_DIR}")
    summary_lines.append(f"classes: {CLASSES}")
    summary_lines.append(f"CONF_THR_LIST: {CONF_THR_LIST}")
    summary_lines.append(f"IOU_THR: {IOU_THR}")
    summary_lines.append(f"CENTROID_TOL_PX: {CENTROID_TOL_PX}")
    summary_lines.append(f"TRACK_IOU_THR: {TRACK_IOU_THR}")
    summary_lines.append(f"TRACK_MAX_AGE: {TRACK_MAX_AGE}")
    summary_lines.append(f"TRACK_MIN_CONF: {TRACK_MIN_CONF}")
    summary_lines.append(f"MIN_TRACK_LEN_FOR_BONUS: {MIN_TRACK_LEN_FOR_BONUS}")
    summary_lines.append(f"SCORE_BONUS_PER_FRAME: {SCORE_BONUS_PER_FRAME}")
    summary_lines.append(f"MAX_SCORE: {MAX_SCORE}")
    summary_lines.append(f"MAX_HOLE_LEN: {MAX_HOLE_LEN}")
    summary_lines.append(f"INTERP_SCORE_PENALTY: {INTERP_SCORE_PENALTY}")
    summary_lines.append(f"HOLE_MIN_EDGE_CONF: {HOLE_MIN_EDGE_CONF}")
    summary_lines.append(f"TRACKPY_SEARCH_RANGE: {TRACKPY_SEARCH_RANGE}")
    summary_lines.append(f"TRACKPY_MEMORY: {TRACKPY_MEMORY}")
    summary_lines.append(f"images_processed: {images_processed}")
    summary_lines.append(f"raw_preds_total: {raw_preds_total}")
    summary_lines.append(f"postproc_preds_total: {postproc_preds_total}")
    summary_lines.append(f"interpolated_detections: {interpolated_count}")
    summary_lines.append(f"total_gt_tracks: {total_gt_tracks}")
    summary_lines.append(f"total_gt_dets_with_id: {total_gt_dets_with_id}")

    if gt_track_lengths:
        gt_lengths_arr = np.array(gt_track_lengths, dtype=float)
        summary_lines.append(
            f"GT tracks: num={len(gt_track_lengths)}, "
            f"mean_len={gt_lengths_arr.mean():.2f}, "
            f"median_len={np.median(gt_lengths_arr):.2f}, "
            f"min_len={gt_lengths_arr.min():.0f}, "
            f"max_len={gt_lengths_arr.max():.0f}"
        )
    else:
        summary_lines.append("GT tracks: none with track_id")
    summary_lines.append("")

    metrics_dict = {
        "config": {
            "checkpoint": checkpoint,
            "ann_file": ANN_FILE,
            "img_dir": IMG_DIR,
            "classes": CLASSES,
            "CONF_THR_LIST": CONF_THR_LIST,
            "IOU_THR": IOU_THR,
            "CENTROID_TOL_PX": CENTROID_TOL_PX,
            "TRACK_IOU_THR": TRACK_IOU_THR,
            "TRACK_MAX_AGE": TRACK_MAX_AGE,
            "TRACK_MIN_CONF": TRACK_MIN_CONF,
            "MIN_TRACK_LEN_FOR_BONUS": MIN_TRACK_LEN_FOR_BONUS,
            "SCORE_BONUS_PER_FRAME": SCORE_BONUS_PER_FRAME,
            "MAX_SCORE": MAX_SCORE,
            "MAX_HOLE_LEN": MAX_HOLE_LEN,
            "INTERP_SCORE_PENALTY": INTERP_SCORE_PENALTY,
            "HOLE_MIN_EDGE_CONF": HOLE_MIN_EDGE_CONF,
            "TRACKPY_SEARCH_RANGE": TRACKPY_SEARCH_RANGE,
            "TRACKPY_MEMORY": TRACKPY_MEMORY,
        },
        "per_threshold": {},
        "gt_track_stats": {
            "total_gt_tracks": total_gt_tracks,
            "total_gt_dets_with_id": total_gt_dets_with_id,
            "lengths": {
                "mean": float(np.mean(gt_track_lengths))
                if gt_track_lengths
                else 0.0,
                "median": float(np.median(gt_track_lengths))
                if gt_track_lengths
                else 0.0,
                "min": int(min(gt_track_lengths)) if gt_track_lengths else 0,
                "max": int(max(gt_track_lengths)) if gt_track_lengths else 0,
            },
        },
    }

    for THR in CONF_THR_LIST:
        TP_thr, FP_thr, FN_thr, TPc_thr, FPc_thr, FNc_thr = results_by_thr[THR]

        summary_lines.append(f"\nThreshold = {THR}")
        summary_lines.append(
            f"{'Class':<10} {'Prec':>8} {'Recall':>8} {'F1':>8} (TP/FP/FN)"
        )
        thr_metrics = {"per_class": {}}

        for cls in CLASSES:
            prec = safe_div(TP_thr[cls], TP_thr[cls] + FP_thr[cls])
            rec = safe_div(TP_thr[cls], TP_thr[cls] + FN_thr[cls])
            f1 = (
                safe_div(2 * prec * rec, (prec + rec))
                if (prec + rec) > 0
                else 0.0
            )

            line = (
                f"{cls:<10} {prec:8.3f} {rec:8.3f} {f1:8.3f} "
                f"({TP_thr[cls]}/{FP_thr[cls]}/{FN_thr[cls]})"
            )
            print(line)
            summary_lines.append(line)

            thr_metrics["per_class"].setdefault(cls, {})
            thr_metrics["per_class"][cls]["iou"] = {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "TP": TP_thr[cls],
                "FP": FP_thr[cls],
                "FN": FN_thr[cls],
            }

        print(f"\nCentroid F1 (tolerance = {CENTROID_TOL_PX}px)")
        summary_lines.append(
            f"\nCentroid F1 (tolerance = {CENTROID_TOL_PX}px)"
        )
        summary_lines.append(
            f"{'Class':<10} {'Prec':>8} {'Recall':>8} {'F1':>8} (TP/FP/FN)"
        )

        for cls in CLASSES:
            prec = safe_div(TPc_thr[cls], TPc_thr[cls] + FPc_thr[cls])
            rec = safe_div(TPc_thr[cls], TPc_thr[cls] + FNc_thr[cls])
            f1 = (
                safe_div(2 * prec * rec, (prec + rec))
                if (prec + rec) > 0
                else 0.0
            )

            line = (
                f"{cls:<10} {prec:8.3f} {rec:8.3f} {f1:8.3f} "
                f"({TPc_thr[cls]}/{FPc_thr[cls]}/{FNc_thr[cls]})"
            )
            print(line)
            summary_lines.append(line)

            thr_metrics["per_class"][cls]["centroid"] = {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "TP": TPc_thr[cls],
                "FP": FPc_thr[cls],
                "FN": FNc_thr[cls],
            }

        clear_metrics = compute_clear_mot_metrics(
            pred_store, gt_tracks_by_img_cls, THR
        )
        mota = clear_metrics["MOTA"]
        motp = clear_metrics["MOTP"]
        TP_c = clear_metrics["TP"]
        FP_c = clear_metrics["FP"]
        FN_c = clear_metrics["FN"]
        IDS_c = clear_metrics["IDS"]
        GT_total_c = clear_metrics["GT_total"]
        matches_c = clear_metrics["matches"]

        clear_line = (
            f"CLEAR MOT at THR={THR}: "
            f"MOTA={mota:.3f}, MOTP(IoU)={motp:.3f}, "
            f"TP={TP_c}, FP={FP_c}, FN={FN_c}, IDS={IDS_c}, "
            f"GT_total={GT_total_c}, matches={matches_c}"
        )
        print()
        print(clear_line)
        summary_lines.append("")
        summary_lines.append(clear_line)
        thr_metrics["tracking_CLEAR"] = clear_metrics

        id_metrics = compute_global_id_metrics(
            pred_store, gt_tracks_by_img_cls, THR
        )
        id_line = (
            f"Global ID metrics at THR={THR}: "
            f"IDP={id_metrics['IDP']:.3f}, IDR={id_metrics['IDR']:.3f}, "
            f"IDF1={id_metrics['IDF1']:.3f}, "
            f"IDTP={id_metrics['IDTP']}, IDFP={id_metrics['IDFP']}, "
            f"IDFN={id_metrics['IDFN']}, "
            f"gt_tracks_with_match={id_metrics['gt_tracks_with_match']}, "
            f"pred_tracks_with_match={id_metrics['pred_tracks_with_match']}"
        )
        print(id_line)
        summary_lines.append(id_line)
        thr_metrics["tracking_IDF1_global"] = id_metrics

        track_len_stats = compute_pred_track_length_stats(
            all_tracks_by_seq_cls, THR
        )
        stats_all = track_len_stats["all"]
        stats_thr = track_len_stats["thr"]

        len_line_all = (
            f"Predicted track lengths (all detections) at THR={THR}: "
            f"num={stats_all['num']}, mean={stats_all['mean']:.2f}, "
            f"median={stats_all['median']:.2f}, "
            f"min={stats_all['min']}, max={stats_all['max']}"
        )
        len_line_thr = (
            f"Predicted track lengths (detections >= THR) at THR={THR}: "
            f"num={stats_thr['num']}, mean={stats_thr['mean']:.2f}, "
            f"median={stats_thr['median']:.2f}, "
            f"min={stats_thr['min']}, max={stats_thr['max']}"
        )
        print(len_line_all)
        print(len_line_thr)
        summary_lines.append(len_line_all)
        summary_lines.append(len_line_thr)
        thr_metrics["pred_track_length_stats"] = track_len_stats

        metrics_dict["per_threshold"][str(THR)] = thr_metrics

    ap_per_class = {}
    for cls in CLASSES:
        ap, recs, pres = compute_ap_for_class(
            pred_store, gt_by_img_cls, cls, IOU_THR
        )
        ap_per_class[cls] = ap

    mAP = float(np.mean(list(ap_per_class.values()))) if len(ap_per_class) > 0 else 0.0

    summary_lines.append("\n================ mAP summary (postprocessed) ================")
    summary_lines.append(f"IoU threshold: {IOU_THR}")
    print("\n================ mAP summary (postprocessed) ================")
    print(f"IoU threshold: {IOU_THR}")
    for cls in CLASSES:
        line = f"AP[{cls}] = {ap_per_class[cls]:.4f}"
        print(line)
        summary_lines.append(line)
    line = f"mAP = {mAP:.4f}"
    print(line)
    summary_lines.append(line)
    summary_lines.append("============================================")
    print("============================================")

    metrics_dict["ap_fixed_iou"] = {
        "iou": IOU_THR,
        "per_class": ap_per_class,
        "mAP": mAP,
    }

    IOU_LIST = [round(x, 2) for x in np.arange(0.25, 0.951, 0.05)]
    ap_by_iou_per_class = {cls: [] for cls in CLASSES}
    mAP_by_iou = []

    for iou in IOU_LIST:
        ap_per_class_iou = {}
        for cls in CLASSES:
            ap_iou, _, _ = compute_ap_for_class(
                pred_store, gt_by_img_cls, cls, iou
            )
            ap_per_class_iou[cls] = ap_iou
        for cls in CLASSES:
            ap_by_iou_per_class[cls].append(ap_per_class_iou[cls])

        mAP_iou = (
            float(np.mean(list(ap_per_class_iou.values())))
            if len(ap_per_class_iou) > 0
            else 0.0
        )
        mAP_by_iou.append(mAP_iou)

    print(
        "\n============= COCO mAP@[0.25:0.95] (postprocessed) ============="
    )
    summary_lines.append(
        "\n============= COCO mAP@[0.25:0.95] (postprocessed) ============="
    )
    hdr = "IoU " + " ".join([f"AP[{cls}]" for cls in CLASSES]) + " mAP"
    print(hdr)
    summary_lines.append(hdr)
    for idx, iou in enumerate(IOU_LIST):
        row_vals = [
            f"{ap_by_iou_per_class[cls][idx]:.4f}" for cls in CLASSES
        ]
        line = f"{iou:>4.2f} " + " ".join(row_vals) + f" {mAP_by_iou[idx]:.4f}"
        print(line)
        summary_lines.append(line)

    avg_per_class = {
        cls: float(np.mean(ap_by_iou_per_class[cls]))
        if len(ap_by_iou_per_class[cls]) > 0
        else 0.0
        for cls in CLASSES
    }
    COCO_mAP = float(np.mean(mAP_by_iou)) if len(mAP_by_iou) > 0 else 0.0

    print("----------------------------------------------")
    summary_lines.append("----------------------------------------------")
    for cls in CLASSES:
        line = f"AP[{cls}]@[0.25:0.95] = {avg_per_class[cls]:.4f}"
        print(line)
        summary_lines.append(line)
    line = f"COCO mAP@[0.25:0.95] = {COCO_mAP:.4f}"
    print(line)
    summary_lines.append(line)
    summary_lines.append("==============================================")
    print("==============================================")

    metrics_dict["coco_style"] = {
        "ious": IOU_LIST,
        "per_class_ap_by_iou": ap_by_iou_per_class,
        "mAP_by_iou": mAP_by_iou,
        "avg_per_class": avg_per_class,
        "COCO_mAP": COCO_mAP,
    }

    # =====================================================
    # --- write logs to disk ------------------------------
    # =====================================================
    summary_path = os.path.join(
        RESULTS_DIR,
        f"summary_MODEL_{CHECKPOINT_NAME}__GT{GT_ANNOTATION}_{TRACK_MERGE_DIST_PX}.txt",
    )
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    metrics_path = os.path.join(
        RESULTS_DIR,
        f"metrics_MODEL_{CHECKPOINT_NAME}__GT{GT_ANNOTATION}_{TRACK_MERGE_DIST_PX}.json",
    )

    def _to_serializable(o):
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        return o

    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2, default=_to_serializable)

    print(f"\n[INFO] Results written to:")
    print(f" {summary_path}")
    print(f" {metrics_path}")

        # =====================================================
    # --- visualization videos (GT green, pred blue, ID switch red) ------
    # =====================================================
    all_img_ids_sorted = sorted(
        imgid_to_info.keys(), key=lambda i: imgid_to_info[i]["file_name"]
    )

    if all_img_ids_sorted:
        first_img_id = all_img_ids_sorted[0]
        first_path = os.path.join(
            IMG_DIR, imgid_to_info[first_img_id]["file_name"]
        )
        first_img = Image.open(first_path).convert("RGB")
        w, h = first_img.size
        first_img.close()
    else:
        w, h = 512, 512  # default guess

    # decide final video size
    if UPSCALE_VIDEO:
        out_w, out_h = UPSCALED_SIZE
    else:
        out_w, out_h = w, h

    gt_color = (0, 255, 0)       # green (BGR)
    pred_color = (255, 0, 0)     # blue (BGR)
    idswitch_color = (0, 0, 255) # red (BGR)

    for THR in CONF_THR_LIST:
        video_path = os.path.join(
            RESULTS_DIR,
            f"video_MODEL_{CHECKPOINT_NAME}__GT{GT_ANNOTATION}"
            f"__thr{THR:.2f}__{TRACK_MERGE_DIST_PX}.mp4",
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (out_w, out_h))

        print(f"[INFO] Writing video for THR={THR} to: {video_path}")

        for idx, img_id in enumerate(all_img_ids_sorted):
            im_meta = imgid_to_info[img_id]
            img_path = os.path.join(IMG_DIR, im_meta["file_name"])
            if not os.path.exists(img_path):
                continue

            pil_img = Image.open(img_path).convert("RGB")
            frame = np.array(pil_img)
            pil_img.close()

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # draw GT
            gt_boxes = []
            for cls in CLASSES:
                gt_boxes.extend(gt_by_img_cls[img_id].get(cls, []))

            for box in gt_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), gt_color, 2)

            # draw predictions
            per_cls_preds = pred_store.get(img_id, {})
            for cls in CLASSES:
                for det in per_cls_preds.get(cls, []):
                    if det["score"] < THR:
                        continue
                    box = det["box"]
                    x1, y1, x2, y2 = map(int, box)

                    color = idswitch_color if det.get("id_switch", False) else pred_color
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label = (
                        f"id{det['track_id']}"
                        if det["track_id"] is not None
                        else ""
                    )
                    if det.get("interp", False):
                        label += " (interp)"
                    if det.get("id_switch", False):
                        label += " [SW]"
                    if label:
                        cv2.putText(
                            frame,
                            label,
                            (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

            # annotations text
            cv2.putText(
                frame,
                "GT: green, Pred: blue, ID switch: red",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"frame {idx+1}/{len(all_img_ids_sorted)} thr={THR:.2f}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # --- upscaling happens here ---
            if UPSCALE_VIDEO:
                frame = cv2.resize(
                    frame,
                    (out_w, out_h),
                    interpolation=cv2.INTER_CUBIC,
                )

            video_writer.write(frame)

        video_writer.release()
        print(f"[INFO] Video saved: {video_path}")
