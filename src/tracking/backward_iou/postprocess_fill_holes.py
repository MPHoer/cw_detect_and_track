"""
Detect objects, track backwards by IoU, and fill internal detection gaps.

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

Set TRACK_IOU_THR (minimum linking overlap), TRACK_MAX_AGE (maximum linking
distance), MAX_HOLE_LEN (missing positions to fill), HOLE_MIN_EDGE_CONF
(required at both endpoints), and INTERP_SCORE_PENALTY.
MIN_TRACK_LEN_FOR_BONUS, SCORE_BONUS_PER_FRAME, and MAX_SCORE control score
boosting and capping; the bonus defaults to zero.

Greedily associate each detection with the eligible track of greatest
last-box IoU, processing frames backwards. Then linearly interpolate boxes
and scores across qualifying internal gaps. No track-end extrapolation
occurs. Endpoint checks use scores after boosting. Backwards processing was
chosen for the strong signal of round dying cells late in the sequences;
reversing direction can change the resulting associations.

Report detection F1/AP, custom MOTA/MOTP, approximate global IDF1, and track
lengths. AP is confidence-truncated and uses IoUs 0.25–0.95, not official COCO AP.

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

# =========================
# --- CONFIG -------------
# =========================

CHECKPOINT_NAME = "ASM_20x20"
GT_ANNOTATION = "_annotations_20x20.coco"

BASE_DIR = "D:/cw_rf_detr_dataset_raw"
checkpoint = f"{BASE_DIR}/output/{CHECKPOINT_NAME}/checkpoint_best_total.pth" #checkpoint_best_ema.pth
IMG_DIR = f"{BASE_DIR}/valid"                                  # folder with the images
ANN_FILE = f"{BASE_DIR}/valid/{GT_ANNOTATION}.json"            # COCO-style GT

# where to save results
RESULTS_DIR = os.path.join(BASE_DIR, "results_rfdetr_eval", "fill_holes")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"  checkpoint: {checkpoint}")
print(f"  ANN_FILE: {ANN_FILE}")
print(f"  IMG_DIR: {IMG_DIR}")
print(f"  RESULTS_DIR: {RESULTS_DIR}")
print()

# evaluation hyperparams
CONF_THR_LIST = [0.4]
IOU_THR = 0.5          # IoU threshold for detection eval and tracking
CENTROID_TOL_PX = 10   # centroid matching tolerance in pixels

# tracking hyperparams
TRACK_IOU_THR = 0.2       # IoU threshold for associating detections into tracks
TRACK_MAX_AGE = 60         # max frame gap (in frames) for association (backwards)
TRACK_MIN_CONF = 0.4       # min detection score used for tracking
MIN_TRACK_LEN_FOR_BONUS = 2   # track length from which we start boosting scores
SCORE_BONUS_PER_FRAME = 0.0   # how much score bonus per extra frame in the track
MAX_SCORE = 1.0               # cap on boosted scores

# hole filling hyperparams
MAX_HOLE_LEN = 10          # maximum number of consecutive missing frames to fill
INTERP_SCORE_PENALTY = 0.0 # score penalty applied to interpolated detections
# NEW: minimum confidence required for BOTH ends of a hole to be interpolated
HOLE_MIN_EDGE_CONF = 0.6
# video config
VIDEO_FPS = 5

# =====================================================
# --- helpers -----------------------------------------
# =====================================================

def xywh_to_xyxy(b):
    x, y, w, h = b
    return np.array([x, y, x + w, y + h], dtype=np.float32)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1); ih = max(0.0, inter_y2 - inter_y1)
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
    Flatten detections for a given class across the dataset, sort by score, and
    perform 1-to-1 greedy matching against ground-truths within each image.
    pred_store: dict[img_id][cls] -> list of det dicts {"box": np.array, "score": float, ...}
    """
    # Collect all detections of this class: list of (img_id, score, box)
    dets = []
    for img_id, per_cls in pred_store.items():
        for det in per_cls.get(class_name, []):
            dets.append((img_id, float(det["score"]), np.asarray(det["box"], dtype=np.float32)))

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
print(f"  cat_id_to_name: {cat_id_to_name}")
print(f"  name_to_cat_id: {name_to_cat_id}")
print(f"  #images in GT: {len(imgid_to_info)}")
print(f"[DEBUG] Using CLASSES from COCO (ignoring {IGNORE_CLASSES}): {CLASSES}")
print(f"  CONF_THR_LIST: {CONF_THR_LIST}")
print(f"  IOU_THR: {IOU_THR}")
print(f"  CENTROID_TOL_PX: {CENTROID_TOL_PX}")
print(f"  TRACK_IOU_THR: {TRACK_IOU_THR}")
print(f"  TRACK_MAX_AGE: {TRACK_MAX_AGE}")
print(f"  TRACK_MIN_CONF: {TRACK_MIN_CONF}")
print(f"  MIN_TRACK_LEN_FOR_BONUS: {MIN_TRACK_LEN_FOR_BONUS}")
print(f"  SCORE_BONUS_PER_FRAME: {SCORE_BONUS_PER_FRAME}")
print(f"  MAX_SCORE: {MAX_SCORE}")
print(f"  MAX_HOLE_LEN: {MAX_HOLE_LEN}")
print(f"  INTERP_SCORE_PENALTY: {INTERP_SCORE_PENALTY}")
print(f"  HOLE_MIN_EDGE_CONF: {HOLE_MIN_EDGE_CONF}")
sys.stdout.flush()

# build GT per image per class (xyxy boxes) and GT tracks (for tracking metrics)
gt_by_img_cls = defaultdict(lambda: defaultdict(list))   # img_id -> cls -> [box]
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
# NOTE: we skip model.optimize_for_inference() because torch.jit.trace
# cannot handle non-tensor outputs from the model's forward

model_class_names = getattr(model, "class_names", None)
print("[DEBUG] Model loaded.")
print(f"  Has class_names: {model_class_names is not None}")
if model_class_names is not None:
    print(f"  model.class_names (len={len(model_class_names)}): {model_class_names}")
sys.stdout.flush()

# =====================================================
# --- inference: collect raw detections ---------------
# =====================================================

# store raw predictions per image (no thresholding yet)
# pred_store[img_id][cls] -> list of det dict:
#   {"box": np.array(4,), "score": float, "base_score": float, "track_id": None, "interp": bool}
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
        gt_counts = {c: len(gt_by_img_cls[img_id].get(c, [])) for c in CLASSES}
        if images_processed < 3:
            print(f"[DEBUG] Image {images_processed+1}: id={img_id} path={img_path}")
            print(f"  GT counts: {gt_counts}")
            sys.stdout.flush()

        # run inference (use lowest conf thr just to prune complete garbage)
        preds = model.predict(image, threshold=CONF_THR_LIST[0])
        pred0 = preds[0] if isinstance(preds, (list, tuple)) else preds

        # Inspect raw prediction structure
        if images_processed < 3:
            print(f"  pred type: {type(preds)} -> first elem type: {type(pred0)}")
            if isinstance(pred0, dict):
                print(f"  pred0 keys: {list(pred0.keys())}")
            elif hasattr(pred0, "predictions"):
                print("  pred0 has attribute .predictions")
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
        if isinstance(pred0, dict) and all(k in pred0 for k in ("scores", "labels", "boxes")):
            used_branch = "dict"
            scores = np.array(pred0["scores"]).astype(float)
            labels = pred0["labels"]
            boxes = np.array(pred0["boxes"], dtype=np.float32)
            raw_preds_total += len(scores)
            for s, lbl, box in zip(scores, labels, boxes):
                cls = label_to_name(lbl)
                if cls in CLASSES:
                    # ensure xyxy ordering
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
                if isinstance(model_class_names, (list, tuple)) and 0 <= cid < len(model_class_names):
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
                cls = label_to_name(getattr(p, "class_name", getattr(p, "label", None)))
                conf = float(getattr(p, "confidence", getattr(p, "score", 1.0)))
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
                        box = np.array([x1, y1, x2, y2], dtype=np.float32)
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
            print(f"  used_branch: {used_branch}")
            if used_branch == "supervision.Detections":
                cid_raw = getattr(pred0, "class_id", None)
                if cid_raw is not None and len(cid_raw) > 0:
                    cid_preview = list(map(int, np.array(cid_raw)[:2]))
                else:
                    cid_preview = []
                print(f"  raw class_id preview: {cid_preview}")
            # quick preview of per-class counts
            debug_counts = {c: len(pred_store[img_id][c]) for c in CLASSES}
            print(f"  pred counts by class: {debug_counts}")
            for c in CLASSES:
                if pred_store[img_id][c]:
                    d0 = pred_store[img_id][c][0]
                    print(f"    sample {c} pred[0]: box={d0['box']}, score={d0['score']:.3f}")
            sys.stdout.flush()

        images_processed += 1
        if images_processed % 100 == 0:
            print(
                f"[DEBUG] Processed {images_processed} images. "
                f"raw_preds_total={raw_preds_total}, dropped_by_conf={preds_dropped_by_conf}"
            )
            sys.stdout.flush()

        image.close()
    except Exception as e:
        print(
            f"[ERROR] Exception on image id={img_id} path={im_meta.get('file_name','?')}: {e}"
        )
        traceback.print_exc()
        sys.stdout.flush()
        continue

print(
    f"[DEBUG] Inference DONE: images_processed={images_processed}, "
    f"raw_preds_total={raw_preds_total}, dropped_by_conf={preds_dropped_by_conf}"
)
sys.stdout.flush()

# =====================================================
# --- tracking (backwards, per sequence) --------------
# =====================================================

# We'll track per sequence and per class, backwards in time.
# We only use detections with base_score >= TRACK_MIN_CONF for building tracks,
# but all detections remain in pred_store for detection evaluation.
all_tracks_by_seq_cls = defaultdict(lambda: defaultdict(list))  # seq -> cls -> [track_dict]

for seq_name, seq_img_ids in seq_to_img_ids.items():
    if not seq_img_ids:
        continue

    # map: local frame index (0..N-1) <-> img_id
    frame_idx_to_img_id = {fi: img_id for fi, img_id in enumerate(seq_img_ids)}
    img_id_to_frame_idx = {img_id: fi for fi, img_id in frame_idx_to_img_id.items()}

    # tracking structures
    tracks_by_cls = {cls: [] for cls in CLASSES}
    # Each track dict:
    #   {"id": int, "cls": str, "detections": [(frame_idx, img_id, det_dict)], "last_box": np.array, "last_frame": int}
    next_track_id_by_cls = {cls: 1 for cls in CLASSES}

    for cls in CLASSES:
        # backwards through frames
        for fi in reversed(range(len(seq_img_ids))):
            img_id = frame_idx_to_img_id[fi]
            dets = pred_store[img_id][cls]

            # candidate detections for tracking (filter by base_score)
            cand_dets = [d for d in dets if d["base_score"] >= TRACK_MIN_CONF]

            # For association, we keep all existing tracks of this class
            for det in cand_dets:
                box = det["box"]
                # find best matching track
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
                    # attach to existing track
                    det["track_id"] = best_trk["id"]
                    best_trk["detections"].append((fi, img_id, det))
                    best_trk["last_box"] = box
                    best_trk["last_frame"] = fi
                else:
                    # start new track
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

    # store tracks for this sequence
    for cls in CLASSES:
        all_tracks_by_seq_cls[seq_name][cls] = tracks_by_cls[cls]

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

                # --- NEW: only fill this hole if both ends are high-confidence ---
                # You can use det_i["score"] or det_i["base_score"]; here I use score.
                if det_i["score"] < HOLE_MIN_EDGE_CONF or det_j["score"] < HOLE_MIN_EDGE_CONF:
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
# --- count detections after postproc -----------------
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
            # filter by threshold and extract boxes
            dets = [d["box"].copy() for d in dets_scored if d["score"] >= THR]

            # IoU-based greedy match
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

            # centroid-distance greedy matching (fresh pass)
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

    results_by_thr[THR] = (TP_thr, FP_thr, FN_thr, TPc_thr, FPc_thr, FNc_thr)

print(
    f"[DEBUG] Detection evaluation DONE: images_processed={images_processed}, "
    f"raw_preds_total={raw_preds_total}, postproc_preds_total={postproc_preds_total}"
)
sys.stdout.flush()

# =====================================================
# --- tracking metrics (CLEAR MOT + global IDF1) ------
# =====================================================

def compute_clear_mot_metrics(pred_store, gt_tracks_by_img_cls, THR):
    """
    Compute CLEAR MOT-like metrics:
      MOTA, MOTP, ID switches, TP, FP, FN
    using greedy IoU matching per frame and tracking GT->pred ID mapping.
    """
    TP = 0
    FP = 0
    FN = 0
    IDS = 0
    iou_sum = 0.0
    match_cnt = 0

    # total GT detections with track_id (we already computed globally)
    GT_total = total_gt_dets_with_id

    for seq_name, seq_img_ids in seq_to_img_ids.items():
        if not seq_img_ids:
            continue
        for cls in CLASSES:
            # map GT track key -> previous predicted track key
            prev_match_for_gt = {}

            for img_id in seq_img_ids:
                gt_objs = [
                    g for g in gt_tracks_by_img_cls[img_id][cls]
                    if g["track_id"] is not None
                ]
                pred_objs = [
                    d for d in pred_store[img_id][cls]
                    if d["score"] >= THR and d["track_id"] is not None
                ]

                if not gt_objs and not pred_objs:
                    continue

                # build IoU candidates
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
    Returns IDP, IDR, IDF1, IDTP, IDFP, IDFN, plus track coverage stats.
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
                    g for g in gt_tracks_by_img_cls[img_id][cls]
                    if g["track_id"] is not None
                ]
                pred_objs = [
                    d for d in pred_store[img_id][cls]
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

    # Greedy maximum-weight matching between GT and Pred track IDs
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

    # coverage: how many GT tracks get any assigned pred track
    gt_tracks_with_match = len(used_gt_tracks)
    pred_tracks_with_match = len(used_pred_tracks)
    total_gt_tracks_local = len(set(k[2] for k in gt_track_keys))  # across seq+cls this is approximate
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
                # all detections in this track (including interpolated ones)
                L_all = len(trk["detections"])
                # detections above the threshold (effective length at THR)
                L_thr = sum(1 for (_, _, d) in trk["detections"] if d["score"] >= THR)
                lengths_all.append(L_all)
                lengths_thr.append(L_thr)

    def stats_from_list(x):
        if not x:
            return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0, "num": 0}
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
summary_lines.append("==== RF-DETR EVAL + TRACKING (score-boosted + hole-filled, per-sequence) ====\n")
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
summary_lines.append(f"images_processed: {images_processed}")
summary_lines.append(f"raw_preds_total: {raw_preds_total}")
summary_lines.append(f"postproc_preds_total: {postproc_preds_total}")
summary_lines.append(f"interpolated_detections: {interpolated_count}")
summary_lines.append(f"total_gt_tracks: {total_gt_tracks}")
summary_lines.append(f"total_gt_dets_with_id: {total_gt_dets_with_id}")

# GT track length stats
if gt_track_lengths:
    gt_lengths_arr = np.array(gt_track_lengths, dtype=float)
    summary_lines.append(f"GT tracks: num={len(gt_track_lengths)}, "
                         f"mean_len={gt_lengths_arr.mean():.2f}, "
                         f"median_len={np.median(gt_lengths_arr):.2f}, "
                         f"min_len={gt_lengths_arr.min():.0f}, "
                         f"max_len={gt_lengths_arr.max():.0f}")
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

    },
    "per_threshold": {},
    "gt_track_stats": {
        "total_gt_tracks": total_gt_tracks,
        "total_gt_dets_with_id": total_gt_dets_with_id,
        "lengths": {
            "mean": float(np.mean(gt_track_lengths)) if gt_track_lengths else 0.0,
            "median": float(np.median(gt_track_lengths)) if gt_track_lengths else 0.0,
            "min": int(min(gt_track_lengths)) if gt_track_lengths else 0,
            "max": int(max(gt_track_lengths)) if gt_track_lengths else 0,
        },
    },
}

for THR in CONF_THR_LIST:
    TP_thr, FP_thr, FN_thr, TPc_thr, FPc_thr, FNc_thr = results_by_thr[THR]
    summary_lines.append(f"\nThreshold = {THR}")
    summary_lines.append(f"{'Class':<10} {'Prec':>8} {'Recall':>8} {'F1':>8}  (TP/FP/FN)")
    thr_metrics = {"per_class": {}}

    for cls in CLASSES:
        prec = safe_div(TP_thr[cls], TP_thr[cls] + FP_thr[cls])
        rec = safe_div(TP_thr[cls], TP_thr[cls] + FN_thr[cls])
        f1 = safe_div(2 * prec * rec, (prec + rec)) if (prec + rec) > 0 else 0.0
        line = f"{cls:<10} {prec:8.3f} {rec:8.3f} {f1:8.3f}  ({TP_thr[cls]}/{FP_thr[cls]}/{FN_thr[cls]})"
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
    summary_lines.append(f"\nCentroid F1 (tolerance = {CENTROID_TOL_PX}px)")
    summary_lines.append(f"{'Class':<10} {'Prec':>8} {'Recall':>8} {'F1':>8}  (TP/FP/FN)")
    for cls in CLASSES:
        prec = safe_div(TPc_thr[cls], TPc_thr[cls] + FPc_thr[cls])
        rec = safe_div(TPc_thr[cls], TPc_thr[cls] + FNc_thr[cls])
        f1 = safe_div(2 * prec * rec, (prec + rec)) if (prec + rec) > 0 else 0.0
        line = f"{cls:<10} {prec:8.3f} {rec:8.3f} {f1:8.3f}  ({TPc_thr[cls]}/{FPc_thr[cls]}/{FNc_thr[cls]})"
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

    # CLEAR MOT metrics
    clear_metrics = compute_clear_mot_metrics(pred_store, gt_tracks_by_img_cls, THR)
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

    # Global ID metrics (IDF1-like)
    id_metrics = compute_global_id_metrics(pred_store, gt_tracks_by_img_cls, THR)
    id_line = (
        f"Global ID metrics at THR={THR}: "
        f"IDP={id_metrics['IDP']:.3f}, IDR={id_metrics['IDR']:.3f}, "
        f"IDF1={id_metrics['IDF1']:.3f}, "
        f"IDTP={id_metrics['IDTP']}, IDFP={id_metrics['IDFP']}, IDFN={id_metrics['IDFN']}, "
        f"gt_tracks_with_match={id_metrics['gt_tracks_with_match']}, "
        f"pred_tracks_with_match={id_metrics['pred_tracks_with_match']}"
    )
    print(id_line)
    summary_lines.append(id_line)

    thr_metrics["tracking_IDF1_global"] = id_metrics

    # Pred track length stats
    track_len_stats = compute_pred_track_length_stats(all_tracks_by_seq_cls, THR)
    stats_all = track_len_stats["all"]
    stats_thr = track_len_stats["thr"]

    len_line_all = (
        f"Predicted track lengths (all detections) at THR={THR}: "
        f"num={stats_all['num']}, mean={stats_all['mean']:.2f}, "
        f"median={stats_all['median']:.2f}, min={stats_all['min']}, max={stats_all['max']}"
    )
    len_line_thr = (
        f"Predicted track lengths (detections >= THR) at THR={THR}: "
        f"num={stats_thr['num']}, mean={stats_thr['mean']:.2f}, "
        f"median={stats_thr['median']:.2f}, min={stats_thr['min']}, max={stats_thr['max']}"
    )

    print(len_line_all)
    print(len_line_thr)
    summary_lines.append(len_line_all)
    summary_lines.append(len_line_thr)

    thr_metrics["pred_track_length_stats"] = track_len_stats

    metrics_dict["per_threshold"][str(THR)] = thr_metrics

# --- mAP at fixed IOU_THR ---
ap_per_class = {}
for cls in CLASSES:
    ap, recs, pres = compute_ap_for_class(pred_store, gt_by_img_cls, cls, IOU_THR)
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

# --- COCO-style mAP@[0.25:0.95] ---
IOU_LIST = [round(x, 2) for x in np.arange(0.25, 0.951, 0.05)]

ap_by_iou_per_class = {cls: [] for cls in CLASSES}
mAP_by_iou = []

for iou in IOU_LIST:
    ap_per_class_iou = {}
    for cls in CLASSES:
        ap_iou, _, _ = compute_ap_for_class(pred_store, gt_by_img_cls, cls, iou)
        ap_per_class_iou[cls] = ap_iou
        ap_by_iou_per_class[cls].append(ap_iou)
    mAP_iou = float(np.mean(list(ap_per_class_iou.values()))) if len(ap_per_class_iou) > 0 else 0.0
    mAP_by_iou.append(mAP_iou)

print("\n============= COCO mAP@[0.25:0.95] (postprocessed) =============")
summary_lines.append("\n============= COCO mAP@[0.25:0.95] (postprocessed) =============")
hdr = "IoU  " + "  ".join([f"AP[{cls}]" for cls in CLASSES]) + "    mAP"
print(hdr)
summary_lines.append(hdr)
for idx, iou in enumerate(IOU_LIST):
    row_vals = [f"{ap_by_iou_per_class[cls][idx]:.4f}" for cls in CLASSES]
    line = f"{iou:>4.2f}  " + "  ".join(row_vals) + f"    {mAP_by_iou[idx]:.4f}"
    print(line)
    summary_lines.append(line)

avg_per_class = {
    cls: float(np.mean(ap_by_iou_per_class[cls])) if len(ap_by_iou_per_class[cls]) > 0 else 0.0
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
    f"summary_MODEL_{CHECKPOINT_NAME}__GT{GT_ANNOTATION}.txt",
)
with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines))

metrics_path = os.path.join(
    RESULTS_DIR,
    f"metrics_MODEL_{CHECKPOINT_NAME}__GT{GT_ANNOTATION}.json",
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
print(f"  {summary_path}")
print(f"  {metrics_path}")

# =====================================================
# --- visualization videos (GT green, pred blue) ------
# =====================================================

# stable image order: sort by file_name over entire dataset
all_img_ids_sorted = sorted(
    imgid_to_info.keys(),
    key=lambda i: imgid_to_info[i]["file_name"],
)

# determine frame size from first existing image
if all_img_ids_sorted:
    first_img_id = all_img_ids_sorted[0]
    first_path = os.path.join(IMG_DIR, imgid_to_info[first_img_id]["file_name"])
    first_img = Image.open(first_path).convert("RGB")
    w, h = first_img.size
    first_img.close()
else:
    w, h = 512, 512  # default guess

gt_color = (0, 255, 0)   # green (BGR)
pred_color = (255, 0, 0) # blue (BGR)

for THR in CONF_THR_LIST:
    video_path = os.path.join(
        RESULTS_DIR,
        f"video_MODEL_{CHECKPOINT_NAME}__GT{GT_ANNOTATION}__thr{THR:.2f}.mp4",
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (w, h))

    print(f"[INFO] Writing video for THR={THR} to: {video_path}")

    for idx, img_id in enumerate(all_img_ids_sorted):
        im_meta = imgid_to_info[img_id]
        img_path = os.path.join(IMG_DIR, im_meta["file_name"])
        if not os.path.exists(img_path):
            continue

        # load image
        pil_img = Image.open(img_path).convert("RGB")
        frame = np.array(pil_img)
        pil_img.close()

        # convert RGB -> BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # GT boxes (all classes)
        gt_boxes = []
        for cls in CLASSES:
            gt_boxes.extend(gt_by_img_cls[img_id].get(cls, []))

        # draw GT (green)
        for box in gt_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), gt_color, 2)

        # Pred detections above THR (all classes)
        per_cls_preds = pred_store.get(img_id, {})
        for cls in CLASSES:
            for det in per_cls_preds.get(cls, []):
                if det["score"] < THR:
                    continue
                box = det["box"]
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), pred_color, 2)

                # optionally display track_id and interp flag
                label = f"id{det['track_id']}" if det["track_id"] is not None else ""
                if det.get("interp", False):
                    label += " (interp)"
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

        # legend + frame index
        cv2.putText(
            frame,
            "GT: green, Pred: blue (id, interp)",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"frame {idx+1}/{len(all_img_ids_sorted)}  thr={THR:.2f}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        video_writer.write(frame)

    video_writer.release()
    print(f"[INFO] Video saved: {video_path}")
