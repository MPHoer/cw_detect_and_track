"""
Evaluate RF-DETR Medium with original + horizontal/vertical flip TTA.

Configure BASE_DIR, CHECKPOINT_NAME (output folder containing the checkpoint),
GT_ANNOTATION (COCO filename without .json), and derived paths IMG_DIR, ANN_FILE,
and checkpoint as needed. Use exactly the same model, images, and ground truth
as eval_rfdetr.py to compare classic inference with TTA. No ASM is performed;
images must already match the representation used to train the checkpoint.

TTA_NMS_IOU_THR controls class-aware duplicate suppression after mapping all
three views back to original coordinates. Reuse predict_image() from the
adjacent inference.py with confidence=0.0 so AP retains low-confidence outputs.
NMS keeps the highest score, without averaging boxes or scores. CONF_THR_LIST
is applied only for F1 evaluation; IOU_THR and CENTROID_TOL_PX govern matching.

Metrics and ground-truth handling are unchanged from eval_rfdetr.py: per-class
IoU/centre precision, recall, F1, and custom AP at IOU_THR and across 0.50–0.95.
Ignore dummy/crowd labels. Missing or failed images still contribute GT misses.
AP is not official COCO AP. TTA requires three model passes per image and its
suppression may remove nearby distinct cells; improvement is not guaranteed.

Write text and JSON metric reports to RESULTS_DIR (default separate directory
results_rfdetr_eval_tta). Reports include the TTA views and NMS threshold.
Existing reports in that directory are overwritten. No tracking, prediction
table, or video export occurs. Like the original evaluator, runs on import.
"""

import os, json
from collections import defaultdict
from PIL import Image
import numpy as np
from rfdetr import RFDETRMedium
from inference import predict_image
from types import SimpleNamespace
import sys
import traceback

# --- CONFIG ---

#CHECKPOINT_NAME = "ASM_perf_gt20"
CHECKPOINT_NAME = "ASM_old_gt"
GT_ANNOTATION = "_perfect_gt_annotations_20x20.coco"

BASE_DIR = "D:/rf_detr_dataset"
checkpoint = f"{BASE_DIR}/output/{CHECKPOINT_NAME}/checkpoint_best_ema.pth"
IMG_DIR  = f"{BASE_DIR}/valid"                                  # folder with the images
ANN_FILE = f"{BASE_DIR}/valid/{GT_ANNOTATION}.json"  # COCO-style GT

# where to save results
RESULTS_DIR = os.path.join(BASE_DIR, "results_rfdetr_eval_tta")
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"  checkpoint: {checkpoint}")
print(f"  ANN_FILE: {ANN_FILE}")
print(f"  IMG_DIR: {IMG_DIR}")
print(f"  RESULTS_DIR: {RESULTS_DIR}")
print()

# evaluation hyperparams
CONF_THR_LIST = [0.4]   # used ONLY for F1 etc., NOT for AP computation
IOU_THR  = 0.5          # IoU threshold for AP@0.5
CENTROID_TOL_PX = 10    # centroid matching tolerance in pixels

TTA_NMS_IOU_THR = 0.5  # duplicate suppression, independent of evaluation IOU_THR

# --- helpers ---
def xywh_to_xyxy(b):
    x, y, w, h = b
    return np.array([x, y, x+w, y+h], dtype=np.float32)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1); ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, (ax2-ax1)) * max(0.0, (ay2-ay1))
    area_b = max(0.0, (bx2-bx1)) * max(0.0, (by2-by1))
    union = area_a + area_b - inter + 1e-9
    return inter / union

# centroid helpers
def box_center_xy(b):
    x1, y1, x2, y2 = b
    return (0.5*(x1+x2), 0.5*(y1+y2))

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
    Returns (AP, recalls, precisions) where recalls/precisions are numpy arrays.
    """
    # Collect all detections of this class: list of (img_id, score, box)
    dets = []
    for img_id, per_cls in pred_store.items():
        for (box, score) in per_cls.get(class_name, []):
            dets.append((img_id, float(score), np.asarray(box, dtype=np.float32)))

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

# --- load GT COCO ---
with open(ANN_FILE, "r") as f:
    coco = json.load(f)

# maps
cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
name_to_cat_id = {v: k for k, v in cat_id_to_name.items()}
imgid_to_info = {im["id"]: im for im in coco["images"]}

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
sys.stdout.flush()

# build GT per image per class (xyxy boxes)
gt_by_img_cls = defaultdict(lambda: defaultdict(list))
for ann in coco["annotations"]:
    if ann.get("iscrowd", 0) == 1:
        continue
    img_id = ann["image_id"]
    cat_id = ann["category_id"]
    cls_name = cat_id_to_name[cat_id]
    if cls_name not in CLASSES:
        continue
    gt_by_img_cls[img_id][cls_name].append(xywh_to_xyxy(ann["bbox"]))

# --- model (RFDETR native) ---
model = RFDETRMedium(pretrain_weights=checkpoint)
# NOTE: we skip model.optimize_for_inference() because torch.jit.trace
# cannot handle non-tensor outputs from the model's forward

model_class_names = getattr(model, "class_names", None)
print("[DEBUG] Model loaded.")
print(f"  Has class_names: {model_class_names is not None}")
if model_class_names is not None:
    print(f"  model.class_names (len={len(model_class_names)}): {model_class_names}")
sys.stdout.flush()

# store raw predictions per image (no *evaluation* thresholding yet)
pred_store = {}

images_processed = 0
preds_total = 0
preds_kept = 0
preds_dropped_by_conf = 0  # only used in some branches

# --- counters (for F1 at given thresholds) ---
TP = {c: 0 for c in CLASSES}
FP = {c: 0 for c in CLASSES}
FN = {c: 0 for c in CLASSES}

# --- evaluate ---
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

        # run inference
        # IMPORTANT: use a very low threshold here so AP is computed over the full score range
        boxes, scores, class_ids = predict_image(
            model, image, use_tta=True, confidence=0.0, nms_iou=TTA_NMS_IOU_THR
        )
        # Preserve the evaluator's usual sv.Detections conversion path.
        preds = SimpleNamespace(xyxy=boxes, confidence=scores, class_id=class_ids)
        pred0 = preds[0] if isinstance(preds, (list, tuple)) else preds

        # Inspect raw prediction structure
        if images_processed < 3:
            print(f"  pred type: {type(preds)} -> first elem type: {type(pred0)}")
            if isinstance(pred0, dict):
                print(f"  pred0 keys: {list(pred0.keys())}")
            elif hasattr(pred0, 'predictions'):
                print("  pred0 has attribute .predictions")
            sys.stdout.flush()

        # Convert predictions into per-class xyxy lists and collect scores
        pred_by_cls = defaultdict(list)  # list of (box, score)

        def label_to_name(lbl):
            if isinstance(lbl, str):
                return lbl
            if isinstance(lbl, (int, np.integer)):
                if hasattr(model, "class_names") and model.class_names is not None and 0 <= lbl < len(model.class_names):
                    return model.class_names[lbl]
                if lbl in cat_id_to_name:
                    return cat_id_to_name[lbl]
            return None

        used_branch = None
        # Branch A: Common RFDETR output: dict with 'scores', 'labels', 'boxes' (xyxy)
        if isinstance(pred0, dict) and all(k in pred0 for k in ("scores", "labels", "boxes")):
            used_branch = 'dict'
            scores = np.array(pred0["scores"]).astype(float)
            labels = pred0["labels"]
            boxes = np.array(pred0["boxes"], dtype=np.float32)
            preds_total += len(scores)
            for s, lbl, box in zip(scores, labels, boxes):
                cls = label_to_name(lbl)
                if cls in CLASSES:
                    if box[2] < box[0] or box[3] < box[1]:
                        box = xywh_to_xyxy(box)
                    pred_by_cls[cls].append((box.astype(np.float32), float(s)))

        # Branch B: Supervision Detections (sv.Detections) from model.predict
        elif hasattr(pred0, "xyxy") and hasattr(pred0, "class_id"):
            used_branch = 'supervision.Detections'
            boxes = np.array(pred0.xyxy, dtype=np.float32)
            class_ids = np.array(getattr(pred0, "class_id"))
            if class_ids is None:
                class_ids = np.zeros((len(boxes),), dtype=int)
            scores = getattr(pred0, "confidence", None)
            if scores is None:
                scores = np.ones((len(boxes),), dtype=float)
            else:
                scores = np.array(scores, dtype=float)
            preds_total += len(scores)

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
                    pred_by_cls[cls].append((box.astype(np.float32), float(s)))

        # Branch C: Roboflow/other wrappers exposing an object with `.predictions`
        elif hasattr(pred0, "predictions"):
            used_branch = 'obj.predictions'
            plist = list(getattr(pred0, "predictions"))
            preds_total += len(plist)
            for p in plist:
                cls = label_to_name(getattr(p, "class_name", getattr(p, "label", None)))
                conf = float(getattr(p, "confidence", getattr(p, "score", 1.0)))
                if cls in CLASSES:
                    if hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "width") and hasattr(p, "height"):
                        cx, cy, w, h = p.x, p.y, p.width, p.height
                        x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
                        box = np.array([x1, y1, x2, y2], dtype=np.float32)
                    elif hasattr(p, "bbox"):
                        box = xywh_to_xyxy(p.bbox)
                    else:
                        continue
                    pred_by_cls[cls].append((box, conf))
                else:
                    preds_dropped_by_conf += 1
        else:
            used_branch = 'unrecognized'

        if images_processed < 3:
            print(f"  used_branch: {used_branch}")
            if used_branch == 'supervision.Detections':
                cid_raw = getattr(pred0, 'class_id')
                cid_preview = list(map(int, np.array(cid_raw)[:2])) if cid_raw is not None and len(cid_raw) > 0 else []
                print(f"  raw class_id preview: {cid_preview}")
            print(f"  pred counts by class: {{c: len(pred_by_cls[c]) for c in CLASSES}} -> "
                  f"{ {c: len(pred_by_cls[c]) for c in CLASSES} }")
            for c in CLASSES:
                if pred_by_cls[c]:
                    print(f"    sample {c} pred[0]: box={pred_by_cls[c][0][0]}, score={pred_by_cls[c][0][1]:.3f}")
            sys.stdout.flush()

        # persist predictions (all scores) for this image
        pred_store[img_id] = pred_by_cls

        images_processed += 1
        if images_processed % 100 == 0:
            print(f"[DEBUG] Processed {images_processed} images. preds_total={preds_total}, kept={preds_kept}, dropped_by_conf={preds_dropped_by_conf}")
            sys.stdout.flush()
    except Exception as e:
        print(f"[ERROR] Exception on image id={img_id} path={im_meta.get('file_name','?')}: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        continue

print(f"[DEBUG] DONE: images_processed={images_processed}, preds_total={preds_total}, kept={preds_kept}, dropped_by_conf={preds_dropped_by_conf}")
sys.stdout.flush()

# --- evaluate across thresholds (for F1 etc.) ---

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
            # filter by threshold and sort by score desc for stable greedy matching
            dets_scored = sorted(dets_scored, key=lambda x: x[1], reverse=True)
            dets = [b for (b, s) in dets_scored if s >= THR]

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

# --- metrics + logging ---
def safe_div(a, b):
    return a / b if b > 0 else 0.0

summary_lines = []
summary_lines.append("==== RF-DETR TTA EVAL SUMMARY ====\n")
summary_lines.append("TTA views: original, horizontal, vertical")
summary_lines.append(f"TTA_NMS_IOU_THR: {TTA_NMS_IOU_THR}")
summary_lines.append(f"checkpoint: {checkpoint}")
summary_lines.append(f"ann_file: {ANN_FILE}")
summary_lines.append(f"img_dir: {IMG_DIR}")
summary_lines.append(f"classes: {CLASSES}")
summary_lines.append(f"CONF_THR_LIST: {CONF_THR_LIST}")
summary_lines.append(f"IOU_THR: {IOU_THR}")
summary_lines.append(f"CENTROID_TOL_PX: {CENTROID_TOL_PX}")
summary_lines.append(f"images_processed: {images_processed}")
summary_lines.append(f"preds_total: {preds_total}")
summary_lines.append("")

metrics_dict = {
    "config": {
        "tta_views": ["original", "horizontal", "vertical"],
        "tta_nms_iou_threshold": TTA_NMS_IOU_THR,
        "checkpoint": checkpoint,
        "ann_file": ANN_FILE,
        "img_dir": IMG_DIR,
        "classes": CLASSES,
        "CONF_THR_LIST": CONF_THR_LIST,
        "IOU_THR": IOU_THR,
        "CENTROID_TOL_PX": CENTROID_TOL_PX,
    },
    "per_threshold": {},
}

for THR in CONF_THR_LIST:
    TP_thr, FP_thr, FN_thr, TPc_thr, FPc_thr, FNc_thr = results_by_thr[THR]
    summary_lines.append(f"\nThreshold = {THR}")
    summary_lines.append(f"{'Class':<10} {'Prec':>8} {'Recall':>8} {'F1':>8}  (TP/FP/FN)")
    thr_metrics = {"per_class": {}}

    for cls in CLASSES:
        prec = safe_div(TP_thr[cls], TP_thr[cls] + FP_thr[cls])
        rec  = safe_div(TP_thr[cls], TP_thr[cls] + FN_thr[cls])
        f1   = safe_div(2*prec*rec, (prec + rec)) if (prec+rec) > 0 else 0.0
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
        rec  = safe_div(TPc_thr[cls], TPc_thr[cls] + FNc_thr[cls])
        f1   = safe_div(2*prec*rec, (prec + rec)) if (prec+rec) > 0 else 0.0
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

    metrics_dict["per_threshold"][str(THR)] = thr_metrics

# --- mAP at fixed IOU_THR (AP50) ---
ap_per_class = {}
for cls in CLASSES:
    ap, recs, pres = compute_ap_for_class(pred_store, gt_by_img_cls, cls, IOU_THR)
    ap_per_class[cls] = ap

mAP = float(np.mean(list(ap_per_class.values()))) if len(ap_per_class) > 0 else 0.0

summary_lines.append("\n================ mAP summary ================")
summary_lines.append(f"IoU threshold: {IOU_THR}")
print("\n================ mAP summary ================")
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

# --- COCO-style mAP@[0.50:0.95] ---
IOU_LIST = [round(x, 2) for x in np.arange(0.50, 0.951, 0.05)]

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

print("\n============= COCO mAP@[0.50:0.95] =============")
summary_lines.append("\n============= COCO mAP@[0.50:0.95] =============")
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
    line = f"AP[{cls}]@[0.50:0.95] = {avg_per_class[cls]:.4f}"
    print(line)
    summary_lines.append(line)
line = f"mAP@[0.50:0.95] = {COCO_mAP:.4f}"
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

# --- write logs to disk ---

summary_path = os.path.join(RESULTS_DIR, f"summary_MODEL_{CHECKPOINT_NAME}__GT{GT_ANNOTATION}.txt")
with open(summary_path, "w") as f:
    f.write("\n".join(summary_lines))

metrics_path = os.path.join(RESULTS_DIR, f"metrics_MODEL_{CHECKPOINT_NAME}__GT{GT_ANNOTATION}.json")

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
