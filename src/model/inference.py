"""
Run RF-DETR Medium on ordinary lensfree images, optionally with flip TTA.

Set CHECKPOINT_PATH to a model trained on the same raw-image representation,
IMAGE_ROOT to a sequence folder or a root containing sequence subfolders,
and OUTPUT_FILE to a new JSON path. Set RESOLUTION to the training resolution,
CONF_THRESHOLD to the detection cutoff, and USE_TTA to enable/disable TTA.
NMS_IOU_THRESHOLD controls duplicate suppression across the three TTA views.
No ASM reconstruction or ground-truth annotations are used.

With TTA, predict on the original, horizontally flipped, and vertically flipped
RGB images. Map boxes back to original pixel coordinates and apply class-aware
non-maximum suppression (NMS), keeping the highest-scoring overlapping box.
Scores are not averaged or boosted. With USE_TTA=False, return the original
model detections without additional NMS. TTA takes three inference passes.

Recursively read supported image files, excluding hidden files/directories.
Write JSON with run settings, the model class mapping, and one record per image
(including images with zero detections). Each record contains relative filename,
image dimensions, and detections with bbox_xyxy, score, class_id, and class_name.
There are no track IDs. Coordinates refer to the original, unflipped image.

Output parents are created if needed; an existing OUTPUT_FILE is refused.
Unreadable images or inference errors stop the run instead of silently omitting
frames. Sources remain unchanged. This script does not evaluate or track;
existing tracking scripts still perform their own inference and do not import
this JSON. predict_image() can be reused by future detector adapters.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

CHECKPOINT_PATH = Path(r'D:\cw_rf_detr_dataset_raw\output\RAW_20x20\checkpoint_best_ema.pth')
IMAGE_ROOT = Path(r'D:\cw_rf_detr_dataset_raw\valid')
OUTPUT_FILE = Path(r'D:\cw_rf_detr_dataset_raw\inference\raw_20x20_tta.json')
RESOLUTION = 512
USE_TTA = True
CONF_THRESHOLD = 0.4
NMS_IOU_THRESHOLD = 0.5
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def restore_boxes(boxes, width, height, view):
    """Undo a horizontal or vertical flip for continuous xyxy box coordinates."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4).copy()
    if view == 'horizontal':
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
    elif view == 'vertical':
        boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
    elif view != 'original':
        raise ValueError(f'Unknown view: {view}')
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    return boxes


def class_aware_nms(boxes, scores, class_ids, iou_threshold):
    """Return indices of highest-score boxes, suppressing same-class IoU > threshold."""
    order = np.argsort(-scores, kind='stable')
    keep = []
    while len(order):
        index = order[0]
        keep.append(int(index))
        rest = order[1:]
        if not len(rest):
            break
        intersection_wh = np.maximum(0, np.minimum(boxes[index, 2:], boxes[rest, 2:])
                                     - np.maximum(boxes[index, :2], boxes[rest, :2]))
        intersection = intersection_wh.prod(axis=1)
        area = (boxes[index, 2:] - boxes[index, :2]).prod()
        other_area = (boxes[rest, 2:] - boxes[rest, :2]).prod(axis=1)
        union = area + other_area - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        suppress = (class_ids[rest] == class_ids[index]) & (iou > iou_threshold)
        order = rest[~suppress]
    return np.asarray(keep, dtype=int)


def predict_image(model, image, use_tta=True, confidence=0.4, nms_iou=0.5):
    """Return original-coordinate boxes, scores, and IDs from RF-DETR sv.Detections."""
    if not 0 <= confidence <= 1 or not 0 <= nms_iou <= 1:
        raise ValueError('Confidence and NMS IoU must be in [0, 1]')
    image = image.convert('RGB')
    views = [('original', image)]
    if use_tta:
        views += [('horizontal', image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)),
                  ('vertical', image.transpose(Image.Transpose.FLIP_TOP_BOTTOM))]
    all_boxes, all_scores, all_ids = [], [], []
    for view, transformed in views:
        prediction = model.predict(transformed, threshold=confidence)
        if isinstance(prediction, (tuple, list)):
            if len(prediction) != 1:
                raise ValueError('Expected predictions for exactly one image')
            prediction = prediction[0]
        boxes = restore_boxes(prediction.xyxy, image.width, image.height, view)
        if len(boxes) == 0:
            continue
        if prediction.confidence is None or prediction.class_id is None:
            raise ValueError('Model returned detections without confidence or class IDs')
        scores = np.asarray(prediction.confidence, dtype=float)
        ids = np.asarray(prediction.class_id, dtype=int)
        if len(scores) != len(boxes) or len(ids) != len(boxes):
            raise ValueError('Model output lengths do not match')
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError('Non-finite model output')
        valid = (scores >= confidence) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        all_boxes.append(boxes[valid]); all_scores.append(scores[valid]); all_ids.append(ids[valid])
    if not all_boxes:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    boxes, scores, ids = np.concatenate(all_boxes), np.concatenate(all_scores), np.concatenate(all_ids)
    if use_tta:
        keep = class_aware_nms(boxes, scores, ids, nms_iou)
        boxes, scores, ids = boxes[keep], scores[keep], ids[keep]
    return boxes, scores, ids


def list_images(root):
    """List supported images in relative-filename order, excluding dot artefacts."""
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted(p for p in root.rglob('*') if p.is_file()
                  and p.suffix.lower() in IMAGE_EXTENSIONS
                  and not any(part.startswith('.') for part in p.relative_to(root).parts))


def run_inference(model, image_root, output_file, use_tta=True, confidence=0.4,
                  nms_iou=0.5, checkpoint=None, resolution=None):
    """Predict a folder and save every image, including empty results, as JSON."""
    image_root, output_file = Path(image_root), Path(output_file)
    if output_file.exists():
        raise FileExistsError(f'Choose a new output file: {output_file}')
    paths = list_images(image_root)
    if not paths:
        raise ValueError(f'No supported images found in {image_root}')
    names = getattr(model, 'class_names', {}) or {}
    if not isinstance(names, dict):
        raise ValueError('Expected RF-DETR class_names as an ID-to-name dictionary')
    result = {'config': {'checkpoint': str(checkpoint) if checkpoint else None,
                         'image_root': str(image_root), 'resolution': resolution,
                         'tta_views': ['original', 'horizontal', 'vertical'] if use_tta else ['original'],
                         'confidence_threshold': confidence,
                         'nms_iou_threshold': nms_iou if use_tta else None},
              'class_names': {str(k): v for k, v in names.items()}, 'images': []}
    for number, path in enumerate(paths, 1):
        with Image.open(path) as image:
            width, height = image.size
            boxes, scores, ids = predict_image(model, image, use_tta, confidence, nms_iou)
        detections = [{'bbox_xyxy': box.tolist(), 'score': float(score),
                       'class_id': int(cid), 'class_name': names.get(int(cid))}
                      for box, score, cid in zip(boxes, scores, ids)]
        result['images'].append({'file_name': path.relative_to(image_root).as_posix(),
                                 'width': width, 'height': height, 'detections': detections})
        print(f'[{number}/{len(paths)}] {path.name}: {len(detections)} detections', flush=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f'Saved {len(paths)} images to {output_file}')


if __name__ == '__main__':
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f'Set CHECKPOINT_PATH to your trained model: {CHECKPOINT_PATH}')
    if OUTPUT_FILE.exists():
        raise FileExistsError(f'Choose a new OUTPUT_FILE: {OUTPUT_FILE}')
    if not list_images(IMAGE_ROOT):
        raise ValueError(f'No supported images in {IMAGE_ROOT}')
    from rfdetr import RFDETRMedium
    model = RFDETRMedium(pretrain_weights=str(CHECKPOINT_PATH), resolution=RESOLUTION)
    run_inference(model, IMAGE_ROOT, OUTPUT_FILE, USE_TTA, CONF_THRESHOLD,
                  NMS_IOU_THRESHOLD, CHECKPOINT_PATH, RESOLUTION)
