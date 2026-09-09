"""
Run RF-DETR Medium once per raw lensfree image and export detections as JSON.

Set CHECKPOINT_PATH, IMAGE_ROOT, OUTPUT_FILE, RESOLUTION (match training),
and CONF_THRESHOLD. Images are loaded as RGB without ASM, flips, or extra NMS.
No ground-truth annotations are needed. Sources remain unchanged.

Recursively discover images excluding hidden files/directories. Save settings,
class names, and every image's filename, dimensions, and detections, including
empty results. Each detection has bbox_xyxy in original pixels, score, class_id,
and class_name. No track IDs, metrics, videos, or image overlays are generated.
Existing output files are refused; read or prediction failures stop the run.
Importing this module does not run inference. GT-free tracking scripts reuse
predict_image() and provide their own video and track-table outputs.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

CHECKPOINT_PATH = Path(r'D:\cw_rf_detr_dataset_raw\output\RAW_20x20\checkpoint_best_ema.pth')
IMAGE_ROOT = Path(r'D:\cw_rf_detr_dataset_raw\valid')
OUTPUT_FILE = Path(r'D:\cw_rf_detr_dataset_raw\inference\raw_20x20_predictions.json')
RESOLUTION = 512
CONF_THRESHOLD = 0.4
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def predict_image(model, image, confidence=0.4):
    """Run one RGB prediction; return boxes, scores, and class IDs without NMS."""
    if not 0 <= confidence <= 1:
        raise ValueError('Confidence must be in [0, 1]')
    prediction = model.predict(image.convert('RGB'), threshold=confidence)
    if isinstance(prediction, (tuple, list)):
        if len(prediction) != 1:
            raise ValueError('Expected predictions for exactly one image')
        prediction = prediction[0]
    boxes = np.asarray(prediction.xyxy, dtype=float).reshape(-1, 4)
    if len(boxes) == 0:
        return boxes, np.empty(0), np.empty(0, dtype=int)
    if prediction.confidence is None or prediction.class_id is None:
        raise ValueError('Model returned detections without confidence or class IDs')
    scores = np.asarray(prediction.confidence, dtype=float)
    ids = np.asarray(prediction.class_id, dtype=int)
    if len(scores) != len(boxes) or len(ids) != len(boxes):
        raise ValueError('Model output lengths do not match')
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError('Non-finite model output')
    valid = (scores >= confidence) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[valid], scores[valid], ids[valid]


def list_images(root):
    """List supported images in relative-filename order, excluding dot artefacts."""
    if not root.is_dir():
        raise NotADirectoryError(root)
    return sorted(p for p in root.rglob('*') if p.is_file()
                  and p.suffix.lower() in IMAGE_EXTENSIONS
                  and not any(part.startswith('.') for part in p.relative_to(root).parts))


def run_inference(model, image_root, output_file, confidence=0.4,
                  checkpoint=None, resolution=None):
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
                         'confidence_threshold': confidence},
              'class_names': {str(k): v for k, v in names.items()}, 'images': []}
    for number, path in enumerate(paths, 1):
        with Image.open(path) as image:
            width, height = image.size
            boxes, scores, ids = predict_image(model, image, confidence)
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
    run_inference(model, IMAGE_ROOT, OUTPUT_FILE, CONF_THRESHOLD,
                  CHECKPOINT_PATH, RESOLUTION)
