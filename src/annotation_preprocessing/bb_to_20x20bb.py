"""
Create fixed-size bounding-box annotations for RF-DETR training.

Context
-------
This project uses COCO JSON annotations: `images` describes image files,
`categories` defines object classes, and `annotations` links each object
to an image, class, and bounding box in [x, y, width, height] pixel format.

The input is already in COCO format. This script standardises box sizes
to TARGET_SIZE × TARGET_SIZE pixels (20×20 by default). Fixed-size boxes
are a project-specific choice, not an RF-DETR requirement.

Behaviour
---------
Reads `_annotations.coco.json` from each configured split (train, valid,
test) and writes a separate `_annotations_20x20.coco.json`.

Each box is resized around its original centre. Boxes near image edges
are shifted inward to fit, which may move their centres. Dimensions are
reduced only when the image itself is smaller than TARGET_SIZE.

All other annotation fields are preserved, including category IDs,
track IDs, segmentation, and area. The `area` field is not recalculated
and may therefore differ from the new box area. Annotations with a
missing bbox or a bbox whose length is not four are copied unchanged.

Source files and images remain unchanged. Existing destination files
are overwritten; splits without a source annotation file are skipped.

Usage
-----
Configure DATASET_ROOT, SPLITS, and TARGET_SIZE before running.
Changing TARGET_SIZE does not automatically change the output filename.

Before training, activate the generated annotations by copying them to
`_annotations.coco.json` in the prepared training dataset. The recovered
training script expects the variant name
`_perfect_gt_annotations_20x20.coco.json`, so its filename configuration
must be reconciled with this script's output.
"""

import json
from pathlib import Path

# Root of your RF-DETR dataset
DATASET_ROOT = Path(r"D:\rf_detr_dataset")

SPLITS = ["train", "valid", "test"]
SRC_ANN_FILENAME = "_annotations.coco.json"
DST_ANN_FILENAME = "_annotations_20x20.coco.json"

TARGET_SIZE = 20  # 20x20 boxes


def make_20x20_boxes(coco):
    """
    Return a new list of annotations where each bbox is resized to 20x20
    around its original center, clipped to image boundaries.
    """
    images = coco.get("images", [])
    anns = coco.get("annotations", [])

    # Map image_id -> (width, height)
    img_size = {img["id"]: (img["width"], img["height"]) for img in images}

    new_anns = []
    for ann in anns:
        ann_new = ann.copy()
        bbox = ann.get("bbox", None)

        if bbox is None or len(bbox) != 4:
            # Just keep as is if something weird happens
            new_anns.append(ann_new)
            continue

        x, y, w, h = bbox
        img_w, img_h = img_size[ann["image_id"]]

        # Center of original bbox
        cx = x + w / 2.0
        cy = y + h / 2.0

        # Start with desired size
        bw = bh = float(TARGET_SIZE)

        # If image is smaller than TARGET_SIZE in any dimension, shrink accordingly
        if img_w < TARGET_SIZE:
            bw = float(img_w)
        if img_h < TARGET_SIZE:
            bh = float(img_h)

        # New top-left around the same center
        new_x = cx - bw / 2.0
        new_y = cy - bh / 2.0

        # Clip to image boundaries
        if bw > img_w:
            bw = float(img_w)
        if bh > img_h:
            bh = float(img_h)

        new_x = max(0.0, min(new_x, img_w - bw))
        new_y = max(0.0, min(new_y, img_h - bh))

        ann_new["bbox"] = [new_x, new_y, bw, bh]
        new_anns.append(ann_new)

    return new_anns


def process_split(split: str):
    src_path = DATASET_ROOT / split / SRC_ANN_FILENAME
    dst_path = DATASET_ROOT / split / DST_ANN_FILENAME

    if not src_path.exists():
        print(f"[{split}] No source annotation file at {src_path}, skipping.")
        return

    print(f"[{split}] Loading {src_path}")
    with src_path.open("r") as f:
        coco = json.load(f)

    print(f"[{split}] Creating 20x20 bboxes...")
    new_anns = make_20x20_boxes(coco)

    coco_20 = coco.copy()
    coco_20["annotations"] = new_anns

    print(f"[{split}] Writing {dst_path}")
    with dst_path.open("w") as f:
        json.dump(coco_20, f)

    print(f"[{split}] Done.")


def main():
    for split in SPLITS:
        process_split(split)

    print("\nAll splits processed. New files:")
    for split in SPLITS:
        print(f"  {split}/{DST_ANN_FILENAME}")


if __name__ == "__main__":
    main()
