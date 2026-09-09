"""
Patch COCO annotation files for the original RF-DETR training setup.

Add missing dataset metadata and category supercategories.
Add an unused dummy category with ID 2 when absent.

The dummy category was introduced as a workaround for training with
a single real class: nucleus (category ID 1). It increases the category
count to two, allowing the original training setup to retain an output
for class ID 1. This does not imply that two real object classes are
required.

Object annotations and bounding boxes remain unchanged.
Back up each source file to .json.bak, then overwrite it with the
patched content. Repeated runs overwrite the previous backup.
"""

import json
import shutil
from pathlib import Path

# Root of your RF-DETR dataset
DATASET_ROOT = Path("/Volumes/Max-HDD/cw_rf_detr_dataset_raw")

SPLITS = ["train", "valid", "test"]
ANN_FILENAME = "_annotations.coco.json"


def patch_categories(categories):
    """
    - Ensure every category has 'supercategory'
    - Ensure dummy class exists as category_id=2
    """
    new_cats = []
    seen_ids = set()

    for c in categories:
        c_new = c.copy()

        # ensure supercategory exists
        if "supercategory" not in c_new:
            c_new["supercategory"] = c_new.get("name", "nucleus")

        new_cats.append(c_new)
        seen_ids.add(c_new["id"])

    # Add dummy class if missing
    if 2 not in seen_ids:
        dummy = {
            "id": 2,
            "name": "dummy",
            "supercategory": "dummy",
        }
        new_cats.append(dummy)
        print("  Added dummy category (id=2).")

    return new_cats


def patch_coco_file(path: Path):
    if not path.exists():
        print(f"[SKIP] {path} does not exist")
        return

    print(f"\n[PATCH] {path}")

    with path.open("r") as f:
        coco = json.load(f)

    # ---------- info ----------
    if "info" not in coco:
        coco["info"] = {
            "description": "RF-DETR nuclei dataset",
            "version": "1.0"
        }
        print("  Added 'info' field.")

    # ---------- licenses (optional, but nice to have) ----------
    if "licenses" not in coco:
        coco["licenses"] = []
        print("  Added empty 'licenses' list.")

    # ---------- categories ----------
    categories = coco.get("categories", [])
    if not categories:
        print("  No categories found → nothing to patch in categories.")
    else:
        categories = patch_categories(categories)
        coco["categories"] = categories

    # Backup original
    backup_path = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup_path)
    print(f"  Backup saved to {backup_path}")

    # Save patched file
    with path.open("w") as f:
        json.dump(coco, f)
    print("  Patched COCO file saved.")


def main():
    for split in SPLITS:
        ann_path = DATASET_ROOT / split / ANN_FILENAME
        patch_coco_file(ann_path)

    print("\nDone patching all RF-DETR annotation files (info + categories).")


if __name__ == "__main__":
    main()
