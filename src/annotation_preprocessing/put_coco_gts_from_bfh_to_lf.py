"""
Reuse brightfield ground-truth annotations for matching lensfree images.

The ground truth is stored in COCO JSON files in the brightfield (BF)
dataset folders. This script creates a corresponding annotation file
in each lensfree (LF) folder, so the existing labels can be used with
the lensfree images for subsequent dataset preparation or evaluation.

This assumes that BF and LF frames with the same frame number correspond
in time and already share the same spatial coordinates. The script does
not align images, transform annotation coordinates, or verify alignment.

For each dataset listed in DATASETS:
- Read BF_ROOT/<dataset>/coco_gt.json.
- Find the corresponding LF folder by replacing the first '_xy' in
  the dataset name with 'xy'.
- Extract frame numbers from BF filenames using 'img_XXXX', falling
  back to the last group of digits in the filename.
- Keep only BF image records whose frame number also exists among
  the LF folder's cycle_*.png files.
- Change retained image filenames to 'cycle_XXXX.png', using at least
  four digits, and remove annotations belonging to dropped images.
- Write the result to LF_COCO_OUT_NAME inside the LF folder.

Retained image IDs, dimensions, bounding boxes, segmentations, class
labels, track IDs, and other annotation fields remain unchanged.
Source information is added to the output's 'info' metadata.

The source JSON and all image files remain unchanged. Existing output
JSON files are overwritten without a backup. Missing LF folders,
sequences without matching LF filenames, and missing BF annotation
files are reported and skipped.

Configure DATASETS, BF_ROOT, LF_ROOT, and the annotation filenames
before running. LF images must follow the generated filename convention.
"""

import json
import re
from pathlib import Path

from tqdm import tqdm

# =======================
# CONFIG
# =======================

DATASETS = [
    # "20230323_livedead_A549_xy3_0_0",
    # "20230323_livedead_A549_xy5_512_512",
    # "20230323_livedead_A549_xy6_0_0",
    # "20230323_livedead_A549_xy6_0_512",
    # "20230323_livedead_A549_xy6_512_0",
    # "20230324_A549_livedead_xy2_0_512",
    # "20230324_A549_livedead_xy6_0_512",
    # "20230324_A549_livedead_xy6_512_512",
    # "20230328_livedead_3T3_xy6_0_512",
    # "20230328_livedead_3T3_xy5_512_512",
    # "20230328_livedead_3T3_xy2_512_512",
    # "20230322_3t3_livedead_xy5_0_0",
    # "20230328_livedead_3T3_xy4_512_512",
    # "20230328_livedead_3T3_xy3_512_512",
    # "20230414_HFF_livedead_xy1_512_512",
    # "20230414_HFF_livedead_xy1_0_0",
    # "20230414_HFF_livedead_xy1_0_512",
    # "20230414_HFF_livedead_xy1_512_0",
    # "20230412_HFF_livedead_xy1_0_0",
    # "20230412_HFF_livedead_xy1_0_512",
    # "20230412_HFF_livedead_xy1_512_0",
    # "20230412_HFF_livedead_xy1_512_512",
    "20230323_livedead_A549_xy2_0_512",
    "20230323_livedead_A549_xy2_0_0"
]

BF_ROOT = Path(r"D:\aligned_basic_corrected")
LF_ROOT = Path(r"D:\all")

BF_COCO_NAME = "coco_gt.json"
LF_COCO_OUT_NAME = "coco_corrected_191125.json"


# =======================
# HELPERS
# =======================

def get_lensfree_dataset_name(bf_name: str) -> str:
    """
    Remove the '_' immediately before 'xy' once:
      '..._livedead_xy6_0_512' -> '..._livedeadxy6_0_512'
      '..._A549_xy3_0_0'       -> '..._A549xy3_0_0'
    """
    if "_xy" not in bf_name:
        return bf_name  # fallback, but all your names have '_xy'
    return bf_name.replace("_xy", "xy", 1)


def load_coco(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"COCO file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_lensfree_indices(lf_dir: Path) -> set[int]:
    """
    Look for cycle_XXXX.png and return the set of integer indices XXXX.
    """
    indices: set[int] = set()
    for img_path in lf_dir.glob("cycle_*.png"):
        m = re.search(r"cycle_(\d+)", img_path.name)
        if m:
            indices.add(int(m.group(1)))
    return indices


def get_bf_index_from_filename(file_name: str) -> int | None:
    """
    Extract BF frame index from a filename that contains 'img_XXXX'.
    Falls back to last group of digits if that pattern is missing.
    """
    base = Path(file_name).name
    m = re.search(r"img_(\d+)", base)
    if m:
        return int(m.group(1))

    # fall back: last digit group
    m = re.search(r"(\d+)(?!.*\d)", base)
    if m:
        return int(m.group(1))

    return None


def make_lensfree_coco_for_dataset(dataset: str):
    # ---- Paths ----
    bf_dir = BF_ROOT / dataset
    bf_coco_path = bf_dir / BF_COCO_NAME

    lf_dataset_name = get_lensfree_dataset_name(dataset)
    lf_dir = LF_ROOT / lf_dataset_name
    lf_coco_path = lf_dir / LF_COCO_OUT_NAME

    print(f"\n=== Dataset: {dataset} ===")
    print(f"BF COCO:      {bf_coco_path}")
    print(f"Lensfree dir: {lf_dir}")
    print(f"LF COCO out:  {lf_coco_path}")

    if not lf_dir.exists():
        print(f"  [WARNING] Lensfree directory does not exist, skipping.")
        return

    # ---- Load BF COCO ----
    coco = load_coco(bf_coco_path)

    images = coco.get("images", [])
    anns = coco.get("annotations", [])

    # ---- Get which frames exist in lensfree ----
    lf_indices = get_lensfree_indices(lf_dir)
    if not lf_indices:
        print(f"  [WARNING] No 'cycle_*.png' found in {lf_dir}, skipping.")
        return

    print(f"  Lensfree frames found: {len(lf_indices)} (min={min(lf_indices)}, max={max(lf_indices)})")

    # ---- Filter & remap images ----
    new_images = []
    valid_image_ids = set()

    for img in tqdm(images, desc="  Processing images"):
        bf_idx = get_bf_index_from_filename(img["file_name"])
        if bf_idx is None:
            print(f"    [WARNING] Could not parse BF index from {img['file_name']}, dropping this image.")
            continue

        if bf_idx not in lf_indices:
            # BF frame exists but lensfree frame missing (usually late dead frames)
            continue

        # Keep this image, but point to lensfree filename
        img_new = dict(img)  # shallow copy
        img_new["file_name"] = f"cycle_{bf_idx:04d}.png"
        new_images.append(img_new)
        valid_image_ids.add(img["id"])

    print(f"  Kept {len(new_images)} images out of {len(images)}")

    # ---- Filter annotations ----
    new_anns = [ann for ann in anns if ann.get("image_id") in valid_image_ids]
    print(f"  Kept {len(new_anns)} annotations out of {len(anns)}")

    # ---- Build new COCO dict ----
    new_coco = dict(coco)  # shallow copy of top-level
    new_coco["images"] = new_images
    new_coco["annotations"] = new_anns

    # Optionally: note the modality mapping in the top-level metadata
    info = new_coco.get("info", {}) or {}
    info["cross_modality_source"] = str(bf_coco_path)
    info["cross_modality_note"] = (
        "Images remapped from BF 'img_XXXX.png' to lensfree 'cycle_XXXX.png', "
        "dropping frames not present in lensfree sequence."
    )
    new_coco["info"] = info

    # ---- Save in lensfree dir ----
    lf_coco_path.parent.mkdir(parents=True, exist_ok=True)
    with lf_coco_path.open("w", encoding="utf-8") as f:
        json.dump(new_coco, f, indent=2)
    print("  -> Saved.")


def main():
    for dataset in DATASETS:
        try:
            make_lensfree_coco_for_dataset(dataset)
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")


if __name__ == "__main__":
    main()
