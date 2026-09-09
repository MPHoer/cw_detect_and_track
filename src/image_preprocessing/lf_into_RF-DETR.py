"""
Build a detection dataset from lensfree holograms and existing COCO labels.

This script prepares the images and annotations used for RF-DETR training
or evaluation. It reconstructs each lensfree hologram into a three-channel
image and combines per-sequence COCO files into one annotation file per
configured dataset split. It does not train a model or track cells.

The source annotations must already describe the lensfree images in
their existing coordinate system, for example after the brightfield-to-
lensfree annotation mapping step. No image alignment or bounding-box
transformation is performed here.

Image preparation:
- Load each source image as grayscale.
- Reconstruct its complex optical field using fringe's AngularSpectrumSolver.
- Use pixel_size=2.8, wavelength=0.628, and propagation distance=-2100
  with the current settings. These values must use consistent length units.
  Only the first entry in heights is used, with its sign reversed.
- Extract amplitude and phase, then subtract the calculated phase background.
- Normalize amplitude by its per-image maximum and encode phase using
  sine and cosine, producing three channels scaled to 0–255.
- Invert all channels and save the result as an 8-bit image.

The array channels are [amplitude, sin(phase), cos(phase)] before inversion.
Although the variable is named 'rgb', cv2.imwrite interprets this array
as BGR. Reading the saved file as RGB therefore gives the reversed channel
order. Preserve this convention when reproducing the original model inputs.

Dataset preparation:
- Read SRC_COCO_NAME from each selected sequence under SRC_ROOT.
- Resolve source folder names by replacing '_xy' with 'xy'.
- Save reconstructed images under DST_ROOT/<split>/<sequence>/.
- Assign new image and annotation IDs, starting at 1 within each split,
  and update annotation image references accordingly.
- Write split-relative image paths and merged labels to
  <split>/_annotations.coco.json.

Bounding boxes, category IDs, track IDs, and other annotation fields
remain unchanged. Image metadata, including dimensions, is copied from
the source without verification. Track and video IDs are not made unique
across sequences.

Categories are taken from the first sequence, with missing supercategory
fields added. Category consistency across sequences is assumed rather
than checked. Other top-level COCO fields, such as info and licenses,
are not copied. No dummy category is added by this script.

Only splits enabled in SPLITS are generated. As currently configured,
the six hold_out sequences populate 'valid'; 'train' and 'test' are
disabled, despite comments and the final printed directory example.
An enabled split with an empty sequence list receives an empty COCO file.

Source files remain unchanged. Existing destination images and annotation
files are overwritten, but stale destination files are not removed.
Missing source folders, annotation files, or images stop execution and
may leave partially generated output.

Configure paths, sequence lists, SPLITS, and reconstruction parameters
before running. Requires OpenCV, NumPy, and the fringe implementation
providing fringe.solvers.AngularSpectrum.AngularSpectrumSolver.
"""

import json
import shutil
from pathlib import Path
from typing import List
import cv2
import numpy as np
from numpy import angle, pi
from fringe.solvers.AngularSpectrum import AngularSpectrumSolver as AsSolver

# -----------------------------
# CONFIG
# -----------------------------

# Source root with your original dirs
SRC_ROOT = Path(r"D:\all")

# Target root for RF-DETR style dataset
DST_ROOT = Path(r"D:\rf_detr_dataset")  # <- change if you like

# Your splits
hold_out = [
    # A549 (2)
    "20230323_livedead_A549_xy3_0_0",
    "20230324_A549_livedead_xy6_512_512",

    # 3T3 (2)
    "20230322_3t3_livedead_xy5_0_0",
    "20230328_livedead_3T3_xy4_512_512",

    # HFF (2)
    "20230414_HFF_livedead_xy1_0_0",
    "20230412_HFF_livedead_xy1_512_512",
]

train = [
    # # A549
    # "20230323_livedead_A549_xy2_0_512",
    # "20230323_livedead_A549_xy2_0_0",
    # "20230323_livedead_A549_xy5_512_512",
    # "20230323_livedead_A549_xy6_0_0",
    # "20230323_livedead_A549_xy6_0_512",
    # "20230323_livedead_A549_xy6_512_0",
    # "20230324_A549_livedead_xy2_0_512",
    # "20230324_A549_livedead_xy6_0_512",

    # # 3T3
    # "20230328_livedead_3T3_xy6_0_512",
    # "20230328_livedead_3T3_xy5_512_512",
    # "20230328_livedead_3T3_xy2_512_512",
    # "20230328_livedead_3T3_xy3_512_512",

    # # HFF
    # "20230414_HFF_livedead_xy1_512_512",
    # "20230414_HFF_livedead_xy1_0_512",
    # "20230414_HFF_livedead_xy1_512_0",
    # "20230412_HFF_livedead_xy1_0_0",
    # "20230412_HFF_livedead_xy1_0_512",
    # "20230412_HFF_livedead_xy1_512_0",
]

# Map split names that RF-DETR expects to your lists
# We now use hold_out as TEST, and keep VALID empty.
SPLITS = {
    #"train": train,
    "valid": hold_out,       # no validation set, just train + test
    #"test": hold_out,  # fixed test set for final evaluation
}

# Name of the per-video coco file
SRC_COCO_NAME = "coco_corrected_191125.json"

# Name RF-DETR expects in each split directory
DST_COCO_NAME = "_annotations.coco.json"

# -----------------------------
# RECONSTRUCTION PARAMS
# -----------------------------

pixel_size: float = 2.8          # µm? doesn't matter here, just keep consistent
wavelength = 628.0 * 1e-3        # same units as height
heights = [2100]                 # you can add more, we currently use the first / only one


def reconstruct_hologram(src_img_path: Path) -> np.ndarray:
    """
    Load a hologram image, reconstruct field using Angular Spectrum method,
    and return an 8-bit RGB image (amp, sin_phase, cos_phase) as in your code.
    """
    hologram = cv2.imread(str(src_img_path))
    if hologram is None:
        raise FileNotFoundError(f"Failed to read image: {src_img_path}")
    hologram = cv2.cvtColor(hologram, cv2.COLOR_BGR2GRAY)

    # Initialize solver ASM
    pad = float(np.mean(hologram))
    asm_solver = AsSolver(
        shape=hologram.shape,
        dr=pixel_size,
        is_batched=False,
        padding="same",
        pad_fill_value=pad,
        backend="Numpy",
    )

    # We'll just use the first height in the list (you can extend this if you want z-stacks)
    height = heights[0]
    height = -1.0 * float(height)

    # Angular Spectrum Method
    position_in_wave_cycle: float = np.round((height / wavelength) % 1, 3)

    field_reconstructed = asm_solver.solve(hologram, 2 * np.pi / wavelength, height)
    amplitude, phase = np.abs(field_reconstructed), angle(field_reconstructed)

    # Background correction for phase
    # For numerical stability
    if position_in_wave_cycle == 0.5:
        height += 0.03
        position_in_wave_cycle = 0.53

    if position_in_wave_cycle == 0.5:
        background_value = -pi
    elif position_in_wave_cycle < 0.5:
        background_value: float = position_in_wave_cycle * 2 * np.pi
    else:
        background_value: float = -np.pi + (position_in_wave_cycle - 0.5) * 2 * np.pi

    phase = phase - background_value

    # convert to right range
    amp = amplitude / np.amax(amplitude) * 255.0
    sin_phase = (np.sin(phase) + 1.0) * (255.0 / 2.0)
    cos_phase = (np.cos(phase) + 1.0) * (255.0 / 2.0)

    rgb = 255.0 - np.stack([amp, sin_phase, cos_phase], axis=-1)  # inverted as in your code
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return rgb


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def logical_to_actual_dir_name(logical_name: str) -> str:
    """
    Real dirs miss '_' before 'xy', e.g.
    '20230414_HFF_livedead_xy1_512_512' -> '20230414_HFF_livedeadxy1_512_512'
    """
    return logical_name.replace("_xy", "xy")


def _normalize_categories(cats_raw):
    """
    Ensure every category has a 'supercategory' key so RF-DETR doesn't crash.
    For your 'nucleus' class, we just set supercategory='nucleus' if missing.
    """
    cats = []
    for c in cats_raw:
        c_new = c.copy()
        if "supercategory" not in c_new:
            c_new["supercategory"] = c_new.get("name", "nucleus")
        cats.append(c_new)
    return cats


def merge_split(split_name: str, logical_dirs: List[str]):
    """
    For one split ('train', 'valid', 'test'):
    - read all source COCO files
    - reconstruct and save their images into DST_ROOT/split_name/<video_dir>/
    - write merged COCO into DST_ROOT/split_name/_annotations.coco.json
    """
    split_dir = DST_ROOT / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    if not logical_dirs:
        # Create an empty-ish coco file to keep structure consistent
        empty_coco = {"images": [], "annotations": [], "categories": []}
        with open(split_dir / DST_COCO_NAME, "w") as f:
            json.dump(empty_coco, f)
        print(f"[{split_name}] No videos - created empty structure.")
        return

    merged_images = []
    merged_annotations = []
    categories = None

    next_image_id = 1
    next_ann_id = 1

    for logical_name in logical_dirs:
        actual_name = logical_to_actual_dir_name(logical_name)
        src_video_dir = SRC_ROOT / actual_name

        if not src_video_dir.exists():
            raise FileNotFoundError(f"Source dir not found: {src_video_dir}")

        src_coco_path = src_video_dir / SRC_COCO_NAME
        if not src_coco_path.exists():
            raise FileNotFoundError(f"COCO file not found: {src_coco_path}")

        with open(src_coco_path, "r") as f:
            coco = json.load(f)

        # Keep categories from first file, assume consistent, but normalize them
        if categories is None:
            raw_cats = coco.get("categories", [])
            categories = _normalize_categories(raw_cats)
        else:
            # Could assert consistency here if you want to be strict.
            pass

        image_id_map = {}

        # Reconstruct & save images
        for img in coco["images"]:
            old_img_id = img["id"]
            new_img_id = next_image_id
            image_id_map[old_img_id] = new_img_id

            src_img_path = src_video_dir / img["file_name"]

            if not src_img_path.exists():
                raise FileNotFoundError(f"Image not found: {src_img_path}")

            # Destination path (keep video dirs separate)
            rel_img_path = Path(actual_name) / img["file_name"]
            dst_img_path = split_dir / rel_img_path
            dst_img_path.parent.mkdir(parents=True, exist_ok=True)

            # --- RECONSTRUCTION INSTEAD OF COPY ---
            rgb = reconstruct_hologram(src_img_path)
            # Save reconstructed image
            cv2.imwrite(str(dst_img_path), rgb)

            # Update COCO image entry
            new_img = img.copy()
            new_img["id"] = new_img_id
            # Use path relative to split dir in COCO file_name
            new_img["file_name"] = str(rel_img_path).replace("\\", "/")

            merged_images.append(new_img)
            next_image_id += 1

        # Remap annotations to new image & annotation ids
        for ann in coco["annotations"]:
            new_ann = ann.copy()
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = image_id_map[ann["image_id"]]
            merged_annotations.append(new_ann)
            next_ann_id += 1

        print(f"[{split_name}] processed & reconstructed video dir: {actual_name}")

    merged_coco = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": categories if categories is not None else [],
        # RF-DETR only needs these fields.
    }

    dst_coco_path = split_dir / DST_COCO_NAME
    with open(dst_coco_path, "w") as f:
        json.dump(merged_coco, f)

    print(
        f"[{split_name}] wrote {len(merged_images)} images, "
        f"{len(merged_annotations)} annotations to {dst_coco_path}"
    )


# -----------------------------
# MAIN
# -----------------------------


def main():
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    for split_name, logical_dirs in SPLITS.items():
        print(f"\n=== Building split: {split_name} ===")
        merge_split(split_name, logical_dirs)

    print("\nDone. Final structure should look like:")
    print(str(DST_ROOT))
    print("  train/")
    print("    _annotations.coco.json")
    print("    <video_dir1>/cycle_0001.png ...  (reconstructed)")
    print("  valid/  (empty, just a placeholder)")
    print("    _annotations.coco.json")
    print("  test/")
    print("    _annotations.coco.json")
    print("    <hold_out_video_dir>/cycle_0001.png ...  (reconstructed)")


if __name__ == "__main__":
    main()
