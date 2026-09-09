"""
Train RF-DETR Medium on prepared images using a selected annotation variant.

Configure before running:
- dataset_path: dataset root containing train/, valid/, and test/.
- SELECT_GT_VARIANT: choose which annotations to activate:
    "perf_gt"   -> _perfect_gt_annotations.coco.json
    "perf_gt20" -> _perfect_gt_annotations_20x20.coco.json
    "old_gt"    -> _annotations_oldpoints.coco.json
    "20x20"     -> _annotations_20x20.coco.json
- SELECT_ASM: keep True; the False case does not define the required paths.
  This flag selects the dataset path and run name; it performs no reconstruction.
- model.train(...): adjust epochs, batch_size, grad_accum_steps, lr,
  device, and W&B logging settings as needed.
- RFDETRMedium(resolution=512): change the model input resolution if needed.

Before training, the annotation file corresponding to SELECT_GT_VARIANT is
copied to _annotations.coco.json in each of train/, valid/, and test/.
An existing _annotations.coco.json is removed and replaced without creating
a backup.

Missing split directories or missing annotation files produce warnings and
are skipped. Training still proceeds, so a skipped split may retain previously
active annotations if _annotations.coco.json already exists there.

Train at resolution 512 for up to 10 epochs with early stopping enabled,
batch size 2, gradient accumulation over 8 steps, and learning rate 1e-4.
The configured device is CUDA, requiring a compatible GPU setup.

Save training outputs under dataset_path/output/ASM_<variant>/ and log
to Weights & Biases. The W&B project name is fixed to 'rf_detr_perf_gt20',
regardless of the selected annotation variant.

Images and annotation variants must already be prepared. This script performs
no hologram reconstruction, annotation generation, annotation correction,
or tracking.
"""

from rfdetr import RFDETRMedium
import os
import shutil
import wandb


def select_annotations(dataset_root: str, variant: str):
    """Activate a selected COCO annotation variant for each dataset split.

    For each of ``train``, ``valid``, and ``test``, copy the annotation file
    corresponding to ``variant`` to ``_annotations.coco.json``.

    Supported variants:
    - ``"perf_gt"``   -> ``_perfect_gt_annotations.coco.json``
    - ``"perf_gt20"`` -> ``_perfect_gt_annotations_20x20.coco.json``
    - ``"old_gt"``    -> ``_annotations_oldpoints.coco.json``
    - ``"20x20"``     -> ``_annotations_20x20.coco.json``

    If ``_annotations.coco.json`` already exists, it is removed before the
    selected annotation file is copied into its place. No backup is created.

    Missing split directories or source annotation files are skipped with a
    warning.

    Args:
        dataset_root: Root directory containing the train/, valid/, and test/
            split directories.
        variant: Annotation variant to activate.

    Raises:
        ValueError: If ``variant`` is not one of the supported variants.
    """
    name_map = {
        "perf_gt": "_perfect_gt_annotations.coco.json",
        "perf_gt20": "_perfect_gt_annotations_20x20.coco.json",
        "old_gt": "_annotations_oldpoints.coco.json",
        "20x20": "_annotations_20x20.coco.json",
    }

    if variant not in name_map:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose one of {list(name_map.keys())}."
        )

    for split in ("train", "valid", "test"):
        split_dir = os.path.join(dataset_root, split)

        if not os.path.exists(split_dir):
            print(f"[warn] Split directory not found: {split_dir}")
            continue

        src = os.path.join(split_dir, name_map[variant])
        dst = os.path.join(split_dir, "_annotations.coco.json")

        if not os.path.exists(src):
            print(f"[warn] Desired annotations for {split} not found: {src}")
            continue

        if os.path.exists(dst):
            os.remove(dst)

        shutil.copy2(src, dst)
        print(f"[info] Using {src} for {split} -> {dst}")


if __name__ == '__main__':
    # Variablen
    SELECT_ASM = False
    SELECT_GT_VARIANT = "20x20" # Choose which annotations to activate for training: one of {"Alex", "bf_hoechst", "hoechst"}

   
    # paths
    if SELECT_ASM == True:
        print(f"ASM: {SELECT_ASM}")
        dataset_path = r"D:\cw_rf_detr_dataset"
        run_name = f"ASM_{SELECT_GT_VARIANT}"

    # paths
    else:
        print(f"ASM: {SELECT_ASM}")
        dataset_path = r"D:\cw_rf_detr_dataset_raw"
        run_name = f"RAW_{SELECT_GT_VARIANT}"
    
    # Choose which set to activate for training in-place per dir
    select_annotations(dataset_path, variant=SELECT_GT_VARIANT)

    # output dir    
    output_path = os.path.join(dataset_path, "output", run_name)
    os.makedirs(output_path, exist_ok=True)

    print(f"Ground Truth: {SELECT_GT_VARIANT}")
    print(f"Datset Path: {dataset_path}")
    print(f"Output Path: {output_path}")

    # get model
    model = RFDETRMedium(resolution=512)

    # train model
    model.train(
        dataset_dir=dataset_path,
        epochs=1,
        batch_size=2,
        grad_accum_steps=8,
        lr=1e-4,
        device="cuda",
        output_dir=output_path,
        early_stopping=True,
        wandb=False,
        project="rf_detr_perf_gt20", #f"rf_detr_{SELECT_GT_VARIANT}",
        run=run_name,
    )
