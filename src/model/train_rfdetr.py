"""
Train RF-DETR Medium on prepared images using a selected annotation variant.

Configure before running:
- dataset_path: dataset root containing train/, valid/, and test/.
- SELECT_GT_VARIANT: choose which annotations to activate:
    "perf_gt"   -> _perfect_gt_annotations.coco.json
    "perf_gt20" -> _perfect_gt_annotations_20x20.coco.json
    "old_gt"    -> _annotations_oldpoints.coco.json
- SELECT_ASM: keep True; the False case does not define the required paths.
  This flag selects the dataset path and run name; it performs no reconstruction.
- model.train(...): adjust epochs, batch_size, grad_accum_steps, lr,
  device, and W&B logging settings as needed.
- RFDETRMedium(resolution=512): change the model input resolution if needed.

Before training, copy the selected annotation file in each split to
_annotations.coco.json, overwriting the active annotations without backup.
Missing splits or variant files produce warnings and are skipped; training
still proceeds, potentially using previously active annotations.

Train at resolution 512 for up to 10 epochs with early stopping enabled,
batch size 2, gradient accumulation over 8 steps, and learning rate 1e-4.
The configured device is CUDA, requiring a compatible GPU setup.

Save training outputs under dataset_path/output/ASM_<variant>/ and log
to Weights & Biases. The W&B project name is fixed to 'rf_detr_perf_gt20',
regardless of the selected annotation variant.

Images must already be prepared for the model. This script performs no
hologram reconstruction, annotation generation, or tracking.
"""

from rfdetr import RFDETRMedium
import os
import shutil
import wandb


def select_annotations(dataset_root: str, variant: str):
    """Activate one of the available annotations by copying it to `_annotations.coco.json`.
    `variant` must be one of: "Alex", "bf_hoechst", "hoechst".
    - Alex      -> `_annotation_alex.json`
    - bf_hoechst-> `_bf_hoechst_coco.json`
    - hoechst   -> `_hoechst_coco_corrected.json`
    If `_annotations.coco.json` already exists, it will be overwritten (no backups).
    """
    name_map = {
        "perf_gt": "_perfect_gt_annotations.coco.json",
        "perf_gt20": "_perfect_gt_annotations_20x20.coco.json",
        "old_gt": "_annotations_oldpoints.coco.json",
    }
    if variant not in name_map:
        raise ValueError(f"Unknown variant '{variant}'. Choose one of {list(name_map.keys())}.")

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
    SELECT_ASM = True
    SELECT_GT_VARIANT = "old_gt" # Choose which annotations to activate for training: one of {"Alex", "bf_hoechst", "hoechst"}

   

    # paths
    if SELECT_ASM == True:
        print(f"ASM: {SELECT_ASM}")
        dataset_path = r"D:\rf_detr_dataset"
        run_name = f"ASM_{SELECT_GT_VARIANT}"
    
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
        epochs=10,
        batch_size=2,
        grad_accum_steps=8,
        lr=1e-4,
        device="cuda",
        output_dir=output_path,
        early_stopping=True,
        wandb=True,
        project="rf_detr_perf_gt20", #f"rf_detr_{SELECT_GT_VARIANT}",
        run=run_name,
    )
