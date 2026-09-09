# detect_and_track

Train a detector to locate cells in individual images, then link those detections across a sequence to obtain tracks. The detector predicts **where an object is**; tracking assigns **which object it is over time** and can fill short gaps in its detections.

```text
Your images + object annotations
       │
       ├─ Ordinary images: prepare a COCO dataset directly
       └─ Lensfree holograms: optionally reconstruct with ASM first
       │
       ▼
Train RF-DETR → saved model checkpoint
       │
       ├─ Evaluate: predict boxes and compare with labelled boxes → metrics
       └─ Track: predict boxes, link identities, optionally fill gaps → metrics + videos
```

ASM is optional image preprocessing, not part of RF-DETR or a requirement for tracking. This repository contains original experiment scripts with editable settings, rather than a single automated pipeline.

## 1. Prepare the model input

### Images and folders

Arrange your dataset like this, with a separate annotation JSON for each split:

```text
data/my_dataset/
├── train/
│   ├── _annotations.coco.json
│   └── sequence_A/
│       ├── cycle_0001.png
│       └── cycle_0002.png
├── valid/
│   ├── _annotations.coco.json
│   └── sequence_B/...
└── test/
    ├── _annotations.coco.json
    └── sequence_C/...
```

`train` supplies examples for learning, `valid` supports model selection and tuning, and `test` is reserved for final assessment. Split by whole sequence or experiment to avoid placing nearly identical neighbouring frames in different splits.

Use PNG/JPEG images readable as RGB. Grayscale microscopy images can be represented as three identical channels. Annotations must use the pixel coordinates of the actual saved images; if you crop or resize images yourself, update their labels accordingly. The trainer uses resolution 512 internally, so source images do not have to be manually resized to 512×512.

The sequence subfolders are useful for detection and **required by these tracking scripts' grouping convention**. Filenames in JSON are relative to their split folder, using `/`. The folder layout follows [RF-DETR's COCO dataset convention](https://rfdetr.roboflow.com/learn/train/#dataset-structure); this repository pins RF-DETR 1.3.0, so newer API examples may differ.

### Annotation JSON

Example `_annotations.coco.json` for one image and one nucleus:

```json
{
  "info": {"description": "My cell dataset"},
  "licenses": [],
  "images": [
    {"id": 1, "file_name": "sequence_A/cycle_0001.png",
     "width": 512, "height": 512, "frame_id": 1}
  ],
  "categories": [
    {"id": 1, "name": "nucleus", "supercategory": "nucleus"},
    {"id": 2, "name": "dummy", "supercategory": "dummy"}
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1,
     "bbox": [100, 150, 20, 20], "area": 400,
     "iscrowd": 0, "track_id": 7}
  ]
}
```

- `images` lists filenames, dimensions, and unique image IDs within the split. `frame_id` supplies ordering for tracking.
- `categories` defines the class IDs and names, consistently across splits.
- `annotations` contains one entry per labelled object. Each has a unique annotation ID, an `image_id` referencing its image, and a `category_id` referencing its class.
- `bbox` is **[left x, top y, width, height] in pixels**, not normalised coordinates or two corners. The example occupies x=100…120 and y=150…170.
- `area` describes object area; for box-only labels, use width × height. `iscrowd=0` marks an ordinary individual annotation.
- `track_id` is unnecessary for detector training, but needed for meaningful tracking identity evaluation. Keep it stable for the same cell across frames within a sequence. Tracking assigns its own predicted IDs.

The example preserves this project's single-class `nucleus=1` / unused `dummy=2` convention, used as a compatibility workaround with the original RF-DETR setup. It does not mean two real classes are required. Do not annotate objects as dummy. For other class sets, verify the model/annotation ID mapping instead of blindly applying the helper that reserves ID 2.

Label all target objects, including complete image records for frames containing none. For your own data, export bounding-box labels in COCO format from your annotation tool and check the paths and IDs above. This repository does not contain a general annotation editor or converter for every label format. Boxes need not be 20×20; that was an experimental choice.

### Choose ordinary images or ASM images

**Without ASM:** place your ordinary images and their matching annotations directly in the layout above. Skip `src/image_preprocessing/lf_into_RF-DETR.py`. Train a checkpoint on this representation and use the same representation during evaluation and tracking. The historical ASM checkpoint should not be assumed suitable for ordinary images.

**With ASM:** `src/image_preprocessing/lf_into_RF-DETR.py` reads lensfree images and per-sequence `coco_corrected_191125.json`, reconstructs amplitude/phase channels, and writes split images plus merged `_annotations.coco.json`. Configure `SRC_ROOT`, `DST_ROOT`, the sequence lists, `SPLITS`, and reconstruction parameters. Currently only `valid` is enabled; populate/enable training and test splits yourself. This step needs the original `fringe.solvers.AngularSpectrum` implementation, whose exact distribution/version has not been recovered. Existing prepared ASM images avoid that dependency. Preserve the original saved channel convention described in that file's docstring.

### Build a raw lensfree dataset from sequence folders

Use [build_lensfree_dataset.py](src/image_preprocessing/build_lensfree_dataset.py). It takes source sequence folders and explicit split lists; no existing ASM dataset is needed.

Set these variables at the top of the script:

| Variable | Input |
|---|---|
| `SOURCE_ROOT` | Base directory containing all lensfree sequence folders |
| `TRAIN_DATASETS`, `VALID_DATASETS`, `TEST_DATASETS` | Lists of exact folder names for each split |
| `OUTPUT_ROOT` | Destination; created if absent, or may be an existing directory with no conflicting files |
| `ANNOTATION_IDENTIFIER` | Full filename or case-sensitive substring; default `corrected_191125` |
| `ALLOW_SPLIT_OVERLAP` | Whether to permit sequences shared across splits |

Set `ANNOTATION_IDENTIFIER = 'corrected_191125'` to select names such as `coco_corrected_191125.json` or `sequence_A_corrected_191125.json`. Selection searches non-hidden JSON files directly in each sequence folder. A full filename matches exactly first; otherwise exactly one filename must contain the identifier. The script prints the selection and stops with candidate names on missing or ambiguous matches. `--check-only` applies the same rules. The selected file must still have COCO structure; matching a name does not convert other JSON formats. Original JSON copies remain available alongside the images.

Paths are prefilled for `/Volumes/Max-HDD/cw_aligned_lensfree` and the new `/Volumes/Max-HDD/cw_rf_detr_dataset_raw`. The lists reproduce the existing dataset: 18 training sequences and the same six sequences in both validation and test. `ALLOW_SPLIT_OVERLAP=True` explicitly preserves this historical setup and prints warnings. For an independent final test, edit the lists to be disjoint and set it to `False`.

```bash
python src/image_preprocessing/build_lensfree_dataset.py --check-only
python src/image_preprocessing/build_lensfree_dataset.py
```

`--check-only` validates annotations, references, source file existence, split overlap, and destination conflicts without writing. It does not decode images or verify alignment. The script uses only the Python standard library.

Output per split:

```text
train/
├── _annotations.coco.json          # merged file used for training
├── sequence_A/
│   ├── cycle_0001.png              # raw bytes copied unchanged
│   ├── coco_corrected_191125.json  # original per-sequence JSON copy
│   └── other_annotation.json       # other non-hidden JSONs preserved
└── sequence_B/...
```

Only images referenced by the source COCO are copied; unrelated or unlabelled images are not silently included. All non-hidden JSON files in the selected sequence trees are copied for provenance. Files and directories starting with `.` are excluded, including external-drive artefacts. If a hidden image is listed in COCO, its record and associated labels are omitted from the merged file. Source JSON copies remain unchanged.

Merged image, annotation, and video IDs are made unique per split; image paths gain the sequence directory. Boxes, class definitions, track IDs, and annotation attributes are preserved. Track IDs remain local to their sequence/video. Per-sequence metadata is stored under `source_metadata` because fluorescence thresholds can differ between sequences. No dummy class or fixed-size boxes are added here.

Next prepare the annotation variant and select it using the workflow below. No manual copy to the active filename is needed.

Source correction (2026-09-09): in `20230414_HFF_livedeadxy1_512_512/coco_corrected_191125.json`, the user confirmed image ID 133 (frame ID 132), an unannotated duplicate of `cycle_0133.png`, was incorrect. That record was removed from the external-drive source after backing it up beside the JSON. Image ID 134 and its 21 annotations were retained. The full builder check subsequently passed: 2,646 training images and 866 images each in validation and test. Other copies of the original source may still contain this duplicate; the builder will flag it rather than automatically deleting empty frames.

Continue with the annotation-selection training route below and a new run name such as `RAW_perf_gt20`. The builder refuses to overwrite existing output files; an interrupted copy can leave partial results, so inspect them and use a fresh destination for a retry.

Optional historical annotation helpers:

| File in `src/annotation_preprocessing/` | Purpose |
|---|---|
| `put_coco_gts_from_bfh_to_lf.py` | Transfer existing, already aligned brightfield labels to corresponding LF filenames; no registration is performed. |
| `fix_rf_detr_anno.py` | Add metadata and the historical dummy category in place; backs up to `.json.bak`. |
| `bb_to_20x20bb.py` | Produce `_annotations_20x20.coco.json`; shifts boxes inside image bounds but does not recalculate `area`. |

### Prepare and activate an annotation variant

After building a dataset from your own sequence folders, use this order:

**Build split COCO files → optionally resize boxes → patch the selected variant → activate it through `select_annotations()` at training time.**

The examples below use `D:/my_dataset_raw`; replace it with your output directory in every script. On this Mac, use `/Volumes/Max-HDD/cw_rf_detr_dataset_raw`. Run commands from the repository root.

1. **Build the dataset.** Configure and run `build_lensfree_dataset.py` as described above. Each split now contains `_annotations.coco.json` with the original box sizes. Confirm `--check-only` passes before copying.

2. **Create 20×20 boxes, if wanted.** In `src/annotation_preprocessing/bb_to_20x20bb.py`, set:

   ```python
   DATASET_ROOT = Path("D:/my_dataset_raw")
   SRC_ANN_FILENAME = "_annotations.coco.json"
   DST_ANN_FILENAME = "_perfect_gt_annotations_20x20.coco.json"
   TARGET_SIZE = 20
   ```

   Then run:

   ```bash
   python src/annotation_preprocessing/bb_to_20x20bb.py
   ```

   This output name deliberately matches the trainer's existing `perf_gt20` mapping. It is a historical filename, not a claim that your labels are perfect. The original merged annotation files remain unchanged. The helper shifts boxes inside image bounds and preserves other fields, including `area`; it does not recalculate area.

3. **Patch that variant's metadata before selecting it.** In `src/annotation_preprocessing/fix_rf_detr_anno.py`, set:

   ```python
   DATASET_ROOT = Path("D:/my_dataset_raw")
   ANN_FILENAME = "_perfect_gt_annotations_20x20.coco.json"
   ```

   Then run:

   ```bash
   python src/annotation_preprocessing/fix_rf_detr_anno.py
   ```

   This edits the named variant and saves a `.json.bak` backup. Patch the variant itself: patching only the active `_annotations.coco.json` would lose those changes when the trainer later activates an unpatched variant. The helper's dummy-ID-2 convention is specific to the historical nucleus-ID-1 setup; adapt it for other class definitions.

4. **Select the variant during training.** Set `SELECT_GT_VARIANT = "perf_gt20"` and call `select_annotations(dataset_path, SELECT_GT_VARIANT)` as shown in the next section. The existing mapping copies the patched `_perfect_gt_annotations_20x20.coco.json` to `_annotations.coco.json` in each split. This is the activation step; do not manually duplicate or rename files afterwards.

The three relevant filenames have different roles:

| Filename | Role |
|---|---|
| `_annotations.coco.json` | Active annotations read by training; initially the builder's original boxes, later replaced by the selector |
| `_perfect_gt_annotations_20x20.coco.json` | Named 20×20 variant that you resize, patch, and select |
| `<sequence>/coco_corrected_191125.json` | Unchanged source annotation copy for provenance; not the merged training input |

If you already generated `_annotations_20x20.coco.json` with the helper's old default, you do not need to resize again. Set `ANN_FILENAME` to that filename when patching, and add `"20x20": "_annotations_20x20.coco.json"` to the trainer's `name_map`; select `"20x20"`. Both approaches work, provided the generated, patched, and selected filename is the same.

**To retain variable-size boxes**, skip resizing, patch the builder's `_annotations.coco.json` directly if needed, and use the already-active-file training route below. Do not map a selector variant to `_annotations.coco.json` itself: the current selector deletes the destination before copying.

The settings above are edits for the user to make; the scripts' existing defaults have not been changed by this guide.

## 2. Train a model

Use Python 3.11 and a virtual environment. From the repository root, install `python -m pip install -r requirements.txt`, with a PyTorch build suitable for your hardware. The pinned combined environment still needs end-to-end validation; `results/training_environment_windows.txt` records the historical environment. The current training settings require a CUDA GPU.

Open [src/model/train_rfdetr.py](src/model/train_rfdetr.py). For the named variant workflow above, replace the main block's existing setup from `SELECT_ASM = True` through the `select_annotations(...)` call with:

```python
dataset_path = r"D:\my_dataset_raw"  # the dataset you just built
SELECT_GT_VARIANT = "perf_gt20"
run_name = "RAW_perf_gt20"
select_annotations(dataset_path, variant=SELECT_GT_VARIANT)
```

Keep the subsequent output-directory creation and training code. The selector uses the existing `name_map` entry for `perf_gt20`; there is no extra activation command. Check that it reports the expected variant for **all three splits**. Missing variant files only produce warnings and may leave old labels active. Activation overwrites the active file without backup, but the named variant and per-sequence source copies remain available.

This path works with either raw images or already-prepared ASM images: set `dataset_path` and `run_name` accordingly. Training itself performs no ASM reconstruction. **Setting `SELECT_ASM=False` alone does not work** in the original script because paths would be undefined. The replacement above removes that incomplete conditional setup.

If your intended annotations are **already active** as `_annotations.coco.json` (for example, you kept variable-size boxes), use the same setup with `SELECT_GT_VARIANT = "prepared"` for the status print and **omit the `select_annotations(...)` call**. `prepared` is not a selector key.

To reproduce the historical ASM run, point to its prepared dataset and use `SELECT_GT_VARIANT="perf_gt20"` with `run_name="ASM_perf_gt20"`. Other existing selector keys are `perf_gt` and `old_gt`; the function's older Alex/hoechst comment is stale, so use its actual `name_map`.

In `model.train(...)`, set `epochs`, `batch_size`, `grad_accum_steps`, `lr`, and `device`. Defaults are 10 epochs, batch size 2, accumulation 8, learning rate 0.0001, and early stopping. Change the W&B `project` name, or set `wandb=False` if you do not want cloud logging. Then run:

```bash
python src/model/train_rfdetr.py
```

### Where the trained model goes

Training saves checkpoints and logs under `<dataset_path>/output/<run_name>/`. The evaluation/tracking scripts select `checkpoint_best_ema.pth`, the checkpoint chosen using validation performance with averaged model weights. Keep the checkpoint together with its class definitions, preprocessing choices, and training settings. A checkpoint is the learned model, not a file of detected cells. RF-DETR starts from pretrained weights by default; reproducing the original run also requires its original starting weights and environment.

## 3. Evaluate detections and understand model output

Edit [src/model/eval_rfdetr.py](src/model/eval_rfdetr.py):

```python
BASE_DIR = "D:/my_dataset"
CHECKPOINT_NAME = "RGB_cells"  # the training run_name
GT_ANNOTATION = "_annotations.coco"  # no .json suffix
```

The derived paths read `valid/` and `output/<CHECKPOINT_NAME>/checkpoint_best_ema.pth`. Set `IMG_DIR` and `ANN_FILE` explicitly to evaluate `test/` or another location. Set `RESULTS_DIR`, `CONF_THR_LIST`, `IOU_THR`, and `CENTROID_TOL_PX`, then run:

```bash
python src/model/eval_rfdetr.py
```

For each image, `model.predict(...)` returns detections containing:

| Field | Meaning |
|---|---|
| `xyxy` | Box corners `[left, top, right, bottom]` in the input image's pixels |
| `confidence` | Model score for each detection |
| `class_id` | Predicted class ID, interpreted using the model's class mapping |

These are in-memory predictions; they have no temporal identity. The evaluator compares them with GT and saves `summary_MODEL_…txt` and `metrics_MODEL_…json` in `RESULTS_DIR` (default `<BASE_DIR>/results_rfdetr_eval_corrected/`). It does **not** save detection tables or annotated videos. Reusing output names overwrites reports.

Reports include overlap-based and centre-distance precision/recall/F1, AP at the selected IoU, and AP averaged over IoUs 0.50–0.95. The AP calculation is custom, not the official COCO evaluator. Detector evaluation runs inference at threshold 0.0 before applying F1 cutoffs; tracking scripts retain different historical evaluation behavior.

## 4. Run tracking postprocessing

Choose one script using the [tracking comparison guide](src/tracking/README.md). The backward gap-filling experiment is `backward_iou/postprocess_fill_holes.py`; the later forward workflow with pruning is `forward_trackpy/pp_fh_ss_prune.py`.

**Input:** a trained checkpoint, prepared image sequences, and a COCO file supplying image paths/classes and evaluation labels. Configure the selected script's `BASE_DIR`, `CHECKPOINT_NAME`, `GT_ANNOTATION`, paths, tracking parameters, and `RESULTS_DIR` as for evaluation. The scripts run detection themselves; they do not read the detector evaluation report or a saved prediction file.

```bash
python src/tracking/backward_iou/postprocess_fill_holes.py
```

**Output:** predicted track IDs and optional interpolated detections in memory, plus text/JSON metric reports and annotated MP4 videos in that script's `RESULTS_DIR`. There is currently no CSV/JSON export of individual tracks. The tracking guide explains how to choose parameters and interpret the historical metrics.

The linking algorithms use detection geometry rather than RF-DETR internals, but detector loading, GT evaluation, and tracking remain combined in these files. Using another detector, tracking unlabelled data, or exporting reusable track tables requires separating those parts. All tracking scripts execute on import; run the chosen file as a script.

## Repository map

- `src/annotation_preprocessing/`: optional annotation adaptations.
- `src/image_preprocessing/`: optional ASM reconstruction and dataset assembly.
- `src/model/`: training and detector evaluation.
- `src/tracking/`: alternative detection-plus-tracking experiments and their comparison guide.
- `results/`: recovered training configuration, environment, tuning results, and historical summary.
- `data/` and `weights/`: optional local storage excluded from Git; scripts use configured paths and do not automatically discover these folders.
