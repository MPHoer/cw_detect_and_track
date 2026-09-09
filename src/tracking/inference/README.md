# Tracking without ground truth

These are inference counterparts of the historical tracking experiments. They take **a folder of images and a trained RF-DETR Medium checkpoint**, then save tracks for inspection. No annotation JSON or GT is required. The original evaluation scripts in the parent folders remain unchanged.

## Choose one

| Inference file | Original experiment | What it does |
|---|---|---|
| `backward_iou_inference.py` | `backward_iou/postprocess.py` | Backwards IoU association and confidence boosting; no gap filling |
| `backward_iou_fill_holes_inference.py` | `backward_iou/postprocess_fill_holes.py` | Backwards IoU association plus filling confident internal gaps |
| `trackpy_fill_holes_inference.py` | `forward_trackpy/postprocess_fill_holles_trackpy.py` | Forward velocity-based linking and gap filling; historical memory=0 |
| `trackpy_static_switch_inference.py` | `forward_trackpy/pp_fill_holes_static_switch.py` | Forward linking/filling plus nearby new-ID continuation merging |
| `trackpy_prune_inference.py` | `forward_trackpy/pp_fh_ss_prune.py` | Adds short-segment pruning and removal of duplicate predictions sharing an ID |
| `trackpy_parameter_sweep_inference.py` | `forward_trackpy/pp_fill_holes_optim.py` | Runs the 54 historical parameter combinations for visual comparison |

Use `trackpy_prune_inference.py` for the later forward workflow, or `backward_iou_fill_holes_inference.py` for the backward dying-cell experiment. See the [parent comparison guide](../README.md) for the algorithm tradeoffs and backward-processing rationale. These are alternatives, not pipeline stages.

The sweep cannot compute IDF1 or choose the most accurate configuration without GT. It caches detections once, writes a separate `exp_000`, `exp_001`, … directory for each combination, and records each configuration in its JSON. It creates videos for every combination, which can take considerable time and disk space. Reduce `HYPERPARAM_SEARCH_SPACE` for a small visual comparison. Use the original GT-based optimizer if you want accuracy-based ranking.

## Input and configuration

Use a flat folder for one sequence, or a directory containing one image folder per sequence:

```text
timeseries/
├── sequence_A/frame_0001.png, frame_0002.png, ...
└── sequence_B/frame_0001.png, frame_0002.png, ...
```

Each directory containing images is treated as its own sequence, including nested directories. Frames are naturally sorted by filename (`frame_2` before `frame_10`), and dot files/directories are ignored. Inputs must already share the image representation used for training. There is no ASM reconstruction. Each image is opened as RGB. Use individual 2D image files; multi-page TIFF time stacks are not unpacked.

In the selected file, edit:

```python
IMAGE_ROOT = Path(r"D:\my_timeseries")
CHECKPOINT_PATH = Path(r"D:\cw_rf_detr_dataset_raw\output\RAW_20x20\checkpoint_best_ema.pth")
RESULTS_DIR = Path(r"D:\tracking_results\raw_prune_run1")
RESOLUTION = 512
DETECTION_MIN_CONF = 0.4
OUTPUT_MIN_CONF = 0.4
VIDEO_FPS = 5
```

Use your actual checkpoint folder and training resolution. Detection runs once per RGB frame through `src/model/inference.py`, without flips or extra NMS. `DETECTION_MIN_CONF` filters inference; `TRACK_MIN_CONF` filters which retained detections are linked; `OUTPUT_MIN_CONF` filters saved/displayed detections after score boosts or interpolation. A later lower cutoff cannot recover a prediction removed earlier. The original class name `dummy` is excluded; other model classes, including unmapped numeric IDs, are retained. Classes come from the checkpoint, never from GT.

Tracking constants are directly below the input settings in each file. The single-run ID-cleanup scripts use one `TRACK_MERGE_DIST_PX` value (default 0, permitting exact-position matches), rather than rerunning detection in an outer radius loop. `MIN_FINAL_SEGMENT_LEN` in the pruning version is configurable at the top. `UPSCALE_VIDEO` only changes display resolution, not exported coordinates.

From the repository root:

```bash
python src/tracking/inference/trackpy_prune_inference.py
```

Use a fresh `RESULTS_DIR`; existing directories are refused. Parent directories are created. Read failures stop execution rather than silently dropping frames. Partial output may remain if video writing or processing is interrupted.

## Outputs

Each run writes:

- **One MP4 per sequence:** blue boxes for detected objects, orange for interpolated boxes, with class and track ID labels. No GT overlay. Frames with no detections still appear. Frame sizes must be constant within each sequence.
- **One CSV per sequence:** `sequence`, zero-based `frame_index`, `file_name`, `class_id`, `class_name`, `track_id`, `x1`, `y1`, `x2`, `y2`, `score`, `interpolated`.
- **`tracks.json`:** settings, class mapping, and all image records with retained detections. Empty frames have an empty detection list. This is the complete timeline; an empty frame has no CSV detection row.

Coordinates are original-image pixel corners, even if the video is enlarged. IDs are local to **sequence and class**; use all three fields to identify a track. Unlinked detections can remain visible/exported with a blank CSV ID or null JSON ID. Interpolated detections are estimates, not additional model observations. Changing `OUTPUT_MIN_CONF` may hide low-score portions of an existing track without relinking it.

There are no accuracy metrics, precision/recall scores, or GT comparisons. These files run detection directly; they do not read the standalone inference JSON.

## Differences from the historical evaluation versions

Association, score boosting, interpolation, ID merging, and pruning were retained in the corresponding script. Two deliberate input-handling differences matter when comparing results:

1. Frame order comes from image filenames rather than COCO `frame_id` metadata. Every on-disk frame is retained; entirely missing image files are not invented. Time is a sorted-frame index, not timestamps or filename-number differences.
2. Trackpy receives every frame position, including frames with no detections, so those gaps consume memory. Each linking attempt gets a fresh velocity predictor. The original `link_df` path did not explicitly feed empty frames; results may differ across such gaps.

Only input/output is shared in `_shared.py`; each named file contains its tracking algorithm. `_shared.py` also imports the single-pass prediction function from `src/model/inference.py`. Keep those files together when transferring the repository. Imports do not start inference.

Tests use synthetic detections and require the tracking dependencies, but no RF-DETR checkpoint or GPU:

```bash
python -m unittest discover -s tests -p "test_tracking_inference.py"
```
