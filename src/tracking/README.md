# Choosing a tracking script

**For time series without GT, use the [inference counterparts](inference/README.md): image folders and checkpoint in, annotated videos and track tables out.** This page describes the original GT-based evaluation experiments.

These six files are alternative complete experiments: each loads RF-DETR detections, tracks them, evaluates against COCO annotations, and writes reports and videos. **Run one script, not all six in sequence.** They preserve the historical workflow; they are not yet a model-independent tracking API.

## Which file should I use?

| File | Use when | Distinct behavior | Main tradeoff |
|---|---|---|---|
| [backward_iou/postprocess.py](backward_iou/postprocess.py) | Reproducing the original backward association and confidence-boost experiment | Greedy box-overlap linking; increases confidence for longer tracks; no interpolation | Simple, but box-size changes and displacement reduce overlap. Its numeric-ID comparison is not valid IDF1. |
| [backward_iou/postprocess_fill_holes.py](backward_iou/postprocess_fill_holes.py) | Reproducing backward tracking with the hole-filling parameter | Adds interpolated boxes inside sufficiently short gaps with confident endpoints | Recovers missing detections but can invent an incorrect path when a link is wrong. |
| [forward_trackpy/postprocess_fill_holles_trackpy.py](forward_trackpy/postprocess_fill_holles_trackpy.py) | Comparing the basic forward velocity-prediction approach | Links centres with Trackpy and fills internal gaps; no subsequent ID cleanup or pruning | Less dependent on box dimensions; current memory=0 restricts reconnection after disappearance. |
| [forward_trackpy/pp_fill_holes_static_switch.py](forward_trackpy/pp_fill_holes_static_switch.py) | Testing continuity cleanup for cells that barely move | Adds a rule merging an existing track with a nearby new ID in the next frame | May repair fragmentation, but nearby different cells can be incorrectly merged. No short-segment removal. |
| [forward_trackpy/pp_fh_ss_prune.py](forward_trackpy/pp_fh_ss_prune.py) | Reproducing the later forward workflow with short-segment removal | Adds pruning of contiguous segments shorter than 2 detections and removes duplicate predictions sharing an ID in a frame | Removes transient false positives but also real brief appearances. Interpolated detections count toward segment length. |
| [forward_trackpy/pp_fill_holes_optim.py](forward_trackpy/pp_fill_holes_optim.py) | Selecting forward linking and interpolation parameters using annotated tracks | Reuses one inference pass across 54 combinations and selects the highest custom IDF1 | More evaluation time and risk of tuning to the evaluation data; does not tune merging or pruning. |

For the remembered **backward tracking of dying cells with hole filling**, use `postprocess_fill_holes.py`. For the later forward workflow with pruning, use `pp_fh_ss_prune.py`. Neither is established here as universally better. The search file tunes the forward tracker only.

## Why backwards?

Max's historical rationale: dying cells became round and produced a good detection signal late in the sequence. Processing backwards allowed association to begin from these clearer detections and extend toward earlier appearances. This is an experimental rationale, not an explicit death model: the code does not recognise death or enforce a biological lifecycle.

Direction is not neutral in these implementations. The backward tracker greedily uses the last box already linked; the forward tracker uses motion estimated from earlier linked detections. Changing the processing order changes the available history and can change identities and fragmentation. For scenes whose end is harder than the beginning, the original reason for starting at the end may not apply. Compare directions on representative annotated sequences rather than assuming equal results. Simply comparing these folders also changes the association algorithm (IoU versus Trackpy), not just direction. There is no configuration flag to reverse either implementation.

## Current defaults that distinguish the scripts

All select `ASM_perf_gt20`, `_perfect_gt_annotations_20x20.coco.json`, inference/evaluation cutoff 0.4, and video playback at 5 fps.

| Script | Association settings | Filling | Other processing |
|---|---|---|---|
| `postprocess.py` | IoU 0.25; maximum index distance 30; tracking confidence 0.3 | None | Score bonus 0.03 × (track detection count − 1), starting at length 2, capped at 1 |
| `postprocess_fill_holes.py` | IoU 0.2; maximum index distance 60; tracking confidence 0.4 | Up to 10 missing positions; both endpoint scores ≥0.6 | Score bonus off |
| `postprocess_fill_holles_trackpy.py` | Search radius 40 px; memory 0; tracking confidence 0.4 | Up to 20 missing positions; endpoints ≥0.6 | Score bonus off |
| `pp_fill_holes_static_switch.py` | Search radius 30 px; memory 10; tracking confidence 0.4 | Up to 5 missing positions; endpoints ≥0.6 | Merge radius list `[0]`; bonus off |
| `pp_fh_ss_prune.py` | Search radius 30 px; memory 10; tracking confidence 0.4 | Up to 5 missing positions; endpoints ≥0.6 | Merge radius list `[0]`; minimum segment length 2; bonus off; video enlarged to 2048×2048 |
| `pp_fill_holes_optim.py` | Searches radius 20/30/40 px and memory 10/20 | Searches hole length 5/10/20 and endpoint confidence 0.5/0.6/0.7 | 54 combinations; best custom IDF1 at the first confidence cutoff |

A zero merge radius is not an explicit off switch: identical centres can still merge. “Static switch” refers to ID cleanup, not switching motion models. The outer merge-radius loops rerun inference for each value. In Trackpy files, `TRACK_IOU_THR` and `TRACK_MAX_AGE` are retained for logging; change `TRACKPY_SEARCH_RANGE` and `TRACKPY_MEMORY` to control association.

## Inputs and settings shared by all scripts

1. Set `BASE_DIR` to the prepared dataset root. By default, images and annotations are read from `valid/`, and weights from `output/<CHECKPOINT_NAME>/checkpoint_best_ema.pth`.
2. Set `CHECKPOINT_NAME` and `GT_ANNOTATION` independently. The annotation variable excludes `.json`. Override `checkpoint`, `IMG_DIR`, or `ANN_FILE` if your layout differs.
3. Set `RESULTS_DIR` to a distinct experiment directory, then adjust the tracking settings described in the chosen file's module docstring. There is no command-line argument parser.
4. Set `CONF_THR_LIST`, `IOU_THR`, and `CENTROID_TOL_PX` for evaluation; set `VIDEO_FPS` for playback. Run the chosen Python file in an environment with RF-DETR and its dependencies; forward scripts additionally import pandas and Trackpy.

Images must already match the model's training preprocessing; these scripts load RGB images and do not reconstruct holograms. The COCO file supplies image paths, boxes, classes, and track IDs for identity evaluation. Classes are derived from non-crowd annotations, excluding `dummy`; model and annotation labels must agree. Tracking uses predictions, not GT identity assignments, but the scripts still require the COCO file and derive the processed images/classes from it.

Use filenames such as `sequence_name/cycle_0001.png`. The first path component defines the sequence; a bare filename is treated as its own sequence. Frames are ordered by `frame_id`, then filename. Keep filenames zero-padded if filename sorting determines chronology. Time is represented by positions in this sorted list, not original frame-number differences or elapsed seconds. All-empty detection frames are not explicitly supplied as separate frames to the Trackpy `link_df` call; verify gap behavior in the installed Trackpy environment before relying on precise elapsed-frame memory.

## How the parameters affect results

- **Association:** increasing IoU tolerance by lowering `TRACK_IOU_THR`, or widening the Trackpy search radius around predicted positions, allows more links but introduces more competing cells. Longer linking memory permits reconnection after longer absences but can attach a different cell. Trackpy adaptively shrinks difficult search regions and retries oversized subnetworks at a smaller radius.
- **Gap filling:** `MAX_HOLE_LEN` only fills gaps inside an already linked track; it cannot join separate tracks. Both endpoint scores must meet `HOLE_MIN_EDGE_CONF`. Boxes and scores are linearly interpolated; a score penalty is applied and scores are clamped. No track-end extrapolation occurs. Filling may smooth over real disappearance or nonlinear motion.
- **Confidence:** inference discards predictions below `CONF_THR_LIST[0]` before tracking. Lowering `TRACK_MIN_CONF` afterwards cannot restore them. A positive score bonus rewards temporal persistence; it can also reinforce a persistent false detection. Endpoint tests use boosted scores.
- **Pruning:** `MIN_FINAL_SEGMENT_LEN` is defined near the pruning function, not in the top configuration. It acts on contiguous segments after filling and merging, not only whole-track lengths. It removes tracked detections from evaluation too, trading recall for fewer transient predictions.
- **Tuning:** the grid-search objective is the script's approximate global IDF1. Use separate annotated sequences for final assessment. Its best setting does not automatically transfer to a different model, image scale, frame rate, or cell type.

## Outputs and limits of the reported metrics

Each ordinary run writes a text summary, a JSON metrics report, and one MP4 per confidence cutoff, concatenating sequences in filename order. Videos show GT boxes in green and predictions in blue, with IDs and interpolation labels where applicable. The optimizer writes per-experiment reports, `hp_search_results.txt`, and videos for the winning configuration. Matching output names are overwritten. Individual detection/track tables are held in memory and are **not exported**.

Detection precision/recall/F1 are measured using independent IoU and centre-distance matching passes. AP uses a custom precision-envelope integral on predictions already filtered at inference and modified by postprocessing. The scripts average over IoUs **0.25–0.95**, so the “COCO mAP” label does not mean official COCO AP. The older evaluation limitations have not been fixed by this documentation change.

`postprocess.py` directly compares numeric GT and predicted IDs, which are arbitrary and independently assigned; its IDF1 must not be used to judge identity quality. The other five files use custom greedy frame matching and global track assignment for approximate IDF1, plus custom MOTA/MOTP and track-length statistics. These are historical comparison metrics, not a validated standard evaluator. In the pruning file, duplicate removal updates the prediction store but leaves entries in the track lists with cleared IDs, so track-list length statistics can differ from the retained predictions. Missing images and inference errors are logged without removing their GT; gap filling can subsequently add predictions at those positions.

## Reusing another detector

The association logic consumes boxes, scores, class labels, and frame membership, so it can be extracted for other detectors. However, these files currently instantiate `RFDETRMedium` and combine inference, association, evaluation, and video generation in one script. Supporting several RF-DETR output formats is not a model adapter interface. Another detector requires replacing the inference/conversion section; an unlabelled production workflow also requires separating image/class discovery from GT evaluation and adding track export.

All six scripts execute on import. Keep them as standalone historical experiments until that separation is implemented. The docstrings and this guide describe the existing code; no tracking algorithm or default has been changed.
