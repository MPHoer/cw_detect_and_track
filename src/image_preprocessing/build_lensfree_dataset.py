"""
Assemble lensfree images and per-sequence COCO labels into train/valid/test.

Edit SOURCE_ROOT, OUTPUT_ROOT, TRAIN_DATASETS, VALID_DATASETS, TEST_DATASETS,
and ANNOTATION_IDENTIFIER (full JSON filename or case-sensitive substring).
Search non-hidden .json files directly inside each dataset folder. A full
filename takes precedence over substring matches; otherwise exactly one match
is required. Print the selected path, or stop and list candidates when there
are zero or multiple matches. --check-only uses the same selection rules.
Dataset names must exactly match source folder names.
Defaults use the recovered external-drive data and historical split lists.
Historical validation and test lists overlap: ALLOW_SPLIT_OVERLAP=True
preserves that experiment, but prints a warning. For an independent test,
choose disjoint lists and set ALLOW_SPLIT_OVERLAP=False.

Copy images referenced by each source COCO file into <output>/<split>/<dataset>/,
and copy all non-hidden .json files in each selected dataset folder tree there
as provenance. Unreferenced images and unrelated files are not copied. Skip
files/directories whose names start with '.', including AppleDouble artefacts;
hidden image records and their annotations are excluded from merged COCO.

Write one <split>/_annotations.coco.json with remapped image, annotation, and
video IDs and split-relative image paths. Preserve boxes, categories, track IDs,
and annotation attributes. Track IDs remain local to their sequence/video.
Original per-sequence JSON copies are unchanged; train with the merged file.
Categories and license definitions must agree between source sequences.
Other source metadata is preserved per sequence in source_metadata.

--check-only validates JSON references, filenames, split assignments, source
file existence, and destination conflicts, then reports counts without writing.
It does not decode images or verify their alignment. All splits are checked
before copying. Create OUTPUT_ROOT if absent; an existing directory is allowed
only if planned output files do not already exist. Never overwrite outputs.
An interrupted copy may leave partial output, requiring a fresh destination.

No ASM reconstruction, 20x20 resizing, dummy category addition, or training is
performed. Apply annotation helpers afterwards if needed. Uses only Python's
standard library; importing the file does not create a dataset.
"""

import argparse
import copy
import json
import shutil
from pathlib import Path, PurePosixPath

SOURCE_ROOT = Path('/Volumes/Max-HDD/cw_aligned_lensfree')
OUTPUT_ROOT = Path('/Volumes/Max-HDD/cw_rf_detr_dataset_raw')
# Full filename or case-sensitive substring; exactly one JSON must match per folder.
ANNOTATION_IDENTIFIER = 'corrected_191125'

TRAIN_DATASETS = [
    '20230323_livedead_A549xy2_0_0',
    '20230323_livedead_A549xy2_0_512',
    '20230323_livedead_A549xy5_512_512',
    '20230323_livedead_A549xy6_0_0',
    '20230323_livedead_A549xy6_0_512',
    '20230323_livedead_A549xy6_512_0',
    '20230324_A549_livedeadxy2_0_512',
    '20230324_A549_livedeadxy6_0_512',
    '20230328_livedead_3T3xy2_512_512',
    '20230328_livedead_3T3xy3_512_512',
    '20230328_livedead_3T3xy5_512_512',
    '20230328_livedead_3T3xy6_0_512',
    '20230412_HFF_livedeadxy1_0_0',
    '20230412_HFF_livedeadxy1_0_512',
    '20230412_HFF_livedeadxy1_512_0',
    '20230414_HFF_livedeadxy1_0_512',
    '20230414_HFF_livedeadxy1_512_0',
    '20230414_HFF_livedeadxy1_512_512',
]
VALID_DATASETS = [
    '20230322_3t3_livedeadxy5_0_0',
    '20230323_livedead_A549xy3_0_0',
    '20230324_A549_livedeadxy6_512_512',
    '20230328_livedead_3T3xy4_512_512',
    '20230412_HFF_livedeadxy1_512_512',
    '20230414_HFF_livedeadxy1_0_0',
]
TEST_DATASETS = [
    '20230322_3t3_livedeadxy5_0_0',
    '20230323_livedead_A549xy3_0_0',
    '20230324_A549_livedeadxy6_512_512',
    '20230328_livedead_3T3xy4_512_512',
    '20230412_HFF_livedeadxy1_512_512',
    '20230414_HFF_livedeadxy1_0_0',
]
# Historical reproduction only: validation and test are the same sequences.
ALLOW_SPLIT_OVERLAP = True


def relative_path(value):
    """Accept portable relative paths; reject parent traversal and drive paths."""
    if not isinstance(value, str) or not value:
        raise ValueError(f'Invalid relative path: {value!r}')
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or '\\' in value or ':' in value or not p.parts:
        raise ValueError(f'Unsafe relative path: {value!r}')
    return Path(*p.parts)


def hidden(path):
    return any(part.startswith('.') for part in path.parts)


def source_file(root, rel):
    """Resolve a required file without permitting symlinks outside the dataset."""
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f'Missing file or path outside source dataset: {root / rel}')
    return path


def select_annotation_file(folder, identifier):
    """Select an exact JSON filename or one unambiguous filename substring."""
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError('ANNOTATION_IDENTIFIER must be a nonempty string')
    if '/' in identifier or '\\' in identifier:
        raise ValueError('Use a filename or substring, not a path')
    if not folder.is_dir():
        raise FileNotFoundError(f'Dataset folder not found: {folder}')
    candidates = sorted(p for p in folder.iterdir()
                        if p.is_file() and not p.name.startswith('.')
                        and p.suffix.lower() == '.json')
    exact = [p for p in candidates if p.name == identifier]
    matches = exact or [p for p in candidates if identifier in p.name]
    if len(matches) != 1:
        available = ', '.join(p.name for p in (matches or candidates)) or '(none)'
        raise ValueError(f'{folder}: expected one JSON matching {identifier!r}, '
                         f'found {len(matches)}. Candidates: {available}')
    selected = source_file(folder, Path(matches[0].name))
    print(f'[{folder.name}] Selected annotations: {selected.name}', flush=True)
    return selected


def unique_ids(records, label):
    ids = [r['id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate {label} IDs')
    return set(ids)


def plan_split(source_root, split, names, annotation_filename):
    """Merge one split in memory and collect its image/JSON copy operations."""
    merged = None
    copies = {}
    skipped = 0
    selected_annotations = []
    for name in names:
        folder = source_root / name
        ann_path = select_annotation_file(folder, annotation_filename)
        selected_annotations.append(f'{name}/{ann_path.name}')
        coco = json.loads(ann_path.read_text(encoding='utf-8'))
        image_ids = unique_ids(coco['images'], 'image')
        unique_ids(coco['annotations'], 'annotation')
        category_ids = unique_ids(coco['categories'], 'category')
        unique_ids(coco.get('videos', []), 'video')
        if merged is None:
            merged = {'categories': copy.deepcopy(coco['categories']),
                      'licenses': copy.deepcopy(coco.get('licenses', [])),
                      'images': [], 'annotations': [], 'videos': [], 'source_metadata': {}}
        else:
            if sorted(coco['categories'], key=lambda x: x['id']) != sorted(merged['categories'], key=lambda x: x['id']):
                raise ValueError(f'Category definitions differ in {ann_path}')
            if coco.get('licenses', []) != merged.get('licenses', []):
                raise ValueError(f'License definitions differ in {ann_path}')
        merged['source_metadata'][name] = {
            key: copy.deepcopy(value) for key, value in coco.items()
            if key not in ('images', 'annotations', 'videos', 'categories')
        }
        # Image/annotation IDs are made unique within the split. Video IDs are
        # also remapped so repeated source video_id=1 cannot merge sequences.
        videos = {v['id']: v for v in coco.get('videos', [])}
        video_map = {}
        image_map = {}
        paths_seen = set()
        for im in coco['images']:
            rel = relative_path(im['file_name'])
            if hidden(rel):
                skipped += 1
                continue
            if rel in paths_seen:
                raise ValueError(f'Duplicate image filename in {ann_path}: {rel}')
            paths_seen.add(rel)
            src = source_file(folder, rel)
            dst = Path(split) / name / rel
            copies[dst] = src
            new = copy.deepcopy(im)
            new['id'] = len(merged['images']) + 1
            new['file_name'] = (Path(name) / rel).as_posix()
            old_video = im.get('video_id')
            if old_video not in video_map:
                video = copy.deepcopy(videos.get(old_video, {}))
                video['id'] = len(merged['videos']) + 1
                video['name'] = name if old_video is None else f'{name}:{old_video}'
                merged['videos'].append(video)
                video_map[old_video] = video['id']
            new['video_id'] = video_map[old_video]
            image_map[im['id']] = new
            merged['images'].append(new)
        for ann in coco['annotations']:
            if ann['image_id'] not in image_ids or ann['category_id'] not in category_ids:
                raise ValueError(f'Broken annotation reference in {ann_path}: {ann["id"]}')
            if ann['image_id'] not in image_map:
                continue  # label belongs to an excluded hidden image
            new = copy.deepcopy(ann)
            new['id'] = len(merged['annotations']) + 1
            new['image_id'] = image_map[ann['image_id']]['id']
            if 'video_id' in new:
                new['video_id'] = image_map[ann['image_id']]['video_id']
            merged['annotations'].append(new)
        for path in folder.rglob('*.json'):
            rel = path.relative_to(folder)
            if not hidden(rel):
                dst = Path(split) / name / rel
                src = source_file(folder, rel)
                if dst in copies:
                    raise ValueError(f'Image/JSON destination conflict: {dst}')
                copies[dst] = src
    if merged is None:
        raise ValueError(f'{split} has no datasets; provide a nonempty list')
    # Per-sequence source info can differ; keep it in the unchanged JSON copies
    # and make the merged file explicitly describe its sources.
    merged['info'] = {'description': 'Raw lensfree dataset assembled from per-sequence COCO',
                      'source_annotations': selected_annotations}
    print(f'[{split}] {len(names)} sequences, {len(merged["images"])} images, '
          f'{len(merged["annotations"])} annotations; {skipped} hidden image records excluded', flush=True)
    return merged, copies


def build_dataset(source_root, output_root, split_datasets, annotation_filename=ANNOTATION_IDENTIFIER,
                  allow_overlap=False, check_only=False):
    """Validate the full plan first, then copy sources and write merged split JSONs."""
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if output_root == source_root or output_root.is_relative_to(source_root) or source_root.is_relative_to(output_root):
        raise ValueError('Source and output must be separate directory trees')
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f'Output is not a directory: {output_root}')
    seen = {}
    for split, names in split_datasets.items():
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f'Unknown split: {split}')
        if len(names) != len(set(names)):
            raise ValueError(f'Duplicate dataset names in {split}')
        for name in names:
            rel = relative_path(name)
            if len(rel.parts) != 1 or hidden(rel):
                raise ValueError(f'Expected a non-hidden dataset folder name: {name}')
            if name in seen:
                if not allow_overlap:
                    raise ValueError(f'Split overlap: {name} in {seen[name]} and {split}')
                print(f'WARNING: {name} appears in {seen[name]} and {split}; these are not independent splits', flush=True)
            seen[name] = split
    plans = {}
    for split, names in split_datasets.items():
        plans[split] = plan_split(source_root, split, names, annotation_filename)
    for split, (_, copies) in plans.items():
        for relative in [*copies, Path(split) / '_annotations.coco.json']:
            dst = output_root / relative
            if not dst.resolve().is_relative_to(output_root):
                raise ValueError(f'Destination escapes output: {dst}')
            if dst.exists() or dst.is_symlink():
                raise FileExistsError(f'Will not overwrite: {dst}')
            for parent in dst.parents:
                if parent.exists() and not parent.is_dir():
                    raise ValueError(f'Destination parent is not a directory: {parent}')
    if check_only:
        print(f'Check passed. No files written. Destination: {output_root}', flush=True)
        return
    output_root.mkdir(parents=True, exist_ok=True)
    for split, (coco, copies) in plans.items():
        for relative, source in copies.items():
            dst = output_root / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as inp, dst.open('xb') as out:
                shutil.copyfileobj(inp, out)
        ann_path = output_root / split / '_annotations.coco.json'
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        with ann_path.open('x', encoding='utf-8') as out:
            json.dump(coco, out, indent=2)
        print(f'[{split}] Saved images, source JSONs, and {ann_path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copy raw lensfree sequences and merge COCO annotations.')
    parser.add_argument('--check-only', action='store_true', help='Validate and report without writing')
    args = parser.parse_args()
    build_dataset(SOURCE_ROOT, OUTPUT_ROOT,
                  {'train': TRAIN_DATASETS, 'valid': VALID_DATASETS, 'test': TEST_DATASETS},
                  allow_overlap=ALLOW_SPLIT_OVERLAP, check_only=args.check_only)
