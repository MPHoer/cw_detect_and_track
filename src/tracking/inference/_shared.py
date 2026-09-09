"""Shared GT-free frame loading, single-pass inference, and video/CSV/JSON output.

Tracking algorithms remain in the named inference scripts. Frames are naturally
sorted within each image directory, which is treated as one sequence. All
frames remain in the manifest, including frames with no detections. This is
not an evaluation module and does not read COCO annotations.
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'model'))
from inference import list_images, predict_image


def natural_key(value):
    return tuple((1, int(p)) if p.isdigit() else (0, p.lower())
                 for p in re.split(r'(\d+)', str(value)))


def load_detections(image_root, checkpoint, resolution, confidence,
                    model=None):
    """Return predictions, ordered frame IDs, image metadata, and model class names."""
    root = Path(image_root)
    paths = sorted(list_images(root), key=lambda p: natural_key(p.relative_to(root)))
    if not paths:
        raise ValueError(f'No images in {root}')
    if model is None:
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        from rfdetr import RFDETRMedium
        model = RFDETRMedium(pretrain_weights=str(checkpoint), resolution=resolution)
    names = getattr(model, 'class_names', {}) or {}
    if not isinstance(names, dict):
        raise ValueError('Expected model class_names as an ID-to-name dictionary')
    names = {int(k): str(v) for k, v in names.items()}
    store = defaultdict(lambda: defaultdict(list))
    sequences, images = defaultdict(list), {}
    for img_id, path in enumerate(paths, 1):
        relative = path.relative_to(root)
        sequence = relative.parent.as_posix()  # '.' means a single flat sequence
        with Image.open(path) as image:
            width, height = image.size
            boxes, scores, ids = predict_image(model, image, confidence)
        images[img_id] = {'file_name': relative.as_posix(), 'width': width, 'height': height,
                          'sequence': sequence, 'frame_index': len(sequences[sequence])}
        sequences[sequence].append(img_id)
        store[img_id]  # retain completely empty frames
        for box, score, cid in zip(boxes, scores, ids):
            cid = int(cid)
            if names.get(cid) == 'dummy':
                continue
            names.setdefault(cid, f'class_{cid}')
            store[img_id][cid].append({'box': box, 'score': float(score),
                                      'base_score': float(score), 'track_id': None, 'interp': False})
        print(f'[{img_id}/{len(paths)}] Detected {relative}', flush=True)
    classes = sorted(k for k in names if names[k] != 'dummy')
    return store, sequences, images, classes, names


def link_complete_frames(df, frame_count, **kwargs):
    """Link every time step, including empty frames, with a fresh velocity predictor."""
    import pandas as pd
    import trackpy as tp
    import trackpy.predict as tpp
    pred = tpp.NearestVelocityPredict()
    pred.pos_columns = ['y', 'x']
    pred.t_column = 'frame'
    frames = [df[df['frame'] == i].copy() for i in range(frame_count)]
    coords = (frame[['y', 'x']].to_numpy(dtype=float) for frame in frames)
    linked = []
    for i, ids in tp.link_iter(coords, predictor=pred.predict, **kwargs):
        frame = frames[i]
        frame['particle'] = ids
        if len(frame):
            pred.observe(frame)
            linked.append(frame)
    return pd.concat(linked, ignore_index=True) if linked else df.assign(particle=np.nan)


def export_results(store, sequences, images, names, image_root, output_dir, settings):
    """Save one CSV/MP4 per sequence and a JSON manifest of all retained detections."""
    import cv2
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(f'Choose a new RESULTS_DIR (never overwritten): {out}')
    fps = settings['VIDEO_FPS']
    if fps <= 0:
        raise ValueError('VIDEO_FPS must be positive')
    out.mkdir(parents=True)
    manifest = {'settings': settings, 'class_names': names, 'images': []}
    threshold = settings['OUTPUT_MIN_CONF']
    for seq_index, (sequence, img_ids) in enumerate(sequences.items()):
        first = images[img_ids[0]]
        size = (first['width'], first['height'])
        if any((images[i]['width'], images[i]['height']) != size for i in img_ids):
            raise ValueError(f'Frame dimensions vary within sequence: {sequence}')
        video_size = tuple(settings.get('UPSCALED_SIZE', size)) if settings.get('UPSCALE_VIDEO') else size
        stem = f'{seq_index:03d}_' + re.sub(r'[^A-Za-z0-9_.-]', '_', sequence if sequence != '.' else 'sequence')
        video_path, csv_path = out/f'{stem}.mp4', out/f'{stem}.csv'
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, video_size)
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f'Cannot open video writer: {video_path}')
        fields = ['sequence', 'frame_index', 'file_name', 'class_id', 'class_name',
                  'track_id', 'x1', 'y1', 'x2', 'y2', 'score', 'interpolated']
        try:
            with csv_path.open('x', newline='', encoding='utf-8') as stream:
                csv_writer = csv.DictWriter(stream, fieldnames=fields); csv_writer.writeheader()
                for img_id in img_ids:
                    info = images[img_id]
                    with Image.open(Path(image_root)/info['file_name']) as image:
                        frame = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)
                    record = {**info, 'detections': []}
                    for cid, detections in store.get(img_id, {}).items():
                        for det in detections:
                            if det['score'] < threshold:
                                continue
                            box = np.asarray(det['box'], dtype=float)
                            tid = det['track_id']
                            tid = int(tid) if tid is not None else None
                            interpolated = bool(det.get('interp', False))
                            row = {'sequence': sequence, 'frame_index': info['frame_index'],
                                   'file_name': info['file_name'], 'class_id': int(cid),
                                   'class_name': names.get(cid, str(cid)), 'track_id': tid,
                                   **dict(zip(['x1','y1','x2','y2'], box.tolist())),
                                   'score': float(det['score']), 'interpolated': interpolated}
                            csv_writer.writerow(row)
                            record['detections'].append({'bbox_xyxy': box.tolist(),
                                'score': row['score'], 'class_id': int(cid), 'class_name': row['class_name'],
                                'track_id': tid, 'interpolated': interpolated})
                            color = (0,165,255) if interpolated else (255,0,0)
                            x1,y1,x2,y2 = map(int, box)
                            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 1)
                            label = f'{names.get(cid,cid)} id={tid if tid is not None else "untracked"}'
                            if interpolated: label += ' interp'
                            cv2.putText(frame,label,(x1,max(12,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
                    cv2.putText(frame,f'{sequence} frame={info["frame_index"]}',(8,18),
                                cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                    if video_size != size:
                        frame = cv2.resize(frame, video_size)
                    writer.write(frame)
                    manifest['images'].append(record)
        finally:
            writer.release()
        print(f'Saved {video_path} and {csv_path}', flush=True)
    with (out/'tracks.json').open('x',encoding='utf-8') as stream:
        json.dump(manifest,stream,indent=2,allow_nan=False)
