"""GT-free tracking tests with synthetic detections; no model weights needed."""
from collections import defaultdict
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

FOLDER = Path(__file__).resolve().parents[1]/'src/tracking/inference'
sys.path.insert(0,str(FOLDER))
from _shared import link_complete_frames, export_results, load_detections


def module(name):
    spec=importlib.util.spec_from_file_location(name,FOLDER/(name+'.py'))
    result=importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def detections():
    store=defaultdict(lambda:defaultdict(list))
    for i in [1,2,3]:store[i]
    for i in [1,3]:
        store[i][1].append({'box':np.array([10.,5.,20.,15.]),'score':.9,
                            'base_score':.9,'track_id':None,'interp':False})
    return store


class TrackingTests(unittest.TestCase):
    def test_all_six_algorithms_without_gt(self):
        variants=['backward_iou','backward_iou_fill_holes','trackpy_fill_holes',
                  'trackpy_static_switch','trackpy_prune','trackpy_parameter_sweep']
        for variant in variants:
            with self.subTest(variant=variant), contextlib.redirect_stdout(io.StringIO()):
                m=module(variant+'_inference')
                cfg={'TRACKPY_SEARCH_RANGE':30,'TRACKPY_MEMORY':10,
                     'MAX_HOLE_LEN':5,'HOLE_MIN_EDGE_CONF':.6}
                processed=m.track(detections(),{'seq':[1,2,3]},[1],cfg)
                if variant in ['backward_iou','trackpy_fill_holes']:
                    self.assertEqual(len(processed[2][1]),0)
                else:
                    self.assertEqual(len(processed[2][1]),1)
                    self.assertTrue(processed[2][1][0]['interp'])
                    self.assertEqual(processed[1][1][0]['track_id'],processed[3][1][0]['track_id'])

    def test_all_empty_sequences(self):
        variants=['backward_iou','backward_iou_fill_holes','trackpy_fill_holes',
                  'trackpy_static_switch','trackpy_prune','trackpy_parameter_sweep']
        for variant in variants:
            m=module(variant+'_inference')
            store=defaultdict(lambda:defaultdict(list))
            for i in [1,2,3]:store[i]
            cfg={'TRACKPY_SEARCH_RANGE':30,'TRACKPY_MEMORY':10,
                 'MAX_HOLE_LEN':5,'HOLE_MIN_EDGE_CONF':.6}
            with contextlib.redirect_stdout(io.StringIO()):
                processed=m.track(store,{'seq':[1,2,3]},[1],cfg)
            self.assertFalse(any(processed[i][1] for i in [1,2,3]))

    def test_memory_counts_empty_frames(self):
        df=pd.DataFrame([{'frame':0,'x':10.,'y':10.},{'frame':3,'x':10.,'y':10.}])
        with contextlib.redirect_stdout(io.StringIO()):
            short=link_complete_frames(df,4,search_range=30,memory=1)
            long=link_complete_frames(df,4,search_range=30,memory=2)
        self.assertNotEqual(short.particle.iloc[0],short.particle.iloc[1])
        self.assertEqual(long.particle.iloc[0],long.particle.iloc[1])

    def test_export_frames_csv_video_and_json(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); images={}
            for i in [1,2,3]:
                name=f'frame_{i}.png'; Image.new('RGB',(80,40)).save(root/name)
                images[i]={'file_name':name,'width':80,'height':40,'sequence':'.','frame_index':i-1}
            m=module('backward_iou_fill_holes_inference')
            store=m.track(detections(),{'.':[1,2,3]},[1])
            settings={'VIDEO_FPS':5,'OUTPUT_MIN_CONF':.4}
            export_results(store,{'.':[1,2,3]},images,{1:'nucleus'},root,root/'out',settings)
            report=json.loads((root/'out/tracks.json').read_text())
            self.assertEqual(len(report['images']),3)
            self.assertTrue(report['images'][1]['detections'][0]['interpolated'])
            video=cv2.VideoCapture(str(next((root/'out').glob('*.mp4'))))
            count=0
            while video.read()[0]:count+=1
            video.release();self.assertEqual(count,3)
            self.assertEqual(len(next((root/'out').glob('*.csv')).read_text().splitlines()),4)

    def test_input_order_empty_frames_hidden_and_no_gt(self):
        from types import SimpleNamespace
        class EmptyModel:
            class_names={1:'nucleus'}
            def predict(self,image,threshold):
                return SimpleNamespace(xyxy=np.empty((0,4)),confidence=None,class_id=None)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ['frame_10.png','frame_2.png','frame_1.png','._hidden.png']:
                Image.new('RGB',(40,40)).save(root/name)
            store,seq,ims,classes,names=load_detections(root,'unused',512,.4,model=EmptyModel())
            self.assertEqual([i['file_name'] for i in ims.values()],['frame_1.png','frame_2.png','frame_10.png'])
            self.assertEqual(seq['.'],[1,2,3]);self.assertEqual(len(store),3)

if __name__=='__main__':unittest.main()
