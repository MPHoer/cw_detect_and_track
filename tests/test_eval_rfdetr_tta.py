"""Run both complete evaluators with synthetic detections, without a GPU."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class MarkerModel:
    class_names = {1: 'nucleus', 2: 'dummy'}
    thresholds = []

    def __init__(self, **kwargs):
        pass

    def predict(self, image, threshold):
        self.thresholds.append(threshold)
        y, x = np.where(np.asarray(image)[:, :, 0] > 0)
        if len(x):
            return SimpleNamespace(xyxy=np.array([[x.min(),y.min(),x.max()+1,y.max()+1]]),
                                   confidence=np.array([.2]), class_id=np.array([1]))
        return SimpleNamespace(xyxy=np.empty((0,4)), confidence=np.empty(0), class_id=np.empty(0,dtype=int))


class EvaluatorTests(unittest.TestCase):
    def test_identical_metrics_for_identical_restored_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory); valid=base/'valid'; valid.mkdir()
            pixels=np.zeros((40,80,3),dtype=np.uint8); pixels[5:15,10:25]=255
            Image.fromarray(pixels).save(valid/'one.png')
            Image.new('RGB',(80,40)).save(valid/'empty.png')
            coco={'images':[{'id':1,'file_name':'one.png'},{'id':2,'file_name':'empty.png'}],
                  'categories':[{'id':1,'name':'nucleus'},{'id':2,'name':'dummy'}],
                  'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[10,5,15,10]}]}
            (valid/'_perfect_gt_annotations_20x20.coco.json').write_text(json.dumps(coco))
            fake=ModuleType('rfdetr'); fake.RFDETRMedium=MarkerModel
            spec=importlib.util.spec_from_file_location('inference',ROOT/'src/model/inference.py')
            helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
            reports=[]
            for filename,outdir,expected_calls in [('eval_rfdetr.py','results_rfdetr_eval_corrected',2),
                                                  ('eval_rfdetr_tta.py','results_rfdetr_eval_tta',6)]:
                source=ROOT/'src/model'/filename
                tree=ast.parse(source.read_text())
                for node in tree.body:
                    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BASE_DIR' for t in node.targets):
                        node.value=ast.Constant(str(base))
                MarkerModel.thresholds=[]
                with patch.dict(sys.modules,{'rfdetr':fake,'inference':helper}), contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(ast.fix_missing_locations(tree),str(source),'exec'),{'__name__':'__main__'})
                self.assertEqual(MarkerModel.thresholds,[0.0]*expected_calls)
                report=json.loads(next((base/outdir).glob('metrics_*.json')).read_text())
                reports.append(report)
            for key in ['per_threshold','ap_fixed_iou','coco_style']:
                self.assertEqual(reports[0][key],reports[1][key])
            self.assertGreater(reports[1]['ap_fixed_iou']['mAP'],.99)
            self.assertEqual(reports[1]['per_threshold']['0.4']['per_class']['nucleus']['iou']['TP'],0)
            self.assertEqual(reports[1]['config']['tta_views'],['original','horizontal','vertical'])


if __name__ == '__main__':
    unittest.main()
