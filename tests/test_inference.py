"""Geometry and export tests; no model download or GPU required."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from PIL import Image

# This test is installed in tests/ beside the repository's src/ directory.
module_path = Path(__file__).resolve().parents[1] / 'src/model/inference.py'
spec = importlib.util.spec_from_file_location('raw_inference', module_path)
inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)


class FakeModel:
    class_names = {1: 'nucleus', 2: 'dummy'}

    def __init__(self, empty=False):
        self.calls = 0
        self.empty = empty

    def predict(self, image, threshold):
        self.calls += 1
        self.last_size = image.size
        if self.empty:
            return SimpleNamespace(xyxy=np.empty((0, 4)), confidence=None, class_id=None)
        # Actual flips change the location of this marker in a rectangular image.
        y, x = np.where(np.asarray(image)[:, :, 0] > 0)
        return SimpleNamespace(xyxy=np.array([[x.min(), y.min(), x.max()+1, y.max()+1]]),
                               confidence=np.array([0.9]), class_id=np.array([1]))


class InferenceTests(unittest.TestCase):
    def test_single_pass_preserves_overlapping_boxes(self):
        class OverlapModel:
            calls = 0
            def predict(self, image, threshold):
                self.calls += 1
                self.mode = image.mode
                self.threshold = threshold
                return SimpleNamespace(xyxy=np.array([[10,5,25,15],[11,5,26,15]]),
                                       confidence=np.array([.9,.8]), class_id=np.array([1,1]))
        model=OverlapModel()
        boxes,scores,ids=inference.predict_image(model,Image.new('L',(80,40)),confidence=.4)
        self.assertEqual(model.calls,1)
        self.assertEqual(model.mode,'RGB')
        self.assertEqual(model.threshold,.4)
        np.testing.assert_allclose(boxes,[[10,5,25,15],[11,5,26,15]])
        np.testing.assert_allclose(scores,[.9,.8])

    def test_empty_frames_export_and_hidden_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root/'images'; inputs.mkdir()
            (inputs/'seq').mkdir(); (inputs/'.hidden').mkdir()
            for name in ['seq/frame_0001.png', 'seq/frame_0002.png', '.hidden/a.png', '._artefact.png']:
                Image.new('RGB', (80,40)).save(inputs/name)
            output = root/'results/predictions.json'
            inference.run_inference(FakeModel(empty=True), inputs, output)
            data=json.loads(output.read_text())
            self.assertEqual(len(data['images']),2)
            self.assertEqual(data['images'][0]['file_name'],'seq/frame_0001.png')
            self.assertEqual(data['images'][0]['detections'],[])
            self.assertEqual(data['images'][0]['width'],80)
            with self.assertRaises(FileExistsError):
                inference.run_inference(FakeModel(), inputs, output)


if __name__ == '__main__':
    unittest.main()
