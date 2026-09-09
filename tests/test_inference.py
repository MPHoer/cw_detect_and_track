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
    def test_flips_restore_and_merge(self):
        pixels = np.zeros((40, 80, 3), dtype=np.uint8)
        pixels[5:15, 10:25] = 255
        model = FakeModel()
        boxes, scores, ids = inference.predict_image(model, Image.fromarray(pixels))
        np.testing.assert_allclose(boxes, [[10, 5, 25, 15]])
        self.assertEqual(model.calls, 3)
        self.assertEqual(ids.tolist(), [1])
        model.calls = 0
        inference.predict_image(model, Image.fromarray(pixels), use_tta=False)
        self.assertEqual(model.calls, 1)

    def test_nms_keeps_distinct_cells_and_classes(self):
        boxes = np.array([[0,0,10,10], [1,0,11,10], [0,0,10,10], [20,0,30,10]],dtype=float)
        indices = inference.class_aware_nms(boxes, np.array([.9,.8,.7,.6]), np.array([1,1,2,1]), .5)
        self.assertEqual(indices.tolist(), [0,2,3])

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
