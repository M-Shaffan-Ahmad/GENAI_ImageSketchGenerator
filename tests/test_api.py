import os
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from starlette.datastructures import UploadFile
from fastapi import HTTPException
from backend.app import restore, health, samples


@unittest.skipUnless(Path(os.environ.get('MODEL_PATH', 'models/universal.onnx')).exists(), 'Export an ONNX model to run API integration tests')
class ApiTests(unittest.IsolatedAsyncioTestCase):
    def arguments(self, condition='clean'):
        return dict(sample_id=None, condition=condition, probability=.08, kernel=5, sigma=1.5,
                    area_ratio=.2, num_boxes=2, seed=42)

    async def test_upload_and_all_corruptions(self):
        for condition in ['clean','salt_and_pepper','gaussian_blur','rectangular_occlusion']:
            with tempfile.SpooledTemporaryFile(max_size=1024*1024) as f:
                Image.new('RGB',(160,160),(120,80,60)).save(f,format='PNG'); f.seek(0)
                result = await restore(file=UploadFile(file=f,filename='image.png'), **self.arguments(condition))
                self.assertTrue(result['output'].startswith('data:image/png;base64,'))
                self.assertGreaterEqual(result['inference_ms'],0)
                self.assertEqual(result['error_map'] is None, condition=='clean')
        self.assertTrue(health()['model_ready'])

    async def test_sample_reference(self):
        args = self.arguments(); args['sample_id'] = samples()[0]['id']
        result = await restore(file=None, **args)
        self.assertIsNotNone(result['error_map'])

    async def test_bad_image_and_parameters(self):
        with tempfile.SpooledTemporaryFile(max_size=1024) as f:
            f.write(b'not an image'); f.seek(0)
            with self.assertRaises(HTTPException) as context:
                await restore(file=UploadFile(file=f,filename='bad.png'), **self.arguments())
            self.assertEqual(context.exception.status_code,422)
        args = self.arguments(); args['probability'] = .9
        with self.assertRaises(HTTPException) as context:
            await restore(file=None, **args)
        self.assertEqual(context.exception.status_code,422)
