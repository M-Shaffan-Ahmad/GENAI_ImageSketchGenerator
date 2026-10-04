import json
import unittest
from pathlib import Path
import numpy as np
import torch
from corruptions import CONDITIONS, SEVERITIES, apply_corruption, specification
from dataset_loaders import RestorationDataset, SketchDataset
from task1.model import UniversalAutoencoder
from task1.metrics import reconstruction_loss, ssim


class PipelineTests(unittest.TestCase):
    def test_fixed_corruptions_reproduce_and_preserve_target(self):
        clean = np.full((128, 128, 3), .5, np.float32)
        for c in CONDITIONS:
            for params in SEVERITIES.get(c, [{}]):
                spec = specification(np.random.default_rng(7), c, params, seed=7)
                a, b = apply_corruption(clean, spec), apply_corruption(clean, spec)
                np.testing.assert_array_equal(a, b)
                self.assertTrue(np.all(clean == .5))
                if c == 'rectangular_occlusion':
                    fraction = np.all(a == 0, axis=2).mean()
                    self.assertAlmostEqual(fraction, params['area_ratio'], delta=.005)
                    self.assertEqual(len(spec['parameters']['boxes']), params['num_boxes'])
                    self.assertGreaterEqual(fraction, .1)
                    self.assertLessEqual(fraction, .35)

    def test_training_resamples(self):
        ds = RestorationDataset(limit=1, condition='salt_and_pepper')
        self.assertFalse(torch.equal(ds[0]['input'], ds[0]['input']))
        ds.set_epoch(0)
        x = ds[0]['input']; ds.set_epoch(0)
        self.assertTrue(torch.equal(x, ds[0]['input']))

    def test_uniform_conditions(self):
        rng = np.random.default_rng(42)
        counts = {c: 0 for c in CONDITIONS}
        for _ in range(4000):
            counts[specification(rng)['condition']] += 1
        self.assertTrue(all(850 < n < 1150 for n in counts.values()))

    def test_split_disjointness_and_complete_evaluation(self):
        root = Path('data/prepared_v2/restoration')
        splits = {s: json.loads((root/f'{s}.json').read_text()) for s in ['train','val','test']}
        from scipy.io import loadmat
        official = loadmat('raw_data/setid.mat')
        self.assertEqual({r['source_id'] for r in splits['test']}, set(map(int, official['tstid'].ravel())))
        for a,b in [('train','val'),('train','test'),('val','test')]:
            self.assertFalse({r['source_id'] for r in splits[a]} & {r['source_id'] for r in splits[b]})
            self.assertFalse({r['sha256'] for r in splits[a]} & {r['sha256'] for r in splits[b]})
        rows = json.loads((root/'test_manifest.json').read_text())
        self.assertEqual(len(rows), 10*len(splits['test']))
        self.assertTrue(all('seed' in r for r in rows))
        self.assertTrue(all('boxes' in r['parameters'] for r in rows if r['condition']=='rectangular_occlusion'))
        paired = json.loads(Path('data/prepared_v2/sketch/paired_samples.json').read_text())
        by_id = {}
        for r in paired:
            by_id.setdefault(r['source_id'], set()).add(r['split'])
        self.assertTrue(all(len(s)==1 for s in by_id.values()))
        self.assertEqual(len(by_id), 2104)
        self.assertEqual(SketchDataset(augment=False)[0]['target'].shape, (3,128,128))

    def test_autoencoder_compression_and_gradients(self):
        torch.set_num_threads(2)
        model = UniversalAutoencoder(base_channels=8, bottleneck_dim=8192, dropout=0.)
        target = torch.rand(2,3,128,128)
        output = model(target)
        self.assertEqual(output.shape, target.shape)
        latent = model.encode(target)
        self.assertEqual(latent.shape, (2,32,16,16))
        self.assertLess(latent[0].numel(), target[0].numel())
        reconstruction_loss(output, target, .8).backward()
        self.assertIsNotNone(model.compress.weight.grad)
        self.assertIsNotNone(model.encoder[0].weight.grad)
        self.assertTrue(torch.allclose(ssim(target,target), torch.ones(2), atol=1e-5))


if __name__ == '__main__':
    unittest.main()
