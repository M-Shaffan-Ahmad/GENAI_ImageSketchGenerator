import unittest
from unittest.mock import patch
import torch
from dataset_loaders import SketchDataset,image_array,tensor
from task4.model import SketchGenerator,PatchDiscriminator


class Task4Tests(unittest.TestCase):
    def test_style_conditions_both_networks_and_gradient_paths(self):
        torch.set_num_threads(2)
        g=SketchGenerator(base_channels=4,embedding_dim=4,dropout=0.)
        d=PatchDiscriminator(base_channels=4,embedding_dim=4)
        x=torch.rand(1,3,128,128).expand(3,-1,-1,-1);style=torch.arange(3)
        y=g(x,style);logits=d(x,y,style)
        self.assertEqual(y.shape,(3,3,128,128));self.assertEqual(logits.shape,(3,1,14,14))
        self.assertTrue(torch.all(y>=0) and torch.all(y<=1))
        self.assertFalse(torch.allclose(y[0],y[1]))
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.ones_like(logits))+(y-.5).abs().mean()
        loss.backward()
        self.assertGreater(float(g.style_embedding.weight.grad.abs().sum()),0)
        self.assertGreater(float(d.style_embedding.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in g.parameters()))

    def test_spatial_augmentation_applies_identical_flip_to_pair(self):
        dataset=SketchDataset(split='train');row=dataset.rows[0]
        with patch('torch.rand',return_value=torch.tensor(0.)):
            item=dataset[0]
        self.assertTrue(torch.equal(item['input'],tensor(image_array(row['path'])[:,::-1])))
        self.assertTrue(torch.equal(item['target'],tensor(image_array(row['sketch_path'])[:,::-1])))
        self.assertEqual(item['style'],row['style_id']-1)

    def test_styles_of_each_photo_stay_in_one_split(self):
        datasets=[SketchDataset(split=split,augment=False) for split in ['train','val','test']]
        groups={}
        for split,ds in zip(['train','val','test'],datasets):
            for r in ds.rows:groups.setdefault(r['source_id'],set()).add(split)
        self.assertTrue(all(len(splits)==1 for splits in groups.values()))
        for ds in datasets:
            from collections import Counter
            self.assertEqual(set(Counter(r['source_id'] for r in ds.rows).values()),{3})
