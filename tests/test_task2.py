import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from PIL import Image
from starlette.datastructures import UploadFile
from corruptions import CONDITIONS
from dataset_loaders import RestorationDataset
from task2.data import ClassifierDataset, BalancedBatchSampler
from task2.metrics import classification_metrics
from task2.model import CorruptionClassifier, route_images
from backend.task2_runtime import Task2Runtime
from backend.app import hard_restore


class ConstantExpert(nn.Module):
    def __init__(self,value):super().__init__();self.value=value;self.calls=0
    def forward(self,x):self.calls+=1;return torch.full_like(x,self.value)


class Task2Tests(unittest.TestCase):
    def test_balanced_batches_cover_all_views_and_resample_epochs(self):
        sampler=BalancedBatchSampler(11,16,seed=42)
        batches=list(sampler);flat=[i for batch in batches for i in batch]
        self.assertEqual(sorted(flat),list(range(44)))
        for batch in batches:self.assertEqual(np.bincount(np.array(batch)%4,minlength=4).tolist(),[len(batch)//4]*4)
        self.assertEqual(batches,list(sampler))
        sampler.set_epoch(1);self.assertNotEqual(batches,list(sampler))
        ds=ClassifierDataset(limit=2)
        for label in range(4):self.assertEqual(ds[label]['label'],label)
        first=ds[1]['input'];ds.set_epoch(1);self.assertFalse(torch.equal(first,ds[1]['input']))
        ds.set_epoch(0);self.assertTrue(torch.equal(first,ds[1]['input']))
        with self.assertRaises(ValueError):BalancedBatchSampler(10,6)

    def test_specialist_filters_before_limit(self):
        for condition in CONDITIONS[1:]:
            ds=RestorationDataset(split='val',condition=condition,limit=8)
            self.assertEqual(len(ds),8)
            self.assertTrue(all(r['condition']==condition for r in ds.rows))
            train=RestorationDataset(condition=condition,limit=2)
            self.assertEqual({train[i]['condition'] for i in range(2)},{condition})

    def test_routing_dispatches_only_selected_experts_and_preserves_identity(self):
        x=torch.rand(5,3,128,128)
        experts=[ConstantExpert(.1),ConstantExpert(.2),ConstantExpert(.3)]
        result=route_images(x,torch.tensor([0,3,1,0,2]),experts)
        self.assertTrue(torch.equal(result[[0,3]],x[[0,3]]))
        for index,value in [(1,.3),(2,.1),(4,.2)]:self.assertTrue(torch.all(result[index]==value))
        self.assertEqual([e.calls for e in experts],[1,1,1])
        experts=[ConstantExpert(.1),ConstantExpert(.2),ConstantExpert(.3)]
        self.assertTrue(torch.equal(route_images(x,torch.zeros(5,dtype=torch.long),experts),x))
        self.assertEqual([e.calls for e in experts],[0,0,0])

    def test_classifier_gradients_and_metrics(self):
        torch.set_num_threads(2)
        model=CorruptionClassifier(base_channels=8)
        logits=model(torch.rand(4,3,128,128))
        self.assertEqual(logits.shape,(4,4))
        nn.functional.cross_entropy(logits,torch.arange(4)).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        scores=classification_metrics([0,0,1,2,3],[0,1,1,2,2])
        self.assertAlmostEqual(scores['accuracy'],.6)
        self.assertAlmostEqual(scores['macro_f1'],.5)
        np.testing.assert_allclose(np.array(scores['normalized_confusion_matrix']).sum(1),np.ones(4))
        missing=classification_metrics([0],[0]);self.assertEqual(missing['macro_f1'],.25)
        self.assertEqual(missing['normalized_confusion_matrix'][1],[0.]*4)

    def test_routing_summary_counts_cascading_failures_without_duplicate_metrics(self):
        from task2.evaluate import summarize
        rows=[]
        for correct,oracle,predicted in [(False,20.,15.),(True,25.,25.)]:
            row=dict(routing_correct=correct,oracle_minus_predicted_psnr=oracle-predicted)
            for name,psnr in [('input',10.),('oracle',oracle),('predicted',predicted)]:
                row.update({name+'_psnr':psnr,name+'_ssim':.8,name+'_l1':.1})
            rows.append(row)
        result=summarize(rows)
        self.assertEqual(result['n'],2)
        self.assertEqual(result['cascading_failures'],1)
        self.assertEqual(result['oracle_minus_predicted_psnr'],2.5)
        self.assertEqual(result['routing_accuracy'],.5)

    def test_onnx_runtime_clean_bypass_and_specialist_dispatch(self):
        class FakeSession:
            def __init__(self,value):self.value=value;self.calls=0
            def run(self,_,inputs):self.calls+=1;return [self.value]
        x=np.random.default_rng(0).random((1,3,128,128),dtype=np.float32)
        runtime=Task2Runtime.__new__(Task2Runtime)
        runtime.sessions={'classifier':FakeSession(np.array([[1,0,0,0]],np.float32)),
                          **{c:FakeSession(np.full_like(x,i/10)) for i,c in enumerate(CONDITIONS[1:],1)}}
        y,p,route=runtime.run(x);np.testing.assert_array_equal(y,x);self.assertEqual(route,0)
        self.assertTrue(all(runtime.sessions[c].calls==0 for c in CONDITIONS[1:]))
        runtime.sessions['classifier'].value=np.array([[0,0,1,0]],np.float32)
        y,p,route=runtime.run(x);self.assertEqual(route,2);self.assertTrue((y==.2).all())
        self.assertEqual(runtime.sessions['gaussian_blur'].calls,1)
        self.assertEqual(runtime.sessions['salt_and_pepper'].calls,0)


class Task2ApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_api_reports_classifier_route_independent_of_simulation_label(self):
        class Runtime:
            metadata={'smoke':True}
            def run(self,x):return x.copy(),np.array([0,0,1,0],np.float32),2
        with tempfile.SpooledTemporaryFile() as f:
            Image.new('RGB',(128,128),(120,80,60)).save(f,format='PNG');f.seek(0)
            with patch('backend.app.task2_session',return_value=Runtime()):
                result=await hard_restore(file=UploadFile(file=f,filename='flower.png'),sample_id=None,
                    condition='salt_and_pepper',probability=.08,kernel=5,sigma=1.5,area_ratio=.2,num_boxes=2,seed=42)
        self.assertEqual(result['predicted_condition'],'gaussian_blur')
        self.assertFalse(result['identity_bypass']);self.assertTrue(result['smoke_model'])
        self.assertEqual(result['routing_probabilities']['gaussian_blur'],1.)
        self.assertIsNotNone(result['error_map'])
