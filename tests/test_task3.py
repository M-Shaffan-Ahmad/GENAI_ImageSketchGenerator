import unittest
import torch
from torch import nn
from task1.model import UniversalAutoencoder
from task2.model import CorruptionClassifier
from task3.model import SoftMixture,set_stage
from task3.loss import joint_loss


class ConstantGate(nn.Module):
    def forward(self,x):return torch.zeros(len(x),4)


class ConstantExpert(nn.Module):
    def __init__(self,value):super().__init__();self.value=value
    def forward(self,x):return torch.full_like(x,self.value)


class Task3Tests(unittest.TestCase):
    def test_convex_blend_identity_branch_and_weight_order(self):
        x=torch.rand(2,3,128,128)
        model=SoftMixture(ConstantGate(),[ConstantExpert(.1),ConstantExpert(.2),ConstantExpert(.3)],temperature=.7)
        y,w=model(x)
        self.assertTrue(torch.allclose(w,torch.full((2,4),.25)))
        self.assertTrue(torch.allclose(y,.25*x+.15))
        self.assertTrue(torch.all(y>=0) and torch.all(y<=1))
        with self.assertRaises(ValueError):SoftMixture(ConstantGate(),[ConstantExpert(.1)]*3,0)

    def test_warmup_freezes_experts_then_joint_propagates_gradients(self):
        torch.set_num_threads(2)
        experts=[UniversalAutoencoder(base_channels=4,bottleneck_dim=256,dropout=.1) for _ in range(3)]
        model=SoftMixture(CorruptionClassifier(base_channels=4,dropout=.1),experts)
        model.train();set_stage(model,'warmup')
        self.assertTrue(all(not e.training for e in experts))
        self.assertTrue(all(not p.requires_grad for e in experts for p in e.parameters()))
        x=torch.rand(4,3,128,128);target=torch.rand_like(x);labels=torch.arange(4)
        config=dict(lambda_l1=.8,lambda_ssim=.2,lambda_ce=.1,lambda_balance=.01)
        y,w,logits=model.forward_details(x);loss,_=joint_loss(y,target,logits,w,labels,config);loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.gate.parameters()))
        self.assertTrue(all(p.grad is None for e in experts for p in e.parameters()))
        model.zero_grad();model.train();set_stage(model,'joint')
        y,w,logits=model.forward_details(x);loss,_=joint_loss(y,target,logits,w,labels,config);loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_balance_regularizer_uses_batch_mean_not_per_image_uniformity(self):
        x=torch.rand(4,3,128,128)
        weights=torch.eye(4)
        logits=torch.zeros(4,4)
        config=dict(lambda_l1=.8,lambda_ssim=.2,lambda_ce=.1,lambda_balance=.01)
        _,terms=joint_loss(x,x,logits,weights,torch.arange(4),config)
        self.assertEqual(float(terms['balance']),0.)
        collapsed=torch.tensor([[1.,0,0,0]]*4)
        _,terms=joint_loss(x,x,logits,collapsed,torch.arange(4),config)
        self.assertAlmostEqual(float(terms['balance']),.75)
