import torch.nn.functional as F
from task1.metrics import ssim


def joint_loss(output, target, logits, weights, labels, config):
    terms=dict(l1=F.l1_loss(output,target), structural=1-ssim(output,target).mean(),
               classification=F.cross_entropy(logits,labels),
               balance=(weights.mean(0)-.25).square().sum())
    total=sum(config[key]*terms[name] for key,name in [('lambda_l1','l1'),('lambda_ssim','structural'),
              ('lambda_ce','classification'),('lambda_balance','balance')])
    return total,terms
