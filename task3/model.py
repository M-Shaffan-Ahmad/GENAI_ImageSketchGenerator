import torch
from torch import nn
from task1.model import UniversalAutoencoder
from task1.train import model_config
from task2.model import CorruptionClassifier, classifier_config
from corruptions import CONDITIONS


class SoftMixture(nn.Module):
    def __init__(self, gate, specialists, temperature=1.):
        super().__init__()
        if temperature<=0 or len(specialists)!=3:
            raise ValueError('Positive temperature and three specialists required')
        self.gate=gate
        self.specialists=nn.ModuleList(specialists)
        self.temperature=float(temperature)

    def forward_details(self, image):
        logits=self.gate(image)
        weights=(logits/self.temperature).softmax(1)
        branches=[image,*[expert(image) for expert in self.specialists]]
        restored=sum(weights[:,i,None,None,None]*branch for i,branch in enumerate(branches))
        return restored,weights,logits

    def forward(self, image):
        output,weights,_=self.forward_details(image)
        return output,weights


def set_stage(model, stage):
    if stage not in ['warmup','joint']:raise ValueError(stage)
    for p in model.gate.parameters():p.requires_grad_(True)
    for expert in model.specialists:
        for p in expert.parameters():p.requires_grad_(stage=='joint')
        # Frozen expert dropout must also be disabled during warm-up.
        if stage=='warmup':expert.eval()


def load_checkpoint(path, device='cpu'):
    saved=torch.load(path,map_location=device,weights_only=True)
    if saved.get('component')!='soft_moe' or saved['conditions']!=list(CONDITIONS):
        raise ValueError('Not a compatible Task 3 checkpoint')
    configs=saved['initialization_configs']
    gate=CorruptionClassifier(**classifier_config(configs['classifier']))
    experts=[UniversalAutoencoder(**model_config(configs[c])) for c in CONDITIONS[1:]]
    model=SoftMixture(gate,experts,saved['config']['temperature']).to(device)
    model.load_state_dict(saved['model'])
    return model,saved
