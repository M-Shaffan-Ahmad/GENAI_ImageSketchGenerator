import torch
from torch import nn
from pathlib import Path
from corruptions import CONDITIONS
from task1.train import load_checkpoint


class CorruptionClassifier(nn.Module):
    """Logits in the shared clean, salt, blur, occlusion order."""
    def __init__(self, base_channels=32, dropout=0.05):
        super().__init__()
        layers, previous = [], 3
        for multiple in [1,2,4,8]:
            channels = base_channels*multiple
            layers += [nn.Conv2d(previous, channels, 3, 2, 1), nn.ReLU(), nn.Dropout2d(dropout)]
            previous = channels
        self.features = nn.Sequential(*layers)
        self.average = nn.AdaptiveAvgPool2d(1)
        self.maximum = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Linear(previous*2, 4)

    def forward(self, image):
        features = self.features(image)
        pooled = torch.cat([self.average(features).flatten(1), self.maximum(features).flatten(1)], dim=1)
        return self.head(pooled)


def classifier_config(config):
    return {key: config[key] for key in ['base_channels','dropout']}


def load_classifier(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved.get('component') != 'classifier' or saved.get('conditions') != list(CONDITIONS):
        raise ValueError('Not a Task 2 classifier checkpoint or class order differs')
    model = CorruptionClassifier(**classifier_config(saved['config'])).to(device)
    model.load_state_dict(saved['model'])
    return model, saved


def load_pipeline(root, device='cpu'):
    root = Path(root)
    classifier, metadata = load_classifier(root/'classifier/final/best.pt', device)
    experts, checkpoints = [], {'classifier':metadata}
    for condition in CONDITIONS[1:]:
        model, checkpoint = load_checkpoint(root/condition/'final/best.pt', device)
        if checkpoint['config'].get('condition') != condition:
            raise ValueError(f'Wrong specialist checkpoint: {condition}')
        experts.append(model); checkpoints[condition] = checkpoint
    for key in ['train_manifest_sha256','validation_manifest_sha256']:
        if len({c[key] for c in checkpoints.values()}) != 1:
            raise ValueError('Pipeline checkpoints use different data splits')
    return HardRoutingPipeline(classifier, experts).to(device).eval(), checkpoints


def route_images(image, route, specialists):
    if route.shape != (len(image),) or ((route < 0)|(route > 3)).any():
        raise ValueError('Routes must be one label from 0 to 3 per image')
    restored = image.clone()
    for label, expert in enumerate(specialists, start=1):
        mask = route == label
        if mask.any():
            restored[mask] = expert(image[mask])
    return restored


class HardRoutingPipeline(nn.Module):
    def __init__(self, classifier, specialists):
        super().__init__()
        if len(specialists) != 3:
            raise ValueError('Exactly salt, blur and occlusion specialists are required')
        self.classifier = classifier
        self.specialists = nn.ModuleList(specialists)

    def forward(self, image):
        probabilities = self.classifier(image).softmax(1)
        route = probabilities.argmax(1)
        return route_images(image, route, self.specialists), probabilities, route
