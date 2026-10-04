"""Runtime restoration corruption and grouped paired sketch loaders."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from corruptions import CONDITIONS, apply_corruption, specification


@lru_cache(maxsize=128)
def image_array(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB').resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32) / 255


def tensor(array):
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


class RestorationDataset(Dataset):
    def __init__(self, root='data/prepared_v2/restoration', split='train', seed=42, limit=None, condition=None, manifest=None):
        name = {'train': 'train.json', 'val': 'validation_manifest.json', 'test': 'test_manifest.json'}[split]
        self.rows = json.loads((Path(manifest) if manifest else Path(root) / name).read_text())
        if condition is not None and condition not in CONDITIONS:
            raise ValueError(f'Unknown condition: {condition}')
        if split != 'train' and condition is not None:
            self.rows = [row for row in self.rows if row['condition'] == condition]
        if limit:
            self.rows = self.rows[:limit]
        self.split, self.seed, self.epoch, self.condition = split, seed, 0, condition
        self.rng = None

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.rng = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        clean = image_array(row['path'])
        if self.split == 'train':
            if self.rng is None:
                worker = torch.utils.data.get_worker_info()
                self.rng = np.random.default_rng(self.seed + self.epoch * 100003 + (worker.id * 1009 if worker else 0))
            spec = specification(self.rng, self.condition)
        else:
            spec = row
        return dict(input=tensor(apply_corruption(clean, spec)), target=tensor(clean),
                    label=CONDITIONS.index(spec['condition']), condition=spec['condition'],
                    severity=spec.get('severity', 'sampled'), source_id=row['source_id'])


class SketchDataset(Dataset):
    def __init__(self, manifest='data/prepared_v2/sketch/paired_samples.json', split='train', augment=True):
        self.rows = [r for r in json.loads(Path(manifest).read_text()) if r['split'] == split]
        self.augment = augment and split == 'train'

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        photo, sketch = image_array(r['path']), image_array(r['sketch_path'])
        if self.augment and torch.rand(()) < .5:
            photo, sketch = photo[:, ::-1], sketch[:, ::-1]
        return dict(input=tensor(photo), target=tensor(sketch), style=r['style_id']-1, source_id=r['source_id'])
