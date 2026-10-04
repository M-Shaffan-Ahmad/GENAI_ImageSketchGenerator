"""Exactly balanced classifier batches and fixed severity validation cases."""
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset, Sampler
from corruptions import CONDITIONS, SEVERITIES, apply_corruption, specification
from dataset_loaders import image_array, tensor


class ClassifierDataset(Dataset):
    """Four runtime views per training photo, with reproducible epoch resampling."""
    def __init__(self, root='data/prepared_v2/restoration', seed=42, limit=None):
        self.rows = json.loads((Path(root)/'train.json').read_text())
        if limit:
            self.rows = self.rows[:limit]
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return 4*len(self.rows)

    def __getitem__(self, index):
        photo, label = divmod(index, 4)
        row = self.rows[photo]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
        spec = specification(rng, CONDITIONS[label])
        clean = image_array(row['path'])
        return dict(input=tensor(apply_corruption(clean, spec)), target=tensor(clean),
                    label=label, source_id=row['source_id'], condition=CONDITIONS[label], severity='sampled')


class BalancedBatchSampler(Sampler):
    """Equal condition counts even in the final partial batch; each view once/epoch."""
    def __init__(self, photos, batch_size, seed=42):
        if photos <= 0 or batch_size < 4 or batch_size % 4:
            raise ValueError('Need photos and a batch size divisible by four')
        self.photos, self.per_class, self.seed, self.epoch = photos, batch_size//4, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return (self.photos+self.per_class-1)//self.per_class

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        orders = [rng.permutation(self.photos) for _ in CONDITIONS]
        for start in range(0, self.photos, self.per_class):
            indices = [int(photo)*4+label for label, order in enumerate(orders)
                       for photo in order[start:start+self.per_class]]
            rng.shuffle(indices)
            yield indices


def severity_manifest(root, split, output, seed=42):
    """One clean + three severities per corruption per photo; leaves existing manifests intact."""
    photos = json.loads((Path(root)/f'{split}.json').read_text())
    rows = []
    for photo in photos:
        for label, condition in enumerate(CONDITIONS):
            for severity, parameters in enumerate(SEVERITIES.get(condition, [{}])):
                rng = np.random.default_rng(np.random.SeedSequence([seed, photo['source_id'], label, severity]))
                spec = specification(rng, condition, parameters)
                rows.append(dict(photo, **spec, severity='none' if label == 0 else ['low','medium','high'][severity]))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(rows, indent=2))
    return Path(output)
