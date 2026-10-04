"""Rebuild provenance-preserving manifests without overwriting the legacy data."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat
from corruptions import CONDITIONS, SEVERITIES, specification


def read_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB').resize((128, 128), Image.Resampling.BILINEAR))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def partition(records, fraction, rng):
    groups = {}
    for r in records:
        groups.setdefault(r['sha256'], []).append(r)
    keys = sorted(groups)
    rng.shuffle(keys)
    target = round(len(records) * fraction)
    left, right = [], []
    for k in keys:
        (left if len(left) < target else right).extend(groups[k])
    return left, right


def prepare(raw, output):
    split_path = raw / 'setid.mat'
    if hashlib.md5(split_path.read_bytes()).hexdigest() != 'a5357ecc9cb78c4bef273ce3793fc85c':
        raise ValueError('Oxford setid.mat checksum mismatch')
    official = loadmat(split_path)
    ids = {s: sorted(map(int, official[k].ravel())) for s, k in [('dev', 'trnid'), ('val', 'valid'), ('test', 'tstid')]}
    development = ids['dev'] + ids['val']
    assert not set(development) & set(ids['test'])
    records = {}
    for i in sorted(development + ids['test']):
        path = raw / 'jpg' / f'image_{i:05d}.jpg'
        pixels = read_rgb(path)
        records[i] = dict(source_id=i, path=str(path), sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
    test = [records[i] for i in ids['test']]
    test_hashes = {r['sha256'] for r in test}
    excluded = [records[i] for i in development if records[i]['sha256'] in test_hashes]
    dev = [records[i] for i in development if records[i]['sha256'] not in test_hashes]
    train, val = partition(dev, .8, np.random.default_rng(42))
    for name, rows in [('train', train), ('val', val), ('test', test)]:
        save_json(output / 'restoration' / f'{name}.json', rows)
    validation = []
    for i, row in enumerate(val):
        seed = 10000 + row['source_id']
        spec = specification(np.random.default_rng(seed), CONDITIONS[i % 4], seed=seed)
        validation.append(dict(**row, **spec, severity='sampled'))
    testing = []
    for row in test:
        for c in CONDITIONS:
            for j, params in enumerate(SEVERITIES.get(c, [{}])):
                seed = 100000 + row['source_id'] * 10 + CONDITIONS.index(c) * 3 + j
                spec = specification(np.random.default_rng(seed), c, params, seed)
                testing.append(dict(**row, **spec, severity='none' if c == 'clean' else ['low', 'medium', 'high'][j]))
    save_json(output / 'restoration' / 'validation_manifest.json', validation)
    save_json(output / 'restoration' / 'test_manifest.json', testing)
    # Independent task; same official boundary. All styles of a photo stay together.
    unique_dev = list({r['sha256']: r for r in dev}.values())
    unique_test = list({r['sha256']: r for r in test}.values())
    rng = np.random.default_rng(42)
    rng.shuffle(unique_dev); rng.shuffle(unique_test)
    sketch_dev = unique_dev[:1748]
    sketch_train, sketch_val = partition(sketch_dev, .85, np.random.default_rng(42))
    pairs = []
    for split, rows in [('train', sketch_train), ('val', sketch_val), ('test', unique_test[:356])]:
        for row in rows:
            rgb = read_rgb(row['path'])
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            smooth = cv2.GaussianBlur(255-gray, (21, 21), 0)
            pencil = cv2.divide(gray, 255-smooth, scale=256)
            ink = 255-cv2.dilate(cv2.Canny(gray, 60, 120), np.ones((2, 2), np.uint8))
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            charcoal = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
            for style, pixels in enumerate([pencil, ink, charcoal], 1):
                dest = output / 'sketch' / 'targets' / f"{row['source_id']:05d}_s{style}.png"
                dest.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).convert('RGB').save(dest)
                pairs.append(dict(**row, split=split, style_id=style, sketch_path=str(dest)))
    save_json(output / 'sketch' / 'paired_samples.json', pairs)
    metadata = dict(seed=42, resize='PIL bilinear RGB 128x128', official_split_md5=hashlib.md5(split_path.read_bytes()).hexdigest(),
                    split_policy='Official train+validation repartitioned 80/20 by content group; official test preserved; development duplicates of test excluded',
                    counts={s: len(r) for s, r in [('train', train), ('val', val), ('test', test)]}, excluded_development=excluded,
                    sketch_photos={s: sum(p['split']==s for p in pairs)//3 for s in ['train','val','test']})
    save_json(output / 'metadata.json', metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=Path('raw_data'))
    parser.add_argument('--output', type=Path, default=Path('data/prepared_v2'))
    args = parser.parse_args()
    prepare(args.raw, args.output)
