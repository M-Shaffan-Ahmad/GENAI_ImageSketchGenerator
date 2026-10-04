"""Install reviewed Colab models only after provenance and ONNX parity checks.

Run from the repository root: python integrate_models.py
Existing deployed files are backed up before replacement. No training or test data
is used; parity checks use fixed validation inputs only.
"""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from corruptions import CONDITIONS
from dataset_loaders import RestorationDataset, SketchDataset
from task1.train import load_checkpoint as load_task1
from task2.model import load_pipeline
from task2.train import fingerprints
from task3.model import load_checkpoint as load_task3
from task4.model import load_checkpoint as load_task4, STYLES
from task4.export import PARITY_ATOL, PARITY_RTOL


ROOT = Path('runs/integration')
STAGE = ROOT / 'staging'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def extract(archive, member, relative):
    target = STAGE / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exact member allowlist, never arbitrary archive paths. ZipFile checks CRC
    # while reading each selected entry without extracting the entire archive.
    with ZipFile(archive) as source, source.open(member) as stream, target.open('wb') as output:
        shutil.copyfileobj(stream, output)
    return target


def metadata(path):
    return json.loads(Path(path).read_text())


def verify_checkpoint(saved, expected):
    if saved.get('smoke', True):
        raise ValueError('Refusing to install a smoke checkpoint')
    for key, value in expected.items():
        if saved[key] != value:
            raise ValueError(f'Checkpoint differs from local data: {key}')


def runtime(path):
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def parity(model, path, datasets, sketch=False):
    model.eval()
    session = runtime(path)
    errors, auxiliary_errors, counts = [], [], [0, 0, 0]
    atol = PARITY_ATOL if sketch else 1e-5
    with torch.no_grad():
        for dataset in datasets:
            for item in dataset:
                x = item['input'][None]
                feed = {'image': x.numpy()}
                if sketch:
                    style = torch.tensor([item['style']], dtype=torch.long)
                    feed['style'] = style.numpy()
                    expected = (model(x, style),)
                    counts[item['style']] += 1
                else:
                    value = model(x)
                    expected = value if isinstance(value, tuple) else (value,)
                actual = session.run(None, feed)
                if len(actual) != len(expected):
                    raise ValueError('Unexpected graph output count')
                for index, (a, e) in enumerate(zip(actual, expected)):
                    e = e.numpy()
                    np.testing.assert_allclose(a, e, rtol=PARITY_RTOL, atol=atol)
                    if not np.isfinite(a).all():
                        raise ValueError('Non-finite graph output')
                    error = float(np.abs(a-e).max())
                    if sketch and error > PARITY_ATOL:
                        raise ValueError('Sketch graph exceeds absolute error budget')
                    (errors if index == 0 else auxiliary_errors).append(error)
                    if e.shape == (1, 4):
                        np.testing.assert_array_equal(a.argmax(1), e.argmax(1))
    result = dict(validation_images=len(errors), max_absolute_error=max(errors),
                  parity_atol=atol, parity_rtol=PARITY_RTOL, passed=True)
    if auxiliary_errors:
        result['weights_max_absolute_error'] = max(auxiliary_errors)
    if sketch:
        result.update(validation_by_style=counts, max_absolute_error_budget=PARITY_ATOL)
    return result


def main():
    torch.set_num_threads(4)
    STAGE.mkdir(parents=True, exist_ok=True)
    archives = {name: Path('colab') / filename for name, filename in {
        'task1': 'task1_run_a_results.zip', 'task2': 'task2_results.zip',
        'task3': 'task3_results.zip', 'task4': 'task4_results.zip'}.items()}
    staged = []

    def model_file(task, member, destination):
        path = extract(archives[task], member, 'models/' + destination)
        staged.append((path, Path('models') / destination))
        return path

    checkpoints = {}
    checkpoints['task1'] = extract(archives['task1'], 'runs/task1/run_a/final/best.pt', 'checkpoints/task1/best.pt')
    for name in ['classifier', *CONDITIONS[1:]]:
        extract(archives['task2'], f'runs/task2/{name}/final/best.pt', f'checkpoints/task2/{name}/final/best.pt')
        for suffix in ['onnx', 'json']:
            model_file('task2', f'models/task2/{name}.{suffix}', f'task2/{name}.{suffix}')
    pipeline_path = model_file('task2', 'models/task2/pipeline.json', 'task2/pipeline.json')
    for task, basename in [('task3', 'soft_moe'), ('task4', 'sketch_generator')]:
        checkpoints[task] = extract(archives[task], f'runs/{task}/final/best.pt', f'checkpoints/{task}/best.pt')
        for suffix in ['onnx', 'json']:
            model_file(task, f'models/{task}/{basename}.{suffix}', f'{task}/{basename}.{suffix}')
    model_file('task1', 'models/universal_run_a.onnx', 'universal.onnx')
    task1_meta_path = model_file('task1', 'models/universal_run_a.json', 'universal.json')

    expected = fingerprints('data/prepared_v2/restoration')
    datasets = lambda n: [RestorationDataset(split='val', condition=condition, limit=n) for condition in CONDITIONS]
    reports = {}
    model, saved = load_task1(checkpoints['task1'])
    verify_checkpoint(saved, expected)
    entry = metadata(task1_meta_path)
    if entry['config'] != saved['config']:
        raise ValueError('Task 1 export configuration differs')
    entry.update(parity(model, STAGE/'models/universal.onnx', datasets(4)),
                 epoch=saved['epoch'], smoke=False, checkpoint_sha256=sha(checkpoints['task1']),
                 onnx_sha256=sha(STAGE/'models/universal.onnx'), selection='Task 1 Run A', **expected)
    write_json(task1_meta_path, entry)
    reports['task1'] = entry
    print('Task 1 provenance and parity passed', flush=True)

    pipeline, saved_components = load_pipeline(STAGE/'checkpoints/task2')
    meta = metadata(pipeline_path)
    for name, saved in saved_components.items():
        verify_checkpoint(saved, expected)
        checkpoint = STAGE/f'checkpoints/task2/{name}/final/best.pt'
        path = STAGE/f'models/task2/{name}.onnx'
        entry = meta['components'][name]
        if sha(checkpoint) != entry['checkpoint_sha256'] or sha(path) != entry['onnx_sha256']:
            raise ValueError(f'Task 2 artifact hash differs: {name}')
        component = nn.Sequential(pipeline.classifier, nn.Softmax(dim=1)) if name == 'classifier' else pipeline.specialists[list(CONDITIONS[1:]).index(name)]
        entry.update(parity(component, path, datasets(4) if name == 'classifier' else [RestorationDataset(split='val', condition=name, limit=16)]))
        write_json(path.with_suffix('.json'), entry)
    meta.update(smoke=False, **expected)
    write_json(pipeline_path, meta)
    reports['task2'] = meta
    print('Task 2 provenance and parity passed', flush=True)

    model, saved = load_task3(checkpoints['task3'])
    verify_checkpoint(saved, expected)
    if saved['initialization_sha256'] != {name: entry['checkpoint_sha256'] for name, entry in meta['components'].items()}:
        raise ValueError('Task 3 initialization differs from selected Task 2')
    path = STAGE/'models/task3/soft_moe.onnx'
    entry = metadata(path.with_suffix('.json'))
    if sha(path) != entry['onnx_sha256']:
        raise ValueError('Task 3 export hash differs')
    entry.update(parity(model, path, datasets(4)), config=saved['config'], epoch=saved['epoch'],
                 stage=saved['stage'], smoke=False, checkpoint_sha256=sha(checkpoints['task3']),
                 initialization_sha256=saved['initialization_sha256'], **expected)
    write_json(path.with_suffix('.json'), entry)
    reports['task3'] = entry
    print('Task 3 provenance and parity passed', flush=True)

    model, saved = load_task4(checkpoints['task4'])
    verify_checkpoint(saved, {'manifest_sha256': sha('data/prepared_v2/sketch/paired_samples.json')})
    path = STAGE/'models/task4/sketch_generator.onnx'
    entry = metadata(path.with_suffix('.json'))
    if sha(path) != entry['onnx_sha256'] or sha(checkpoints['task4']) != entry['checkpoint_sha256'] or entry['styles'] != list(STYLES):
        raise ValueError('Task 4 artifact provenance differs')
    entry.update(parity(model, path, [SketchDataset(split='val', augment=False)], sketch=True))
    write_json(path.with_suffix('.json'), entry)
    reports['task4'] = entry
    print('Task 4 provenance and all 786 validation parity cases passed', flush=True)

    # All artifacts have passed before any installed model is replaced.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = ROOT/'backups'/stamp
    for source, target in staged:
        if target.exists():
            old = backup/target
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, old)
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_suffix(target.suffix+'.pending')
        shutil.copy2(source, pending)
        pending.replace(target)
    release = dict(installed_at_utc=stamp, selections={'task1': 'Run A', 'task2': 'reviewed classifier + three specialists',
                   'task3': 'joint epoch '+str(reports['task3']['epoch']), 'task4': 'epoch '+str(reports['task4']['epoch'])},
                   archives={name: dict(path=str(path), sha256=sha(path)) for name, path in archives.items()},
                   files={str(target): sha(target) for _, target in staged}, verification=reports, backup_directory=str(backup))
    if Path('models/release.json').exists():
        backup.mkdir(parents=True, exist_ok=True)
        shutil.copy2('models/release.json', backup/'release.json')
    write_json('models/release.json', release)
    write_json(ROOT/'release.json', release)
    print('Installed verified models; release manifest: models/release.json')


if __name__ == '__main__':
    main()
