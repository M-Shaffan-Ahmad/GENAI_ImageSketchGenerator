"""Per-severity results, input baseline, example grids and worst-case candidates."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from dataset_loaders import RestorationDataset
from task1.metrics import measurements
from task1.train import load_checkpoint


def panel(x, target, prediction):
    canvas = Image.new('RGB', (512, 150), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (title, values) in enumerate(zip(['Clean', 'Input', 'Restored', 'Absolute error'], [target, x, prediction, (prediction-target).abs()])):
        pixels = (values.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)*255).astype(np.uint8)
        canvas.paste(Image.fromarray(pixels), (i*128, 22)); draw.text((i*128+3, 3), title, fill='black')
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default='data/prepared_v2/restoration')
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--output', type=Path, default=Path('runs/task1/evaluation'))
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    torch.set_num_threads(4)
    model, checkpoint = load_checkpoint(args.checkpoint); model.eval()
    ds = RestorationDataset(args.data, args.split, limit=args.limit)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, representative, worst = [], {}, []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=32):
            output = model(b['input'])
            restored = measurements(output, b['target']); baseline = measurements(b['input'], b['target'])
            for i in range(len(output)):
                row = dict(source_id=int(b['source_id'][i]), condition=b['condition'][i], severity=b['severity'][i])
                row.update({k: float(v[i]) for k, v in restored.items()})
                row.update({'input_'+k: float(v[i]) for k, v in baseline.items()})
                rows.append(row)
                key = row['condition']+'/'+row['severity']
                if representative.get(key, 0) < 3:
                    panel(b['input'][i], b['target'][i], output[i]).save(args.output/f"example_{row['source_id']}_{row['condition']}_{row['severity']}.png")
                    representative[key] = representative.get(key, 0)+1
                existing = next((entry for entry in worst if entry[1]['source_id']==row['source_id']), None)
                qualifies = row['l1'] > existing[0] if existing else len(worst) < 4 or row['l1'] > worst[-1][0]
                if qualifies:
                    if existing:
                        worst.remove(existing)
                    worst.append((row['l1'], row, panel(b['input'][i], b['target'][i], output[i])))
                    worst.sort(key=lambda r: r[0], reverse=True); worst = worst[:4]
            if len(rows) % 8192 == 0:
                print(f'Evaluated {len(rows)}/{len(ds)} cases', flush=True)
    with (args.output/'per_image.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row['condition']+'/'+row['severity']].append(row)
    summary = {k: dict(n=len(v), **{m: float(np.mean([r[m] for r in v])) for m in ['l1','ssim','psnr','input_l1','input_ssim','input_psnr']}) for k, v in groups.items()}
    grouped = [dict(group=k, **v) for k, v in summary.items()]
    with (args.output/'grouped_metrics.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(grouped[0])); w.writeheader(); w.writerows(grouped)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    positions = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(positions-.2, [v['input_ssim'] for v in summary.values()], .4, label='Input baseline')
    ax.bar(positions+.2, [v['ssim'] for v in summary.values()], .4, label='Restored')
    ax.set_xticks(positions, list(summary), rotation=35, ha='right'); ax.set_ylabel('SSIM'); ax.legend()
    fig.tight_layout(); fig.savefig(args.output/'ssim_comparison.png'); plt.close(fig)
    for i, (_, row, picture) in enumerate(worst):
        picture.save(args.output/f'failure_candidate_{i+1}.png')
    result = dict(split=args.split, limited=bool(args.limit), smoke_checkpoint=checkpoint.get('smoke', False), metrics=summary,
                  failure_candidates=[r for _, r, _ in worst])
    (args.output/'results.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
    import mlflow
    mlflow.set_tracking_uri('sqlite:///'+str(Path('mlflow.db').resolve()))
    mlflow.set_experiment('Task1 Universal Restoration')
    with mlflow.start_run(run_name='evaluation-'+args.split):
        mlflow.log_params(dict(split=args.split, checkpoint=args.checkpoint, limited=bool(args.limit)))
        for group, values in summary.items():
            mlflow.log_metrics({group.replace('/', '_')+'_'+k: v for k, v in values.items() if k != 'n'})
        mlflow.log_artifacts(str(args.output), artifact_path='evaluation')


if __name__ == '__main__':
    main()
