"""Fixed paired blur validation: severity metrics, controls, grids and review report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader
from corruptions import SEVERITIES
from dataset_loaders import RestorationDataset
from task1.metrics import measurements
from task1.train import load_checkpoint


def fixed_manifest(clean_rows):
    """Three specified blur severities and a clean control for each validation photo."""
    rows = []
    for source in clean_rows:
        rows.append(dict(**source, condition='clean', severity='none', parameters={}, seed=420000+source['source_id']*4))
        for i, parameters in enumerate(SEVERITIES['gaussian_blur']):
            rows.append(dict(**source, condition='gaussian_blur', severity=['low','medium','high'][i],
                             parameters=dict(parameters), seed=420000+source['source_id']*4+i+1))
    return rows


def summarize(rows):
    result = {}
    for severity in ['none','low','medium','high']:
        group = [r for r in rows if r['severity'] == severity]
        metrics = ['input_psnr','psnr','psnr_delta','input_ssim','ssim','ssim_delta','input_l1','l1','l1_improvement']
        result[severity] = dict(n=len(group), **{k: float(np.mean([r[k] for r in group])) for k in metrics},
                               median_psnr_delta=float(np.median([r['psnr_delta'] for r in group])),
                               psnr_improved_n=sum(r['psnr_delta'] > 1e-6 for r in group),
                               ssim_improved_n=sum(r['ssim_delta'] > 1e-6 for r in group),
                               both_improved_n=sum(r['psnr_delta'] > 1e-6 and r['ssim_delta'] > 1e-6 for r in group))
    return result


def write_csv(path, rows):
    with path.open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def pixels(tensor):
    return (tensor.cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)


def comparison_panel(x, target, output, row):
    """Full images and a shared edge-rich crop; error uses a fixed 4x display gain."""
    clean = pixels(target)
    gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gradient = cv2.Sobel(gray, cv2.CV_32F,1,0)**2+cv2.Sobel(gray,cv2.CV_32F,0,1)**2
    cy, cx = np.unravel_index(np.argmax(gradient[24:104,24:104]), (80,80))
    cy, cx = int(cy+24), int(cx+24)
    box = [cx-24,cy-24,cx+24,cy+24]
    canvas = Image.new('RGB',(512,322),'white'); draw = ImageDraw.Draw(canvas)
    draw.text((4,2), f"ID {row['source_id']} / {row['severity']} | PSNR gain {row['psnr_delta']:+.2f} dB | SSIM gain {row['ssim_delta']:+.3f}",fill='black')
    arrays = [clean,pixels(x),pixels(output),pixels((output-target).abs()*4)]
    for i, (title, array) in enumerate(zip(['Clean target','Input','Restored','Absolute error x4'],arrays)):
        draw.text((i*128+3,22),title,fill='black')
        canvas.paste(Image.fromarray(array),(i*128,40))
        draw.text((i*128+3,178),'Same edge crop',fill='black')
        crop = Image.fromarray(array).crop(box).resize((128,128),Image.Resampling.NEAREST)
        canvas.paste(crop,(i*128,194))
    return canvas, box


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--results-zip', type=Path, default=Path('colab/task1_spatial_results.zip'))
    parser.add_argument('--data', type=Path, default=Path('data/prepared_v2/restoration'))
    parser.add_argument('--output', type=Path, default=Path('runs/task1/spatial/blur_validation'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        source_name = str(checkpoint_path)
    else:
        checkpoint_path = args.output/'review_checkpoint.pt'
        with ZipFile(args.results_zip) as archive:
            checkpoint_path.write_bytes(archive.read('runs/task1/spatial/final/best.pt'))
        source_name = str(args.results_zip)
    model, checkpoint = load_checkpoint(checkpoint_path); model.eval()
    if checkpoint.get('smoke',False):
        raise ValueError('Use a fully trained checkpoint for this investigation')
    base_manifest = args.data/'validation_manifest.json'
    expected = checkpoint.get('validation_manifest_sha256')
    if not expected or expected != hashlib.sha256(base_manifest.read_bytes()).hexdigest():
        raise ValueError('Validation manifest does not match the checkpoint training provenance')
    clean_rows = json.loads((args.data/'val.json').read_text())
    if {r['source_id'] for r in clean_rows} != {r['source_id'] for r in json.loads(base_manifest.read_text())}:
        raise ValueError('Clean validation IDs do not match the original validation manifest')
    for source in clean_rows:
        from dataset_loaders import image_array
        current = (image_array(source['path'])*255).round().astype(np.uint8)
        if hashlib.sha256(current.tobytes()).hexdigest() != source['sha256']:
            raise ValueError(f"Source image content changed: {source['source_id']}")
    manifest = fixed_manifest(clean_rows)
    manifest_path = args.output/'manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    ds = RestorationDataset(args.data,'val',manifest=manifest_path)
    rows = []
    with torch.no_grad():
        for batch in DataLoader(ds,batch_size=32):
            predictions = model(batch['input'])
            baseline = measurements(batch['input'],batch['target'])
            restored = measurements(predictions,batch['target'])
            for i in range(len(predictions)):
                row = dict(source_id=int(batch['source_id'][i]),severity=batch['severity'][i],condition=batch['condition'][i])
                row.update({k: float(v[i]) for k,v in restored.items()})
                row.update({'input_'+k: float(v[i]) for k,v in baseline.items()})
                row.update(psnr_delta=row['psnr']-row['input_psnr'],ssim_delta=row['ssim']-row['input_ssim'],
                           l1_improvement=row['input_l1']-row['l1'])
                rows.append(row)
            if len(rows)%256 == 0:
                print(f'Evaluated {len(rows)}/{len(ds)} validation cases',flush=True)
    metrics = summarize(rows)
    write_csv(args.output/'per_image.csv',rows)
    write_csv(args.output/'grouped_metrics.csv',[dict(severity=k,**v) for k,v in metrics.items()])
    # Fixed representative photos are chosen without inspecting model outcomes.
    ids = sorted(r['source_id'] for r in clean_rows)
    representative = sorted(map(int,np.random.default_rng(42).choice(ids,min(4,len(ids)),replace=False)))
    selected = []
    for severity in ['low','medium','high']:
        group = [r for r in rows if r['severity']==severity]
        selected += [('representative',r) for r in group if r['source_id'] in representative]
        selected += [('largest_gain',r) for r in sorted(group,key=lambda r:(-r['psnr_delta'],r['source_id']))[:2]]
        selected += [('largest_loss',r) for r in sorted(group,key=lambda r:(r['psnr_delta'],r['source_id']))[:2]]
    loss_ids = {r['source_id'] for category,r in selected if category == 'largest_loss'}
    selected += [('clean_control',r) for r in rows if r['severity']=='none' and r['source_id'] in loss_ids]
    lookup = {(r['source_id'],r['severity']):i for i,r in enumerate(manifest)}
    examples, panels = [], {}
    with torch.no_grad():
        for category, row in selected:
            batch = ds[lookup[(row['source_id'],row['severity'])]]
            prediction = model(batch['input'][None])[0]
            panel, crop = comparison_panel(batch['input'],batch['target'],prediction,row)
            filename = f"{category}_{row['severity']}_{row['source_id']}.png"
            panel.save(args.output/filename)
            panels[(category,row['severity'],row['source_id'])] = panel
            examples.append(dict(category=category,filename=filename,crop_box=crop,**row))
    for category in ['representative','largest_gain','largest_loss']:
        group = [e for e in examples if e['category']==category]
        grid = Image.new('RGB',(1536,322*(len(group)//3)),'white')
        # Each column is a severity. Representatives use identical photos across columns.
        for col,severity in enumerate(['low','medium','high']):
            chosen = [e for e in group if e['severity']==severity]
            for row_index, e in enumerate(chosen):
                grid.paste(panels[(category,severity,e['source_id'])],(col*512,row_index*322))
        grid.save(args.output/(category+'_grid.png'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes = plt.subplots(1,2,figsize=(11,4))
    for axis,key in zip(axes,['psnr_delta','ssim_delta']):
        for severity in ['low','medium','high']:
            axis.hist([r[key] for r in rows if r['severity']==severity],bins=30,alpha=.45,label=severity)
        axis.axvline(0,color='black',linestyle='--');axis.set_xlabel('Restored minus input '+key.split('_')[0].upper())
        axis.set_ylabel('Validation photos');axis.legend()
    figure.tight_layout();figure.savefig(args.output/'gain_distributions.png');plt.close(figure)
    result = dict(split='validation_only',source=source_name,checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  checkpoint_epoch=checkpoint['epoch'],config=checkpoint['config'],photos=len(clean_rows),
                  blur_cases=len(clean_rows)*3,clean_controls=len(clean_rows),representative_source_ids=representative,
                  metrics=metrics,examples=examples)
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    report = ['# Fixed blur validation investigation','',
              f"Evaluated {len(clean_rows)} validation photographs at all three assignment blur settings ({len(clean_rows)*3} cases), plus {len(clean_rows)} clean controls. No training or test-set evaluation was performed. Checkpoint epoch: {checkpoint['epoch']}.", '',
              '| Severity | Input PSNR | Restored PSNR | Gain | Input SSIM | Restored SSIM | PSNR improved | Both improved |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for severity in ['low','medium','high']:
        m = metrics[severity]
        report.append(f"| {severity} | {m['input_psnr']:.2f} | {m['psnr']:.2f} | {m['psnr_delta']:+.2f} | {m['input_ssim']:.3f} | {m['ssim']:.3f} | {m['psnr_improved_n']}/{m['n']} | {m['both_improved_n']}/{m['n']} |")
    report += ['', 'Positive PSNR/SSIM differences indicate improvement. Positive L1 improvement means lower pixel error. Means are calculated per photo; all levels use identical photo IDs. The clean identity PSNR baseline is capped at 120 dB.', '',
               f"Clean controls: reconstructed PSNR {metrics['none']['psnr']:.2f} dB and SSIM {metrics['none']['ssim']:.3f}; the model introduces reconstruction error even when no blur is added.", '',
               '## Visual review', '',
               'Open `representative_grid.png` for four fixed photographs across all severities (columns: low, medium, high). Their IDs were selected with seed 42 before inspecting scores. `largest_gain_grid.png` and `largest_loss_grid.png` show two metric-selected cases per severity; these are deliberately selected extremes, not representative samples.', '',
               'Each panel contains clean target, blurred input, restored output and absolute error displayed with a fixed 4x gain. The second row enlarges the same 48x48 edge-rich target crop with nearest-neighbor scaling for all images. Compare boundaries, thin structures and texture; apparent sharpness alone does not establish faithful recovery.', '',
               'Individual `clean_control_none_*.png` panels show unblurred inputs for the photos with the largest blur losses. Use these to distinguish general reconstruction artifacts from failures introduced specifically by blurred input.', '',
               'These are validation diagnostics, not additional test results. Use them to design training experiments; keep model-selection decisions separate from final test reporting.']
    (args.output/'report.md').write_text('\n'.join(report)+'\n')
    import mlflow
    mlflow.set_tracking_uri('sqlite:///'+str(Path('mlflow.db').resolve()))
    mlflow.set_experiment('Task1 Universal Restoration')
    with mlflow.start_run(run_name='fixed-blur-validation'):
        mlflow.log_params(dict(checkpoint_source=source_name,checkpoint_epoch=checkpoint['epoch'],photos=len(clean_rows),blur_cases=len(clean_rows)*3))
        for severity,values in metrics.items():
            mlflow.log_metrics({severity+'_'+k:v for k,v in values.items()})
        for path in args.output.iterdir():
            if path.suffix != '.pt':
                mlflow.log_artifact(str(path),artifact_path='blur_validation')
    print(json.dumps(metrics,indent=2),flush=True)


if __name__ == '__main__':
    main()
