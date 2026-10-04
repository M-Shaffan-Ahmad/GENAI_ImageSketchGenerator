"""Classifier metrics, oracle/predicted restoration and routing failure evidence."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from corruptions import CONDITIONS
from dataset_loaders import RestorationDataset
from task1.metrics import measurements
from task1.evaluate import panel
from task2.data import severity_manifest
from task2.model import load_pipeline, route_images
from task2.metrics import classification_metrics
from task2.train import fingerprints


def write_csv(path, rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summarize(rows):
    columns=[f'{route}_{metric}' for route in ['input','oracle','predicted'] for metric in ['l1','ssim','psnr']]
    return dict(n=len(rows),routing_accuracy=float(np.mean([r['routing_correct'] for r in rows])),
                **{key:float(np.mean([r[key] for r in rows])) for key in columns},
                oracle_minus_predicted_psnr=float(np.mean([r['oracle_psnr']-r['predicted_psnr'] for r in rows])),
                cascading_failures=sum(not r['routing_correct'] and r['predicted_psnr']<r['oracle_psnr'] for r in rows),
                predicted_psnr_better_than_input=sum(r['predicted_psnr']>r['input_psnr'] for r in rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',type=Path,default=Path('runs/task2'))
    p.add_argument('--data',default='data/prepared_v2/restoration')
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--severity-sweep',action='store_true')
    p.add_argument('--output',type=Path,default=Path('runs/task2/validation'))
    p.add_argument('--limit',type=int);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
    args=p.parse_args();torch.set_num_threads(args.threads)
    model,checkpoints=load_pipeline(args.checkpoints,args.device)
    expected=fingerprints(args.data)
    if any(c[k]!=v for c in checkpoints.values() for k,v in expected.items()):
        raise ValueError('Evaluation data differs from training/validation checkpoint manifests')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=severity_manifest(args.data,args.split,args.output/'manifest.json') if args.severity_sweep else None
    ds=RestorationDataset(args.data,args.split,limit=args.limit,manifest=manifest)
    if not ds:raise ValueError('No evaluation cases')
    rows,representative,failures,worst=[],{},[],[]
    def remember(collection,score,row,picture):
        existing=next((entry for entry in collection if entry[1]['source_id']==row['source_id']),None)
        if existing and existing[0]>=score:return
        if not existing and len(collection)>=4 and collection[-1][0]>=score:return
        if existing:collection.remove(existing)
        collection.append((score,row,picture));collection.sort(key=lambda e:e[0],reverse=True)
        del collection[4:]
    with torch.no_grad():
        for batch in DataLoader(ds,batch_size=args.batch_size):
            x,target=batch['input'].to(args.device),batch['target'].to(args.device)
            predicted,probabilities,routes=model(x)
            labels=batch['label'].to(args.device)
            oracle=route_images(x,labels,model.specialists)
            values={name:measurements(image,target) for name,image in [('input',x),('oracle',oracle),('predicted',predicted)]}
            for i in range(len(x)):
                label,route=int(labels[i]),int(routes[i])
                row=dict(source_id=int(batch['source_id'][i]),condition=batch['condition'][i],severity=batch['severity'][i],
                         true_label=label,predicted_label=route,predicted_condition=CONDITIONS[route],
                         confidence=float(probabilities[i,route]),routing_correct=label==route)
                for name in values:
                    row.update({name+'_'+metric:float(scores[i]) for metric,scores in values[name].items()})
                row.update({condition+'_probability':float(probabilities[i,j]) for j,condition in enumerate(CONDITIONS)})
                row['oracle_minus_predicted_psnr']=row['oracle_psnr']-row['predicted_psnr']
                rows.append(row)
                key=row['condition']+'/'+row['severity']
                picture=None
                if representative.get(key,0)<3:
                    picture=panel(x[i],target[i],predicted[i])
                    picture.save(args.output/f"example_{row['source_id']}_{row['condition']}_{row['severity']}.png")
                    representative[key]=representative.get(key,0)+1
                if not row['routing_correct'] or len(worst)<4 or row['predicted_l1']>worst[-1][0]:
                    if picture is None:picture=panel(x[i],target[i],predicted[i])
                    remember(worst,row['predicted_l1'],row,picture)
                    if not row['routing_correct'] and row['oracle_minus_predicted_psnr']>0:
                        remember(failures,row['oracle_minus_predicted_psnr'],row,picture)
    classifier=classification_metrics([r['true_label'] for r in rows],[r['predicted_label'] for r in rows])
    groups=defaultdict(list);pairs=defaultdict(list)
    for row in rows:
        groups[row['condition']+'/'+row['severity']].append(row)
        pairs[row['condition']+' -> '+row['predicted_condition']].append(row)
    grouped={key:summarize(group) for key,group in groups.items()}
    for category,collection in [('restoration_failure',worst),('routing_failure',failures)]:
        for index,(_,row,picture) in enumerate(collection,1):picture.save(args.output/f'{category}_{index}.png')
    manifest_path=manifest or Path(args.data)/('validation_manifest.json' if args.split=='val' else 'test_manifest.json')
    checkpoint_hashes={component:hashlib.sha256((args.checkpoints/component/'final/best.pt').read_bytes()).hexdigest() for component in checkpoints}
    result=dict(split=args.split,severity_sweep=args.severity_sweep,limited=bool(args.limit),
                smoke_checkpoint=any(c['smoke'] for c in checkpoints.values()),conditions=list(CONDITIONS),
                checkpoint_sha256=checkpoint_hashes,checkpoint_epochs={k:c['epoch'] for k,c in checkpoints.items()},
                configs={k:c['config'] for k,c in checkpoints.items()},manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                classifier=classifier,overall=summarize(rows),metrics=grouped,
                routing_pairs={key:summarize(group) for key,group in pairs.items()},
                restoration_failure_candidates=[row for _,row,_ in worst],routing_failure_candidates=[row for _,row,_ in failures])
    (args.output/'results.json').write_text(json.dumps(result,indent=2))
    write_csv(args.output/'per_image.csv',rows)
    write_csv(args.output/'grouped_metrics.csv',[dict(group=key,**scores) for key,scores in grouped.items()])
    write_csv(args.output/'routing_pairs.csv',[dict(route=key,**scores) for key,scores in result['routing_pairs'].items()])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,6))
    matrix=np.asarray(classifier['normalized_confusion_matrix'])
    chart=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues');fig.colorbar(chart,ax=ax)
    for i in range(4):
        for j in range(4):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white' if matrix[i,j]>.5 else 'black')
    ax.set_xticks(range(4),CONDITIONS,rotation=25,ha='right');ax.set_yticks(range(4),CONDITIONS)
    ax.set_xlabel('Predicted');ax.set_ylabel('Ground truth');fig.tight_layout();fig.savefig(args.output/'confusion_matrix.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5));positions=np.arange(len(grouped))
    for offset,name in [(-.25,'input'),(0,'oracle'),(.25,'predicted')]:
        ax.bar(positions+offset,[v[name+'_ssim'] for v in grouped.values()],.25,label=name)
    ax.set_xticks(positions,list(grouped),rotation=35,ha='right');ax.set_ylabel('SSIM');ax.legend()
    fig.tight_layout();fig.savefig(args.output/'routing_comparison.png');plt.close(fig)
    (args.output/'report.md').write_text(
        '# Task 2 routing evaluation\n\n'
        f"Split: {args.split}; severity sweep: {args.severity_sweep}; cases: {len(rows)}; smoke: {result['smoke_checkpoint']}.\n\n"
        f"Classifier accuracy: {classifier['accuracy']:.4f}; macro F1: {classifier['macro_f1']:.4f}.\n\n"
        f"Oracle PSNR: {result['overall']['oracle_psnr']:.3f}; predicted-routing PSNR: {result['overall']['predicted_psnr']:.3f}.\n\n"
        f"Misrouted cases with worse PSNR than oracle: {result['overall']['cascading_failures']}. Wrong routes can also improve scores accidentally, especially when blur is bypassed; see per-image data.\n\n"
        'Rows of the confusion matrix are true classes; columns are predictions. Missing-class F1 and normalized rows are zero. '
        'The severity sweep has one clean and nine corrupted cases per photo, so accuracy is not class-balanced; inspect macro F1 and class support. '
        'PSNR for exact clean identity is capped at 120 dB. Compare clean and corrupted groups separately; identity cases inflate overall PSNR.\n\n'
        'Example panels show clean target, input, predicted output and unamplified absolute error. '
        'Routing failures are selected by oracle-minus-predicted PSNR; restoration failures by predicted L1, with distinct source photos. '
        'Failure filenames and source/route metadata appear in results.json.\n')
    import mlflow
    mlflow.set_tracking_uri('sqlite:///'+str(Path('mlflow.db').resolve()));mlflow.set_experiment('Task2 Hard Routing')
    with mlflow.start_run(run_name='evaluation-'+args.split+('-severity' if args.severity_sweep else '')):
        mlflow.log_params(dict(split=args.split,severity_sweep=args.severity_sweep,limited=bool(args.limit),smoke=result['smoke_checkpoint']))
        mlflow.log_metrics({k:classifier[k] for k in ['accuracy','macro_f1']})
        mlflow.log_metrics(result['overall']);mlflow.log_artifacts(str(args.output),artifact_path='evaluation')
    print(json.dumps(dict(classifier=classifier,overall=result['overall']),indent=2))


if __name__=='__main__':main()
