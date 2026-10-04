"""Frozen-checkpoint test comparison for Tasks 1-3, sharing inputs/baselines.

Run from the repository root. --limit is for verification only and is recorded.
Task 4 uses its existing paired evaluator. No tuning, training or model changes.
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from corruptions import CONDITIONS
from dataset_loaders import RestorationDataset
from task1.evaluate import panel
from task1.metrics import measurements
from task1.train import load_checkpoint as load1
from task2.model import load_pipeline
from task2.metrics import classification_metrics
from task2.train import fingerprints
from task3.model import load_checkpoint as load3


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',type=Path,default=Path('runs/integration/staging/checkpoints'))
    p.add_argument('--output',type=Path,default=Path('runs/final_test/restoration'))
    p.add_argument('--limit',type=int)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args();torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True,exist_ok=True)
    release=json.loads(Path('models/release.json').read_text())
    universal,c1=load1(args.checkpoints/'task1/best.pt');universal.eval()
    hard,c2=load_pipeline(args.checkpoints/'task2')
    soft,c3=load3(args.checkpoints/'task3/best.pt');soft.eval()
    expected=fingerprints('data/prepared_v2/restoration')
    for c in [c1,*c2.values(),c3]:
        if c.get('smoke',True) or any(c[k]!=v for k,v in expected.items()):
            raise ValueError('Checkpoint smoke flag or data provenance mismatch')
    for task,checkpoint in [('task1',args.checkpoints/'task1/best.pt'),('task3',args.checkpoints/'task3/best.pt')]:
        if sha(checkpoint)!=release['verification'][task]['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed')
    for name in c2:
        if sha(args.checkpoints/f'task2/{name}/final/best.pt')!=release['verification']['task2']['components'][name]['checkpoint_sha256']:
            raise ValueError('Task 2 selected checkpoint changed')
    ds=RestorationDataset(split='test',limit=args.limit)
    if not args.limit and len(ds)!=61490:raise ValueError('Unexpected full test size')
    models=['input','universal','oracle','hard','soft'];rows=[];representative=defaultdict(int);worst={m:[] for m in models[1:]}
    start=time.perf_counter()
    with torch.inference_mode():
        for batch in DataLoader(ds,batch_size=args.batch_size):
            x,target=batch['input'],batch['target'];labels=batch['label']
            probabilities=hard.classifier(x).softmax(1);routes=probabilities.argmax(1)
            branches=torch.stack([x,*[expert(x) for expert in hard.specialists]],dim=1)
            index=torch.arange(len(x))
            soft_output,weights=soft(x)
            outputs=dict(input=x,universal=universal(x),oracle=branches[index,labels],hard=branches[index,routes],soft=soft_output)
            scores={name:measurements(value,target) for name,value in outputs.items()}
            for i in range(len(x)):
                row=dict(source_id=int(batch['source_id'][i]),condition=batch['condition'][i],severity=batch['severity'][i],
                         true_label=int(labels[i]),predicted_label=int(routes[i]),dominant_label=int(weights[i].argmax()))
                for name,values in scores.items():
                    row.update({f'{name}_{metric}':float(value[i]) for metric,value in values.items()})
                row.update({f'weight_{condition}':float(weights[i,j]) for j,condition in enumerate(CONDITIONS)})
                row.update({f'probability_{condition}':float(probabilities[i,j]) for j,condition in enumerate(CONDITIONS)})
                rows.append(row);group=row['condition']+'/'+row['severity']
                if representative[group]<2:
                    for name in ['universal','hard','soft']:
                        destination=args.output/name;destination.mkdir(exist_ok=True)
                        panel(x[i],target[i],outputs[name][i]).save(destination/f"example_{row['source_id']}_{row['condition']}_{row['severity']}.png")
                    representative[group]+=1
                for name in worst:
                    old=next((entry for entry in worst[name] if entry[1]['source_id']==row['source_id']),None)
                    score=row[name+'_l1']
                    qualifies=score>old[0] if old else len(worst[name])<4 or score>worst[name][-1][0]
                    if qualifies:
                        if old:worst[name].remove(old)
                        worst[name].append((score,row,panel(x[i],target[i],outputs[name][i])))
                        worst[name].sort(key=lambda entry:entry[0],reverse=True);del worst[name][4:]
            if len(rows)%1024==0 or len(rows)==len(ds):
                elapsed=time.perf_counter()-start
                progress=dict(completed=len(rows),total=len(ds),elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed*(len(ds)-len(rows))/len(rows))
                (args.output/'progress.json').write_text(json.dumps(progress,indent=2))
                print(f"Evaluated {len(rows)}/{len(ds)}; elapsed {elapsed:.0f}s; remaining ~{progress['estimated_remaining_seconds']:.0f}s",flush=True)
    with (args.output/'per_image.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups=defaultdict(list)
    for row in rows:groups[row['condition']+'/'+row['severity']].append(row)
    metrics={}
    for group,values in groups.items():
        stats=dict(n=len(values))
        for name in models:
            stats[name]={metric:float(np.mean([r[f'{name}_{metric}'] for r in values])) for metric in ['l1','ssim','psnr']}
            if name!='input':
                stats[name]['psnr_better_than_input']=sum(r[f'{name}_psnr']>r['input_psnr']+1e-5 for r in values)
                stats[name]['both_metrics_better_than_input']=sum(r[f'{name}_psnr']>r['input_psnr']+1e-5 and r[f'{name}_ssim']>r['input_ssim']+1e-6 for r in values)
        stats['mean_weights']={c:float(np.mean([r['weight_'+c] for r in values])) for c in CONDITIONS}
        stats['routing_accuracy']=float(np.mean([r['true_label']==r['predicted_label'] for r in values]))
        stats['cascading_failures']=sum(r['true_label']!=r['predicted_label'] and r['hard_psnr']<r['oracle_psnr']-1e-5 for r in values)
        metrics[group]=stats
    failures={}
    for name,entries in worst.items():
        destination=args.output/name;destination.mkdir(exist_ok=True)
        failures[name]=[row for _,row,_ in entries]
        for i,(_,row,picture) in enumerate(entries,1):picture.save(destination/f'failure_candidate_{i}.png')
    result=dict(split='test',limited=bool(args.limit),smoke_checkpoint=False,cases=len(rows),unique_photos=len({r['source_id'] for r in rows}),
        manifest_sha256=sha('data/prepared_v2/restoration/test_manifest.json'),selection_frozen_release_sha256=sha('models/release.json'),
        selected_epochs={'task1':c1['epoch'],'task2':{k:v['epoch'] for k,v in c2.items()},'task3':c3['epoch']},
        checkpoint_sha256={'task1':release['verification']['task1']['checkpoint_sha256'],'task2':{k:v['checkpoint_sha256'] for k,v in release['verification']['task2']['components'].items()},'task3':release['verification']['task3']['checkpoint_sha256']},
        environment=dict(python=platform.python_version(),torch=torch.__version__,processor=platform.processor(),device='cpu',threads=args.threads,batch_size=args.batch_size),
        elapsed_seconds=time.perf_counter()-start,classifier=classification_metrics([r['true_label'] for r in rows],[r['predicted_label'] for r in rows]),
        gate_dominant_class_metrics=classification_metrics([r['true_label'] for r in rows],[r['dominant_label'] for r in rows]),metrics=metrics,failure_candidates=failures)
    (args.output/'results.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matrix=np.asarray(result['classifier']['normalized_confusion_matrix'])
    fig,ax=plt.subplots(figsize=(6,5));chart=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues');fig.colorbar(chart,ax=ax)
    for i in range(4):
        for j in range(4):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='white' if matrix[i,j]>.5 else 'black')
    ax.set_xticks(range(4),['Clean','Noise','Blur','Occlusion'],rotation=25);ax.set_yticks(range(4),['Clean','Noise','Blur','Occlusion']);ax.set_xlabel('Predicted');ax.set_ylabel('True');fig.tight_layout();fig.savefig(args.output/'confusion_matrix.png',dpi=180);plt.close(fig)
    matrix=np.asarray([[v['mean_weights'][c] for c in CONDITIONS] for v in metrics.values()])
    fig,ax=plt.subplots(figsize=(7,6));chart=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto');fig.colorbar(chart,ax=ax)
    for i in range(len(matrix)):
        for j in range(4):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white' if matrix[i,j]>.5 else 'black')
    ax.set_xticks(range(4),['Identity','Noise','Blur','Occlusion']);ax.set_yticks(range(len(metrics)),list(metrics));fig.tight_layout();fig.savefig(args.output/'routing_heatmap.png',dpi=180);plt.close(fig)
    import mlflow
    mlflow.set_tracking_uri('sqlite:///'+str(Path('mlflow.db').resolve()));mlflow.set_experiment('Final Frozen Test Comparison')
    with mlflow.start_run(run_name='tasks1-3-test'):
        mlflow.log_params(dict(split='test',limited=bool(args.limit),cases=len(rows),threads=args.threads,release_sha256=result['selection_frozen_release_sha256']))
        mlflow.log_metrics(dict(classifier_accuracy=result['classifier']['accuracy'],classifier_macro_f1=result['classifier']['macro_f1']))
        for group,values in metrics.items():
            for name in models:
                mlflow.log_metrics({group.replace('/','_')+'_'+name+'_'+k:v for k,v in values[name].items()})
        mlflow.log_artifacts(str(args.output),artifact_path='final_test')
    print('Completed frozen test comparison:',len(rows),'cases',flush=True)


if __name__=='__main__':main()
