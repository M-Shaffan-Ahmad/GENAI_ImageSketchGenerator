"""Soft vs hard restoration, paired input comparisons, gate heatmaps and failures."""
import argparse
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
from task2.model import load_pipeline
from task2.metrics import classification_metrics
from task2.train import fingerprints
from task2.evaluate import write_csv
from task3.model import load_checkpoint


def summarize(rows):
    result=dict(n=len(rows),**{f'{name}_{metric}':float(np.mean([r[f'{name}_{metric}'] for r in rows]))
                            for name in ['input','hard','soft'] for metric in ['l1','ssim','psnr']})
    result.update(soft_psnr_minus_hard=float(np.mean([r['soft_psnr']-r['hard_psnr'] for r in rows])),
                  soft_psnr_minus_input=float(np.mean([r['soft_psnr']-r['input_psnr'] for r in rows])),
                  soft_psnr_better_than_input=sum(r['soft_psnr']>r['input_psnr']+1e-5 for r in rows),
                  soft_both_metrics_better_than_input=sum(r['soft_psnr']>r['input_psnr']+1e-5 and r['soft_ssim']>r['input_ssim']+1e-6 for r in rows),
                  mean_weights={c:float(np.mean([r['weight_'+c] for r in rows])) for c in CONDITIONS})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=Path('runs/task3/final/best.pt'))
    p.add_argument('--initialization',type=Path,default=Path('pretrained/task2'))
    p.add_argument('--data',default='data/prepared_v2/restoration')
    p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--severity-sweep',action='store_true')
    p.add_argument('--output',type=Path,default=Path('runs/task3/validation'))
    p.add_argument('--limit',type=int);p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args();torch.set_num_threads(4)
    soft,checkpoint=load_checkpoint(args.checkpoint,args.device);soft.eval()
    hard,_=load_pipeline(args.initialization,args.device)
    for k,v in fingerprints(args.data).items():
        if checkpoint[k]!=v:raise ValueError('Evaluation manifests changed')
    for c,digest in checkpoint['initialization_sha256'].items():
        if hashlib.sha256((args.initialization/c/'final/best.pt').read_bytes()).hexdigest()!=digest:
            raise ValueError('Hard-routing comparison does not match original Task 2 initialization')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=severity_manifest(args.data,args.split,args.output/'manifest.json') if args.severity_sweep else None
    ds=RestorationDataset(args.data,args.split,limit=args.limit,manifest=manifest)
    rows=[];representative={};worst=[];gate_examples={'dominant':None,'distributed':None}
    with torch.no_grad():
        for b in DataLoader(ds,batch_size=args.batch_size):
            x,target=b['input'].to(args.device),b['target'].to(args.device)
            output,weights=soft(x);hard_output,_,_=hard(x)
            scores={name:measurements(values,target) for name,values in [('input',x),('hard',hard_output),('soft',output)]}
            for i in range(len(x)):
                w=weights[i].cpu().numpy();dominant=int(w.argmax())
                row=dict(source_id=int(b['source_id'][i]),condition=b['condition'][i],severity=b['severity'][i],
                         true_label=int(b['label'][i]),dominant_label=dominant,dominant_condition=CONDITIONS[dominant],
                         entropy=float(-(w*np.log(np.clip(w,1e-12,1))).sum()),max_weight=float(w.max()))
                row.update({'weight_'+c:float(w[j]) for j,c in enumerate(CONDITIONS)})
                for name in scores:row.update({name+'_'+k:float(v[i]) for k,v in scores[name].items()})
                rows.append(row);key=row['condition']+'/'+row['severity'];picture=None
                if representative.get(key,0)<3:
                    picture=panel(x[i],target[i],output[i]);picture.save(args.output/f"example_{row['source_id']}_{row['condition']}_{row['severity']}.png")
                    representative[key]=representative.get(key,0)+1
                existing=next((entry for entry in worst if entry[1]['source_id']==row['source_id']),None)
                qualifies=row['soft_l1']>existing[0] if existing else len(worst)<4 or row['soft_l1']>worst[-1][0]
                if qualifies:
                    if existing:worst.remove(existing)
                    if picture is None:picture=panel(x[i],target[i],output[i])
                    worst.append((row['soft_l1'],row,picture));worst.sort(key=lambda r:r[0],reverse=True);del worst[4:]
                for category,score in [('dominant',row['max_weight']),('distributed',row['entropy'])]:
                    previous=gate_examples[category]
                    if previous is None or score>previous[0]:
                        if picture is None:picture=panel(x[i],target[i],output[i])
                        gate_examples[category]=(score,row,picture)
    if not rows:raise ValueError('No evaluation cases')
    groups=defaultdict(list)
    for row in rows:groups[row['condition']+'/'+row['severity']].append(row)
    metrics={key:summarize(group) for key,group in groups.items()};overall=summarize(rows)
    gate=classification_metrics([r['true_label'] for r in rows],[r['dominant_label'] for r in rows])
    result=dict(split=args.split,severity_sweep=args.severity_sweep,limited=bool(args.limit),smoke_checkpoint=checkpoint['smoke'],
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),epoch=checkpoint['epoch'],stage=checkpoint['stage'],
                config=checkpoint['config'],initialization_sha256=checkpoint['initialization_sha256'],
                gate_dominant_class_metrics=gate,overall=overall,metrics=metrics,
                branches_below_one_percent_mean=[c for c,v in overall['mean_weights'].items() if v<.01],
                gate_examples={c:row for c,(_,row,_) in gate_examples.items()},failure_candidates=[r for _,r,_ in worst])
    result['manifest_sha256']=hashlib.sha256((manifest or Path(args.data)/('validation_manifest.json' if args.split=='val' else 'test_manifest.json')).read_bytes()).hexdigest()
    (args.output/'results.json').write_text(json.dumps(result,indent=2));write_csv(args.output/'per_image.csv',rows)
    write_csv(args.output/'grouped_metrics.csv',[dict(group=key,**{k:v for k,v in values.items() if k!='mean_weights'},
                    **{'weight_'+c:values['mean_weights'][c] for c in CONDITIONS}) for key,values in metrics.items()])
    for index,(_,row,picture) in enumerate(worst,1):picture.save(args.output/f'failure_candidate_{index}.png')
    for category,(_,row,picture) in gate_examples.items():picture.save(args.output/f'gate_{category}.png')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matrix=np.array([[values['mean_weights'][c] for c in CONDITIONS] for values in metrics.values()])
    fig,ax=plt.subplots(figsize=(9,max(4,len(metrics)*.5)));chart=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto');fig.colorbar(chart,ax=ax)
    for i in range(len(matrix)):
        for j in range(4):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white' if matrix[i,j]>.5 else 'black')
    ax.set_xticks(range(4),CONDITIONS,rotation=25,ha='right');ax.set_yticks(range(len(metrics)),list(metrics));ax.set_xlabel('Branch mean weight');fig.tight_layout();fig.savefig(args.output/'routing_heatmap.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    for c in CONDITIONS:ax.hist([r['weight_'+c] for r in rows],bins=20,alpha=.4,label=c)
    ax.set_xlabel('Branch weight');ax.legend();fig.tight_layout();fig.savefig(args.output/'weight_distributions.png');plt.close(fig)
    (args.output/'report.md').write_text('# Task 3 validation\n\n'
        f"Stage: {checkpoint['stage']}, epoch {checkpoint['epoch']}; cases {len(rows)}; smoke {checkpoint['smoke']}.\n\n"
        f"Soft PSNR {overall['soft_psnr']:.3f}, hard PSNR {overall['hard_psnr']:.3f}.\n\n"
        'Inspect condition/severity scores and paired input comparisons: clean identity inflates hard-routing overall PSNR. '
        'Dominant-class accuracy is descriptive; a soft gate can appropriately blend branches without choosing the true class as its largest weight. '
        'Gate example panels select maximum confidence and maximum entropy, not necessarily a successful blend. '
        'A branch with mean weight below 1% is flagged for inspection; this threshold alone does not prove collapse. '
        'Training balance targets equal batch-average usage, not uniform per-image routing.\n')
    import mlflow
    mlflow.set_tracking_uri('sqlite:///'+str(Path('mlflow.db').resolve()));mlflow.set_experiment('Task3 Soft MoE')
    with mlflow.start_run(run_name='evaluation-'+args.split+('-severity' if args.severity_sweep else '')):
        mlflow.log_params(dict(split=args.split,severity_sweep=args.severity_sweep,stage=checkpoint['stage'],smoke=checkpoint['smoke']))
        mlflow.log_metrics({k:v for k,v in overall.items() if isinstance(v,(int,float))})
        mlflow.log_metrics({f'weight_{c}':v for c,v in overall['mean_weights'].items()})
        mlflow.log_artifacts(str(args.output),artifact_path='evaluation')
    print(json.dumps(dict(overall=overall,gate_dominant_class_metrics=gate),indent=2))


if __name__=='__main__':main()
