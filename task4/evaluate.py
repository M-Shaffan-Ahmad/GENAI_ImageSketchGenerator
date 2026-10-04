"""Per-style paired validation, blank baseline, style diversity and failures."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image,ImageDraw
from task1.metrics import measurements
from task2.evaluate import write_csv
from task4.model import load_checkpoint,STYLES
from task4.train import limited_dataset,manifest_hash


def panel(photo,target,prediction,style):
    canvas=Image.new('RGB',(512,150),'white');draw=ImageDraw.Draw(canvas)
    for i,(title,tensor) in enumerate(zip(['Photo',STYLES[style]+' target','Generated','Absolute error'],[photo,target,prediction,(prediction-target).abs()])):
        pixels=(tensor.detach().cpu().permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
        canvas.paste(Image.fromarray(pixels),(i*128,22));draw.text((i*128+2,3),title,fill='black')
    return canvas


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=Path('runs/task4/final/best.pt'))
    p.add_argument('--manifest',type=Path,default=Path('data/prepared_v2/sketch/paired_samples.json'))
    p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--output',type=Path,default=Path('runs/task4/validation'))
    p.add_argument('--limit',type=int);p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args();torch.set_num_threads(4)
    model,c=load_checkpoint(args.checkpoint,args.device);model.eval()
    if c['manifest_sha256']!=manifest_hash(args.manifest):raise ValueError('Evaluation manifest differs from training')
    dataset=limited_dataset(args.manifest,args.split,args.limit);args.output.mkdir(parents=True,exist_ok=True)
    rows=[];pending={};diversity=[];representative=defaultdict(int);worst=[]
    with torch.no_grad():
        for b in DataLoader(dataset,batch_size=args.batch_size):
            photo,target,style=b['input'].to(args.device),b['target'].to(args.device),b['style'].to(args.device)
            output=model(photo,style);scores=measurements(output,target);blank=measurements(torch.ones_like(target),target)
            for i in range(len(output)):
                source=int(b['source_id'][i]);s=int(style[i]);row=dict(source_id=source,style_id=s+1,style=STYLES[s])
                row.update({k:float(v[i]) for k,v in scores.items()});row.update({'blank_'+k:float(v[i]) for k,v in blank.items()});rows.append(row)
                if representative[s]<4:
                    panel(photo[i],target[i],output[i],s).save(args.output/f'example_{source}_style_{s+1}.png');representative[s]+=1
                existing=next((e for e in worst if e[1]['source_id']==source),None)
                qualifies=row['l1']>existing[0] if existing else len(worst)<4 or row['l1']>worst[-1][0]
                if qualifies:
                    if existing:worst.remove(existing)
                    worst.append((row['l1'],row,panel(photo[i],target[i],output[i],s)));worst.sort(key=lambda e:e[0],reverse=True);del worst[4:]
                pending.setdefault(source,{})[s]=(output[i].cpu(),target[i].cpu())
                if len(pending[source])==3:
                    group=pending.pop(source)
                    for a,d in [(0,1),(0,2),(1,2)]:
                        diversity.append(dict(source_id=source,style_pair=f'{a+1}-{d+1}',
                          generated_l1=float((group[a][0]-group[d][0]).abs().mean()),target_l1=float((group[a][1]-group[d][1]).abs().mean())))
    groups=defaultdict(list)
    for row in rows:groups[row['style']].append(row)
    metrics={style:dict(n=len(group),**{k:float(np.mean([r[k] for r in group])) for k in ['l1','ssim','psnr','blank_l1','blank_ssim','blank_psnr']}) for style,group in groups.items()}
    result=dict(split=args.split,limited=bool(args.limit),smoke_checkpoint=c['smoke'],epoch=c['epoch'],config=c['config'],
                manifest_sha256=c['manifest_sha256'],checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),metrics=metrics,
                complete_style_comparison_photos=len(diversity)//3,
                style_diversity={pair:dict(generated_l1=float(np.mean([r['generated_l1'] for r in diversity if r['style_pair']==pair])),
                                         target_l1=float(np.mean([r['target_l1'] for r in diversity if r['style_pair']==pair])))
                                for pair in sorted({r['style_pair'] for r in diversity})},failure_candidates=[r for _,r,_ in worst])
    (args.output/'results.json').write_text(json.dumps(result,indent=2));write_csv(args.output/'per_image.csv',rows)
    write_csv(args.output/'grouped_metrics.csv',[dict(style=style,**values) for style,values in metrics.items()]);write_csv(args.output/'style_diversity.csv',diversity)
    for i,(_,row,picture) in enumerate(worst,1):picture.save(args.output/f'failure_candidate_{i}.png')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    positions=np.arange(len(metrics));fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(positions-.2,[v['blank_ssim'] for v in metrics.values()],.4,label='Blank white baseline');ax.bar(positions+.2,[v['ssim'] for v in metrics.values()],.4,label='Generator')
    ax.set_xticks(positions,list(metrics));ax.set_ylabel('SSIM');ax.legend();fig.tight_layout();fig.savefig(args.output/'style_metrics.png');plt.close(fig)
    (args.output/'report.md').write_text('# Task 4 paired sketch evaluation\n\n'
      f"Split {args.split}; pairs {len(rows)}; epoch {c['epoch']}; smoke {c['smoke']}.\n\n"
      'Targets are deterministic synthetic pencil, ink and charcoal approximations, not human-drawn sketch annotations. '
      'Blank white baseline helps expose background-driven scores. Pairwise generated-style L1 versus target-style L1 measures output diversity, not style correctness. '
      'Inspect the same photograph across all three generated styles and the separate target/error panels. Failure candidates use distinct source photos. '
      'Test evaluation is separate and does not run in the supplied Colab notebook.\n')
    import mlflow
    mlflow.set_tracking_uri('sqlite:///'+str(Path('mlflow.db').resolve()));mlflow.set_experiment('Task4 Style Sketch cGAN')
    with mlflow.start_run(run_name='evaluation-'+args.split):
        mlflow.log_params(dict(split=args.split,limited=bool(args.limit),smoke=c['smoke']))
        for style,values in metrics.items():mlflow.log_metrics({style.replace(' ','_')+'_'+k:v for k,v in values.items()})
        mlflow.log_artifacts(str(args.output),artifact_path='evaluation')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
