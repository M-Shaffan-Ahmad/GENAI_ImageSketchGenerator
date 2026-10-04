"""Paired cGAN training with Optuna, tracked losses, fixed samples and resume."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import mlflow
import numpy as np
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset_loaders import SketchDataset
from task1.metrics import measurements
from task1.train import seed_all
from task4.model import SketchGenerator,PatchDiscriminator,generator_config,STYLES


def manifest_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def limited_dataset(manifest,split,limit=None,augment=False):
    dataset=SketchDataset(manifest,split,augment)
    if limit:dataset.rows=dataset.rows[:limit]
    if not len(dataset):raise ValueError('No paired samples')
    return dataset


@torch.no_grad()
def validate(generator,loader,device):
    generator.eval();totals={k:0. for k in ['l1','ssim','psnr']};n=0
    for batch in loader:
        image,target,style=batch['input'].to(device),batch['target'].to(device),batch['style'].to(device)
        for k,v in measurements(generator(image,style),target).items():totals[k]+=v.sum().item()
        n+=len(image)
    return {k:v/n for k,v in totals.items()}


@torch.no_grad()
def fixed_samples(generator,dataset,device,path):
    from PIL import Image,ImageDraw
    generator.eval();panels=[]
    # Four fixed photos, each with all styles. No random augmentation in validation.
    for index in range(min(12,len(dataset))):
        row=dataset[index];prediction=generator(row['input'][None].to(device),torch.tensor([row['style']],device=device))[0].cpu()
        canvas=Image.new('RGB',(512,150),'white');draw=ImageDraw.Draw(canvas)
        for i,(name,tensor) in enumerate(zip(['Photo',STYLES[row['style']]+' target','Generated','Absolute error'],
                                            [row['input'],row['target'],prediction,(prediction-row['target']).abs()])):
            pixels=(tensor.permute(1,2,0).numpy().clip(0,1)*255).astype(np.uint8)
            canvas.paste(Image.fromarray(pixels),(i*128,22));draw.text((i*128+2,3),name,fill='black')
        panels.append(canvas)
    grid=Image.new('RGB',(512,150*len(panels)),'white')
    for i,panel in enumerate(panels):grid.paste(panel,(0,i*150))
    grid.save(path)


def train(config,args,output,trial=None):
    seed_all(config['seed']);device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    train_set=limited_dataset(args.manifest,'train',args.train_limit,True)
    val_set=limited_dataset(args.manifest,'val',args.val_limit)
    shuffle=torch.Generator().manual_seed(config['seed'])
    loader=DataLoader(train_set,batch_size=config['batch_size'],shuffle=True,num_workers=args.workers,generator=shuffle)
    val_loader=DataLoader(val_set,batch_size=config['batch_size'],num_workers=args.workers)
    generator=SketchGenerator(**generator_config(config)).to(device)
    discriminator=PatchDiscriminator(config['base_channels'],config['embedding_dim']).to(device)
    g_optimizer=torch.optim.Adam(generator.parameters(),lr=config['generator_learning_rate'],betas=(.5,.999))
    d_optimizer=torch.optim.Adam(discriminator.parameters(),lr=config['discriminator_learning_rate'],betas=(.5,.999))
    criterion=nn.BCEWithLogitsLoss();fingerprint=manifest_hash(args.manifest)
    history=[];best=float('inf');start=0;output.mkdir(parents=True,exist_ok=True)
    if args.resume:
        saved=torch.load(args.resume,map_location=device,weights_only=True)
        if saved.get('component')!='sketch_cgan' or saved.get('styles')!=list(STYLES):raise ValueError('Wrong resume checkpoint')
        if saved['manifest_sha256']!=fingerprint or saved['train_pairs']!=len(train_set) or saved['validation_pairs']!=len(val_set):raise ValueError('Resume data changed')
        if any(saved['config'].get(k)!=v for k,v in config.items() if k!='epochs'):raise ValueError('Resume configuration changed')
        generator.load_state_dict(saved['generator']);discriminator.load_state_dict(saved['discriminator'])
        g_optimizer.load_state_dict(saved['g_optimizer']);d_optimizer.load_state_dict(saved['d_optimizer'])
        history=saved['history'];best=saved['best_objective'];start=saved['epoch']
        torch.set_rng_state(saved['torch_rng'].cpu());shuffle.set_state(saved['shuffle_rng'].cpu())
        if device.startswith('cuda') and saved.get('cuda_rng'):torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
        previous=args.resume.parent/'best.pt'
        if not previous.exists():raise ValueError('Keep best.pt next to last.pt')
        if previous.resolve()!=(output/'best.pt').resolve():shutil.copy2(previous,output/'best.pt')
        if start>=config['epochs']:raise ValueError('Already completed; increase total epochs to extend')
    (output/'config.json').write_text(json.dumps(config,indent=2))
    with mlflow.start_run(run_name=output.name):
        mlflow.log_params(dict(config,manifest_sha256=fingerprint,train_pairs=len(train_set),validation_pairs=len(val_set),device=device))
        mlflow.set_tag('purpose','smoke' if args.train_limit or args.val_limit else 'full')
        for epoch in range(start,config['epochs']):
            generator.train();discriminator.train()
            totals={k:0. for k in ['d_real','d_fake','g_adversarial','g_l1','g_total']};n=0
            for batch in loader:
                x,target,style=batch['input'].to(device),batch['target'].to(device),batch['style'].to(device)
                prediction=generator(x,style)
                for p in discriminator.parameters():p.requires_grad_(True)
                d_optimizer.zero_grad(set_to_none=True)
                real=discriminator(x,target,style);fake=discriminator(x,prediction.detach(),style)
                d_real=criterion(real,torch.ones_like(real));d_fake=criterion(fake,torch.zeros_like(fake))
                (.5*(d_real+d_fake)).backward();d_optimizer.step()
                for p in discriminator.parameters():p.requires_grad_(False)
                g_optimizer.zero_grad(set_to_none=True)
                fake_for_g=discriminator(x,prediction,style)
                adversarial=criterion(fake_for_g,torch.ones_like(fake_for_g));l1=(prediction-target).abs().mean()
                loss=adversarial+config['lambda_l1']*l1;loss.backward();g_optimizer.step()
                for k,value in [('d_real',d_real),('d_fake',d_fake),('g_adversarial',adversarial),('g_l1',l1),('g_total',loss)]:totals[k]+=value.item()*len(x)
                n+=len(x)
            metrics=validate(generator,val_loader,device)
            # Independent of tuned lambda, so trial rankings remain comparable.
            objective=.8*metrics['l1']+.2*(1-metrics['ssim'])
            record=dict(epoch=epoch+1,objective=objective,**metrics,**{k:v/n for k,v in totals.items()})
            history.append(record);improved=objective<best
            if improved:best=objective
            saved=dict(component='sketch_cgan',styles=list(STYLES),generator=generator.state_dict(),discriminator=discriminator.state_dict(),
                       g_optimizer=g_optimizer.state_dict(),d_optimizer=d_optimizer.state_dict(),config=config,epoch=epoch+1,
                       objective=objective,best_objective=best,history=history,manifest_sha256=fingerprint,
                       train_pairs=len(train_set),validation_pairs=len(val_set),smoke=bool(args.train_limit or args.val_limit),
                       torch_rng=torch.get_rng_state(),shuffle_rng=shuffle.get_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [])
            for name in (['best.pt','last.pt'] if improved else ['last.pt']):
                temporary=output/(name+'.tmp');torch.save(saved,temporary);temporary.replace(output/name)
            (output/'history.json').write_text(json.dumps(history,indent=2))
            if epoch==0 or (epoch+1)%5==0 or epoch+1==config['epochs']:
                fixed_samples(generator,val_set,device,output/f'samples_epoch_{epoch+1:03d}.png')
            mlflow.log_metrics({k:v for k,v in record.items() if k!='epoch'},step=epoch+1)
            print(json.dumps(record),flush=True)
            if trial:
                trial.report(objective,epoch)
                if trial.should_prune():mlflow.set_tag('pruned',True);raise optuna.TrialPruned()
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for k in ['d_real','d_fake','g_adversarial','g_l1']:axes[0].plot([r['epoch'] for r in history],[r[k] for r in history],label=k)
        axes[1].plot([r['epoch'] for r in history],[r['objective'] for r in history],label='Fixed validation objective')
        for ax in axes:ax.set_xlabel('Epoch');ax.legend()
        fig.tight_layout();fig.savefig(output/'curves.png');plt.close(fig);mlflow.log_artifacts(str(output),artifact_path='training')
    return best


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','tune']);p.add_argument('--config',type=Path,default=Path('configs/task4.json'))
    p.add_argument('--manifest',type=Path,default=Path('data/prepared_v2/sketch/paired_samples.json'))
    p.add_argument('--output',type=Path,default=Path('runs/task4/final'))
    p.add_argument('--epochs',type=int);p.add_argument('--trials',type=int,default=6);p.add_argument('--resume',type=Path)
    p.add_argument('--train-limit',type=int);p.add_argument('--val-limit',type=int)
    p.add_argument('--workers',type=int,default=0);p.add_argument('--threads',type=int,default=4);p.add_argument('--device')
    args=p.parse_args();torch.set_num_threads(args.threads)
    if args.resume and args.mode!='train':p.error('Resume applies to final training')
    config=json.loads(args.config.read_text());original=dict(config)
    if args.epochs is not None:config['epochs']=args.epochs
    if config['epochs']<1 or args.trials<1:p.error('Positive epoch and trial counts required')
    args.output.mkdir(parents=True,exist_ok=True)
    mlflow.set_tracking_uri('sqlite:///'+str(Path('mlflow.db').resolve()));mlflow.set_experiment('Task4 Style Sketch cGAN')
    if args.mode=='train':train(config,args,args.output);return
    search=dict(generator_learning_rate=[1e-4,4e-4],discriminator_learning_rate=[5e-5,3e-4],batch_size=[8,16],
                base_channels=[16,32],dropout=[0.,.3],embedding_dim=[4,8,16],lambda_l1=[30.,150.])
    (args.output/'search_space.json').write_text(json.dumps(search,indent=2))
    signature=hashlib.sha256(json.dumps(dict(config=config,search=search,manifest=manifest_hash(args.manifest),
          train_limit=args.train_limit,val_limit=args.val_limit),sort_keys=True).encode()).hexdigest()
    study=optuna.create_study(study_name='task4',storage='sqlite:///'+str((args.output/'study.sqlite3').resolve()),load_if_exists=True,
       direction='minimize',sampler=optuna.samplers.TPESampler(seed=config['seed']),pruner=optuna.pruners.MedianPruner(n_startup_trials=3,n_warmup_steps=2))
    if study.user_attrs.get('signature',signature)!=signature:raise ValueError('Study budget/data/configuration changed; use a new output folder')
    study.set_user_attr('signature',signature)
    def objective(trial):
        chosen=dict(config)
        for key in ['generator_learning_rate','discriminator_learning_rate','lambda_l1']:chosen[key]=trial.suggest_float(key,*search[key],log=True)
        for key in ['batch_size','base_channels','embedding_dim']:chosen[key]=trial.suggest_categorical(key,search[key])
        chosen['dropout']=trial.suggest_float('dropout',*search['dropout'])
        return train(chosen,args,args.output/f'trial_{trial.number:03d}',trial)
    study.optimize(objective,n_trials=args.trials)
    (args.output/'best_config.json').write_text(json.dumps(dict(original,**study.best_params),indent=2))
    study.trials_dataframe().to_csv(args.output/'trials.csv',index=False)
    summary=dict(best_trial=study.best_trial.number,best_objective=study.best_value,total_trials=len(study.trials),complete_trials=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials))
    (args.output/'study_summary.json').write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    completed=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    fig,ax=plt.subplots();ax.scatter([t.number for t in completed],[t.value for t in completed]);ax.plot([t.number for t in completed],np.minimum.accumulate([t.value for t in completed]));ax.set_xlabel('Trial');ax.set_ylabel('Fixed validation objective');fig.savefig(args.output/'optimization_history.png');plt.close(fig)
    with mlflow.start_run(run_name='study'):
        mlflow.log_dict(summary,'study_summary.json')
        for name in ['search_space.json','trials.csv','best_config.json','optimization_history.png']:mlflow.log_artifact(str(args.output/name))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
