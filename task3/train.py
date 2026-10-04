"""Two-stage MoE training, persistent Optuna search, MLflow and exact-epoch resume."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import mlflow
import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader
from corruptions import CONDITIONS
from dataset_loaders import RestorationDataset
from task1.metrics import measurements
from task1.train import seed_all
from task2.data import ClassifierDataset,BalancedBatchSampler
from task2.model import load_pipeline
from task2.train import fingerprints
from task3.model import SoftMixture,set_stage
from task3.loss import joint_loss


def initialization(args):
    pipeline,saved=load_pipeline(args.initialization,args.device or 'cpu')
    expected=fingerprints(args.data)
    if any(c[k]!=v for c in saved.values() for k,v in expected.items()):
        raise ValueError('Task 2 initialization belongs to different data manifests')
    smoke=any(c['smoke'] for c in saved.values())
    if smoke and not args.allow_smoke_initialization:
        raise ValueError('Use fully trained Task 2 checkpoints; smoke initialization is for tests only')
    hashes={c:hashlib.sha256((args.initialization/c/'final/best.pt').read_bytes()).hexdigest() for c in saved}
    return pipeline,saved,hashes


@torch.no_grad()
def validate(model,loader,device):
    model.eval();totals={k:0. for k in ['l1','ssim','psnr']};n=0;weight_sum=np.zeros(4)
    for batch in loader:
        x,target=batch['input'].to(device),batch['target'].to(device)
        output,weights=model(x)
        for k,v in measurements(output,target).items():totals[k]+=v.sum().item()
        weight_sum+=weights.sum(0).cpu().numpy();n+=len(x)
    if not n:raise ValueError('Empty validation set')
    return {k:v/n for k,v in totals.items()},(weight_sum/n).tolist()


def optimizer_for(model,stage,config):
    set_stage(model,stage)
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=config['warmup_learning_rate'] if stage=='warmup' else config['learning_rate'])


def train(config,args,output,trial=None):
    seed_all(config['seed'])
    device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline,source,source_hashes=initialization(args)
    model=SoftMixture(pipeline.classifier,pipeline.specialists,config['temperature']).to(device)
    dataset=ClassifierDataset(args.data,config['seed'],args.train_limit)
    sampler=BalancedBatchSampler(len(dataset.rows),config['batch_size'],config['seed'])
    loader=DataLoader(dataset,batch_sampler=sampler,num_workers=args.workers)
    val_set=RestorationDataset(args.data,'val',limit=args.val_limit)
    val_loader=DataLoader(val_set,batch_size=config['batch_size'],num_workers=args.workers)
    total_epochs=config['warmup_epochs']+config['joint_epochs']
    hashes=fingerprints(args.data)
    history=[];best={'warmup':float('inf'),'joint':float('inf')};start=0;stage='warmup'
    optimizer=optimizer_for(model,stage,config)
    output.mkdir(parents=True,exist_ok=True)
    if args.resume:
        saved=torch.load(args.resume,map_location=device,weights_only=True)
        if saved.get('component')!='soft_moe' or saved.get('conditions')!=list(CONDITIONS):raise ValueError('Wrong resume checkpoint')
        if any(saved.get(k)!=v for k,v in hashes.items()) or saved['initialization_sha256']!=source_hashes:
            raise ValueError('Resume data or initialization changed')
        if saved['train_images']!=len(dataset.rows) or saved['validation_images']!=len(val_set):raise ValueError('Resume data limits changed')
        if any(saved['config'].get(k)!=v for k,v in config.items() if k!='joint_epochs'):raise ValueError('Resume configuration changed')
        model.load_state_dict(saved['model']);stage=saved['stage'];optimizer=optimizer_for(model,stage,config)
        optimizer.load_state_dict(saved['optimizer'])
        history=saved['history'];best=saved['best_objectives'];start=saved['epoch']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.startswith('cuda') and saved.get('cuda_rng'):
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
        for name in ['warmup_best.pt','best.pt']:
            previous=args.resume.parent/name
            if previous.exists() and previous.resolve()!=(output/name).resolve():shutil.copy2(previous,output/name)
        if start>=total_epochs:raise ValueError('Training completed; increase joint epoch target to extend')
    (output/'config.json').write_text(json.dumps(config,indent=2))
    with mlflow.start_run(run_name=output.name):
        mlflow.log_params(dict(config,**hashes,train_images=len(dataset.rows),validation_images=len(val_set),device=device))
        mlflow.log_dict(source_hashes,'initialization_sha256.json')
        mlflow.set_tag('purpose','smoke' if args.train_limit or args.val_limit or any(c['smoke'] for c in source.values()) else 'full')
        for epoch in range(start,total_epochs):
            next_stage='warmup' if epoch<config['warmup_epochs'] else 'joint'
            model.train()
            if next_stage!=stage:stage=next_stage;optimizer=optimizer_for(model,stage,config)
            else:set_stage(model,stage)
            dataset.set_epoch(epoch);sampler.set_epoch(epoch)
            loss_sum,n=0.,0;terms_sum={k:0. for k in ['l1','structural','classification','balance']}
            for batch in loader:
                x,target,labels=batch['input'].to(device),batch['target'].to(device),batch['label'].to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction,weights,logits=model.forward_details(x)
                loss,terms=joint_loss(prediction,target,logits,weights,labels,config)
                loss.backward();optimizer.step();loss_sum+=loss.item()*len(x);n+=len(x)
                for k,v in terms.items():terms_sum[k]+=v.item()*len(x)
            metrics,mean_weights=validate(model,val_loader,device)
            objective=.8*metrics['l1']+.2*(1-metrics['ssim'])
            record=dict(epoch=epoch+1,stage=stage,train_loss=loss_sum/n,objective=objective,**metrics,
                        **{'train_'+k:v/n for k,v in terms_sum.items()},mean_weights=mean_weights)
            history.append(record);improved=objective<best[stage]
            if improved:best[stage]=objective
            saved=dict(component='soft_moe',conditions=list(CONDITIONS),model=model.state_dict(),optimizer=optimizer.state_dict(),
                       config=config,epoch=epoch+1,stage=stage,best_objectives=best,history=history,objective=objective,**hashes,
                       initialization_configs={k:c['config'] for k,c in source.items()},initialization_sha256=source_hashes,
                       train_images=len(dataset.rows),validation_images=len(val_set),
                       smoke=bool(args.train_limit or args.val_limit or any(c['smoke'] for c in source.values())),
                       torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [])
            names=['last.pt']
            if improved:names.insert(0,'warmup_best.pt' if stage=='warmup' else 'best.pt')
            for name in names:
                temporary=output/(name+'.tmp');torch.save(saved,temporary);temporary.replace(output/name)
            (output/'history.json').write_text(json.dumps(history,indent=2))
            mlflow.log_metrics({k:v for k,v in record.items() if isinstance(v,(int,float)) and k!='epoch'},step=epoch+1)
            mlflow.log_metrics({f'weight_{c}':mean_weights[i] for i,c in enumerate(CONDITIONS)},step=epoch+1)
            print(json.dumps(record),flush=True)
            if trial and stage=='joint':
                trial.report(objective,epoch-config['warmup_epochs'])
                if trial.should_prune():mlflow.set_tag('pruned',True);raise optuna.TrialPruned()
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots();ax.plot([h['epoch'] for h in history],[h['objective'] for h in history],label='Fixed validation objective')
        ax.axvline(config['warmup_epochs']+.5,color='gray',linestyle='--',label='Joint fine-tuning begins');ax.set_xlabel('Epoch');ax.legend()
        fig.savefig(output/'curves.png');plt.close(fig);mlflow.log_artifacts(str(output),artifact_path='training')
    return best['joint']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','tune'])
    p.add_argument('--config',type=Path,default=Path('configs/task3.json'))
    p.add_argument('--initialization',type=Path,default=Path('pretrained/task2'))
    p.add_argument('--data',default='data/prepared_v2/restoration')
    p.add_argument('--output',type=Path,default=Path('runs/task3/final'))
    p.add_argument('--warmup-epochs',type=int);p.add_argument('--joint-epochs',type=int)
    p.add_argument('--trials',type=int,default=6);p.add_argument('--resume',type=Path)
    p.add_argument('--train-limit',type=int);p.add_argument('--val-limit',type=int)
    p.add_argument('--workers',type=int,default=0);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device');p.add_argument('--allow-smoke-initialization',action='store_true')
    args=p.parse_args();torch.set_num_threads(args.threads)
    if args.resume and args.mode!='train':p.error('Resume only applies to final training')
    original=json.loads(args.config.read_text());config=dict(original)
    for key in ['warmup_epochs','joint_epochs']:
        if getattr(args,key) is not None:config[key]=getattr(args,key)
        if config[key]<1:p.error('Both training stages need at least one epoch')
    if config['temperature']<=0 or not 0<config['learning_rate']<config['warmup_learning_rate']:p.error('Need positive temperature and lower joint learning rate')
    if args.trials<1:p.error('Positive trial count required')
    args.output.mkdir(parents=True,exist_ok=True)
    mlflow.set_tracking_uri('sqlite:///'+str(Path('mlflow.db').resolve()));mlflow.set_experiment('Task3 Soft MoE')
    if args.mode=='train':train(config,args,args.output);return
    _,_,source_hashes=initialization(args)
    search=dict(learning_rate=[2e-5,2e-4],temperature=[.35,1.5],alpha=[.6,.95],lambda_ce=[.01,.2],lambda_balance=[1e-4,.02])
    (args.output/'search_space.json').write_text(json.dumps(search,indent=2))
    signature=hashlib.sha256(json.dumps(dict(config=config,search=search,initialization=source_hashes,
              train_limit=args.train_limit,val_limit=args.val_limit,**fingerprints(args.data)),sort_keys=True).encode()).hexdigest()
    study=optuna.create_study(study_name='task3',storage='sqlite:///'+str((args.output/'study.sqlite3').resolve()),
          load_if_exists=True,direction='minimize',sampler=optuna.samplers.TPESampler(seed=config['seed']),
          pruner=optuna.pruners.MedianPruner(n_startup_trials=3,n_warmup_steps=1))
    if study.user_attrs.get('signature',signature)!=signature:raise ValueError('Study configuration/data/initialization changed; use a new output')
    study.set_user_attr('signature',signature)
    def objective(trial):
        alpha=trial.suggest_float('alpha',*search['alpha'])
        chosen=dict(config,lambda_l1=alpha,lambda_ssim=1-alpha,
                    learning_rate=trial.suggest_float('learning_rate',*search['learning_rate'],log=True),
                    temperature=trial.suggest_float('temperature',*search['temperature']),
                    lambda_ce=trial.suggest_float('lambda_ce',*search['lambda_ce'],log=True),
                    lambda_balance=trial.suggest_float('lambda_balance',*search['lambda_balance'],log=True))
        if chosen['learning_rate']>=chosen['warmup_learning_rate']:raise ValueError('Joint search LR must remain below warm-up LR')
        return train(chosen,args,args.output/f'trial_{trial.number:03d}',trial)
    study.optimize(objective,n_trials=args.trials)
    selected=dict(original,**{k:v for k,v in study.best_params.items() if k!='alpha'},lambda_l1=study.best_params['alpha'],lambda_ssim=1-study.best_params['alpha'])
    (args.output/'best_config.json').write_text(json.dumps(selected,indent=2))
    study.trials_dataframe().to_csv(args.output/'trials.csv',index=False)
    summary=dict(best_trial=study.best_trial.number,best_objective=study.best_value,total_trials=len(study.trials),
                 complete_trials=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials))
    (args.output/'study_summary.json').write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    completed=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    fig,ax=plt.subplots();ax.scatter([t.number for t in completed],[t.value for t in completed]);ax.plot([t.number for t in completed],np.minimum.accumulate([t.value for t in completed]))
    ax.set_xlabel('Trial');ax.set_ylabel('Fixed validation objective');fig.savefig(args.output/'optimization_history.png');plt.close(fig)
    with mlflow.start_run(run_name='study'):
        mlflow.log_dict(summary,'study_summary.json')
        for name in ['trials.csv','best_config.json','search_space.json','optimization_history.png']:mlflow.log_artifact(str(args.output/name))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
