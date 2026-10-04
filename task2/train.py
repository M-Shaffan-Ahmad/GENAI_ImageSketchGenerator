"""Independent Optuna studies and resumable classifier/specialist training."""
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
from task1.train import seed_all, train as train_specialist
from task2.data import ClassifierDataset, BalancedBatchSampler
from task2.model import CorruptionClassifier, classifier_config
from task2.metrics import classification_metrics


def fingerprints(data):
    return {key:hashlib.sha256((Path(data)/name).read_bytes()).hexdigest()
            for key,name in [('train_manifest_sha256','train.json'),
                             ('validation_manifest_sha256','validation_manifest.json')]}


@torch.no_grad()
def validate_classifier(model, loader, device):
    model.eval(); loss, n, labels, predictions = 0., 0, [], []
    for batch in loader:
        x, y = batch['input'].to(device), batch['label'].to(device)
        logits = model(x)
        loss += torch.nn.functional.cross_entropy(logits,y,reduction='sum').item()
        n += len(x); labels.extend(y.cpu().tolist()); predictions.extend(logits.argmax(1).cpu().tolist())
    if not n:
        raise ValueError('Empty validation set')
    return dict(cross_entropy=loss/n, **classification_metrics(labels,predictions))


def train_classifier(config, args, output, trial=None):
    seed_all(config['seed'])
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ClassifierDataset(args.data, config['seed'], args.train_limit)
    sampler = BalancedBatchSampler(len(dataset.rows), config['batch_size'], config['seed'])
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.workers)
    val_set = RestorationDataset(args.data, 'val', limit=args.val_limit)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'],num_workers=args.workers)
    model = CorruptionClassifier(**classifier_config(config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    output.mkdir(parents=True,exist_ok=True)
    hashes = fingerprints(args.data)
    history, best, start_epoch = [], float('inf'), 0
    if args.resume:
        saved = torch.load(args.resume,map_location=device,weights_only=True)
        if saved.get('component') != 'classifier' or saved.get('conditions') != list(CONDITIONS):
            raise ValueError('Resume requires a compatible classifier checkpoint')
        if any(saved[key]!=value for key,value in hashes.items()):
            raise ValueError('Resume data manifests changed')
        if saved['train_images']!=len(dataset.rows) or saved['validation_images']!=len(val_set):
            raise ValueError('Resume data limits changed')
        if any(saved['config'].get(k)!=v for k,v in config.items() if k!='epochs'):
            raise ValueError('Resume configuration changed')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        history=saved['history'];best=saved['best_objective'];start_epoch=saved['epoch']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.startswith('cuda') and saved.get('cuda_rng'):
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
        previous_best=args.resume.parent/'best.pt'
        if not previous_best.exists():
            raise ValueError('Keep best.pt alongside last.pt to resume')
        if previous_best.resolve()!=(output/'best.pt').resolve():
            shutil.copy2(previous_best,output/'best.pt')
        if start_epoch>=config['epochs']:
            raise ValueError('Already reached --epochs; increase the total target')
    (output/'config.json').write_text(json.dumps(config,indent=2))
    with mlflow.start_run(run_name='classifier-'+output.name):
        mlflow.log_params(dict(config,**hashes,train_images=len(dataset.rows),validation_images=len(val_set),device=device))
        mlflow.set_tag('purpose','smoke' if args.train_limit or args.val_limit else 'full')
        if args.resume:mlflow.log_param('resumed_from',str(args.resume))
        for epoch in range(start_epoch,config['epochs']):
            dataset.set_epoch(epoch);sampler.set_epoch(epoch);model.train()
            total_loss,n=0.,0
            for batch in loader:
                x,y=batch['input'].to(device),batch['label'].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.cross_entropy(model(x),y)
                loss.backward();optimizer.step()
                total_loss+=loss.item()*len(x);n+=len(x)
            metrics=validate_classifier(model,val_loader,device)
            objective=metrics['cross_entropy']
            record=dict(epoch=epoch+1,train_loss=total_loss/n,objective=objective,
                        accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1'])
            history.append(record)
            improved=objective<best
            if improved:best=objective
            saved=dict(component='classifier',conditions=list(CONDITIONS),model=model.state_dict(),
                       optimizer=optimizer.state_dict(),config=config,epoch=epoch+1,objective=objective,
                       best_objective=best,history=history,**hashes,train_images=len(dataset.rows),
                       validation_images=len(val_set),smoke=bool(args.train_limit or args.val_limit),
                       torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [])
            for name in (['best.pt','last.pt'] if improved else ['last.pt']):
                temporary=output/(name+'.tmp');torch.save(saved,temporary);temporary.replace(output/name)
            (output/'history.json').write_text(json.dumps(history,indent=2))
            mlflow.log_metrics({k:v for k,v in record.items() if k!='epoch'},step=epoch+1)
            print(json.dumps(record),flush=True)
            if trial:
                trial.report(objective,epoch)
                if trial.should_prune():
                    mlflow.set_tag('pruned',True);raise optuna.TrialPruned()
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].plot([r['epoch'] for r in history],[r['train_loss'] for r in history],label='Train CE')
        axes[0].plot([r['epoch'] for r in history],[r['objective'] for r in history],label='Validation CE')
        for name in ['accuracy','macro_f1']:
            axes[1].plot([r['epoch'] for r in history],[r[name] for r in history],label=name)
        for ax in axes:ax.set_xlabel('Epoch');ax.legend()
        fig.tight_layout();fig.savefig(output/'curves.png');plt.close(fig)
        mlflow.log_artifacts(str(output),artifact_path='training')
    return best


def search_space(component):
    if component=='classifier':
        return dict(learning_rate=[1e-4,3e-3],base_channels=[16,32,48],dropout=[0.,.2],weight_decay=[1e-6,1e-2])
    return dict(learning_rate=[1e-4,3e-3],batch_size=[16,32],base_channels=[16,32,48],
                bottleneck_dim=[8192,16384],dropout=[0.,.1],alpha=[.5,.9])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['train','tune'])
    p.add_argument('--component',choices=['classifier',*CONDITIONS[1:]],required=True)
    p.add_argument('--config',type=Path,default=Path('configs/task2.json'))
    p.add_argument('--data',default='data/prepared_v2/restoration')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int)
    p.add_argument('--trials',type=int,default=8)
    p.add_argument('--train-limit',type=int);p.add_argument('--val-limit',type=int)
    p.add_argument('--workers',type=int,default=0);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device');p.add_argument('--resume',type=Path)
    p.add_argument('--tracking-uri',default='sqlite:///'+str(Path('mlflow.db').resolve()))
    args=p.parse_args()
    if args.resume and args.mode!='train':p.error('--resume applies to training only')
    if args.epochs is not None and args.epochs<=0:p.error('--epochs must be positive')
    if args.trials<=0:p.error('--trials must be positive')
    torch.set_num_threads(args.threads)
    raw=json.loads(args.config.read_text())
    config=dict(raw[args.component if args.component=='classifier' else 'specialist'] if 'classifier' in raw else raw)
    if args.component!='classifier':
        if config.get('condition',args.component)!=args.component:
            raise ValueError('Selected specialist config belongs to another corruption')
        config['condition']=args.component
    final_config=dict(config)
    if args.epochs is not None:config['epochs']=args.epochs
    args.output.mkdir(parents=True,exist_ok=True)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment('Task2 Hard Routing')
    trainer=train_classifier if args.component=='classifier' else train_specialist
    if args.mode=='train':
        trainer(config,args,args.output)
        return
    search=search_space(args.component)
    (args.output/'search_space.json').write_text(json.dumps(search,indent=2))
    signature=hashlib.sha256(json.dumps(dict(component=args.component,config=config,search=search,
                          train_limit=args.train_limit,val_limit=args.val_limit,**fingerprints(args.data)),sort_keys=True).encode()).hexdigest()
    study=optuna.create_study(study_name='task2-'+args.component,storage='sqlite:///'+str((args.output/'study.sqlite3').resolve()),
          load_if_exists=True,direction='minimize',sampler=optuna.samplers.TPESampler(seed=config['seed']),
          pruner=optuna.pruners.MedianPruner(n_startup_trials=4,n_warmup_steps=2))
    if study.user_attrs.get('signature',signature)!=signature:
        raise ValueError('Study configuration, budget or data changed; choose a new output directory')
    study.set_user_attr('signature',signature)
    def objective(trial):
        chosen=dict(config,learning_rate=trial.suggest_float('learning_rate',*search['learning_rate'],log=True),
                    base_channels=trial.suggest_categorical('base_channels',search['base_channels']),
                    dropout=trial.suggest_float('dropout',*search['dropout']))
        if args.component=='classifier':
            chosen['weight_decay']=trial.suggest_float('weight_decay',*search['weight_decay'],log=True)
        else:
            for key in ['batch_size','bottleneck_dim']:
                chosen[key]=trial.suggest_categorical(key,search[key])
            chosen['alpha']=trial.suggest_float('alpha',*search['alpha'])
        return trainer(chosen,args,args.output/f'trial_{trial.number:03d}',trial)
    study.optimize(objective,n_trials=args.trials)
    (args.output/'best_config.json').write_text(json.dumps(dict(final_config,**study.best_params),indent=2))
    study.trials_dataframe().to_csv(args.output/'trials.csv',index=False)
    summary=dict(component=args.component,best_trial=study.best_trial.number,best_objective=study.best_value,
                 total_trials=len(study.trials),complete_trials=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials))
    (args.output/'study_summary.json').write_text(json.dumps(summary,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    completed=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    fig,ax=plt.subplots()
    ax.scatter([t.number for t in completed],[t.value for t in completed],label='Trial objective')
    ax.plot([t.number for t in completed],np.minimum.accumulate([t.value for t in completed]),label='Best so far')
    ax.set_xlabel('Trial');ax.set_ylabel('Validation CE' if args.component=='classifier' else 'Fixed restoration objective');ax.legend()
    fig.savefig(args.output/'optimization_history.png');plt.close(fig)
    with mlflow.start_run(run_name=args.component+'-study'):
        mlflow.log_dict(summary,'study_summary.json');mlflow.log_dict(search,'search_space.json')
        for name in ['trials.csv','optimization_history.png','best_config.json']:
            mlflow.log_artifact(str(args.output/name),artifact_path='search')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
