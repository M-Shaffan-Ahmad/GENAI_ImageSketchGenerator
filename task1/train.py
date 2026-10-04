"""Training and persistent Optuna search, with MLflow experiment evidence."""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import mlflow
import optuna
from dataset_loaders import RestorationDataset
from task1.model import UniversalAutoencoder
from task1.metrics import measurements, reconstruction_loss


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def model_config(config):
    return dict({k: config[k] for k in ['base_channels', 'bottleneck_dim', 'dropout']},
                architecture=config.get('architecture', 'vector_v1'))


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = UniversalAutoencoder(**model_config(checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['model'])
    return model, checkpoint


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    totals = {k: 0. for k in ['l1', 'ssim', 'psnr']}; n = 0
    for batch in loader:
        x, target = batch['input'].to(device), batch['target'].to(device)
        values = measurements(model(x), target)
        for k, v in values.items():
            totals[k] += v.sum().item()
        n += len(x)
    return {k: v/n for k, v in totals.items()}


def train(config, args, output, trial=None):
    seed_all(config['seed'])
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    train_set = RestorationDataset(args.data, 'train', config['seed'], args.train_limit, condition=config.get('condition'))
    val_set = RestorationDataset(args.data, 'val', limit=args.val_limit, condition=config.get('condition'))
    generator = torch.Generator().manual_seed(config['seed'])
    loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=args.workers, generator=generator)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], num_workers=args.workers)
    model = UniversalAutoencoder(**model_config(config)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    output.mkdir(parents=True, exist_ok=True)
    (output/'config.json').write_text(json.dumps(config, indent=2))
    history, best = [], float('inf')
    fingerprint = hashlib.sha256((Path(args.data)/'train.json').read_bytes()).hexdigest()
    validation_fingerprint = hashlib.sha256((Path(args.data)/'validation_manifest.json').read_bytes()).hexdigest()
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint['train_manifest_sha256'] != fingerprint or checkpoint.get('validation_manifest_sha256', validation_fingerprint) != validation_fingerprint:
            raise ValueError('Resume checkpoint belongs to different data manifests')
        if checkpoint.get('train_images', len(train_set)) != len(train_set) or checkpoint.get('validation_images', len(val_set)) != len(val_set):
            raise ValueError('Resume data limits changed')
        for key, value in config.items():
            if key != 'epochs' and checkpoint['config'].get(key, 'vector_v1' if key == 'architecture' else None) != value:
                raise ValueError(f'Resume configuration changed: {key}')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']; best = checkpoint.get('best_objective', checkpoint['objective'])
        history = checkpoint.get('history', [])
        if 'torch_rng' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng'].cpu())
            generator.set_state(checkpoint['shuffle_rng'].cpu())
            if device.startswith('cuda') and checkpoint.get('cuda_rng'):
                torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint['cuda_rng']])
        previous_best = args.resume.parent/'best.pt'
        if previous_best.exists() and previous_best.resolve() != (output/'best.pt').resolve():
            shutil.copy2(previous_best, output/'best.pt')
        if start_epoch >= config['epochs']:
            raise ValueError('Checkpoint already reached --epochs; increase the total epoch target')
    with mlflow.start_run(run_name=output.name):
        mlflow.log_params(config)
        mlflow.log_params(dict(device=device, train_images=len(train_set), validation_images=len(val_set), train_manifest_sha256=fingerprint))
        mlflow.set_tag('purpose', 'smoke' if args.train_limit or args.val_limit else 'full')
        mlflow.log_dict(config, 'config.json')
        if args.resume:
            mlflow.log_param('resumed_from', str(args.resume))
        for epoch in range(start_epoch, config['epochs']):
            train_set.set_epoch(epoch)
            model.train(); loss_sum = 0.; n = 0
            for batch in loader:
                x, target = batch['input'].to(device), batch['target'].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = reconstruction_loss(model(x), target, config['alpha'])
                loss.backward(); optimizer.step()
                loss_sum += loss.item()*len(x); n += len(x)
            metrics = validate(model, val_loader, device)
            # Fixed search objective: tuning alpha must not change how a trial is scored.
            objective = .8*metrics['l1']+.2*(1-metrics['ssim'])
            record = dict(epoch=epoch+1, train_loss=loss_sum/n, objective=objective, **metrics)
            history.append(record)
            mlflow.log_metrics({k: v for k, v in record.items() if k != 'epoch'}, step=epoch+1)
            improved = objective < best
            if improved:
                best = objective
            checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=config,
                              epoch=epoch+1, objective=objective, best_objective=best,
                              train_manifest_sha256=fingerprint, validation_manifest_sha256=validation_fingerprint,
                              train_images=len(train_set), validation_images=len(val_set),
                              smoke=bool(args.train_limit or args.val_limit), history=history,
                              torch_rng=torch.get_rng_state(), shuffle_rng=generator.get_state(),
                              cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [])
            for name in (['best.pt', 'last.pt'] if improved else ['last.pt']):
                temporary = output/(name+'.tmp')
                torch.save(checkpoint, temporary); temporary.replace(output/name)
            (output/'history.json').write_text(json.dumps(history, indent=2))
            print(json.dumps(record), flush=True)
            if trial:
                trial.report(objective, epoch)
                if trial.should_prune():
                    mlflow.set_tag('pruned', True)
                    raise optuna.TrialPruned()
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([h['epoch'] for h in history], [h['train_loss'] for h in history], label='Training loss')
        ax.plot([h['epoch'] for h in history], [h['objective'] for h in history], label='Validation objective')
        ax.set_xlabel('Epoch'); ax.legend(); fig.savefig(output/'curves.png'); plt.close(fig)
        mlflow.log_artifacts(str(output), artifact_path='training')
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'tune'])
    parser.add_argument('--config', type=Path, default=Path('configs/task1.json'))
    parser.add_argument('--data', default='data/prepared_v2/restoration')
    parser.add_argument('--output', type=Path, default=Path('runs/task1'))
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--train-limit', type=int)
    parser.add_argument('--val-limit', type=int)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--tracking-uri', default='sqlite:///' + str(Path('mlflow.db').resolve()))
    args = parser.parse_args()
    if args.mode == 'tune' and args.resume:
        parser.error('--resume applies to final training; rerun the same tune command to extend its persistent study')
    torch.set_num_threads(args.threads)
    config = json.loads(args.config.read_text())
    if args.epochs:
        config['epochs'] = args.epochs
    args.output.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment('Task1 Universal Restoration')
    if args.mode == 'train':
        train(config, args, args.output)
    else:
        spatial = config.get('architecture', 'vector_v1') == 'spatial_v2'
        search = dict(learning_rate=[1e-4, 3e-3], batch_size=[16, 32, 64],
                      bottleneck_dim=[4096, 8192, 16384] if spatial else [64, 128, 256, 512],
                      base_channels=[16, 32, 48], dropout=[0., .1] if spatial else [0., .3], alpha=[.5, .95])
        (args.output/'search_space.json').write_text(json.dumps(search, indent=2))
        # Prevent silently mixing trials with different data budgets/search definitions.
        signature = hashlib.sha256(json.dumps(dict(search=search, architecture=config.get('architecture', 'vector_v1'), epochs=config['epochs'], seed=config['seed'],
                     train_limit=args.train_limit, val_limit=args.val_limit,
                     train=hashlib.sha256((Path(args.data)/'train.json').read_bytes()).hexdigest(),
                     val=hashlib.sha256((Path(args.data)/'validation_manifest.json').read_bytes()).hexdigest()), sort_keys=True).encode()).hexdigest()
        study = optuna.create_study(study_name='task1', storage='sqlite:///'+str((args.output/'study.sqlite3').resolve()),
                                    load_if_exists=True, direction='minimize', sampler=optuna.samplers.TPESampler(seed=config['seed']),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2))
        if study.user_attrs.get('signature', signature) != signature:
            raise ValueError('Study budget or data changed; choose a new --output directory')
        study.set_user_attr('signature', signature)
        def objective(trial):
            chosen = dict(config,
                learning_rate=trial.suggest_float('learning_rate', 1e-4, 3e-3, log=True),
                batch_size=trial.suggest_categorical('batch_size', search['batch_size']),
                bottleneck_dim=trial.suggest_categorical('bottleneck_dim', search['bottleneck_dim']),
                base_channels=trial.suggest_categorical('base_channels', search['base_channels']),
                dropout=trial.suggest_float('dropout', *search['dropout']), alpha=trial.suggest_float('alpha', .5, .95))
            return train(chosen, args, args.output/f'trial_{trial.number:03d}', trial)
        study.optimize(objective, n_trials=args.trials)
        # Keep the final full schedule rather than the shortened trial schedule.
        final_config = dict(json.loads(args.config.read_text()), **study.best_params)
        (args.output/'best_config.json').write_text(json.dumps(final_config, indent=2))
        study.trials_dataframe().to_csv(args.output/'trials.csv', index=False)
        (args.output/'study_summary.json').write_text(json.dumps(dict(best_trial=study.best_trial.number,
             best_objective=study.best_value, complete_trials=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials),
             total_trials=len(study.trials)), indent=2))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        fig, ax = plt.subplots()
        ax.scatter([t.number for t in completed], [t.value for t in completed], label='Trial objective')
        ax.plot([t.number for t in completed], np.minimum.accumulate([t.value for t in completed]), label='Best so far')
        ax.set_xlabel('Trial'); ax.set_ylabel('Validation objective'); ax.legend()
        fig.savefig(args.output/'optimization_history.png'); plt.close(fig)


if __name__ == '__main__':
    main()
