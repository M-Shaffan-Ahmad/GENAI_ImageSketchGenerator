"""Make portable MLflow viewing records and retain original Optuna databases.

Records are explicitly tagged as retrospective evidence imports. Histories and
configs come from original Colab artifacts; this script does not train models.
"""
import json
import shutil
import sqlite3
from pathlib import Path
from zipfile import ZipFile
import mlflow


def main():
    root=Path('experiments');root.mkdir(exist_ok=True)
    mlflow.set_tracking_uri('sqlite:///'+str((root/'tracking.db').resolve()))
    name='Reviewed Colab Training Evidence'
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name,artifact_location=(root/'tracking_artifacts/1').resolve().as_uri())
    mlflow.set_experiment(name)
    cases=[('task1_run_a','task1/final'),*[(f'task2_{name}',f'task2/{name}/final') for name in ['classifier','salt_and_pepper','gaussian_blur','rectangular_occlusion']],('task3_joint','task3/final'),('task4_cgan','task4/final')]
    evidence=Path('report/evidence')
    for name,relative in cases:
        directory=evidence/relative
        with mlflow.start_run(run_name=name):
            mlflow.set_tags(dict(record_type='retrospective_evidence_import',source='original Colab result ZIP',no_training_performed=True))
            mlflow.log_params(json.loads((directory/'config.json').read_text()))
            history=json.loads((directory/'history.json').read_text())
            for row in history:
                step=row['epoch']
                values={key:float(value) for key,value in row.items() if key!='epoch' and isinstance(value,(float,int))}
                if 'mean_weights' in row:
                    values.update({f'weight_{i}':float(value) for i,value in enumerate(row['mean_weights'])})
                mlflow.log_metrics(values,step=step)
            for file in directory.iterdir():
                if file.suffix in ['.json','.png'] and not file.name.startswith('samples_epoch_'):
                    mlflow.log_artifact(str(file),artifact_path='original_evidence')
            mlflow.log_param('recorded_epochs',len(history))
    for task,filename in [('task1','task1_spatial_results.zip'),('task2','task2_results.zip'),('task3','task3_results.zip'),('task4','task4_results.zip')]:
        with ZipFile('colab/'+filename) as archive:
            for name in archive.namelist():
                if '/search/' in name and name.endswith('.sqlite3'):
                    before,after=name.split('/search/',1)
                    component=before.split('/')[-1] if task=='task2' else task
                    destination=root/'optuna'/component/Path(after).name
                    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(archive.read(name))
                    with sqlite3.connect(destination) as db:
                        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Corrupt study')
    with mlflow.start_run(run_name='release-and-validation'):
        mlflow.set_tag('record_type','verified integration evidence')
        mlflow.log_artifact('models/release.json')
        for path in Path('docs').glob('*validation_findings.md'):mlflow.log_artifact(str(path),artifact_path='reviews')
        for path in [Path('runs/final_test/restoration/results.json'),Path('runs/final_test/sketch/results.json')]:
            if path.exists():mlflow.log_artifact(str(path),artifact_path=path.parent.name)
    print('MLflow viewing records created; original Optuna SQLite integrity passed.')


if __name__=='__main__':main()
