"""Create explicit-allowlist submission packages, without credentials or raw data."""
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'submission'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def safe_files(folder):
    for path in folder.rglob('*'):
        if path.is_file() and not any(part in ['__pycache__','node_modules','.git','.aws','.codex','.agents'] for part in path.parts) and path.suffix not in ['.pyc','.log','.aux','.out']:
            yield path


def archive(path,files):
    count=0
    with ZipFile(path,'w',ZIP_DEFLATED,compresslevel=6) as zipfile:
        for source,relative in sorted(files,key=lambda entry:str(entry[1])):
            zipfile.write(source,str(relative));count+=1
    with ZipFile(path) as zipfile:
        bad=zipfile.testzip()
        if bad:raise ValueError('Archive CRC failure: '+bad)
        if len(zipfile.namelist())!=len(set(zipfile.namelist())):raise ValueError('Duplicate ZIP members')
    print(path.name,count,'files',round(path.stat().st_size/1024**2,2),'MiB',flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    restoration=json.loads((ROOT/'runs/final_test/restoration/results.json').read_text())
    sketch=json.loads((ROOT/'runs/final_test/sketch/results.json').read_text())
    assert restoration['cases']==61490 and not restoration['limited'] and not restoration['smoke_checkpoint']
    assert sum(row['n'] for row in sketch['metrics'].values())==1068 and not sketch['limited'] and not sketch['smoke_checkpoint']
    assert (ROOT/'report/main.pdf').is_file()
    final=ROOT/'experiments/final_test'
    for name in ['restoration','sketch']:
        shutil.copytree(ROOT/'runs/final_test'/name,final/name,dirs_exist_ok=True)
    verify=ROOT/'experiments/integration_verification';verify.mkdir(exist_ok=True)
    for name in ['release.json','http_checks.json','docker_checks.json']:
        shutil.copy2(ROOT/'runs/integration'/name,verify/name)
    shutil.copytree(ROOT/'runs/integration/browser',verify/'browser',dirs_exist_ok=True)
    for name in ['independent_export_check.json','full_validation_export_check.json']:
        shutil.copy2(ROOT/'runs/task4/review'/name,verify/name)
    status=dict(author='Muhammad Shaffan Ahmad',student_id='23i-0673',section='A',university='FAST NUCES',
        completed=['four trained tasks and integrated app','61490 frozen restoration test cases','1068 frozen sketch test pairs',
                   'IEEE double-column LaTeX paper and compiled PDF','ONNX provenance/parity','23 automated tests',
                   'production desktop/mobile browser checks','Docker build/start/inference','data setup container and fingerprints',
                   'original Optuna studies and labeled MLflow evidence imports','source/model/checkpoint/report archives'],
        pending=['authenticated GitHub source publication','published model/checkpoint release URLs',
                 'genuine Google Stitch design evidence','personal demo recording and unlisted YouTube upload',
                 'update pending paper links and evidence then recompile','author review and Google Classroom submission'],
        repository_destination='https://github.com/M-Shaffan-Ahmad/GENAI_ImageSketchGenerator',
        repository_published_in_this_session=False,stitch_provenance_created=False,demo_recorded=False,classroom_submitted=False,
        instructions='docs/submission_steps.md')
    (OUT/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    (ROOT/'experiments/submission_status.json').write_text(json.dumps(status,indent=2)+'\n')
    sources=[]
    roots=['task1','task2','task3','task4','configs','tests','backend','data_prep','scripts','docs','experiments']
    for directory in roots:
        for path in safe_files(ROOT/directory):sources.append((path,path.relative_to(ROOT)))
    for name in ['README.md','requirements.txt','requirements.lock.txt','prepare_dataset.py','dataset_loaders.py','corruptions.py',
                 'package_colab.py','integrate_models.py','docker-compose.yml','run_app.sh','.gitignore','.dockerignore']:
        sources.append((ROOT/name,name))
    for path in (ROOT/'frontend').iterdir():
        if path.is_file():sources.append((path,path.relative_to(ROOT)))
    for path in (ROOT/'colab').iterdir():
        if path.suffix in ['.ipynb','.md']:sources.append((path,path.relative_to(ROOT)))
    for path in safe_files(ROOT/'report'):
        if path.name.startswith('compiler_check'):continue
        sources.append((path,path.relative_to(ROOT)))
    archive(OUT/'github_source.zip',sources)
    reportfiles=[]
    for name in ['main.tex','submission_metadata.tex','IEEEtran.cls','README.md']:
        reportfiles.append((ROOT/'report'/name,name))
    for name in ['assets','generated','fonts']:
        for path in safe_files(ROOT/'report'/name):reportfiles.append((path,path.relative_to(ROOT/'report')))
    archive(OUT/'ieee_report_source.zip',reportfiles)
    release=json.loads((ROOT/'models/release.json').read_text())
    model_files=[]
    for name,digest in release['files'].items():
        source=ROOT/name
        if sha(source)!=digest:raise ValueError('Installed model changed: '+name)
        model_files.append((source,name))
    model_files.append((ROOT/'models/release.json','models/release.json'))
    archive(OUT/'trained_models.zip',model_files)
    checkpoints=ROOT/'runs/integration/staging/checkpoints'
    archive(OUT/'training_checkpoints.zip',[(p,p.relative_to(ROOT)) for p in safe_files(checkpoints)])
    checksums=''.join(sha(OUT/name)+'  '+name+'\n' for name in ['github_source.zip','ieee_report_source.zip','trained_models.zip','training_checkpoints.zip'])
    (OUT/'SHA256SUMS.txt').write_text(checksums)
    with sqlite3.connect(ROOT/'experiments/tracking.db') as database:
        assert database.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    print('All archive CRCs, selected model hashes and tracking DB integrity passed.')


if __name__=='__main__':main()
