"""Build a small source-only upload for Colab; datasets download there."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import argparse


def package(output=Path('colab/task1_source.zip'), task2_results=None):
    output = Path(output)
    output.parent.mkdir(exist_ok=True)
    files = [Path(p) for p in ['corruptions.py','dataset_loaders.py','prepare_dataset.py','requirements.txt','README.md']]
    for directory in ['task1','task2','task3','task4','configs','tests','backend','docs','experiments']:
        files += [p for p in Path(directory).rglob('*') if p.is_file() and p.suffix in ['.py','.json','.md','.txt'] and '__pycache__' not in p.parts]
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.as_posix())
        if task2_results:
            import hashlib, json
            expected=json.loads(Path('experiments/task3/initialization.json').read_text())
            with ZipFile(task2_results) as source:
                for component in ['classifier',*expected['conditions'][1:]]:
                    data=source.read(f'runs/task2/{component}/final/best.pt')
                    if hashlib.sha256(data).hexdigest()!=expected['checkpoint_sha256'][component]:
                        raise ValueError('Task 2 archive differs from expected initialization')
                    archive.writestr(f'pretrained/task2/{component}/final/best.pt',data)
    included='source and four trained Task 2 checkpoints; no images' if task2_results else 'no images or model weights included'
    print(f'{output}: {output.stat().st_size:,} bytes; {included}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('colab/task1_source.zip'))
    parser.add_argument('--task2-results', type=Path, help='Include Task 2 initialization checkpoints for the Task 3 bundle')
    args=parser.parse_args()
    package(args.output,args.task2_results)
