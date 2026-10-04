"""Download official Flowers102 and prepare manifests inside the setup container."""
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile
from urllib.request import urlopen
import shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare_dataset import prepare


def fetch(url,path):
    temporary=path.with_suffix(path.suffix+'.part')
    with urlopen(url,timeout=90) as source,temporary.open('wb') as output:
        shutil.copyfileobj(source,output)
    temporary.replace(path)


def main():
    raw=Path('raw_data');raw.mkdir(exist_ok=True)
    if len(list((raw/'jpg').glob('image_*.jpg')))!=8189:
        archive=raw/'102flowers.tgz'
        if not archive.exists():fetch('https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz',archive)
        with tarfile.open(archive) as tar:
            files=[m for m in tar.getmembers() if m.isfile() and re.fullmatch(r'jpg/image_\d{5}\.jpg',m.name)]
            if len(files)!=8189:raise ValueError('Unexpected Oxford image archive contents')
            tar.extractall(raw,members=files,filter='data')
    if not (raw/'setid.mat').exists():fetch('https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat',raw/'setid.mat')
    prepare(raw,Path('data/prepared_v2'))
    release=Path('models/release.json')
    if release.exists():
        selected=json.loads(release.read_text())['verification']
        checks=[('data/prepared_v2/restoration/train.json',selected['task1']['train_manifest_sha256']),
                ('data/prepared_v2/restoration/validation_manifest.json',selected['task1']['validation_manifest_sha256']),
                ('data/prepared_v2/sketch/paired_samples.json',selected['task4']['manifest_sha256'])]
        for path,expected in checks:
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
                raise ValueError('Prepared data differs from selected model provenance: '+path)
    print('Official data prepared; installed-model provenance checks passed.')


if __name__=='__main__':main()
