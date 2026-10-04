"""HTTP and production-browser verification; starts and stops its own servers.

Requires a built frontend, installed models, Chrome and Node 22+ (NODE_BINARY may
specify a temporary runtime). Evidence is written under runs/integration.
"""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT = ROOT/'runs/integration'
OUT.mkdir(parents=True, exist_ok=True)
BASE = 'http://127.0.0.1:18080'


def get(path):
    with urlopen(BASE+path, timeout=30) as response:
        return json.load(response)


def post(path, fields, upload=None):
    boundary='assignment-integration-boundary'
    parts=[]
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    if upload is not None:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="photo.png"\r\nContent-Type: image/png\r\n\r\n'.encode()+upload+b'\r\n')
    parts.append(f'--{boundary}--\r\n'.encode())
    request=Request(BASE+'/api/'+path,data=b''.join(parts),headers={'Content-Type':f'multipart/form-data; boundary={boundary}'})
    try:
        with urlopen(request,timeout=30) as response:
            return response.status,json.load(response)
    except HTTPError as error:
        return error.code,json.load(error)


def image_hash(data):
    assert data.startswith('data:image/png;base64,')
    raw=base64.b64decode(data.split(',',1)[1],validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        assert image.size==(128,128) and image.mode=='RGB'
    return hashlib.sha256(raw).hexdigest()


def main():
    node=os.environ.get('NODE_BINARY') or shutil.which('node')
    chrome=shutil.which('google-chrome') or shutil.which('chromium')
    if not node or not chrome:
        raise RuntimeError('Node 22+ and Chrome are required for browser verification')
    processes=[]
    logs=[]
    try:
        for port in [18080,9227]:
            try:
                urlopen(f'http://127.0.0.1:{port}',timeout=.3)
            except HTTPError:
                raise RuntimeError(f'Port {port} already has a server; stop it before verification')
            except (URLError,TimeoutError):
                continue
            raise RuntimeError(f'Port {port} is already in use')
        log=(OUT/'web_server.log').open('w');logs.append(log)
        processes.append(subprocess.Popen([sys.executable,'-m','uvicorn','backend.web:app','--host','127.0.0.1','--port','18080'],stdout=log,stderr=log))
        for _ in range(100):
            try:
                health=get('/api/health');break
            except (URLError,TimeoutError):
                time.sleep(.1)
        else:
            raise RuntimeError('App failed to start; see web_server.log')
        assert all(health[k] for k in ['model_ready','task2_ready','task3_ready','task4_ready'])
        sample=get('/api/samples')[0]['id']
        checks=[]
        for endpoint in ['universal-restoration','hard-routed-restoration','soft-moe-restoration']:
            for condition in ['clean','salt_and_pepper','gaussian_blur','rectangular_occlusion']:
                status,body=post(endpoint,dict(sample_id=sample,condition=condition))
                assert status==200,(endpoint,body)
                assert body['smoke_model'] is False
                digest=image_hash(body['output'])
                assert body['error_map'] is not None
                if 'routing_probabilities' in body:
                    probabilities=body['routing_probabilities']
                    assert len(probabilities)==4 and abs(sum(probabilities.values())-1)<1e-5
                    if body.get('identity_bypass'):
                        assert body['output']==body['input']
                checks.append(dict(endpoint=endpoint,condition=condition,status=status,output_sha256=digest,inference_ms=body['inference_ms']))
        sketches=[]
        for style in [1,2,3]:
            status,body=post('object-to-sketch',dict(sample_id=sample,style=style))
            assert status==200 and body['sketch_mode'] and body['style_id']==style and not body['smoke_model']
            assert body['error_map'] is None
            digest=image_hash(body['output']);sketches.append(digest)
            checks.append(dict(endpoint='object-to-sketch',style=style,status=status,output_sha256=digest))
        assert len(set(sketches))==3
        manifest=json.loads(Path('data/prepared_v2/restoration/val.json').read_text())
        raw=Path(next(row['path'] for row in manifest if row['source_id']==sample)).read_bytes()
        for endpoint in ['universal-restoration','hard-routed-restoration','soft-moe-restoration','object-to-sketch']:
            status,body=post(endpoint,{},upload=raw)
            assert status==200 and body['error_map'] is None
            image_hash(body['output'])
            checks.append(dict(endpoint=endpoint,upload=True,status=status))
        for endpoint,fields,upload,expected in [
            ('object-to-sketch',dict(sample_id=sample,style=4),None,422),
            ('universal-restoration',dict(sample_id=-1),None,404),
            ('soft-moe-restoration',{},b'bad image',422),
            ('hard-routed-restoration',dict(sample_id=sample),raw,422),
            ('universal-restoration',dict(sample_id=sample,probability=.9),None,422),
        ]:
            status,body=post(endpoint,fields,upload)
            assert status==expected,(status,body)
            checks.append(dict(endpoint=endpoint,invalid_request=True,status=status))
        write=dict(passed=True,health=health,checks=checks)
        (OUT/'http_checks.json').write_text(json.dumps(write,indent=2))
        print('HTTP checks passed: trained-model readiness, 19 inference requests and five invalid requests.',flush=True)
        with tempfile.TemporaryDirectory(prefix='ass1-chrome-') as profile:
            log=(OUT/'chrome.log').open('w');logs.append(log)
            processes.append(subprocess.Popen([chrome,'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
                                               '--remote-debugging-port=9227','--user-data-dir='+profile,'about:blank'],stdout=log,stderr=log))
            for _ in range(100):
                try:
                    with urlopen('http://127.0.0.1:9227/json/list',timeout=1) as response:
                        if json.load(response):break
                except (URLError,TimeoutError):
                    pass
                time.sleep(.1)
            else:
                raise RuntimeError('Chrome failed to start; see chrome.log')
            subprocess.run([node,'scripts/check_browser.mjs'],check=True,timeout=90)
    finally:
        for process in reversed(processes):
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()
        for log in logs:log.close()


if __name__=='__main__':main()
