import base64
import hashlib
from functools import lru_cache
import io
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from corruptions import CONDITIONS, apply_corruption, specification
from backend.task2_runtime import Task2Runtime
from backend.task3_runtime import Task3Runtime
from backend.task4_runtime import Task4Runtime

app = FastAPI(title='Generative AI Restoration', version='1.2')
MODEL = Path(os.environ.get('MODEL_PATH', 'models/universal.onnx'))
TASK4_MODEL = Path(os.environ.get('TASK4_MODEL_PATH', 'models/task4/sketch_generator.onnx'))
TASK3_MODEL = Path(os.environ.get('TASK3_MODEL_PATH', 'models/task3/soft_moe.onnx'))
TASK2_MODELS = Path(os.environ.get('TASK2_MODEL_DIR', 'models/task2'))
DATA = Path(os.environ.get('DATA_ROOT', 'data/prepared_v2/restoration'))


@lru_cache
def session():
    if not MODEL.exists():
        raise HTTPException(503, 'Export a trained Task 1 ONNX model first.')
    metadata_path = MODEL.with_suffix('.json')
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        expected_hash = metadata.get('onnx_sha256')
        if expected_hash and hashlib.sha256(MODEL.read_bytes()).hexdigest() != expected_hash:
            raise HTTPException(503, 'Task 1 model does not match export metadata.')
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    return ort.InferenceSession(str(MODEL), sess_options=options, providers=['CPUExecutionProvider'])


@lru_cache
def task2_session():
    try:
        return Task2Runtime(TASK2_MODELS)
    except (OSError, ValueError, KeyError) as error:
        raise HTTPException(503, 'Export the trained Task 2 classifier and specialists first. ' + str(error))


@lru_cache
def task3_session():
    try:
        return Task3Runtime(TASK3_MODEL)
    except (OSError, ValueError, KeyError) as error:
        raise HTTPException(503, 'Export a trained Task 3 ONNX model first. ' + str(error))


@lru_cache
def task4_session():
    try:
        return Task4Runtime(TASK4_MODEL)
    except (OSError,ValueError,KeyError) as error:
        raise HTTPException(503, 'Export a trained Task 4 ONNX generator first. ' + str(error))


def png(array):
    buffer = io.BytesIO()
    Image.fromarray((np.clip(array,0,1)*255).astype(np.uint8)).save(buffer, format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()


@app.get('/health')
def health():
    ready = MODEL.exists()
    if ready:
        session()
    task2_ready = all((TASK2_MODELS/name).exists() for name in
                      ['pipeline.json','classifier.onnx',*[condition+'.onnx' for condition in CONDITIONS[1:]]])
    if task2_ready:
        task2_session()
    task3_ready=TASK3_MODEL.exists() and TASK3_MODEL.with_suffix('.json').exists()
    if task3_ready:task3_session()
    task4_ready=TASK4_MODEL.exists() and TASK4_MODEL.with_suffix('.json').exists()
    if task4_ready:task4_session()
    return dict(status='ok', model_ready=ready, task2_ready=task2_ready, task3_ready=task3_ready, task4_ready=task4_ready)


def sample_rows():
    if not (DATA/'val.json').exists():
        return []
    return json.loads((DATA/'val.json').read_text())[:12]


@app.get('/samples')
def samples():
    return [dict(id=r['source_id']) for r in sample_rows()]


async def _restore(file: UploadFile | None = File(None), sample_id: int | None = Form(None),
                  condition: str = Form('clean'), probability: float = Form(.08),
                  kernel: int = Form(5), sigma: float = Form(1.5), area_ratio: float = Form(.2),
                  num_boxes: int = Form(2), seed: int = Form(42), hard_routing=False, soft_routing=False, sketch_mode=False, style_id=1):
    if sketch_mode and style_id not in [1,2,3]:
        raise HTTPException(422, 'Choose style 1, 2 or 3')
    if condition not in CONDITIONS or not 0 <= seed < 2**32:
        raise HTTPException(422, 'Invalid corruption or seed')
    if file is not None and sample_id is not None:
        raise HTTPException(422, 'Choose either an uploaded image or a sample')
    if not (.02 <= probability <= .15 and kernel in [3,5,7] and .5 <= sigma <= 2.5 and .1 <= area_ratio <= .35 and 1 <= num_boxes <= 3):
        raise HTTPException(422, 'Corruption parameters outside supported ranges')
    try:
        if file is not None:
            content = await file.read(5*1024*1024+1)
            if len(content) > 5*1024*1024:
                raise HTTPException(413, 'Maximum upload size is 5 MB')
            source = io.BytesIO(content)
        elif sample_id is not None:
            row = next((r for r in sample_rows() if r['source_id']==sample_id), None)
            if row is None:
                raise HTTPException(404, 'Unknown sample')
            source = row['path']
        else:
            raise HTTPException(422, 'Upload an image or select a sample')
        with Image.open(source) as im:
            if im.width*im.height > 25_000_000:
                raise HTTPException(413, 'Image is too large')
            original = np.asarray(im.convert('RGB').resize((128,128),Image.Resampling.BILINEAR),dtype=np.float32)/255
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(422, 'Upload a valid image')
    params = {'salt_and_pepper': dict(prob=probability), 'gaussian_blur': dict(ksize=kernel,sigma=sigma),
              'rectangular_occlusion': dict(area_ratio=area_ratio,num_boxes=num_boxes)}.get(condition,{})
    spec = specification(np.random.default_rng(seed),condition,params,seed)
    x = apply_corruption(original,spec)
    runtime = task4_session() if sketch_mode else task3_session() if soft_routing else task2_session() if hard_routing else session()
    image = x.transpose(2,0,1)[None].copy()
    start = perf_counter()
    routing = {}
    if sketch_mode:
        output = runtime.run(image, style_id-1)[0].transpose(1,2,0)
        metadata = runtime.metadata
        routing = dict(sketch_mode=True, style_id=style_id, style_name=metadata['styles'][style_id-1])
    elif soft_routing:
        restored, weights = runtime.run(image)
        output = restored[0].transpose(1,2,0)
        metadata = runtime.metadata
        routing = dict(predicted_condition=CONDITIONS[int(weights.argmax())],
                       routing_probabilities={condition: float(weights[i]) for i,condition in enumerate(CONDITIONS)},
                       identity_bypass=False, soft_mixture=True)
    elif hard_routing:
        restored, probabilities, route = runtime.run(image)
        output = restored[0].transpose(1,2,0)
        metadata = runtime.metadata
        routing = dict(predicted_condition=CONDITIONS[route],
                       routing_probabilities={condition: float(probabilities[i]) for i,condition in enumerate(CONDITIONS)},
                       identity_bypass=route==0)
    else:
        output = runtime.run(None, {'image': image})[0][0].transpose(1,2,0)
        metadata_path = MODEL.with_suffix('.json')
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    elapsed = (perf_counter()-start)*1000
    # A target is known only for a selected clean sample or simulated corruption.
    return dict(input=png(x), original=png(original), output=png(output),
                error_map=png(np.abs(output-original)) if not sketch_mode and (sample_id is not None or condition!='clean') else None,
                corruption=spec, inference_ms=elapsed, smoke_model=metadata.get('smoke', False), **routing)


@app.post('/universal-restoration')
async def restore(file: UploadFile | None = File(None), sample_id: int | None = Form(None),
                  condition: str = Form('clean'), probability: float = Form(.08),
                  kernel: int = Form(5), sigma: float = Form(1.5), area_ratio: float = Form(.2),
                  num_boxes: int = Form(2), seed: int = Form(42)):
    return await _restore(**locals())


@app.post('/hard-routed-restoration')
async def hard_restore(file: UploadFile | None = File(None), sample_id: int | None = Form(None),
                  condition: str = Form('clean'), probability: float = Form(.08),
                  kernel: int = Form(5), sigma: float = Form(1.5), area_ratio: float = Form(.2),
                  num_boxes: int = Form(2), seed: int = Form(42)):
    return await _restore(**locals(), hard_routing=True)


@app.post('/soft-moe-restoration')
async def soft_restore(file: UploadFile | None = File(None), sample_id: int | None = Form(None),
                  condition: str = Form('clean'), probability: float = Form(.08),
                  kernel: int = Form(5), sigma: float = Form(1.5), area_ratio: float = Form(.2),
                  num_boxes: int = Form(2), seed: int = Form(42)):
    return await _restore(**locals(), soft_routing=True)


@app.post('/object-to-sketch')
async def sketch(file: UploadFile | None = File(None), sample_id: int | None = Form(None), style: int = Form(1)):
    return await _restore(file=file,sample_id=sample_id,condition='clean',probability=.08,kernel=5,sigma=1.5,
                          area_ratio=.2,num_boxes=2,seed=42,sketch_mode=True,style_id=style)
