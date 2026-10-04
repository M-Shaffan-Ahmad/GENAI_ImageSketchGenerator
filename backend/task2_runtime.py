"""ONNX-only hard routing, shared by FastAPI and parity checks."""
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from corruptions import CONDITIONS


class Task2Runtime:
    def __init__(self, directory):
        directory=Path(directory)
        self.metadata=json.loads((directory/'pipeline.json').read_text())
        if self.metadata['conditions']!=list(CONDITIONS) or self.metadata['clean_route']!='identity':
            raise ValueError('Task 2 metadata class order or clean route is incompatible')
        options=ort.SessionOptions();options.intra_op_num_threads=4
        self.sessions={}
        for name in ['classifier',*CONDITIONS[1:]]:
            path=directory/(name+'.onnx')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=self.metadata['components'][name]['onnx_sha256']:
                raise ValueError(f'Task 2 ONNX file does not match export metadata: {name}')
            self.sessions[name]=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])

    def run(self, image):
        if image.shape!=(1,3,128,128) or image.dtype!=np.float32:
            raise ValueError('Task 2 inference expects one float32 RGB NCHW 128x128 image')
        probabilities=self.sessions['classifier'].run(None,{'image':image})[0][0]
        if probabilities.shape!=(4,) or not np.isfinite(probabilities).all() or (probabilities<0).any() or not np.isclose(probabilities.sum(),1,atol=1e-5):
            raise ValueError('Classifier returned invalid probabilities')
        route=int(probabilities.argmax())
        restored=image.copy() if route==0 else self.sessions[CONDITIONS[route]].run(None,{'image':image})[0]
        return restored,probabilities,route
