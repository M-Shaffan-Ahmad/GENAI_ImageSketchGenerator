"""Single-graph ONNX soft mixture serving."""
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from corruptions import CONDITIONS


class Task3Runtime:
    def __init__(self,path):
        path=Path(path);self.metadata=json.loads(path.with_suffix('.json').read_text())
        if self.metadata['conditions']!=list(CONDITIONS) or hashlib.sha256(path.read_bytes()).hexdigest()!=self.metadata['onnx_sha256']:
            raise ValueError('Task 3 model class order or artifact hash differs')
        options=ort.SessionOptions();options.intra_op_num_threads=4
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])

    def run(self,image):
        if image.shape!=(1,3,128,128) or image.dtype!=np.float32:raise ValueError('Expected one float32 RGB NCHW 128x128 image')
        output,weights=self.session.run(['restored','weights'],{'image':image})
        if weights.shape!=(1,4) or not np.isfinite(weights).all() or (weights<0).any() or not np.isclose(weights.sum(),1,atol=1e-5):
            raise ValueError('Invalid soft routing weights')
        return output,weights[0]
