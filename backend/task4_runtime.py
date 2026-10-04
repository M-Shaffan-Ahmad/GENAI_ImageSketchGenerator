import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort


class Task4Runtime:
    def __init__(self,path):
        path=Path(path);self.metadata=json.loads(path.with_suffix('.json').read_text())
        if len(self.metadata['styles'])!=3 or hashlib.sha256(path.read_bytes()).hexdigest()!=self.metadata['onnx_sha256']:
            raise ValueError('Task 4 model metadata mismatch')
        options=ort.SessionOptions();options.intra_op_num_threads=4
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])

    def run(self,image,style):
        if image.shape!=(1,3,128,128) or image.dtype!=np.float32 or style not in [0,1,2]:raise ValueError('Expected float32 RGB image and style index 0, 1 or 2')
        return self.session.run(['sketch'],{'image':image,'style':np.array([style],dtype=np.int64)})[0]
