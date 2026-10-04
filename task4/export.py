"""ONNX conditional generator with categorical style input and parity for all styles."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from dataset_loaders import SketchDataset
from task4.model import load_checkpoint,STYLES
from task4.train import manifest_hash

# Full trained validation (786 pairs) measured CPU framework/runtime differences
# up to 1.47e-4. This fixed bound is 0.051 of an 8-bit intensity level;
# the smoke model's 2e-5 tolerance did not cover accumulated InstanceNorm error.
PARITY_ATOL = 2e-4
PARITY_RTOL = 1e-4


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=Path('runs/task4/final/best.pt'))
    p.add_argument('--manifest',type=Path,default=Path('data/prepared_v2/sketch/paired_samples.json'))
    p.add_argument('--output',type=Path,default=Path('models/task4/sketch_generator.onnx'))
    args=p.parse_args();torch.set_num_threads(4)
    generator,c=load_checkpoint(args.checkpoint);generator.eval()
    if c['manifest_sha256']!=manifest_hash(args.manifest):raise ValueError('Export manifest changed')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(generator,(torch.rand(1,3,128,128),torch.zeros(1,dtype=torch.long)),str(args.output),
                      input_names=['image','style'],output_names=['sketch'],opset_version=18,dynamo=True,external_data=False)
    onnx.checker.check_model(onnx.load(str(args.output)))
    options=ort.SessionOptions();options.intra_op_num_threads=4
    runtime=ort.InferenceSession(str(args.output),sess_options=options,providers=['CPUExecutionProvider'])
    dataset=SketchDataset(args.manifest,'val',False);errors=[];counts=[0,0,0]
    with torch.no_grad():
        for item in dataset:
            s=item['style']
            if counts[s]>=4:continue
            x=item['input'][None];style=torch.tensor([s],dtype=torch.long)
            expected=generator(x,style).numpy();actual=runtime.run(None,{'image':x.numpy(),'style':style.numpy()})[0]
            np.testing.assert_allclose(actual,expected,rtol=PARITY_RTOL,atol=PARITY_ATOL)
            if np.abs(actual-expected).max()>PARITY_ATOL:
                raise ValueError('Task 4 ONNX exceeds the absolute error budget')
            errors.append(float(np.abs(actual-expected).max()));counts[s]+=1
            if counts==[4,4,4]:break
    if counts!=[4,4,4]:raise ValueError('Need four validation cases for each style')
    metadata=dict(config=c['config'],epoch=c['epoch'],styles=list(STYLES),style_input='int64 [1], indices 0/1/2',
                  smoke=c['smoke'],manifest_sha256=c['manifest_sha256'],checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  onnx_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),validation_images=len(errors),
                  validation_by_style=counts,max_absolute_error=max(errors),
                  parity_atol=PARITY_ATOL,parity_rtol=PARITY_RTOL,max_absolute_error_budget=PARITY_ATOL,
                  input='float32 NCHW [1,3,128,128], RGB [0,1], PIL bilinear resize')
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    import mlflow
    mlflow.set_tracking_uri('sqlite:///'+str(Path('mlflow.db').resolve()));mlflow.set_experiment('Task4 Style Sketch cGAN')
    with mlflow.start_run(run_name='onnx-export'):
        mlflow.log_metric('onnx_max_absolute_error',max(errors));mlflow.log_dict(metadata,'export.json');mlflow.log_artifact(str(args.output))
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
