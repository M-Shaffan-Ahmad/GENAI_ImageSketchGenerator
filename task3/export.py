"""Export the full differentiable MoE with output image and four gate weights."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from corruptions import CONDITIONS
from dataset_loaders import RestorationDataset
from task3.model import load_checkpoint
from task2.train import fingerprints


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=Path('runs/task3/final/best.pt'))
    p.add_argument('--output',type=Path,default=Path('models/task3/soft_moe.onnx'))
    p.add_argument('--data',default='data/prepared_v2/restoration')
    args=p.parse_args();torch.set_num_threads(4)
    model,c=load_checkpoint(args.checkpoint);model.eval()
    if any(c[k]!=v for k,v in fingerprints(args.data).items()):raise ValueError('Export data manifests changed')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(model,(torch.rand(1,3,128,128),),str(args.output),input_names=['image'],
                      output_names=['restored','weights'],opset_version=18,dynamo=True,external_data=False)
    onnx.checker.check_model(onnx.load(str(args.output)))
    options=ort.SessionOptions();options.intra_op_num_threads=4
    runtime=ort.InferenceSession(str(args.output),sess_options=options,providers=['CPUExecutionProvider'])
    errors=[];weight_errors=[]
    with torch.no_grad():
        for condition in CONDITIONS:
            for item in RestorationDataset(args.data,'val',condition=condition,limit=4):
                x=item['input'][None];expected,w=model(x)
                actual,weights=runtime.run(None,{'image':x.numpy()})
                np.testing.assert_allclose(actual,expected.numpy(),rtol=1e-4,atol=1e-5)
                np.testing.assert_allclose(weights,w.numpy(),rtol=1e-4,atol=1e-5)
                errors.append(float(np.abs(actual-expected.numpy()).max()));weight_errors.append(float(np.abs(weights-w.numpy()).max()))
    metadata=dict(config=c['config'],stage=c['stage'],epoch=c['epoch'],conditions=list(CONDITIONS),smoke=c['smoke'],
                  initialization_sha256=c['initialization_sha256'],checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  onnx_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),validation_images=len(errors),
                  max_absolute_error=max(errors),weights_max_absolute_error=max(weight_errors),
                  input='float32 NCHW [1,3,128,128], RGB [0,1], PIL bilinear resize',outputs=['restored','weights'])
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    import mlflow
    mlflow.set_tracking_uri('sqlite:///'+str(Path('mlflow.db').resolve()));mlflow.set_experiment('Task3 Soft MoE')
    with mlflow.start_run(run_name='onnx-export'):
        mlflow.log_metrics(dict(onnx_max_absolute_error=max(errors),weights_max_absolute_error=max(weight_errors)))
        mlflow.log_dict(metadata,'export.json');mlflow.log_artifact(str(args.output))
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
