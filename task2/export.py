"""Export all four Task 2 ONNX components and check runtime parity."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader
from corruptions import CONDITIONS
from dataset_loaders import RestorationDataset
from task2.model import load_pipeline
from task2.train import fingerprints


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',type=Path,default=Path('runs/task2'))
    p.add_argument('--output',type=Path,default=Path('models/task2'))
    p.add_argument('--data',default='data/prepared_v2/restoration')
    args=p.parse_args();torch.set_num_threads(4)
    pipeline,checkpoints=load_pipeline(args.checkpoints)
    if any(c[k]!=v for c in checkpoints.values() for k,v in fingerprints(args.data).items()):
        raise ValueError('Export validation data differs from checkpoint manifests')
    args.output.mkdir(parents=True,exist_ok=True)
    classifier=nn.Sequential(pipeline.classifier,nn.Softmax(dim=1)).eval()
    components={'classifier':classifier,**dict(zip(CONDITIONS[1:],pipeline.specialists))}
    report={}
    for name,model in components.items():
        path=args.output/(name+'.onnx');model.eval()
        output_name='probabilities' if name=='classifier' else 'restored'
        torch.onnx.export(model,(torch.rand(1,3,128,128),),str(path),input_names=['image'],
                          output_names=[output_name],opset_version=18,dynamo=True,external_data=False)
        onnx.checker.check_model(onnx.load(str(path)))
        options=ort.SessionOptions();options.intra_op_num_threads=4
        session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        datasets=[RestorationDataset(args.data,'val',condition=condition,limit=4 if name=='classifier' else 16)
                  for condition in (CONDITIONS if name=='classifier' else [name])]
        errors=[]
        with torch.no_grad():
            for dataset in datasets:
                for batch in DataLoader(dataset,batch_size=1):
                    x=batch['input'];expected=model(x).numpy();actual=session.run(None,{'image':x.numpy()})[0]
                    np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-5)
                    if name=='classifier':np.testing.assert_array_equal(actual.argmax(1),expected.argmax(1))
                    errors.append(float(np.max(np.abs(actual-expected))))
        if not errors:raise ValueError('No validation inputs for ONNX parity')
        checkpoint=args.checkpoints/name/'final/best.pt'
        report[name]=dict(config=checkpoints[name]['config'],epoch=checkpoints[name]['epoch'],
                          checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                          smoke=checkpoints[name]['smoke'],validation_images=len(errors),max_absolute_error=max(errors),
                          onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        path.with_suffix('.json').write_text(json.dumps(report[name],indent=2))
    metadata=dict(conditions=list(CONDITIONS),input='float32 NCHW [1,3,128,128], RGB [0,1], PIL bilinear resize',
                  classifier_output='probabilities',clean_route='identity',components=report,
                  smoke=any(c['smoke'] for c in checkpoints.values()))
    (args.output/'pipeline.json').write_text(json.dumps(metadata,indent=2))
    import mlflow
    mlflow.set_tracking_uri('sqlite:///'+str(Path('mlflow.db').resolve()));mlflow.set_experiment('Task2 Hard Routing')
    with mlflow.start_run(run_name='onnx-export'):
        mlflow.log_dict(metadata,'pipeline.json')
        for name,entry in report.items():mlflow.log_metric(name+'_onnx_max_absolute_error',entry['max_absolute_error'])
        mlflow.log_artifacts(str(args.output),artifact_path='onnx')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
