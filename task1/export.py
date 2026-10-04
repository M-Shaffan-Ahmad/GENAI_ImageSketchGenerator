import argparse
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
from dataset_loaders import RestorationDataset
from task1.train import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', type=Path, default=Path('models/universal.onnx'))
    p.add_argument('--data', default='data/prepared_v2/restoration')
    args = p.parse_args()
    torch.set_num_threads(4)
    model, checkpoint = load_checkpoint(args.checkpoint); model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Deployment uses fixed 128x128 images and batch size one.
    sample = torch.rand(1, 3, 128, 128)
    torch.onnx.export(model, (sample,), str(args.output), input_names=['image'], output_names=['restored'],
                      opset_version=18, dynamo=True, external_data=False)
    onnx.checker.check_model(onnx.load(str(args.output)))
    session = ort.InferenceSession(str(args.output), providers=['CPUExecutionProvider'])
    errors = []
    ds = RestorationDataset(args.data, 'val', limit=16)
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=1):
            x = batch['input']
            expected = model(x).numpy()
            actual = session.run(None, {'image': x.numpy()})[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            errors.append(float(np.max(np.abs(actual-expected))))
    metadata = dict(config=checkpoint['config'], smoke=checkpoint.get('smoke', False),
                    validation_images=len(errors), max_absolute_error=max(errors),
                    input='float32 NCHW [1,3,128,128], RGB [0,1], PIL bilinear resize',
                    checkpoint=args.checkpoint)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)); print(json.dumps(metadata, indent=2))
    import mlflow
    mlflow.set_tracking_uri('sqlite:///'+str(Path('mlflow.db').resolve()))
    mlflow.set_experiment('Task1 Universal Restoration')
    with mlflow.start_run(run_name='onnx-export'):
        mlflow.log_param('checkpoint', args.checkpoint)
        mlflow.log_metric('onnx_max_absolute_error', max(errors))
        mlflow.log_dict(metadata, 'export.json')
        mlflow.log_artifact(str(args.output), artifact_path='onnx')


if __name__ == '__main__':
    main()
