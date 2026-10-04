# Frozen trained artifacts

The Git repository includes `trained_models.zip` (seven selected ONNX graphs and
metadata) and `checkpoints/` (seven selected PyTorch checkpoints). These are the
same artifacts used for the reported final tests, not another training run.
`SHA256SUMS.txt` covers every published model/checkpoint file.

After cloning, extract `trained_models.zip` into the repository root using your
archive manager or `unzip artifacts/trained_models.zip`. This creates `models/`.
Then run the Docker commands in the root README.

To reproduce the final tests, copy the checkpoint tree into the expected staging
location before running the evaluation commands:

```bash
mkdir -p runs/integration/staging
cp -r artifacts/checkpoints runs/integration/staging/
```

The repository currently requires authenticated access. Make it public or grant
access to the evaluator. A raw model URL works for signed-out readers only after
the repository is public. GitHub Releases remain an optional alternative; no
release was created through an API in this session.
