# Generative AI assignment: four integrated workspaces

This checkout implements the adapted Flowers102 pipeline and all four tasks. The reviewed
trained models are packaged: Task 1 Run A, Task 2 classifier and specialists, Task 3 joint
checkpoint, and Task 4 epoch 53. See [final integration and launch instructions](docs/final_integration.md).

The frozen official-test comparison is complete (61,490 restoration views and 1,068
sketch pairs). The [IEEE paper](report/main.pdf), [LaTeX source](report/main.tex),
[submission handoff](docs/submission_steps.md), [demo script](docs/demo_script.md),
and [preserved Stitch export](report/stitch/) are included. The app follows the supplied
Stitch design with four functional workspaces, responsive controls, routing displays,
and real inference results. Source and frozen weights are published in this public
repository. The [demonstration video](https://www.youtube.com/live/1NpOTGTbwuc) is supplied by the author. Final author review and Classroom submission remain.

## Fresh clone: run with Docker

The repository includes the frozen [ONNX model package](artifacts/trained_models.zip)
and [PyTorch checkpoints](artifacts/checkpoints/). The
[direct model download](https://raw.githubusercontent.com/M-Shaffan-Ahmad/GENAI_ImageSketchGenerator/main/artifacts/trained_models.zip)
is public. Extract `artifacts/trained_models.zip` into the repository root with an
archive manager or `unzip artifacts/trained_models.zip`, producing `models/release.json`.
Verify hashes using `sha256sum -c artifacts/SHA256SUMS.txt` from the repository root.
Datasets are downloaded/prepared by the setup container rather than committed.

```bash
docker compose --profile setup run --rm --build prepare-data
docker compose up --build
```

The setup container downloads official Oxford data and reproduces the prepared
manifests, checking them against model fingerprints. Open http://localhost:8080.
No local Python environment is needed for this Docker route. If a port is occupied:

```bash
FRONTEND_PORT=18081 BACKEND_PORT=18082 docker compose up --build
```

Then open http://localhost:18081. Both inference images and the data setup image
were built and exercised. Current verification is recorded in `docs/final_integration.md`.

## Frozen results and experiment records

The submission source package copies final JSON/CSV/figures to
`experiments/final_test/`. Repeating evaluation requires the prepared official dataset and the committed
checkpoints copied to their expected location:

```bash
mkdir -p runs/integration/staging
cp -r artifacts/checkpoints runs/integration/staging/
```

Run evaluation in the research Python environment:

```bash
venv/bin/python scripts/final_evaluation.py --threads 8
venv/bin/python -m task4.evaluate --checkpoint runs/integration/staging/checkpoints/task4/best.pt --split test --output runs/final_test/sketch
```

Portable viewing records preserve the original recorded Colab histories, explicitly
tagged as retrospective imports. Original Optuna studies are in `experiments/optuna/`.
To view tracking after unpacking/cloning (requires the research Python environment):

```bash
venv/bin/python scripts/relocate_tracking.py
venv/bin/mlflow ui --backend-store-uri sqlite:///experiments/tracking.db --host 127.0.0.1 --port 5000
```

For the full remaining account steps and final paper updates, use
[the submission guide](docs/submission_steps.md).

The deployed Task 1 Run A uses a 64×16×16 latent (16,384 values, 3:1 compression) and
2.92% dropout. Training defaults still use an 8,192-value spatial latent and zero dropout;
deployment uses the selected checkpoint's configuration. Legacy checkpoints remain loadable.
The Colab notebooks remain available for reproducible training.

To launch the integrated production build locally without Docker or Node at runtime:

```bash
bash run_app.sh
```

Open **http://localhost:8080**; API documentation is at **http://localhost:8080/api/docs**.
The existing virtual environment and built frontend are required.

## Investigate blur preservation on validation data

```bash
python -m task1.blur_validation --results-zip colab/task1_spatial_results.zip
```

Alternatively, pass `--checkpoint` for a locally trained model. The command verifies
validation provenance and image hashes, creates fixed low/medium/high blur cases for all
408 validation photographs, and evaluates 1,224 blur cases plus 408 clean controls.
It writes its own manifest under `runs/task1/spatial/blur_validation`, preserving the
original validation protocol. No training or test evaluation is performed.
See [the measured findings and visual interpretation](docs/blur_validation_findings.md).

Read `report.md` and `results.json` there for paired input-versus-output metrics,
improvement counts and selected examples. The fixed representative grid compares the
same four photos across all severities. Separate grids show the largest PSNR gains and
losses; these are deliberately selected extremes. Enlarged crops use identical coordinates
and scaling, and absolute-error panels have a fixed 4x display gain. Results and figures
are also logged to MLflow.

The historical vector Task 1 baseline was trained for 20 epochs and tested on all 61,490
official test cases. Read [methodology](docs/task1_methodology.md) and
[actual results and failure analysis](docs/task1_results.md). Its outputs are oversmoothed:
high-noise/occlusion PSNR gains coexist with poor detail preservation, and clean/blurred
inputs are degraded. Do not describe this initial baseline as strong restoration quality.

## Setup and data

Run all Python commands from the repository root. Python 3.12 was used for validation.

```bash
python3 -m venv venv
source venv/bin/activate
# CPU machines: install CPU PyTorch first to avoid downloading CUDA libraries.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
mkdir -p raw_data
curl -fL https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz -o raw_data/102flowers.tgz
tar -xzf raw_data/102flowers.tgz -C raw_data
curl -fL https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat -o raw_data/setid.mat
python prepare_dataset.py
python -m unittest discover -s tests -v
```

`requirements.lock.txt` records the tested environment; its CPU PyTorch build needs the CPU
package index above. The original prepared folders remain untouched, but are not used.
Use `data/prepared_v2` for all new work. Source paths are relative to the repository root.

Official Oxford train and validation IDs form development data, repartitioned 80/20 with
seed 42. Images sharing the same resized RGB SHA256 stay together. Development images
that duplicate official test content are excluded; the official test IDs remain intact.
Current counts are **1,630 train / 408 validation / 6,149 test**, after excluding source
IDs 7307 and 8077. The script records the source ID, path and content hash, and checks
Oxford's split-file MD5. Approximate 80/20 rounding is necessary for grouped duplicates.

Training generates fresh corruptions each load with equal probability across four conditions.
Validation stores one deterministic condition per image, balanced across conditions.
Test manifests store clean plus nine corruption/severity cases per photo: **61,490 cases**.
Every case records its seed, parameters and exact occlusion coordinates. No corrupted image
copies are saved. Non-overlapping boxes have random locations within horizontal bands to
control their total union area; integer-pixel rounding is recorded as actual area ratio.

The sketch dataset uses 1,748 distinct development photos and 356 official test photos,
with all three styles per photo. Fifteen percent of its development photos are held out:
**1,486 train / 262 validation / 356 test**, or 6,312 style pairs. Splitting photographs
before expanding styles prevents leakage and gives exact style balance. All paired
spatial augmentations are synchronized. Sketches are algorithmic targets, not human art;
the GAN's results should be described as approximating those algorithms.

## Task 1 training and tuning

```bash
# Six short trials are a starting study, not an exhaustive architecture comparison.
python -m task1.train tune --epochs 10 --trials 12 --output runs/task1/spatial/search
# Retrain the selected configuration from scratch on all development training data.
python -m task1.train train --config runs/task1/spatial/search/best_config.json --output runs/task1/spatial/final
```

The revised network downsamples 128 → 64 → 32 → 16, compresses channels into a learned
spatial map, then decodes back to RGB 128×128. It has **no skip connections**. The training objective
is `alpha * L1 + (1-alpha) * (1-SSIM)`. SSIM uses an 11×11 Gaussian window with
sigma 1.5 and RGB data range 1. The validation objective uses fixed weights 0.8/0.2
across every trial so changing alpha cannot make a trial win by changing its score scale.

Optuna searches learning rate [1e-4, 3e-3] logarithmically, batch size {16,32,64},
bottleneck {4096,8192,16384}, base channels {16,32,48}, dropout [0,0.1], alpha [0.5,0.95].
Trials use a persistent SQLite study and median pruning. Search settings, completed trials,
best configuration and CSV results are retained. The final schedule defaults to 60 epochs;
`--epochs` overrides it. All data budgets are logged. Use `--train-limit 64 --val-limit 40`
only for smoke tests; such checkpoints are explicitly marked as smoke models.
The 12×10 tuning budget and 60-epoch schedule are configurable starting budgets for GPU
training; their adequacy must be judged using validation curves. Resume interrupted training
with the same command plus `--resume runs/task1/spatial/final/last.pt`. The epoch argument
is the total target, not an additional count. Checkpoints include optimizer and RNG states.

MLflow records configurations, losses, metrics, checkpoints and training curves locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Optuna trials use validation data only. Final testing is a separate explicit command:

```bash
python -m task1.evaluate --checkpoint runs/task1/spatial/final/best.pt --split val --output runs/task1/spatial/validation
python -m task1.export --checkpoint runs/task1/spatial/final/best.pt --output models/universal_spatial.onnx
```

Evaluation writes per-image and per-corruption/severity L1, PSNR and SSIM, including the
unrestored input baseline, representative clean/input/output/error panels, and four worst
L1 cases from distinct photographs for human failure analysis. These candidates need interpretation in the report.
For exact identity comparisons, the finite PSNR ceiling is 120 dB (MSE floor 1e-12).
Export checks the ONNX graph and compares framework/runtime outputs on 16 validation
inputs with `rtol=1e-4, atol=1e-5`. Deployment expects RGB float32 NCHW [1,3,128,128].
The official test set was already evaluated for the vector baseline. Keep revision selection
on validation data; any new final test result is a subsequent comparison, not an unseen test.
The integrated app already uses the reviewed Run A export at `models/universal.onnx`.
To install the same selected artifacts again, run `python integrate_models.py`; it verifies
provenance and parity before replacing files and backs up previous models.

## Browser application

```bash
docker compose up --build
```

Open **http://localhost:8080**. Upload an image or select a validation sample, choose
runtime corruption settings, restore it, inspect output/settings/time, and download PNG.
An error map is shown only when a clean reference is available; an uploaded already
corrupted image does not provide a known target. API docs: http://localhost:8000/docs.
Health and samples endpoints are `/health` and `/samples`. The four inference endpoints
are `/universal-restoration`, `/hard-routed-restoration`, `/soft-moe-restoration`, and
`/object-to-sketch`; the frontend accesses these through `/api`.

The backend uses ONNX Runtime only and loads the selected models under `models/`.
Containers mount raw images, manifests and models read-only. Public hosting is optional.
Docker Desktop must be running. Without Docker, launch the API using:

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

The React + Tailwind frontend builds with `cd frontend && npm ci && npm run build`.
For local development, keep the API running and run `npm run dev` in `frontend`; Vite
proxies `/api` to port 8000. Open the local URL printed by Vite.
The application is implemented, but **Google Stitch design provenance has not been
created**. Create the assignment-required Stitch design and record screenshots before
claiming that submission requirement is satisfied. Model files and data are excluded
from Git; publish documented model download links when submitting.

## Research and remaining submission work

- [Oxford Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- [Official torchvision Flowers102 split implementation](https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/flowers102.html)
- [SSIM paper and implementation resources](https://ece.uwaterloo.ca/~z70wang/research/ssim/)
- [PyTorch ONNX export documentation](https://docs.pytorch.org/docs/stable/onnx.html)

Record the actual trial budget and hardware in the IEEE report; do not describe short
trials as exhaustive optimization. Interpret example grids and failure candidates, explain
the synthetic sketch targets, include Stitch evidence, and document this Codex-assisted
implementation and its tests in the AI-use appendix. Full report and demonstration video
are separate submission deliverables.

## Task 2: hard-routed restoration

Task 2 includes a balanced four-class CNN classifier, independently trained salt/blur/occlusion autoencoders, an exact predicted-clean identity bypass, per-component Optuna studies, MLflow tracking, resumable training, oracle-versus-predicted routing evaluation and four ONNX exports. Its reviewed exports are installed in `models/task2/`. The app displays the selected route and class probabilities.

Use `colab/task2_gpu.ipynb` with `colab/task2_source.zip` for full GPU training. See [Task 2 Colab instructions](docs/task2_colab.md) for budgets, resume behavior, metrics and current verification limits. Local smoke checkpoints are development artifacts, not final trained models.

## Task 3: soft mixture of experts

Task 3 initializes its gate and specialists from trained Task 2 checkpoints, freezes specialists for gate warm-up, then jointly fine-tunes with a reduced learning rate. It adds persistent Optuna studies, MLflow tracking, balanced-batch routing regularization, resumable training, hard-vs-soft validation with gate heatmaps, and full-pipeline ONNX export. The app includes Soft Mixture-of-Experts Restoration with four contribution weights.

Run `colab/task3_gpu.ipynb` with `colab/task3_bundle.zip` to reproduce training using reviewed Task 2 initialization weights. See [Task 3 Colab instructions](docs/task3_colab.md) and [measured validation findings](docs/task3_validation_findings.md). The selected joint checkpoint is installed in `models/task3/`.

## Task 4: style-conditioned object-to-sketch GAN

Task 4 adds a style-conditioned U-Net generator and PatchGAN discriminator trained on the existing paired synthetic sketch foundation. It includes Optuna tuning, MLflow tracking, paired augmentation, resumable generator/discriminator training, fixed visual samples, per-style validation with blank-baseline and diversity checks, and ONNX export with a categorical style input. The app includes the Object-to-Sketch Generator workspace.

Use `colab/task4_gpu.ipynb` with `colab/task4_source.zip` to reproduce training from scratch; see [Task 4 Colab instructions](docs/task4_colab.md). The reviewed epoch 53 generator is installed in `models/task4/`. See [validation findings and limitations](docs/task4_validation_findings.md).
