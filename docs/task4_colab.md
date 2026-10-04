# Task 4 Colab training

1. Open https://colab.research.google.com/ and upload `colab/task4_gpu.ipynb`.
2. Select **Runtime → Change runtime type → GPU**, then **Run all**.
3. Allow Drive access and upload `colab/task4_source.zip` when prompted. No Task 1–3 model ZIP is needed.
4. Let tuning, full training, paired validation and ONNX export finish. Download `task4_results.zip` from the final cell and place it in the project's `colab/` folder for review.

This is the adapted object-to-sketch task, using the existing Flowers foundation. Targets are deterministic synthetic fine-pencil, technical-ink and tonal-charcoal approximations, not human-drawn sketch annotations. The 2104 photos are grouped across all styles: 1486 train, 262 validation and 356 test, with three pairs per photo. Test data stays reserved. Colab regenerates the targets and verifies the paired manifest fingerprint.

The generator is a five-level U-Net with learned three-category style embedding, encoder/decoder skips, InstanceNorm and sigmoid RGB output. The discriminator is a conditional 70-pixel PatchGAN producing 14×14 logits at 128×128 input size; it receives the photo, real/generated sketch and its own learned categorical style embedding. Style indices are 0/1/2 inside PyTorch/ONNX, while API/user style IDs are 1/2/3.

Generator loss is adversarial BCE with logits plus lambda_L1 × paired L1, initially lambda_L1=100. Discriminator loss averages real and detached-fake BCE. Discriminator parameters are frozen during the generator update while gradients through its operations still reach the generator. Photo/sketch flips are identical. Train and validation data never share photo IDs or content groups.

Optuna tunes both learning rates, batch size, base channel count, dropout, embedding dimension and lambda_L1. Starting budget: 6 trials × 5 epochs, followed by 60 epochs from scratch using the selected configuration. The trial ranking uses fixed validation 0.8*L1+0.2*(1-SSIM), independent of tuned lambda. These are starting budgets rather than an exhaustive search.

Every epoch records discriminator real/fake losses, generator adversarial/L1/total losses and validation metrics separately. The first epoch, every fifth epoch and the final epoch save the same validation photos across styles. MLflow stores configuration, metrics, curves, samples and checkpoints.

Drive folder: `MyDrive/GenAI_Ass1/task4_v1`; runtime folder: `/content/ass1_task4`. Best and last checkpoints include both models, both optimizers, history and RNG states. Reconnect and rerun with the same configuration and budgets to restore studies and resume completed epochs. An interrupted epoch repeats. Study/config/data changes require a new experiment folder.

Validation reports all 786 pairs, per-style PSNR/SSIM/L1, a blank-white baseline, four representative photos across three styles, four distinct-photo failures, and generated-versus-target pairwise style diversity. Diversity alone is not style correctness; inspect actual outputs for style collapse, lost edges and nearly blank predictions. These scores measure approximation of synthetic targets, not artistic quality.

ONNX exports the generator alone with fixed batch-one image and int64 style input. Parity checks four validation cases per style. The app has an Object-to-Sketch Generator workspace, three named styles, upload/sample input and generated-image download. Its FastAPI endpoint is `/object-to-sketch`. After reviewing training results, install `models/task4/` to enable local inference; `TASK4_MODEL_PATH` defaults to `models/task4/sketch_generator.onnx`.

Local verification: 23 automated tests, CPU GAN training, exact resumed-versus-uninterrupted generator/discriminator/history comparison, one complete Optuna smoke trial, per-style evaluation, ONNX parity for all three styles, and JSX bundling. Full GPU quality evaluation, browser visual checks and a full Docker/Vite build remain outstanding. Local smoke models are not installed as final application models.

Rebuild the source upload:

```bash
python3 package_colab.py --output colab/task4_source.zip
```
