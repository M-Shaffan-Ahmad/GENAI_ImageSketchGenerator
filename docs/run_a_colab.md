# Run A in Google Colab

1. Open https://colab.research.google.com/ and upload `colab/task1_run_a.ipynb`.
2. Select **Runtime → Change runtime type → GPU**.
3. Choose **Runtime → Run all**. Mount Drive when prompted, then upload `colab/task1_run_a_source.zip` at the source-upload prompt.
4. Allow all 60 epochs, validation, and ONNX export to finish. Download `task1_run_a_results.zip` from the last cell and place it in this project's `colab/` directory for review.

Only latent size changes: 8192 → 16384, reducing compression from 6:1 to 3:1. All other settings match the configuration actually used in `task1_spatial_results.zip`, including dropout 0.029214464853521818 (2.92%). No Optuna retuning is performed. The larger latent is an experiment; improvement requires validation evidence.

Artifacts use `/content/ass1_run_a` and `MyDrive/GenAI_Ass1/run_a_latent16384`. Reconnect and rerun all cells to restore and resume from the last completed epoch. A disconnect during an epoch repeats that unfinished epoch. Keep the same source ZIP and configuration when resuming.

Validation writes the original all-condition metrics, 408 clean controls and 1224 fixed blur cases, representative crops, failures, and `comparison.csv`. Positive PSNR/SSIM changes and negative L1 changes mean improvement over the baseline. This run does not evaluate the test set or replace the app's deployed model.

Rebuild the source archive locally with:

```bash
python3 package_colab.py --output colab/task1_run_a_source.zip
```
