# Run B in Colab

1. Open https://colab.research.google.com/ and upload `colab/task1_run_b.ipynb`.
2. Select **Runtime → Change runtime type → GPU**.
3. Choose **Runtime → Run all**, allow Drive access, and upload `colab/task1_run_b_source.zip` when prompted.
4. Wait for 60 epochs, both validation evaluations, the comparison table and ONNX export. The final cell downloads `task1_run_b_results.zip`. Place that ZIP in the project's `colab/` directory for review.

Run B changes only original-baseline alpha from 0.6648628294821612 to 0.5. Latent stays at 8192, and all other baseline settings are preserved. Equal coefficients do not guarantee equal numerical contributions from L1 and SSIM. Training starts from scratch with no Optuna search. This experiment tests whether increased SSIM weight improves preservation; improvement is not assumed.

Drive backups are separate at `MyDrive/GenAI_Ass1/run_b_alpha05`; runtime files are at `/content/ass1_run_b`. Reconnect and rerun all cells to resume from the last completed epoch with the same configuration. An unfinished epoch repeats after a disconnect.

The notebook checks baseline train/validation fingerprints and compares against baseline and completed Run A. Only baseline vs Run B isolates loss balance; Run A vs Run B also differs in latent size. It evaluates validation data and retains the test set. Results include checkpoints, histories, metrics, crops, comparison.csv and a verified ONNX export.

Rebuild the source ZIP with:

```bash
python3 package_colab.py --output colab/task1_run_b_source.zip
```
