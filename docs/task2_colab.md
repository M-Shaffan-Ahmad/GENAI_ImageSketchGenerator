# Task 2: train in Colab

1. Open https://colab.research.google.com/ and upload `colab/task2_gpu.ipynb`.
2. Select **Runtime → Change runtime type → GPU**, then **Run all**.
3. Allow Drive access, then upload `colab/task2_source.zip` when prompted.
4. Keep the session running through tuning, final training, both validation evaluations and ONNX export. The final cell downloads `task2_results.zip`; place it in this project's `colab/` folder for review.

The notebook trains a four-class corruption classifier and three independent specialists. Predicted-clean images use an exact identity bypass. Classifier batches have equal numbers of all four classes, using four runtime views per training photo. Specialist training uses only its own corruption. All splits and fixed validation cases match the existing data foundation.

Starting budgets are classifier 8 Optuna trials × 5 epochs followed by 20 final epochs; each specialist 6 trials × 8 epochs followed by 60 final epochs. These are editable starting budgets, not an exhaustive search. Each selected model trains from scratch. Classifier tuning covers learning rate, channels, dropout and weight decay; specialists tune learning rate, batch size, channels, compressed latent size, dropout and loss balance. Tracking uses MLflow and separate persistent Optuna studies.

Artifacts are separate at `/content/ass1_task2` and `MyDrive/GenAI_Ass1/task2_v1`. On disconnect, reconnect and rerun all cells with the same configuration and budgets to restore studies and resume final training from the last completed epoch. An unfinished epoch repeats. Completed and pruned trials count toward the trial budget; interrupted trials can leave extra records in the study.

Validation produces classifier accuracy, macro F1 and row-normalized confusion matrices, per-image and per-severity oracle/predicted routing scores, routing penalties and failure panels. The original validation has 408 cases; the separate severity sweep has 4080 cases (408 clean plus 1224 cases per corruption). The latter is not balanced across classes; inspect macro F1 and class support. Exact clean identity has PSNR capped at 120 dB; compare groups separately rather than interpreting the overall average as restoration quality.

All four ONNX components are exported and checked against PyTorch. `pipeline.json` records class order, model hashes, checkpoint provenance and smoke status. The notebook also verifies the ONNX router used by FastAPI. No test-set evaluation occurs in the notebook.

After reviewing the downloaded trained results, install its `models/task2/` folder into the local `models/task2/` directory to enable the app's Hard-Routed Restoration workspace. The backend uses `TASK2_MODEL_DIR` (default `models/task2`) and `/hard-routed-restoration`; it accepts image uploads or samples, simulates the requested corruption, then classifies the actual input without supplying the corruption label to the router.

Local verification completed with small CPU smoke runs: classifier and all three specialists trained; classifier and blur-specialist Optuna studies completed; two-epoch resumed classifier and blur-specialist model states/history matched uninterrupted runs exactly; original and severity validation reports generated; all four ONNX components exported and passed parity. Smoke runs do not demonstrate restoration quality. Full GPU training and final model review remain required. Frontend JSX bundled with esbuild; a full Vite/Docker build and browser visual check remain outstanding.

Rebuild the upload with:

```bash
python3 package_colab.py --output colab/task2_source.zip
```
