# Task 3 Colab training

1. Open https://colab.research.google.com/ and upload `colab/task3_gpu.ipynb`.
2. Select **Runtime → Change runtime type → GPU**, then **Run all**.
3. Allow Drive access and upload `colab/task3_bundle.zip` when prompted. It contains source code and exactly the four trained Task 2 best checkpoints reviewed from your results; uploading the 564 MB Task 2 results archive is unnecessary.
4. Let tuning, final training, evaluation and ONNX export finish. Download `task3_results.zip` from the final cell and place it in `colab/` for review.

The gate is initialized from the Task 2 classifier and the experts from the trained salt, blur and occlusion autoencoders. The model outputs the convex combination of the input identity branch and three expert reconstructions, weighted by softmax(gate logits / temperature). Temperature is positive and tuned; no corruption label is supplied at inference.

Training stages:

- Warm-up: freeze expert parameters and disable their dropout, train only the gate.
- Joint: unfreeze all experts, use a new Adam optimizer with a lower learning rate and fine-tune the complete system.

Loss is lambda_l1 * L1 + lambda_ssim * (1-SSIM) + lambda_ce * cross_entropy(raw gate logits, known training labels) + lambda_balance * sum((batch_mean_weights-0.25)^2). Classifier-style balanced runtime batches use all four conditions equally, four views per training photo. Balance is computed on the batch mean, as described in the original Task 3 specification, rather than forcing every image to have uniform weights. The adapted specification's balance notation is interpreted using this explicit batch-usage definition.

Optuna searches temperature, reduced joint learning rate, CE weight, balance weight and reconstruction balance (lambda_l1=alpha, lambda_ssim=1-alpha). The default budget is 6 trials with 1 warm-up and 3 joint epochs each; final training uses 3 warm-up and 20 joint epochs. These are editable starting budgets. Every trial and the final run start from the same trained Task 2 components. The fixed validation objective is 0.8*L1+0.2*(1-SSIM), so changing train-loss weights does not change the ranking criterion.

`best.pt` selects only among joint epochs. `warmup_best.pt` retains the best gate-only model for comparison; `last.pt` resumes the most recently completed epoch. Configuration, initialization hashes, data fingerprints and data limits are checked on resume. Studies and checkpoints are backed up in `MyDrive/GenAI_Ass1/task3_v1`; runtime files are under `/content/ass1_task3`. Reconnect and rerun using the same bundle and budgets; unfinished epochs repeat. Changing search settings requires a separate study folder.

Validation evaluates original fixed 408 cases, the identical Task 2 4080-case severity sweep, and gate-only warm-up results. Reports include hard-vs-soft PSNR/SSIM/L1, paired input improvements, per-photo gate weights, mean weights by true condition and severity, heatmaps, weight distributions, dominant/distributed gate examples, and failure cases. Dominant gate class accuracy is descriptive, not a guarantee that the best restoration must select the corruption's named expert. Low mean branch usage is flagged for inspection, not declared collapse automatically. Composite corruptions are handled by the mixture's inference formulation but are not included in the single-corruption training or these quantitative evaluations.

The notebook checks original and severity manifest hashes against saved Task 2 results before accepting the comparison. Overall PSNR can be inflated by exact identity outputs in hard routing; inspect clean and mild-blur groups separately. Soft blending can also damage clean inputs, and improved preservation is not guaranteed. Test-set evaluation is not performed.

The full MoE is exported as one `models/task3/soft_moe.onnx` graph with two outputs: restored image and four weights. PyTorch/ONNX image and weight parity is checked on 16 fixed cases. The FastAPI `/soft-moe-restoration` endpoint and Soft Mixture-of-Experts app workspace use this graph and display learned contributions. Install reviewed trained exports under `models/task3/` to enable local inference; development smoke models are not installed automatically.

Rebuild the upload bundle:

```bash
python3 package_colab.py --output colab/task3_bundle.zip --task2-results colab/task2_results.zip
```

Full GPU training, quality assessment, browser visual checks and a full Docker/Vite build remain separate from local smoke verification.

Local verification passed: 20 automated tests; gate warm-up and joint CPU smoke training; expert weights unchanged after warm-up; resumed training across the stage transition exactly matching uninterrupted model weights and history; one complete Optuna smoke trial; original and severity-sweep smoke reports; complete ONNX graph parity on 16 validation cases (image max error 1.7881393432617188e-07, weight max error 2.9802322387695312e-08); actual FastAPI multipart inference with smoke ONNX models for all four conditions; JSX bundling with esbuild. Smoke metrics are not final restoration results.
