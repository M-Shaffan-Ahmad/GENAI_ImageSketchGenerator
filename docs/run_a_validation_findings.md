# Run A: larger latent validation review

Reviewed `colab/task1_run_a_results.zip`. ZIP CRC passed. The checkpoint completed epoch 60 and its SHA256 matches the blur report. Training/validation manifest hashes and image counts match the baseline checkpoint. The fixed blur manifest is byte-identical to the baseline. Only bottleneck_dim changed, 8192 to 16384; all other configuration values match. This is one run per configuration, so it does not establish a repeatable effect across training seeds.

## Fixed blur: 408 photos per severity

| Severity | Input PSNR | Baseline output PSNR | Run A output PSNR | Run A minus input | Baseline both metrics improved | Run A both metrics improved |
|---|---:|---:|---:|---:|---:|---:|
| low | 33.63 | 27.59 | 27.98 | -5.66 | 0/408 | 0/408 |
| medium | 27.33 | 26.35 | 26.68 | -0.65 | 13/408 | 60/408 |
| high | 24.45 | 24.27 | 24.45 | -0.00 | 150/408 | 225/408 |

Mild blur remains worse in both PSNR and SSIM for all 408 photos. Medium blur remains worse on average; photos improving in both metrics increase from 13 to 60. Severe blur has essentially unchanged mean PSNR relative to input (-0.00424 dB), slightly improved mean SSIM (+0.00417), and worsened mean L1 (+0.00213). Photos improving in both PSNR and SSIM increase from 150 to 225. These counts do not imply pixel-exact restoration.

On the original all-condition validation, output PSNR changes are clean +0.283 dB, salt/pepper +0.125 dB, blur +0.299 dB, occlusion -0.115 dB. SSIM improves slightly in every group. Occlusion L1 worsens by 0.000798. Larger latent helps modestly but does not solve preservation.

## Visual checks

Reviewed the saved matched crops for mild blur source 1522, high blur source 6313 and clean control source 6651. Source 1522 still loses petal boundary detail; source 6313 remains visibly blurred. Source 6651 still develops a bright tan block in a naturally dark background even without input corruption. These are selected examples, not average visual-quality estimates.

ONNX export metadata reports 16 validation cases with maximum absolute PyTorch/ONNX error 4.77e-7. That is archive-reported parity; inference was not rerun locally during this review. The deployed app model was not replaced.

## Next controlled experiment

Run the previously proposed Run B: baseline latent 8192, alpha 0.5 instead of 0.6648628294821612, with other baseline settings unchanged and 60 epochs from scratch. This isolates the loss-balance change from latent size. Evaluate the same all-condition and fixed-blur manifests. If subsequently testing alpha 0.5 with latent 16384, label it as a separate combined experiment. No new training or test-set evaluation was performed in this review.

The machine-readable comparison is saved at `runs/task1/run_a/review/comparison.csv`.
