# Task 1 results and limitations

Final training completed 20 epochs on CPU. The lowest validation objective selected epoch 19. The selected model has 18,174,147 trainable parameters, a 512-value latent, and no skip connections.

Final testing used all 6,149 official test photographs in ten conditions each: 61,490 cases. No test subset limit was used. ONNX parity on 16 validation images had a maximum absolute difference of 3.5763e-7 (tolerance 1e-5 absolute, 1e-4 relative).

| Condition / severity | Input PSNR | Restored PSNR | Input SSIM | Restored SSIM |
|---|---:|---:|---:|---:|
| clean/none | 120.00 | 16.14 | 1.000 | 0.349 |
| salt_and_pepper/low | 19.91 | 16.14 | 0.609 | 0.350 |
| salt_and_pepper/medium | 15.64 | 16.10 | 0.344 | 0.349 |
| salt_and_pepper/high | 12.91 | 16.01 | 0.206 | 0.347 |
| gaussian_blur/low | 33.69 | 16.14 | 0.964 | 0.349 |
| gaussian_blur/medium | 27.36 | 16.14 | 0.848 | 0.349 |
| gaussian_blur/high | 24.47 | 16.13 | 0.725 | 0.349 |
| rectangular_occlusion/low | 16.82 | 15.83 | 0.859 | 0.342 |
| rectangular_occlusion/medium | 13.74 | 15.54 | 0.734 | 0.335 |
| rectangular_occlusion/high | 11.23 | 15.07 | 0.549 | 0.326 |

Each row averages 6,149 individual image scores. The clean-input identity baseline has zero MSE; its displayed PSNR is the implementation ceiling of 120 dB.

The model improves PSNR for medium/high salt-and-pepper noise and medium/high occlusion. High-noise SSIM also improves, from 0.206 to 0.347. Occlusion PSNR gains do not imply structural recovery: high-occlusion SSIM drops from 0.549 to 0.326. Clean, blurred and mildly corrupted images are degraded. This is a reproducible initial baseline, not a strong restoration result.

Visual outputs are oversmoothed and sometimes change foreground/background colors. The compressed representation and short training/search budgets are plausible contributors, but this run does not isolate their causal effects. The five completed short trials cannot establish an architecture optimum. Any further model selection must use validation evidence, and this test set can no longer be described as unseen after these reported measurements.

## Four observed failure cases

1. Source 6901, high occlusion: an orange bloom on a white background becomes red/pink with a green/dark background. Missing areas are filled, but colors and petal structure are wrong. PSNR rises from 6.16 to 8.18 dB while SSIM falls from 0.553 to 0.397.
2. Source 7466, high occlusion: saturated red/pink petals and small pale centers become diffuse yellow/green shapes. Visible unmasked features are lost as well as masked detail. PSNR falls from 9.08 to 8.64 dB and SSIM from 0.602 to 0.196.
3. Source 5669, high occlusion: the dark purple petals and thin stem are replaced by broad red/green regions, with a substantial background-color shift. PSNR rises from 6.33 to 9.37 dB but SSIM falls from 0.560 to 0.394; the aggregate error improvement masks semantic distortion.
4. Source 3515, low occlusion: most fringed pink flowers remain visible in the input, yet the output loses their fine edges and pale streaks. PSNR falls from 13.09 to 9.57 dB and SSIM from 0.887 to 0.152. Mild corruption is a particularly clear example of unnecessary reconstruction damage.

The panels are in `runs/task1/evaluation/failure_candidate_1.png` through `failure_candidate_4.png`; the combined grid is `failure_grid.png`. Thirty representative panels and the full per-image CSV are also saved.

## Functional verification

Eight automated data/model/API tests passed against the final ONNX model. The production React build passed. Headless Chrome passed clean/noise/blur/occlusion restoration, sample selection, file upload, PNG download and a 390-pixel layout without horizontal overflow. Desktop/mobile evidence was refreshed against the final model in `runs/task1/ui/`.

Docker Compose parses correctly, but its container build/run could not be exercised because the Docker Desktop daemon was unavailable. Google Stitch design provenance, the final IEEE LaTeX report and submission video still require completion.
