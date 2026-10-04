# Blur preservation: fixed validation investigation

This assessment uses the Colab spatial model at epoch 60 from
`colab/task1_spatial_results.zip`. The checkpoint's validation-manifest fingerprint and
all validation image-content hashes were checked before evaluation.

All **408 validation photos** were evaluated at the three prescribed blur settings:
kernel/sigma (3,0.7), (5,1.5), (7,2.5). This produced **1,224 blur cases**. An additional
408 clean controls measure ordinary reconstruction damage. No training, model change,
or test-set evaluation was performed.

| Severity | Input PSNR | Restored PSNR | Difference | Input SSIM | Restored SSIM | PSNR improved | PSNR and SSIM improved |
|---|---:|---:|---:|---:|---:|---:|---:|
| Low | 33.63 | 27.59 | -6.05 dB | 0.963 | 0.866 | 0/408 | 0/408 |
| Medium | 27.33 | 26.35 | -0.98 dB | 0.846 | 0.819 | 51/408 | 13/408 |
| High | 24.45 | 24.27 | -0.18 dB | 0.721 | 0.720 | 168/408 | 150/408 |

Clean-control reconstruction averages 27.66 dB PSNR, 0.871 SSIM and 0.02979 L1.
L1 also worsens at every blur level on average: low 0.01320→0.03004,
medium 0.02803→0.03421, high 0.04018→0.04311. The clean identity comparison has
zero pixel error; the implementation caps its PSNR at 120 dB.

## What the measurements establish

**Mild blur is a consistent preservation failure.** Every validation image loses both
PSNR and SSIM. The mild input is substantially more accurate than this checkpoint's
ordinary clean reconstruction. Low-blur output quality is close to clean-output quality
(27.59 versus 27.66 dB), suggesting that general reconstruction damage is an important
part of this failure. This observation does not isolate its architectural or training cause.

**Medium blur is usually harmed.** Only 12.5% of photos improve in PSNR, and 3.2%
improve in both PSNR and SSIM. A positive pixel-error result alone cannot establish
structural recovery: source 4912 gains 1.08 dB but loses about 0.002 SSIM.

**High blur has mixed outcomes.** PSNR improves for 41.2% of photos and SSIM improves
for 56.6%; both improve for 36.8%. Mean PSNR and SSIM are still slightly worse, so the
checkpoint does not demonstrate an average high-blur benefit on this fixed set.
The average conceals individual gains and large failures; no significance test was performed.

## Observed visual cases

- Fixed representative source 1522: the bright pink petal boundaries and background
  foliage become softer. PSNR differences are -7.75, -2.07 and -0.62 dB as severity rises.
- Source 6313: the yellow bloom's narrow petal tips and texture are softened. The
  high-blur PSNR difference is almost zero (-0.07 dB), despite visible remaining detail loss.
- Source 5191: the yellow flower boundary and petal separations soften in the enlarged
  edge crop; mild-blur PSNR falls by 7.76 dB.
- High-blur source 4945 is an improvement case: +1.07 dB PSNR and +0.007 SSIM.
  Its white petal boundary becomes more defined, while substantial blur remains.
- Source 5640 is another high-blur improvement case: +1.01 dB and +0.014 SSIM.
  This is a modest recovery, not faithful restoration of all fine flower details.
- Sources 6651 and 5100 show bright/tan rectangular artifacts in naturally dark
  backgrounds. The artifacts also appear in clean-control outputs, before any artificial
  blur is applied. This is a general reconstruction artifact. Confusion between naturally
  dark regions and training occlusion masks is a possible explanation that requires a
  controlled experiment; this evaluation does not prove it.

## Saved evidence and reproduction

Run from the repository root:

```bash
python -m task1.blur_validation --results-zip colab/task1_spatial_results.zip
```

Artifacts are saved in `runs/task1/spatial/blur_validation/`:

- `manifest.json`: fixed cases with source IDs, image hashes, blur settings and seeds.
- `per_image.csv`, `grouped_metrics.csv`, `results.json`: scores, paired differences,
  improvement counts, model provenance and example/crop selections.
- `representative_grid.png`: the same four seed-42-selected photos across every severity.
- `largest_gain_grid.png` and `largest_loss_grid.png`: deliberately selected score extremes.
  The low-blur "largest gain" cases are actually the smallest losses; none improved.
- `clean_control_none_6651.png`, `clean_control_none_5100.png`: evidence that the dark-region
  artifacts exist even on unblurred inputs.
- `gain_distributions.png`: distributions of paired PSNR/SSIM differences.
- `report.md`: generated evaluation summary.

Each visual panel contains clean target, blurred input, restored image and absolute error
with a fixed 4x display gain. All crops use the same 48×48 edge-rich ground-truth region
and nearest-neighbor enlargement. Do not interpret the amplified error display as raw
error magnitude or the selected extremes as average performance. Metrics and artifacts
were logged to MLflow. All ten automated data/model/API/diagnostic tests passed.

This completes the fixed validation set, paired metric comparison and visual investigation.
Further training experiments are separate work and should use these validation diagnostics
to assess preservation alongside the other corruption conditions.
