# Task 4 trained-results review

Reviewed `colab/task4_results.zip`. ZIP CRC passed. Training completed 60 epochs; the selected checkpoint is epoch 53. The checkpoint contains generator/discriminator states, both optimizers and RNG state. Its recorded training size is 4,458 pairs and validation size is 786 pairs. Validation is unrestricted and the checkpoint is not a smoke run. Search records six trials, three complete, with trial 1 selected.

The checkpoint, ONNX and local paired-manifest hashes match their uploaded metadata and validation report. Checkpoint SHA256: `e9c48c296dec0427f4c85d302cae90a4a19eb3d399b5814872d1702efef374a9`. Manifest SHA256: `055d9cef95187e85cb582def57cd03c6871d58df233363b91865c6faac77a432`.

## Validation quality

Each style uses the same 262 validation photos. The figures below are from the uploaded evaluation, not an independent rerun of all predictions.

| Style | L1 (lower is better) | SSIM (higher is better) | PSNR | Blank-white SSIM |
|---|---:|---:|---:|---:|
| Fine Pencil | 0.03487 | 0.9027 | 25.14 dB | 0.2852 |
| Technical Ink | 0.16378 | 0.5252 | 8.29 dB | 0.1396 |
| Tonal Charcoal | 0.03824 | 0.8149 | 14.81 dB | 0.2886 |

All 262 cases in every style have lower L1 than the blank-white baseline. SSIM beats that baseline in 262/262 pencil, 261/262 ink and 262/262 charcoal cases. This rules out a simple all-white output as an explanation for the average quality, but does not establish artistic quality or perfect reconstruction. These are matches to algorithmically generated targets.

Style diversity was measured on all 262 complete photo triplets. Generated pairwise L1 is 0.4284 (pencil/ink), 0.1781 (pencil/charcoal) and 0.4259 (ink/charcoal), versus target differences 0.4451, 0.1852 and 0.4348. Together with the inspected panels, these indicate distinct styles rather than global style collapse. Pairwise distances alone do not establish correct local detail.

## Visual findings

Inspected source 6800 in all three styles, pencil failure source 5100, and ink failure source 6308. Source 6800 retains the main flower and stem structure, with clearly different pencil shading, ink contours and charcoal masses. Ink contains edge-position/detail errors; charcoal loses some small structures.

Source 5100 has a conspicuous grid pattern in the generated dark background and a mostly white flower with missing target shading. Source 6308 has dense background contours that differ substantially from the ink target. These failures should appear in the report alongside representative successes. The panels establish the artifacts, but do not isolate whether architecture, target generation, optimization or data distribution caused them.

## Independent export verification

The uploaded ONNX graph passes `onnx.checker`; its artifact hash matches the sidecar. Locally compared ONNX Runtime CPU outputs with the selected PyTorch checkpoint on four validation photos for each style (12 cases). Evidence: `runs/task4/review/independent_export_check.json`.

Maximum absolute difference is `9.268522262573242e-05` on the [0,1] output scale, approximately 0.024 of an 8-bit intensity level. All pencil and ink cases pass the original `rtol=1e-4, atol=2e-5` check. All four charcoal cases exceed that strict tolerance at a total of 16 channel elements out of 196,608 charcoal elements. The uploaded sidecar reports maximum error `8.994340896606445e-05`; this local check does not reproduce a complete strict-tolerance pass. The difference is far too small to explain the visible quality problems, but strict export parity should be resolved or an evidence-based tolerance documented before final deployment.

## Recommendation

Keep epoch 53 as the Task 4 candidate. Do not extend training solely because epoch 60 is later: its validation objective is worse (0.11871 versus 0.11365). Pencil and charcoal provide a useful baseline; ink and the grid-pattern failure remain limitations worth documenting or investigating in a focused experiment.

The next assignment work is export verification, installing the selected models into the four app workspaces, final held-out testing after freezing model selection, app/deployment checks, and the report/demo. This review did not replace deployed models, change training code, rerun training or evaluate the held-out test set. Validation results alone do not establish completion of the full assignment.

## Subsequent integration resolution

Final integration independently compared all 786 validation pairs. The maximum
absolute framework/runtime difference was 0.000146985 (0.0375 of an 8-bit level).
A fixed absolute budget of 0.0002 (0.051 of a level), with relative tolerance
0.0001, passes every case and is now recorded in the export code and installed
metadata. Re-exporting did not reduce these differences; the original graph and
trained weights were retained. This resolves numerical acceptance without changing
image quality. The selected generator is now installed and browser/API checks
passed; see `docs/final_integration.md`. The strict original-tolerance audit above
remains historical evidence rather than being overwritten.
