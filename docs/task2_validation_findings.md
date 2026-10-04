# Task 2 trained-result review

Archive: `colab/task2_results.zip`. ZIP CRC passed. All four best-checkpoint hashes match the reports; their configurations and training/validation fingerprints match local data. Final classifier training completed 20 epochs; each specialist completed 60. Best selected epochs: classifier 18, salt 55, blur 60, occlusion 58. These earlier selected epochs are normal validation checkpoint selection, not incomplete runs. No smoke checkpoints were used.

## Original fixed validation: 102 cases per condition

| Condition | Run A PSNR | Task 2 oracle PSNR | Task 2 predicted PSNR | Run A SSIM | Task 2 predicted SSIM |
|---|---:|---:|---:|---:|---:|
| clean/sampled | 27.96 | 120.00 | 119.08 | 0.8769 | 0.9988 |
| salt_and_pepper/sampled | 26.63 | 27.66 | 27.66 | 0.8310 | 0.8702 |
| gaussian_blur/sampled | 26.72 | 27.94 | 28.13 | 0.8313 | 0.8740 |
| rectangular_occlusion/sampled | 21.00 | 21.74 | 21.74 | 0.7367 | 0.7686 |

Classifier accuracy is 99.02%, macro F1 0.9902: one clean image routed to occlusion, three blurred images routed to clean. Noise and occlusion routing are correct for all original-manifest cases. The three blurred-image bypasses improve metrics relative to the blur specialist, showing that a technically incorrect route can preserve a mild input better.

Task 2 improves condition-average PSNR and SSIM over Run A on all three corrupted groups. It is not an across-the-board quantitative regression, but this does not contradict a poor appearance on individual images. Clean identity inflates the overall PSNR (49.15 dB); that number should not be compared directly with a model that reconstructs every clean input.

## Severity sweep: 408 cases per condition/severity

| Blur | Input PSNR | Oracle PSNR | Predicted PSNR | Input SSIM | Predicted SSIM | Predicted PSNR better than input |
|---|---:|---:|---:|---:|---:|---:|
| low | 33.63 | 28.54 | 29.53 | 0.9631 | 0.9069 | 0/408 |
| medium | 27.33 | 28.39 | 28.39 | 0.8457 | 0.8823 | 349/408 |
| high | 24.45 | 25.49 | 25.49 | 0.7209 | 0.7717 | 395/408 |

Mild blur remains a preservation failure: none of 408 outputs improves PSNR over input. Medium blur now improves PSNR on 349/408 cases, severe on 395/408. Overall severity classifier accuracy is 97.52%, macro F1 0.9615; clean recall is 398/408, and mild-blur routing accuracy is only 79.66%. These results include natural clean/blur ambiguity and should not be summarized only by aggregate accuracy.

## Visual checks and interpretation

Reviewed source 7013 at low/high blur, high-occlusion restoration failure source 3851 and clean misrouting source 5124. Source 7013 at low blur shows more distinct edges but altered texture; at high blur it remains soft. Source 3851 fills masked areas with smooth approximations rather than the original flower detail and also changes visible regions. Source 5124 is a true clean image misrouted to occlusion, damaging an already sharp flower.

The strongest remaining issue is specialist reconstruction/preservation, not widespread classifier failure. Oracle results already show mild-blur damage. Missing masked content cannot be uniquely recovered from a corrupted image, so plausible reconstruction and faithful ground-truth recovery are different expectations. The observed softness alone does not establish whether bottleneck size, decoder, loss, or training budget is the root cause.

Keep this run as the Task 2 baseline and report the improvements and failures honestly. Before another broad training run, focus on mild-blur preservation and visible-region preservation for occlusion; leave classifier retuning lower priority. Task 3 can learn identity/expert blending, but it should not be assumed to fix weak specialist reconstructions automatically. No model replacement, new training, or test-set evaluation was performed during this review.

Saved detailed evidence under `runs/task2/review/`.
