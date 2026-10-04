# Run B validation review

Run B completed 60 epochs. ZIP CRC passed. Verified checkpoint hash against the blur report, baseline training/validation manifest hashes and image counts, and a byte-identical fixed blur manifest. Only alpha changes from the original baseline, 0.6648628294821612 to 0.5; latent remains 8192. The baseline comparison isolates the configured loss-weight change; the Run A comparison also differs in latent size. Each configuration has only one training seed, so results do not establish repeatability.

| Fixed evaluation | Baseline PSNR | Run A PSNR | Run B PSNR | Run A SSIM | Run B SSIM | Run A / B both metrics better than input |
|---|---:|---:|---:|---:|---:|---:|
| none | 27.66 | 27.93 | 27.37 | 0.8773 | 0.8791 | 0 / 0 of 408 |
| low | 27.59 | 27.98 | 27.39 | 0.8745 | 0.8757 | 0 / 0 of 408 |
| medium | 26.35 | 26.68 | 26.28 | 0.8282 | 0.8292 | 60 / 38 of 408 |
| high | 24.27 | 24.45 | 24.25 | 0.7250 | 0.7268 | 225 / 165 of 408 |

Run B improves average SSIM over the baseline at every fixed blur level, but worsens average PSNR and L1 at every level and for clean controls. Compared with Run A, SSIM gains are small (about 0.0010 to 0.0018), while PSNR decreases by 0.589 dB for mild blur, 0.394 dB for medium and 0.194 dB for severe. Mild blur still worsens both metrics on all 408 photos. Severe blur improves mean SSIM by 0.00593 relative to input but loses 0.199 dB PSNR and increases L1 by 0.00321.

The original 408-case all-condition validation averages PSNR 25.17 dB and SSIM 0.8215 for Run B, compared with 25.58 dB and 0.8190 for Run A, and 25.43 dB and 0.8142 for the baseline. Run B has worse PSNR and L1 than both references in every corruption group. Occlusion SSIM is slightly below Run A.

Visual checks: mild-blur source 1522 still loses petal-edge detail; high-blur source 6313 remains visibly soft despite a slight positive SSIM delta; clean source 6651 retains the yellow/tan rectangular background artifact. The artifact exists without synthetic corruption. Its cause is not established by this experiment. These are selected examples, not a comprehensive visual ranking.

Conclusion: increased SSIM coefficient produces a metric tradeoff rather than a broad restoration improvement. Retain Run A as the stronger pixel-fidelity candidate; Run B should be documented as the loss-weight ablation, not silently substituted as an overall upgrade. None of these runs solves mild-blur preservation. Further work should inspect clean reconstruction and dark-background artifact failures before assuming another loss-weight change will fix them.

The archive reports ONNX parity on 16 validation inputs with maximum absolute error 4.17e-7; that is archive-reported evidence and was not rerun during this review. No training, test-set evaluation or app model replacement was performed.

Machine-readable comparison: `runs/task1/run_b/review/comparison.csv`.
