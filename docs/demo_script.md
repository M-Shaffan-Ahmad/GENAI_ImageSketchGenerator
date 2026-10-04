# Six-minute demonstration outline

Presenter: Muhammad Shaffan Ahmad, 23i-0673, Section A, FAST NUCES.
Use a flower/object image on disk and the built-in samples. Record the real app and
your terminal, with readable browser zoom and your own explanation. The times are
targets, not instructions to fake or speed through unsuccessful actions.

| Time | Show | Suggested explanation |
|---|---|---|
| 0:00–0:40 | Name/ID; repository files; `docker compose up --build`; running containers | “This project implements universal restoration, hard routing, soft expert blending and three conditioned sketch styles. Inference runs through ONNX in a FastAPI backend, with a React frontend.” |
| 0:40–1:30 | Universal tab; upload a photo; select salt-and-pepper; change probability; restore and download | “The first task uses one compressed spatial autoencoder across all conditions. The clean photo is the reference because I simulated this noise. An already-corrupted upload would not supply ground truth.” |
| 1:30–2:20 | Hard routing; same sample/noise; class probabilities; clean input and identity bypass | “The classifier predicts a route from the actual image. Three specialists are trained independently. A clean prediction passes through unchanged. Wrong routing can cause additional error.” |
| 2:20–3:15 | Soft MoE; medium/high blur; four bars; input/output/reference | “The model weights identity and three experts and outputs their convex combination. Warm-up trains the gate first, then joint fine-tuning updates the experts. Stronger blur shifts weight toward the blur branch. Mild blur still loses detail.” |
| 3:15–4:15 | Sketch tab; same photo in pencil, ink, charcoal; download PNG | “A categorical embedding conditions a U-Net generator and its training discriminator. The targets are synthetic algorithms, not human drawings. Ink is the weakest style by test SSIM; the three outputs remain distinct.” |
| 4:15–5:15 | MLflow history curves, parameters and artifacts; Optuna study summaries; test tables/confusion/gate heatmap | “Search and selection used validation. The frozen comparison has 61,490 restoration views and 1,068 sketch pairs. The classifier has about 97.36% test accuracy and .959 macro F1. The viewer imports original Colab evidence and labels it as such.” |
| 5:15–6:00 | One failure panel; IEEE report; actual Stitch evidence if completed; repository/model links | “Soft blending improves all corrupted severity groups over hard routing on average, but mild blur often remains worse than input. Occluded detail is approximated, not perfectly recovered. The report includes failures and AI assistance with its verification.” |

Before recording:

- Start MLflow separately using the commands in `docs/submission_steps.md`.
- Have `report/main.pdf`, `docs/submission_steps.md` and your actual Stitch project
  available. Only show Stitch evidence if you have genuinely completed it.
- Use the same image for comparisons where possible; allow every inference to finish.
- Test the download action by opening one saved PNG.
- Keep desktop notifications and credentials out of the recording.
- If startup is cached, say the images were previously built; do not pretend a
  cached build measures first-run installation time.

Upload the finished recording to YouTube as Unlisted. Test the link signed out,
place it in `report/submission_metadata.tex`, and recompile the paper. No demo video
has been recorded or uploaded by this preparation script.
