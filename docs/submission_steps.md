# Submission handoff for Muhammad Shaffan Ahmad (23i-0673)

CS4065 / Generative AI, Section A, FAST NUCES.
Public repository: https://github.com/M-Shaffan-Ahmad/GENAI_ImageSketchGenerator

The four trained tasks, integrated Stitch-style app, frozen test evaluation,
experiment evidence, Docker setup, IEEE paper and public source/model publication
are complete. The author has supplied the uploaded video URL, which has been
added to the paper. Final review and Classroom submission remain.

## Ready files

- `report/main.pdf`: IEEE double-column paper, author details, all four tasks,
  actual test results, limitations, supplied Stitch design, real app screenshots,
  research references and AI-use appendix.
- `submission/ieee_report_source.zip`: import into Overleaf; select `main.tex`
  and pdfLaTeX. XeLaTeX is also supported with bundled fonts.
- `submission/github_source.zip`: source, notebooks, docs, experiment evidence,
  report and original Stitch export. It excludes dataset and weight binaries.
- `submission/trained_models.zip`: selected seven ONNX graphs and metadata.
- `submission/training_checkpoints.zip`: seven selected PyTorch checkpoints.
- `submission/SHA256SUMS.txt`: ZIP integrity checks.
- `docs/demo_script.md`: six-minute demonstration outline and narration.

## 1. GitHub and model publication — completed

Source is on `main` in the public repository. The model ZIP is committed at
`artifacts/trained_models.zip`; selected PyTorch checkpoints are in
`artifacts/checkpoints/`. Public ONNX download:

https://raw.githubusercontent.com/M-Shaffan-Ahmad/GENAI_ImageSketchGenerator/main/artifacts/trained_models.zip

The package is sufficient for app inference. Checkpoints permit repeating final
evaluation. `artifacts/SHA256SUMS.txt` records their hashes. These are the frozen
models already evaluated, not new training. No GitHub Release is necessary to
use these public artifacts; release publishing remains optional.

## 2. Stitch design and implementation — completed

The user supplied `stitch_ai_image_restoration_sketching_studio.zip`. Original
HTML, screenshots and design notes are preserved in `report/stitch/`, together
with file hashes and an accurate provenance record. The earlier React app existed
before this export; it was subsequently revised to follow the exported design.
The original generation date/prompt were not supplied. The prepared brief is not
represented as the export's original prompt.

The working UI uses the export's lab header, emerald/pale-blue palette, four
workspaces, control/result cards and mobile layout. It displays actual CPU ONNX
inference and 128px outputs; the prototype's example GPU, diffusion and fidelity
numbers are not real model measurements. Desktop/mobile checks, actual browser
upload, sample switching, all four workspaces and three sketch styles passed.
The paper distinguishes the original supplied design from working app screenshots.

## 3. Demonstration link — supplied and added to the paper

https://www.youtube.com/live/1NpOTGTbwuc

The author supplied this uploaded demonstration link. It is included in
`report/submission_metadata.tex` and the recompiled PDF. Automated web access
could not fetch YouTube, so playback, visibility, duration and content were not
independently verified here. Before submission, open the link in an incognito
window while signed out. Confirm it plays completely, meets the assignment's
5–7 minute duration, and shows the required tasks. Set visibility to Unlisted
or Public so the evaluator can view it; Private requires separate viewer access.

## 4. Final paper — recompiled with the video link

Use the latest `report/main.pdf`. It includes repository, model and video links.
Read the PDF personally, verify your identity and all links, and check figures,
methods, losses, measured results, limitations and the AI-use appendix.

The refreshed `submission/ieee_report_source.zip` can be imported into Overleaf
with `main.tex` and pdfLaTeX if you need another edit. No recompilation is needed
unless you change the report or URLs.

## 5. Evaluator's fresh-clone path

```bash
git clone https://github.com/M-Shaffan-Ahmad/GENAI_ImageSketchGenerator.git
cd GENAI_ImageSketchGenerator
sha256sum -c artifacts/SHA256SUMS.txt
unzip artifacts/trained_models.zip
docker compose --profile setup run --rm --build prepare-data
docker compose up --build
```

An archive manager can replace `unzip`. The setup container downloads official
Flowers102 data and prepares manifests/sketch targets, checking model fingerprints.
No local Python installation is required for this Docker route. Open
http://localhost:8080; API documentation is at http://localhost:8000/docs.

Confirm `/api/health` reports all four models ready; upload, run the four tasks
and download results. Dataset files are not committed. Linux users can run the
setup service with `--user "$(id -u):$(id -g)"` after creating `raw_data/` and `data/`
to keep generated files owned by their user.

To repeat final evaluation, copy `artifacts/checkpoints/` into
`runs/integration/staging/checkpoints/` and use the commands in the root README.
Do not retune frozen checkpoints on the test data.

## 6. Submit through Google Classroom

Submit the final IEEE PDF with the public repository/model links, your working
unlisted video link, Stitch evidence and AI-use appendix. Include any extra files
requested by the instructor. Check the actual Classroom deadline; the adapted
assignment PDF contains a historical date. Classroom access was unavailable here,
so this final submission must be performed from your account.
