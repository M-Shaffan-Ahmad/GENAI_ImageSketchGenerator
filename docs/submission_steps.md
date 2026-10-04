# Submission handoff for Muhammad Shaffan Ahmad (23i-0673)

CS4065 / Generative AI, Section A, FAST NUCES.
Public repository: https://github.com/M-Shaffan-Ahmad/GENAI_ImageSketchGenerator

The four trained tasks, integrated Stitch-style app, frozen test evaluation,
experiment evidence, Docker setup, IEEE paper and public source/model publication
are complete. The personal video and Classroom submission remain.

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

## 3. Record and upload the 5–7 minute demonstration

This remains yours to complete: a personal recording/narration and authenticated
YouTube upload were not available in this session. Follow `docs/demo_script.md`;
aim for six minutes and show the real terminal and browser using OBS or your
usual screen recorder.

Include startup, image upload, corruption controls, all four tasks, hard routing,
soft weights, three sketch styles, PNG download, MLflow/Optuna evidence and an
honest limitation. Show actual inference and explain the 128px model resolution.

Start the app:

```bash
# Current machine: 8080 is occupied. These alternate ports were verified.
FRONTEND_PORT=18081 BACKEND_PORT=18082 docker compose up --build
```

Open http://localhost:18081. On a clean evaluator machine, defaults are 8080/8000.
Show tracking in another terminal (research environment required):

```bash
venv/bin/python scripts/relocate_tracking.py
venv/bin/mlflow ui --backend-store-uri sqlite:///experiments/tracking.db --host 127.0.0.1 --port 5000
```

Open http://localhost:5000. Training histories are labeled retrospective imports
of original Colab records; the final test record is identified separately. Original
Optuna databases are preserved in `experiments/optuna/`.

Upload through YouTube Studio and select **Unlisted**. Check the link signed out.
Replace only the video entry in `report/submission_metadata.tex`:

```tex
\newcommand{\DemoVideoLink}{\url{YOUR_ACTUAL_UNLISTED_YOUTUBE_URL}}
```

## 4. Recompile and personally review the paper

After adding your video URL, upload `ieee_report_source.zip` to Overleaf, choose
`main.tex` and pdfLaTeX, compile and download the final PDF. Locally:

```bash
cd report
pdflatex main.tex
pdflatex main.tex
```

The source also works with Tectonic. Review your name/ID, working URLs, figures,
losses, routing, synthetic sketch targets, measured failures and AI-use appendix.
Update the sentence about the pending personal demonstration once it is uploaded.
Commit the updated LaTeX and PDF back to GitHub. The currently compiled PDF is
ready for review but honestly marks the video as pending.

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
