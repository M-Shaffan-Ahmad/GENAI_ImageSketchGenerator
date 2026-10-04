# Submission handoff for Muhammad Shaffan Ahmad (23i-0673)

Course: CS4065 / Generative AI, Section A, FAST NUCES.
Repository destination: https://github.com/M-Shaffan-Ahmad/GENAI_ImageSketchGenerator

The code, four trained tasks, final evaluations, IEEE LaTeX paper, model packages,
experiment records and Docker implementation are prepared. The remaining items
require your authenticated accounts and a personal demonstration. Do not describe
the submission as fully complete until those items are done and their links work.

## Files to use

- `report/main.pdf`: compiled IEEE double-column technical paper. It includes your
  name, ID, all four methodologies, measured final tests, failure analysis, figures,
  primary research references and AI-use appendix.
- `submission/ieee_report_source.zip`: upload this to Overleaf to edit/compile the
  paper. It contains the LaTeX source, IEEE class, generated tables and figures.
- `submission/github_source.zip`: complete repository-ready source, notebooks,
  small experiment evidence, final evaluation CSV/JSON and instructions. It excludes
  datasets, trained weights, credentials, local environment folders and result ZIPs.
- `submission/trained_models.zip`: the selected seven ONNX graphs, metadata and
  release manifest; extract into the repository root, producing `models/`.
- `submission/training_checkpoints.zip`: selected PyTorch checkpoints, separate
  from source. It produces `runs/integration/staging/checkpoints/`, matching the
  final-evaluation scripts. Task 3 initializes from the included Task 2 models.
- `submission/SHA256SUMS.txt`: integrity hashes for those ZIP packages.
- `docs/demo_script.md`: a six-minute recording outline and narration.
- `docs/stitch_design_brief.md`: copyable brief for a real Stitch prototype.

The PDF still clearly marks missing model-download, demonstration and Stitch
evidence. Replace these honest pending entries after completing the account steps.

## 1. Publish source in your GitHub repository

GitHub access was not authenticated in this session; inspecting the supplied remote
with noninteractive Git failed. No repository push or GitHub release occurred.
The GitHub integration can publish after it is installed/connected, or you can use
Git locally. Do not paste passwords or access tokens into this chat.

If the destination repository does not exist yet, create
`GENAI_ImageSketchGenerator` under `M-Shaffan-Ahmad` in GitHub. If it already exists,
use that repository; do not create another or force-push over its history.

Clone it into a **new directory** outside this current project, then extract the
contents of `github_source.zip` into that clone. The archive contains files at its
root, so they should sit beside the clone's `.git`, not inside another wrapper folder.
Review changes before committing. The original project contains private environment
folders and huge data/result files; the prepared archive avoids copying them.

```bash
git clone https://github.com/M-Shaffan-Ahmad/GENAI_ImageSketchGenerator.git
cd GENAI_ImageSketchGenerator
# Extract github_source.zip here using your archive manager or unzip.
git status --short
git add .
git commit -m "Complete four-task restoration and sketch assignment"
git push origin HEAD
```

Authenticate through GitHub's normal browser/credential-manager flow. If the remote
has existing work, review the resulting diff and resolve conflicts; avoid `--force`.
The archive deliberately contains `.gitignore` exceptions for the small original
Optuna databases so the required studies are committed.

## 2. Publish the trained model download links

In the GitHub repository, open **Releases → Draft a new release**. Create tag
`submission-v1` and attach `trained_models.zip`, `training_checkpoints.zip`, and
`SHA256SUMS.txt`. The ONNX package is sufficient for inference. Checkpoints permit
repeating the final evaluation and inspecting trained PyTorch models.

Describe the release as the frozen selected models, not new training. Publish it,
copy the **actual** direct asset URL, and test downloading it from a signed-out
browser. Do not just link a private repository page the evaluator cannot access.

Update `report/submission_metadata.tex`:

```tex
\newcommand{\ModelDownloadLink}{\url{PASTE_ACTUAL_MODEL_ASSET_URL}}
```

Also place the model URL in `README.md`, then commit the change. The expected URL
shape is `https://github.com/M-Shaffan-Ahmad/GENAI_ImageSketchGenerator/releases/download/submission-v1/trained_models.zip`,
but it is **not a working download link until the release asset is published**.

## 3. Complete genuine Google Stitch evidence

Open https://stitch.withgoogle.com/ using your Google account. Create a project,
paste `docs/stitch_design_brief.md`, and generate the four workspaces. Save/export
the actual generated design and capture screens showing the project and designs.
Put the screenshots and exported design into `report/stitch/` with a short record
of the date, prompt and any changes you made.

Compare the prototype with the implemented UI. Apply and verify any design changes
you decide to use, then capture the resulting app screenshots. The React app was
implemented before this Stitch prototype; state that chronology honestly. Generating
a later prototype does not retroactively establish that the app was originally
designed in Stitch. If your instructor enforces the original-design sequence,
clarify whether a documented prototype/revision is acceptable.

Replace the pending Stitch text in `report/submission_metadata.tex` with a short
accurate status. Add an actual figure to `report/main.tex`, for example:

```tex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{stitch/universal.png}
\caption{Actual Google Stitch prototype generated on DATE;
the earlier React implementation was subsequently compared with this design.}
\end{figure}
```

Use a real saved screenshot; do not rename an app screenshot as Stitch provenance.
Commit the evidence and revised report source.

## 4. Record the 5–7 minute demonstration

Use OBS Studio, your operating system's screen recorder, or another recorder you
already know. Record the actual terminal and browser with your narration, following
`docs/demo_script.md`. Aim for six minutes. Include app startup, an image upload,
corruption controls, every task, routing weights, all three styles, PNG downloads,
and MLflow/Optuna evidence. Explain at least one limitation rather than claiming
perfect restoration. Practice first; show actual inference rather than static
screenshots pretending to be live operation.

Open the app for recording:

```bash
# This current machine has 8080 occupied; these free ports were verified.
FRONTEND_PORT=18081 BACKEND_PORT=18082 docker compose up --build
```

Browser: http://localhost:18081. On a clean evaluator machine the standard command
`docker compose up --build` uses frontend 8080 and backend 8000.

In another terminal, show the portable MLflow evidence:

```bash
venv/bin/python scripts/relocate_tracking.py
venv/bin/mlflow ui --backend-store-uri sqlite:///experiments/tracking.db --host 127.0.0.1 --port 5000
```

Open http://localhost:5000. Training histories are tagged as retrospective imports
of the original Colab records; final test evidence is separately identified. This
viewer does not pretend the import was another training run. Original Optuna
SQLite databases remain in `experiments/optuna/`.

Upload your recording through YouTube Studio and select **Unlisted**, rather than
Private. YouTube's official instructions allow anyone with the unlisted link to
view the video: https://support.google.com/youtube/answer/157177?hl=en.
Test the link signed out. Put it in the paper:

```tex
\newcommand{\DemoVideoLink}{\url{PASTE_ACTUAL_UNLISTED_YOUTUBE_URL}}
```

The recording/upload was not performed here because your authenticated YouTube
account and personal narration were unavailable. Submit the link, not the large
video file, as required by the assignment.

## 5. Compile the final paper and review it

For the easiest route, upload `ieee_report_source.zip` to Overleaf, choose `main.tex`
as the main document, and use **pdfLaTeX**. Make the metadata/Stitch changes there,
compile and download the final PDF. The source also supports XeLaTeX; that mode
uses the Nimbus fonts bundled in the report ZIP with their license.

With a local TeX installation:

```bash
cd report
pdflatex main.tex
pdflatex main.tex
```

Two passes resolve references. Tectonic is another supported compiler. The compiled
PDF already includes final results; do not replace test tables with validation
results or change frozen checkpoints after reporting them.

Read the paper personally. Verify your name/ID and all published URLs, check every
figure/table reference, and understand the losses, identity bypass, soft weights,
synthetic targets and stated limitations. Update the availability paragraph so it
accurately says what has been published; a destination URL alone is not publication.

## 6. Check the evaluator's fresh-clone path

On a new clone, extract `trained_models.zip` into the repository root, then run:

```bash
docker compose --profile setup run --rm --build prepare-data
docker compose up --build
```

The setup container downloads official Flowers102 images if absent and creates
the manifests and sketch pairs; it checks the fingerprints against installed-model
metadata. No local Python installation is needed for these two commands. The app
opens at http://localhost:8080, with API documentation at http://localhost:8000/docs.
The raw dataset and generated targets are intentionally excluded from GitHub.

Verify http://localhost:8080/api/health reports all four readiness flags true,
upload an image, run all four tasks and download outputs. Linux users may use
`docker compose --profile setup run --rm --user "$(id -u):$(id -g)" --build prepare-data`
after creating `raw_data/` and `data/` to keep generated files owned by their user.

## 7. Submit through Google Classroom

Submit the revised final IEEE PDF containing the working repository, model and
YouTube links, with actual Stitch evidence and AI-use appendix. Include any extra
files the instructor explicitly requests. Check the actual Classroom deadline;
the adapted assignment PDF contains a historical date and is not a live deadline.
No Classroom submission was made here because your account and class submission
interface were not available. This is the final user-controlled step.
