# Final model integration

The four reviewed trained candidates are installed at the paths used by both the
local API and Docker Compose. No new training ran, and no model was selected using
the official test set during integration.

| Workspace | Selected model | Installed path |
|---|---|---|
| Universal Restoration | Task 1 Run A, best epoch 60 | `models/universal.onnx` |
| Hard-Routed Restoration | Task 2 classifier and three independently trained specialists | `models/task2/` |
| Soft Mixture-of-Experts | Task 3 joint checkpoint, epoch 21 | `models/task3/soft_moe.onnx` |
| Object-to-Sketch | Task 4 generator, epoch 53 | `models/task4/sketch_generator.onnx` |

All seven ONNX graphs are non-smoke artifacts. Their model metadata and the release
manifest record provenance, artifact hashes, selected configurations and parity
evidence. The previous deployed Task 1 artifact is preserved under the backup
directory recorded in `models/release.json`. Original uploaded ZIP archives remain
untouched.

## Run the app locally

From the repository root:

```bash
bash run_app.sh
```

Open **http://localhost:8080**. All four workspaces use the installed ONNX models.
API docs: **http://localhost:8080/api/docs**. Readiness:
**http://localhost:8080/api/health**; all four readiness fields should be true.
Stop with Ctrl+C. This launcher serves the built React application and mounts the
existing API under `/api`; Node and Docker are not required at runtime. It requires
the existing Python virtual environment, models, manifests and sample images.

If port 8080 is occupied, choose a free port:

```bash
PORT=18080 bash run_app.sh
```

Then use **http://localhost:18080**. Verification used port 18080 because 8080 was
already occupied on this machine.

For a fresh checkout, create the Python environment as described in the README,
prepare the data, install the selected models, and build the frontend:

```bash
venv/bin/python integrate_models.py
cd frontend
npm ci
npm run build
```

Return to the repository root before running `run_app.sh`. Reinstallation requires
the four result archives currently under `colab/`; the script verifies every selected
member's CRC, hashes, data fingerprints and runtime parity before replacing models.
It does not extract entire result archives. The built frontend is present in this
workspace, but `frontend/dist` is ignored by Git.

## Verification completed

- Production Vite build passed with Node 22.14.0.
- All 23 existing Python tests passed against the installed models.
- ONNX graph checks passed for all seven graphs.
- Independent parity checks passed: Task 1 16 validation inputs; Task 2 classifier
  16 and each specialist 16; Task 3 image and weight outputs on 16; Task 4 every
  validation pair (786, 262 per style).
- Actual HTTP requests passed for all restoration conditions across Tasks 1–3,
  all three sketch styles, and uploads to every workspace: 19 successful inference
  requests. Five invalid requests returned the expected 404/422 responses.
- HTTP checks verified RGB 128×128 PNG outputs, trained-model status, normalized
  routing probabilities, exact hard clean bypass when selected, and reference/error
  panel rules for samples versus uploads.
- Headless Chrome exercised all four workspaces through the production build, all
  sketch selector choices, routing displays, and PNG download links. Generated
  sketch outputs differed across styles. No browser exceptions were recorded.
- Desktop (1365px) and mobile (390px) screenshots were captured for each workspace;
  horizontal overflow checks passed. Soft-MoE desktop and sketch mobile screenshots
  were also visually inspected. This is browser evidence, not a physical-phone test.

Evidence:

- `models/release.json`: installed files, source archive hashes, backup path and
  per-component parity checks. Also copied to `runs/integration/release.json`.
- `runs/integration/http_checks.json`: actual API verification.
- `runs/integration/browser/results.json`: production UI checks.
- `runs/integration/browser/workspace_*_{desktop,mobile}.png`: eight screenshots.
- `runs/task4/review/full_validation_export_check.json`: original tolerance audit
  on all 786 pairs.

Repeat verification with Chrome and Node 22+ available:

```bash
venv/bin/python scripts/verify_integration.py
```

`NODE_BINARY` can specify a Node executable. The script starts its own local web
server and separate headless Chrome profile and stops both afterward. Ports 18080
and 9227 must be free; `VERIFY_WEB_PORT` and `VERIFY_CHROME_PORT` override them.

## Task 4 numerical export resolution

Re-exporting with the legacy exporter produced the same small CPU differences, so
the uploaded graph was retained. Across all 786 validation pairs, maximum absolute
PyTorch/ONNX Runtime difference was **0.000146985** on [0,1], about **0.0375 of an
8-bit intensity level**. Maximum per-image mean absolute difference was 0.000003101.
These differences cannot explain the visible grid artifacts or ink-detail errors.

The export now uses a documented absolute tolerance and independent maximum-error
budget of **0.0002** (0.051 of an 8-bit level), with relative tolerance 0.0001.
All 786 cases pass that fixed budget. This changes numerical verification only;
the trained weights and ONNX graph were not altered. Future exports must still
pass the absolute budget rather than automatically relaxing it after a failure.

## Docker and remaining submission work

Docker Compose already points to the installed model paths. Both Docker build
contexts exclude dependencies and generated archives that need not be copied.
Docker Desktop was unavailable during the initial integration review. Subsequent
submission work verified both image builds, startup, and all 15 model inference
combinations through the frontend proxy. Start Docker Desktop, then run:

```bash
docker compose up --build
```

Open **http://localhost:8080**; API docs are **http://localhost:8000/docs** for this
two-container mode. Verification used `FRONTEND_PORT=18081 BACKEND_PORT=18082`
because 8080 was occupied by another service. The optional data setup container
also built and reproduced all expected model provenance fingerprints.

Final held-out evaluation of the frozen selected models is complete: 61,490
restoration cases and 1,068 sketch pairs. The IEEE report with failure examples
and AI-use appendix is prepared in `report/main.tex` and compiled to `report/main.pdf`.
The supplied Stitch export is preserved in `report/stitch/`; the app was revised
to match it and verified through production desktop/mobile browser checks, including
actual file upload and switching back to samples. Source, ONNX models and selected
PyTorch checkpoints are published in the public GitHub repository. The author supplied the uploaded demonstration URL, which is now included in
the paper. Final author review, signed-out playback check and Classroom submission remain. See
`docs/submission_steps.md`.
Historical test results for the older vector
baseline must not be presented as measurements of these selected models. Existing
restoration and sketch quality limitations remain unchanged by integration.
