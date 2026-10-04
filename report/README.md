# IEEE paper

Author: Muhammad Shaffan Ahmad, 23i-0673, Section A, FAST NUCES.

`main.tex` uses the official IEEEtran conference class and double-column layout.
`main.pdf` is the compiled paper. Figures and tables are derived from preserved
training/validation artifacts and the frozen final official-test evaluation.
The bibliography cites primary research sources; the AI-use appendix explains
assistance and independent checks. This is an academic assignment report, not a
claim of peer-reviewed publication or a novel architecture.

The supplied Stitch export is preserved in `stitch/` and illustrated in the paper;
the updated application screenshots show the implemented revision. The public
repository includes the selected ONNX model package and PyTorch checkpoints.
Before submission, record/upload the personal demonstration, replace the pending
video link in `submission_metadata.tex`, recompile and personally review the PDF.
Follow `docs/submission_steps.md` for the remaining account steps.

Compile with pdfLaTeX twice, or Tectonic. XeLaTeX mode uses bundled Nimbus fonts. With
Overleaf, import `submission/ieee_report_source.zip`, select `main.tex` and pdfLaTeX.

```bash
pdflatex main.tex
pdflatex main.tex
```

From the original project root, `venv/bin/python scripts/build_report_assets.py`
rebuilds generated tables and figures from evidence, without running training or
changing model selection. It expects original evidence under `report/evidence`
and final test outputs under `runs/final_test` in the original workspace.
