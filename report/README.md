# IEEE paper

Author: Muhammad Shaffan Ahmad, 23i-0673, Section A, FAST NUCES.

`main.tex` uses the official IEEEtran conference class and double-column layout.
`main.pdf` is the compiled paper. Figures and tables are derived from preserved
training/validation artifacts and the frozen final official-test evaluation.
The bibliography cites primary research sources; the AI-use appendix explains
assistance and independent checks. This is an academic assignment report, not a
claim of peer-reviewed publication or a novel architecture.

Before submission, complete the real account steps in `docs/submission_steps.md`,
replace pending links/status in `submission_metadata.tex`, add genuine Stitch
evidence, and review the availability paragraph for accurate publication status.
The current report discloses those gaps explicitly.

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
