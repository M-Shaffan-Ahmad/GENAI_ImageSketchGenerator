# Supplied Stitch export and UI revision

The user provided the original ZIP on 2026-10-04. HTML, screenshots and DESIGN.md
are preserved unchanged in the export directory. The ZIP SHA-256 and every
exported file hash are in provenance.json. The original generation date/prompt
were not supplied; docs/stitch_design_brief.md is a prepared brief, not a verified
record of the prompt used for this export.

The earlier React app predates the export. This revision implements its layout
and visual language while preserving the four real trained model workspaces.
Prototype-only A100, diffusion, 512/1024px and fidelity figures were replaced
with actual CPU ONNX inference, 128px outputs and measured latency.

Real app screenshots and browser checks are in experiments/integration_verification/browser/.
