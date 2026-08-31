# KiwiLM 2 Slim v3 residual-gate 250M comparison

This directory records the exact-250M confirmation of H6/S4 with upper-SwiGLU
residual gates initialized at 0.5. The candidate is not promoted because it
fails residual-health, repetition, and recorded-throughput requirements.

- [Analysis](analysis.md)
- [Machine-readable summary](summary.json)
- [100-batch health audit](health.json)
- [Generation report](generation/report.md)
- [Retrieval report](retrieval/report.md)

The primary checkpoints are:

```text
runs/kiwilm2-architecture/kiwilm2-adamw/latest.pt
runs/kiwilm2-muon-0.01/latest.pt
runs/kiwilm2-slim-v3-architecture2/kiwilm2-slim-v3-h6-s4-adamw/latest.pt
runs/kiwilm2-slim-v3-gate050-architecture/kiwilm2-slim-v3-h6-s4-gate-050-adamw/latest.pt
```

All comparison artifacts use repository-relative paths. The generation suite
is `eval/residual-gate-prompts.json`; retrieval uses the full 512-token context.
