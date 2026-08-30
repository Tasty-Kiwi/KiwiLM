# Slim v3 residual-growth audit

The exact 100-batch audit is complete. It compares the existing exact-250M
Dense-AdamW and ungated H6/S4 `latest.pt` checkpoints under matched controls:

```bash
uv run --locked python scripts/audit_kiwilm2_residual_growth.py \
  --data-dir data/smollm-architecture \
  --smoke-data-dir data/smollm-smoke \
  --dense runs/kiwilm2-architecture/kiwilm2-adamw/latest.pt \
  --h6s4 runs/kiwilm2-slim-v3-architecture2/kiwilm2-slim-v3-h6-s4-adamw/latest.pt \
  --output examples/comparisons/kiwilm2-slim-v3-residual-audit/audit.json \
  --seeds 141 142 --batches-per-seed 50 --batch-size 2 \
  --context-length 512 --device cpu --precision fp32
```

The frozen controls match. H6/S4 block 9 has median amplification 1.526,
p90 1.598, maximum 1.654, and 63/100 batches above 1.5. The result therefore
sets both `residual_growth_reproduced` and `gated_smoke_authorized` to true.
All activations and gradients remain finite and both cached-generation parity
checks pass, so this is a bounded residual-growth problem rather than a
numerical failure.

Dense is a useful calibration: its block-9 p90 is 1.514 with 14/100 failures.
The 1.5 threshold is somewhat strict for this architecture, but H6/S4 exceeds
it much more consistently and still satisfies the predeclared intervention
rule. Full distributions and provenance are in [audit.json](audit.json).
