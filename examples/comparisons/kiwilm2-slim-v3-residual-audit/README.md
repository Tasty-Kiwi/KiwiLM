# Slim v3 residual-growth audit

This directory is intentionally result-free until Dense Muon 0.01 has been
completed and analyzed. Run the audit against the existing exact-250M
Dense-AdamW and ungated H6/S4 `latest.pt` checkpoints:

```bash
uv run --locked python scripts/audit_kiwilm2_residual_growth.py \
  --data-dir data/smollm-architecture \
  --dense runs/kiwilm2-architecture/kiwilm2-adamw/latest.pt \
  --h6s4 runs/kiwilm2-slim-v3-architecture2/latest.pt \
  --output examples/comparisons/kiwilm2-slim-v3-residual-audit/audit.json \
  --seeds 141 142 --batches-per-seed 50 --batch-size 2 \
  --context-length 512 --device cuda --precision bf16
```

`gated_smoke_authorized` becomes true only with the exact 100-batch controls,
block-9 residual-amplification p90 above 1.5, and at least 10 failures above
1.5. If it is false, do not launch gated training.
