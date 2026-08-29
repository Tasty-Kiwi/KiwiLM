# KiwiLM 2 Slim v3 smoke ablation

This directory intentionally contains no model results yet. Populate it only
after both independently trained 50M-token Slim v3 checkpoints are available.
Dense and gated Slim v2 remain the frozen controls.

First validate all four checkpoints against the smoke dataset and one another:

```bash
uv run python scripts/validate_kiwilm2_slim_v3_ablation.py \
  --data-dir data/smollm-smoke \
  --dense runs/kiwilm2-dense-smoke/best.pt \
  --slim-v2 runs/kiwilm2-slim-gated-v2-smoke/best.pt \
  --h7s3 runs/kiwilm2-slim-v3-smoke/kiwilm2-slim-v3-h7-s3-adamw/best.pt \
  --h6s4 runs/kiwilm2-slim-v3-smoke/kiwilm2-slim-v3-h6-s4-adamw/best.pt \
  --output examples/comparisons/kiwilm2-smoke-slim-v3-ablation/provenance.json
```

Generate the four-way prompt report:

```bash
uv run kiwilm compare \
  --data-dir data/smollm-smoke \
  --checkpoints \
    runs/kiwilm2-dense-smoke/best.pt \
    runs/kiwilm2-slim-gated-v2-smoke/best.pt \
    runs/kiwilm2-slim-v3-smoke/kiwilm2-slim-v3-h7-s3-adamw/best.pt \
    runs/kiwilm2-slim-v3-smoke/kiwilm2-slim-v3-h6-s4-adamw/best.pt \
  --labels \
    "KiwiLM 2 Dense" \
    "KiwiLM 2 Slim Gated v2" \
    "KiwiLM 2 Slim v3 H7/S3" \
    "KiwiLM 2 Slim v3 H6/S4" \
  --output-dir examples/comparisons/kiwilm2-smoke-slim-v3-ablation/generation
```

Run the full 512-token retrieval suite:

```bash
uv run python scripts/evaluate_context_retrieval.py \
  --data-dir data/smollm-smoke \
  --context-length 512 \
  --checkpoint runs/kiwilm2-dense-smoke/best.pt \
  --checkpoint runs/kiwilm2-slim-gated-v2-smoke/best.pt \
  --checkpoint runs/kiwilm2-slim-v3-smoke/kiwilm2-slim-v3-h7-s3-adamw/best.pt \
  --checkpoint runs/kiwilm2-slim-v3-smoke/kiwilm2-slim-v3-h6-s4-adamw/best.pt \
  --label "KiwiLM 2 Dense" \
  --label "KiwiLM 2 Slim Gated v2" \
  --label "KiwiLM 2 Slim v3 H7/S3" \
  --label "KiwiLM 2 Slim v3 H6/S4" \
  --output-dir examples/comparisons/kiwilm2-smoke-slim-v3-ablation/retrieval
```

Evaluate all four checkpoints with the same fixed 200-batch validation command
and summarize validation, training-log throughput/memory, profiles, health,
generation, retrieval, and any available external transfer results in
`summary.json` using the existing comparison schema. Then apply the frozen
promotion rule:

```bash
uv run python scripts/select_kiwilm2_slim_v3.py \
  --summary examples/comparisons/kiwilm2-smoke-slim-v3-ablation/summary.json \
  --output examples/comparisons/kiwilm2-smoke-slim-v3-ablation/selection.json
```

When a candidate is selected, `selection.json` also records the exact Windows
PowerShell and Colab commands for its 250M architecture run. The selector never
starts that run itself.

H6/S4 advances only if it beats H7/S3 validation loss by at least 0.03 and
runs at least 1.10x as fast as Dense. Otherwise H7/S3 advances. Missing metrics
or failed health/parity checks intentionally leave the selection unset.
