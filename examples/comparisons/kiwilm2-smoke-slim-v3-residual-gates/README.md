# Slim v3 H6/S4 residual-gate smoke

This comparison is populated only if the preceding 250M audit authorizes the
experiment. The ungated H6/S4 control is reused after checkpoint-provenance
validation; only alpha=0.25 and alpha=0.5 train from scratch.

Generate the complete prompt suite (six prompts, seeds 42 through 46):

```bash
uv run --locked kiwilm compare \
  --data-dir data/smollm-smoke \
  --checkpoints CONTROL_LATEST GATE_025_LATEST GATE_050_LATEST \
  --labels control gate_025 gate_050 \
  --suite eval/residual-gate-prompts.json \
  --output-dir examples/comparisons/kiwilm2-smoke-slim-v3-residual-gates/generation \
  --device cuda
```

Build the aligned 200-batch validation and 100-batch health summary:

```bash
uv run --locked python scripts/evaluate_kiwilm2_residual_gate_smoke.py \
  --data-dir data/smollm-smoke \
  --control CONTROL_LATEST \
  --gate-025 GATE_025_LATEST \
  --gate-050 GATE_050_LATEST \
  --generation-summary examples/comparisons/kiwilm2-smoke-slim-v3-residual-gates/generation/summary.json \
  --output examples/comparisons/kiwilm2-smoke-slim-v3-residual-gates/summary.json \
  --device cuda --precision bf16
```

Apply every frozen promotion rule:

```bash
uv run --locked python scripts/select_kiwilm2_residual_gate.py \
  --summary examples/comparisons/kiwilm2-smoke-slim-v3-residual-gates/summary.json \
  --output examples/comparisons/kiwilm2-smoke-slim-v3-residual-gates/selection.json
```

Retrieval is supporting evidence and must use the full context window:

```bash
uv run --locked python scripts/evaluate_context_retrieval.py \
  --data-dir data/smollm-smoke \
  --checkpoints CONTROL_LATEST GATE_025_LATEST GATE_050_LATEST \
  --labels control gate_025 gate_050 \
  --context-length 512 \
  --output-dir examples/comparisons/kiwilm2-smoke-slim-v3-residual-gates/retrieval \
  --device cuda
```

An eligible candidate needs 95/100 health passes, family gradient ratios of at
least 0.1, block-9 p90 at most 1.5 and maximum at most 1.65, bounded learned
alphas, both cache parity checks, loss within 0.03 of control, at least 95% of
control throughput, no more than 2% extra peak memory, no 20-word repetition
run, and repeated-four-gram rate within 0.05 of control. Lowest loss wins; a
loss difference below 0.01 selects alpha=0.5. A winner advances only to a fresh
250M confirmation.
