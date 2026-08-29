# KiwiLM 2 Slim v3 smoke ablation

This is the four-way 50M-token comparison of Dense, all-Hadamard Slim v2, and
the two Slim v3 hybrid schedules. All primary measurements use each run's
`latest.pt`, because it is the checkpoint that contains exactly 50M training
tokens. The frozen promotion rule selects **H6/S4**: it clears the loss gate and
its user-identified idle-system regime remains more than 10% faster than Dense.

- [Analysis](analysis.md)
- [Machine-readable summary](summary.json)
- [Provenance validation](provenance.json)
- [Promotion decision and commands](selection.json)
- [Generation report](generation/report.md)
- [Full-context retrieval report](retrieval/report.md)

The fixed validation pass uses 50 batches of 8 sequences at 512 tokens, or
204,800 next-token targets per model. TinyStories and SimpleStories transfer
results are absent because those prepared evaluation datasets were not
available locally.

The H6/S4 whole-run throughput median is retained in `summary.json` but is not
used for selection: the training host was concurrently running a game during
the slow regime. A two-cluster analysis of the recorded throughput samples
identifies 94 idle-regime observations with a 39,120 tok/s median, effectively
equal to H7/S3's 39,178 tok/s fast-regime median.

First validate all four checkpoints against the smoke dataset and one another:

```bash
uv run python scripts/validate_kiwilm2_slim_v3_ablation.py \
  --data-dir data/smollm-smoke \
  --dense runs/kiwilm2-dense-smoke/latest.pt \
  --slim-v2 runs/kiwilm2-slim-gated-v2-smoke/latest.pt \
  --h7s3 runs/kiwilm2-slim-v3-smoke/kiwilm2-slim-v3-h7-s3-adamw/latest.pt \
  --h6s4 runs/kiwilm2-slim-v3-smoke/kiwilm2-slim-v3-h6-s4-adamw/latest.pt \
  --output examples/comparisons/kiwilm2-smoke-slim-v3-ablation/provenance.json
```

Generate the four-way prompt report:

```bash
uv run kiwilm compare \
  --data-dir data/smollm-smoke \
  --checkpoints \
    runs/kiwilm2-dense-smoke/latest.pt \
    runs/kiwilm2-slim-gated-v2-smoke/latest.pt \
    runs/kiwilm2-slim-v3-smoke/kiwilm2-slim-v3-h7-s3-adamw/latest.pt \
    runs/kiwilm2-slim-v3-smoke/kiwilm2-slim-v3-h6-s4-adamw/latest.pt \
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
  --checkpoint runs/kiwilm2-dense-smoke/latest.pt \
  --checkpoint runs/kiwilm2-slim-gated-v2-smoke/latest.pt \
  --checkpoint runs/kiwilm2-slim-v3-smoke/kiwilm2-slim-v3-h7-s3-adamw/latest.pt \
  --checkpoint runs/kiwilm2-slim-v3-smoke/kiwilm2-slim-v3-h6-s4-adamw/latest.pt \
  --label "KiwiLM 2 Dense" \
  --label "KiwiLM 2 Slim Gated v2" \
  --label "KiwiLM 2 Slim v3 H7/S3" \
  --label "KiwiLM 2 Slim v3 H6/S4" \
  --output-dir examples/comparisons/kiwilm2-smoke-slim-v3-ablation/retrieval
```

The generated `summary.json` combines the fixed validation, training-log
throughput/memory, static profiles, same-batch health diagnostics, generation,
and retrieval results. Reapply the frozen promotion rule with:

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
