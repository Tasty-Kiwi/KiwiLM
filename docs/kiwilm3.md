# KiwiLM 3: M3 bidirectional backbone

The Phase 3 encoder prototype is implemented. It is **untrained**, and is not
yet a pretrained diffusion language model or classifier. The
[Phase 4 denoising prototype](kiwilm3-denoising.md) now provides mask corruption,
protected-token policy, reconstruction loss and local CPU state recovery.
Iterative generation belongs to M5. No cloud training is started automatically.

## Confirmed baseline

Width 512, context 512, existing 32K vocabulary, eight equal-width attention
heads with RoPE, and width-2048 dense SwiGLUs in every block. Repeat
`Attention → BiConv31 → BiConv63`; the 16-block candidate ends with
`Attention → BiConv31 → BiConv63 → Attention` after four complete cycles.

Both mixers are bidirectional: attention has no causal mask, and depthwise
convolutions use symmetric same-length padding. Pointwise projections provide
the convolution gate. Each block uses ordinary pre-RMSNorm mixer and FFN
residuals. Residual output projections initialize with `0.02 / sqrt(2 * depth)`.
There are no n-gram tables, Hadamard MLPs, GQA, residual gates or AR caches.

| Default configuration | Parameters | Full-forward FLOPs/token at context 512 |
| --- | ---: | ---: |
| 12 blocks: 4 attention / 8 BiConv | 65,123,840 | 133,817,472 |
| 16 blocks: 6 attention / 10 BiConv | 81,430,016 | 168,516,736 |

Parameters count the tied embedding/head once. FLOPs count multiply-add as two
operations, include full bidirectional attention, the reconstruction head and
amortized noise-conditioning projections. They omit norms, RoPE, biases,
nonlinearities, dropout, lookups and backward. These are static estimates,
**not measured throughput or quality**. `profile_encoder` constructs on meta,
without allocating full weights.

## Separate encoder API

```python
import torch
from kiwilm.v3 import KiwiLM3Config, build_encoder

model = build_encoder(KiwiLM3Config(num_blocks=12)).eval()
input_ids = torch.tensor([[2, 100, 200, 3, 0]])  # example IDs, not a text tokenizer
valid = input_ids != 0
with torch.no_grad():
    hidden = model.encode(input_ids, attention_mask=valid, noise_level=0.5)
    logits = model(input_ids, attention_mask=valid, noise_level=0.5)
# hidden: [batch, sequence, 512]; logits: [batch, sequence, 32000]
```

The tied reconstruction head follows final RMSNorm; `encode()` exposes the
same contextual states without computing vocabulary logits. No target IDs are
passed into either API. The `BidirectionalEncoder` interface is separate from
V2's `CausalLanguageModel`, config dispatch, causal trainer and generation CLI.
Do not use V2 next-token losses, AR generation, perplexity or cache utilities
to interpret this encoder.

`attention_mask` is boolean `[batch, sequence]`, **True means valid**. If absent,
it is inferred from `pad_token_id=0`; configure `None` to disable inference.
An explicit mask is authoritative. Masked positions are isolated in attention
and convolution, and their final hidden states/logits are zero. Fully padded
rows stay finite. Sequences must fit the configured context.

`noise_level` is a finite scalar or floating `[batch]` tensor in `[0,1]`;
`None` means zero. Fixed sinusoidal features of `1000 * level` feed a learned
64→512→512 SiLU MLP and are added to token embeddings. This is conditioning
only: it does **not** corrupt tokens or specify a diffusion schedule. The
V2 tokenizer is unchanged. M4 creates a separate tokenizer with an appended real
`[MASK]` and requires an explicitly matching model vocabulary; see the
[denoising policy](kiwilm3-denoising.md).

## Local qualification

These commands work in Bash and PowerShell:

```bash
uv run --locked python scripts/check_kiwilm3_encoder.py --depth 12
uv run --locked python scripts/check_kiwilm3_encoder.py --depth 16
uv run --locked pytest -q tests/test_kiwilm3.py
```

The qualification helper uses the actual 12/16-block schedule at **width 16**
on CPU, without data downloads or optimizer steps. It checks finite nonzero
gradients, right-context influence, padding isolation, noise conditioning,
tied head and exact FP32 weight round-trip. It profiles width 512 separately.
The tests additionally cover symmetric convolution locality, invalid inputs,
all-padding rows, parameter-replacing dtype transfers, config JSON, native
weights and FP32/BF16 Safetensors. GPU/TPU compatibility and speed still require
real backend qualification before any longer run.

## Prototype weight artifacts

```python
from kiwilm.v3.weights import save_encoder_weights, load_encoder_weights

save_encoder_weights(model, "artifacts/v3-prototype.pt")  # default FP32
save_encoder_weights(model, "artifacts/v3-prototype.safetensors")  # default BF16
restored, config = load_encoder_weights("artifacts/v3-prototype.pt")
```

These are independent `kiwilm3-encoder-weights-v1` **inference-only** files,
not resumable checkpoints or complete Hub bundles. They embed config/storage
dtype, store the tied embedding once, reject nonfinite weights and refuse
overwrites. Supply `tokenizer_sha256` on save and the corresponding
`expected_tokenizer_sha256`, `expected_config` and `expected_sha256` on load
when available. Loading reconstructs FP32 parameters; BF16 storage is lossy.
Explicit `dtype="fp32"` enables exact Safetensors parity.

Export uses exclusive atomic hard-link publication on a local filesystem;
export to VM-local disk, not directly onto a Drive/FUSE mount. The format omits
tokenizer bytes, optimizer, schedule, data position and RNG. M4 has a separate
local training-state format including noise RNG; production verified
latest/previous Drive recovery remains a future qualification gate.

## Preserved V2 / notebook boundaries

The local `codex/kiwilm2-frozen` branch remains at
`2ad239b29f3a816795d07903ea160c72fee7074e`. V2 causal code, release artifacts,
historical evidence, checkpoints and browser playground remain unchanged.
The three existing notebooks are still M2 **V2 recovery probes**; this phase
does not repurpose them into diffusion training.

The M2 fresh-Colab/Drive continuation gate is still pending. Local M3 checks
are not that evidence. M2 locks package-code identity, so experiments made
before these new modules must resume with their original reviewed wheel
(available from the frozen branch), not with a changed package digest.
Do not relax resume locks to bypass that distinction.

## Local verification record — 2026-10-05

Both qualification commands pass. The complete locked environment with browser
and notebook extras passes **402 pytest tests** (58 new M3 cases), with two
existing ONNX exporter deprecation warnings. All seven playground runtime tests,
Ruff, `git diff --check`, offline lock validation and wheel build pass.
No training, Colab allocation, Git commit/push or release update was performed.
