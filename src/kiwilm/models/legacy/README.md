# Legacy and comparison models

This package contains KiwiLM's earlier architecture experiments. They remain
registered, tested, trainable, and checkpoint-compatible, but Models X and Y
are the active scaling candidates documented in the project README.

| Model | CLI architecture | Structure | Parameters |
| --- | --- | --- | ---: |
| A | `gated_cnn` | 8 dilated gated CNN blocks | 5,259,776 |
| B | `cnn_attention` | 3 CNNs, attention/GELU FFN, 3 CNNs | 5,261,056 |
| C | `cnn_dual_attention` | Model B plus a final attention/GELU FFN | 6,050,816 |
| D | `cnn_attention_mamba` | Model B plus a portable Mamba block | 6,027,648 |
| E | `cnn_interleaved_attention` | 2 CNNs, attention, 2 CNNs, attention, 2 CNNs | 6,050,816 |
| F | `cnn_deep_interleaved_attention` | Model E plus 3 refinement CNNs and final attention | 8,023,296 |
| G | `cnn_attention_ffn` | Model B plus a residual GELU FFN after every CNN | 8,417,536 |
| GPT | `transformer` | 4 pre-LayerNorm RoPE attention/GELU FFN blocks | 5,264,896 |

All parameter counts assume the default 8,192-token vocabulary, 256-wide
hidden state, 256-token context, and tied token/LM-head weights.

## Model notes

### Model A

Model A established the gated-convolution baseline. Its dilations are
`1, 2, 4, 8, 16, 32, 64, 128`; strict left padding preserves causality.

![Model A](../../../../docs/model-a.svg)

### Model B

Model B demonstrated that one global attention block materially improves
entity and context tracking while retaining most of Model A's training speed.
The later "B2" experiments changed the data and training pipeline, not the
`cnn_attention` parameterization.

![Model B](../../../../docs/model-b.svg)

### Models C and D

Model C adds a second Transformer block after the final CNN stack. Model D
replaces that block with a readable, portable Mamba-1-style selective
state-space implementation. The portable Mamba path is intentionally retained
as a reference; without compatible fused CUDA kernels it was substantially
slower than the CNN/attention alternatives.

![Model C](../../../../docs/model-c.svg)

![Model D](../../../../docs/model-d.svg)

### Models E and F

Model E interleaves two attention blocks throughout six CNN blocks. Model F
adds three local-refinement CNN blocks and one final global attention block.
They produced the strongest validation results among the pre-X/Y hybrids, with
a corresponding throughput cost.

![Model E](../../../../docs/model-e.svg)

![Model F](../../../../docs/model-f.svg)

### Model G

Model G tests whether a residual GELU FFN after every convolution can close the
gap to Transformer-style blocks. Its parameter and compute increase did not
beat the smaller Model B in the smoke comparison.

![Model G](../../../../docs/model-g.svg)

### GPT-style baseline

The original Transformer control uses four pre-LayerNorm decoder blocks with
RoPE causal attention and 1,024-wide GELU FFNs. It is a controlled GPT-style
baseline, not a literal GPT-2 reproduction. Model Y supersedes it as the modern
RMSNorm/SwiGLU Transformer candidate.

![GPT-style Transformer](../../../../docs/transformer-baseline.svg)

## Historical workflows

The associated local and Colab experiment runners remain under `scripts/`:

- `run_transformer_smoke_benchmark.py`
- `run_model_g_smoke_benchmark.py`
- `run_model_x_smoke_benchmark.py`
- `run_colab_b2_smoke.sh` and `run_colab_b2_full.sh`
- `run_colab_model_b_750k.sh` and `run_colab_model_b_2m.sh`
- `run_colab_model_e_full.sh` and `run_colab_model_e_750k.sh`
- `run_colab_model_f_full.sh`
- `run_colab_transformer_750k.sh`

Legacy checkpoints continue to reconstruct from their original architecture
identifiers. Moving these implementations did not change parameter names or
state dictionaries.
