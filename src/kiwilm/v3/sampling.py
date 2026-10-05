"""Fixed-slot confidence reveal: no remasking, autoregressive cache or length model."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from kiwilm.models.kiwilm3 import KiwiLM3Encoder
from kiwilm.v3.tokenizer import MaskBPETokenizer


@dataclass(frozen=True)
class SamplingConfig:
    steps: int = 16
    temperature: float = 0.8
    top_k: int = 40
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("steps", "top_k", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not 1 <= self.steps <= 1024 or self.seed >= 2**32:
            raise ValueError("steps must be 1-1024 and seed below 2**32")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (float, int))
            or not math.isfinite(self.temperature)
            or self.temperature < 0
        ):
            raise ValueError("temperature must be finite and nonnegative; zero is greedy")


@dataclass(frozen=True)
class SampleResult:
    token_ids: Tensor
    trace: tuple[dict, ...]
    config: dict


@torch.no_grad()
def sample_masked(
    model: KiwiLM3Encoder,
    tokenizer: MaskBPETokenizer,
    input_ids: Tensor,
    *,
    config: SamplingConfig | None = None,
    attention_mask: Tensor | None = None,
) -> SampleResult:
    """Preserve every visible token; reveal high-confidence predictions at MASK slots.

    Each iteration proposes all remaining holes, commits a linear cumulative
    quota, then freezes committed IDs forever. Confidence is the sampled token's
    probability under the original content-only softmax (not top-k-renormalized).
    Ties break by sequence index. Noise is remaining / initial hole count per row.
    """
    config = config or SamplingConfig()
    if (
        model.config.vocab_size != tokenizer.vocab_size
        or model.config.pad_token_id != tokenizer.pad_id
    ):
        raise ValueError("sampler model/tokenizer vocabulary or padding mismatch")
    valid = model._mask(input_ids, attention_mask) & (input_ids != tokenizer.pad_id)
    holes = input_ids == tokenizer.mask_id
    if (holes & ~valid).any().item():
        raise ValueError("MASK slots must be valid, not padding or excluded positions")
    if input_ids.device != model.token_embedding.weight.device:
        raise ValueError("sample inputs must be on the encoder device")
    output = input_ids.clone()
    initial = holes.sum(dim=1)
    if not initial.any().item():
        return SampleResult(output, (), asdict(config))
    allowed = torch.ones(tokenizer.vocab_size, dtype=torch.bool, device=input_ids.device)
    allowed[
        [tokenizer.pad_id, tokenizer.unk_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.mask_id]
    ] = False
    if not allowed.any().item():
        raise ValueError("tokenizer has no sampleable content tokens")
    rng = torch.Generator(device="cpu").manual_seed(config.seed)
    # Avoid no-progress passes when a requested schedule has more steps than holes.
    passes = min(config.steps, initial.max().item())
    trace = []
    was_training = model.training
    model.eval()
    try:
        for index in range(passes):
            remaining = output == tokenizer.mask_id
            counts = remaining.sum(dim=1)
            level = counts.float() / initial.clamp_min(1)
            logits = model(output, attention_mask=valid, noise_level=level).float()
            if not torch.isfinite(logits[remaining]).all().item():
                raise FloatingPointError("nonfinite logits at generation slots")
            content = logits.masked_fill(~allowed, -torch.inf)
            for row in range(output.shape[0]):
                positions = remaining[row].nonzero().flatten()
                if not positions.numel():
                    continue
                proposals = content[row, positions]
                probabilities = proposals.softmax(-1)
                if config.temperature == 0:
                    selected = proposals.argmax(-1)
                else:
                    # Subtract before dividing: tiny positive temperatures must not
                    # overflow the largest finite logit into +inf/NaN softmax.
                    # Sampling already owns a CPU RNG; FP64 filtering here also
                    # avoids requiring FP64 support on the encoder accelerator.
                    scaled = proposals.cpu().double()
                    scaled = (scaled - scaled.max(-1, keepdim=True).values) / config.temperature
                    if config.top_k:
                        keep = min(config.top_k, allowed.sum().item())
                        # Stable order keeps exact ties deterministic across repeated calls.
                        ranked = scaled.argsort(dim=-1, descending=True, stable=True)[:, :keep]
                        filtered = torch.full_like(scaled, -torch.inf)
                        filtered.scatter_(1, ranked, scaled.gather(1, ranked))
                        scaled = filtered
                    selected = torch.multinomial(scaled.softmax(-1).cpu(), 1, generator=rng)
                    selected = selected.flatten().to(input_ids.device)
                confidence = probabilities.gather(1, selected[:, None]).flatten()
                rank = confidence.argsort(descending=True, stable=True)
                target = math.ceil(initial[row].item() * (index + 1) / passes)
                already = initial[row].item() - counts[row].item()
                commit = rank[: target - already]
                output[row, positions[commit]] = selected[commit].to(output.dtype)
            trace.append(
                {
                    "iteration": index + 1,
                    "noise_level": level.cpu().tolist(),
                    "remaining": (output == tokenizer.mask_id).sum(1).cpu().tolist(),
                }
            )
    finally:
        model.train(was_training)
    if (output == tokenizer.mask_id).any().item():
        raise RuntimeError("reveal schedule failed to complete its fixed output slots")
    return SampleResult(output, tuple(trace), asdict(config))


def generate_slots(
    model: KiwiLM3Encoder,
    tokenizer: MaskBPETokenizer,
    prompt: str,
    *,
    output_slots: int = 32,
    suffix: str = "",
    config: SamplingConfig | None = None,
) -> dict:
    """Create BOS + protected prompt + fixed MASK slots + protected suffix + EOS."""
    if type(output_slots) is not int or output_slots < 1:
        raise ValueError("output_slots must be a positive integer")
    prefix, ending = tokenizer.encode(prompt), tokenizer.encode(suffix)
    ids = [
        tokenizer.bos_id,
        *prefix,
        *([tokenizer.mask_id] * output_slots),
        *ending,
        tokenizer.eos_id,
    ]
    if len(ids) > model.config.context_length:
        raise ValueError(
            "prompt, slots and suffix exceed context; no silent truncation or rollover"
        )
    tensor = torch.tensor([ids], dtype=torch.long, device=model.token_embedding.weight.device)
    sampled = sample_masked(model, tokenizer, tensor, config=config)
    result = sampled.token_ids[0].cpu().tolist()
    start = len(prefix) + 1
    return {
        "text": tokenizer.decode(result),
        "infill": tokenizer.decode(result[start : start + output_slots]),
        "token_ids": result,
        "trace": list(sampled.trace),
        "sampling": sampled.config,
        "prompt": prompt,
        "suffix": suffix,
        "output_slots": output_slots,
    }
