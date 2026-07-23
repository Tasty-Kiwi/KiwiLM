"""Architecture-independent autoregressive text generation."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


@torch.inference_mode()
def generate_tokens(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 128,
    context_length: int | None = None,
    temperature: float = 0.8,
    top_k: int | None = 50,
    eos_id: int | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Append tokens autoregressively and return the full token sequence.

    Sampling is performed on CPU so a seeded generator behaves consistently
    whether the model itself is on CPU, CUDA, or MPS.
    """

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive or None")
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have shape [batch, sequence] with sequence > 0")

    resolved_context_length = context_length
    if resolved_context_length is None:
        resolved_context_length = getattr(
            getattr(model, "config", None), "context_length", None
        )
    if resolved_context_length is None or resolved_context_length <= 0:
        raise ValueError(
            "context_length must be positive or available as model.config.context_length"
        )

    if generator is not None and seed is not None:
        raise ValueError("pass either generator or seed, not both")
    if generator is None:
        generator = torch.Generator(device="cpu")
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

    generated = input_ids
    finished = torch.zeros(
        generated.shape[0], dtype=torch.bool, device=generated.device
    )
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            model_input = generated[:, -resolved_context_length:]
            logits = model(model_input)
            if logits.ndim != 3 or logits.shape[:2] != model_input.shape:
                raise ValueError(
                    "model must return logits shaped [batch, sequence, vocabulary]"
                )
            next_logits = logits[:, -1, :]
            if temperature == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_token = _sample(
                    next_logits,
                    temperature=temperature,
                    top_k=top_k,
                    generator=generator,
                ).to(generated.device)

            if eos_id is not None:
                eos_tokens = torch.full_like(next_token, eos_id)
                next_token = torch.where(
                    finished.unsqueeze(1), eos_tokens, next_token
                )
            generated = torch.cat((generated, next_token), dim=1)
            if eos_id is not None:
                finished |= next_token.squeeze(1).eq(eos_id)
                if bool(finished.all()):
                    break
    finally:
        model.train(was_training)
    return generated


def generate(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    context_length: int | None = None,
    temperature: float = 0.8,
    top_k: int | None = 50,
    seed: int | None = 42,
    device: str | torch.device | None = None,
    include_prompt: bool = True,
) -> str:
    """Encode a prompt, generate from any KiwiLM architecture, and decode it."""

    resolved_device = _model_device(model) if device is None else torch.device(device)
    model.to(resolved_device)
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    if not prompt_ids:
        bos_id = getattr(tokenizer, "bos_id", None)
        if bos_id is None:
            raise ValueError("the tokenizer produced an empty prompt and has no BOS token")
        prompt_ids = [bos_id]
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=resolved_device)
    output_ids = generate_tokens(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        context_length=context_length,
        temperature=temperature,
        top_k=top_k,
        eos_id=getattr(tokenizer, "eos_id", None),
        seed=seed,
    )[0]
    decoded_ids = output_ids if include_prompt else output_ids[len(prompt_ids) :]
    return tokenizer.decode(decoded_ids.tolist(), skip_special_tokens=True)


def generate_text(*args: Any, **kwargs: Any) -> str:
    """Descriptive alias for :func:`generate`."""

    return generate(*args, **kwargs)


def _sample(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    generator: torch.Generator,
) -> torch.Tensor:
    scaled = logits.float().cpu() / temperature
    if top_k is not None:
        effective_k = min(top_k, scaled.shape[-1])
        values, indices = torch.topk(scaled, effective_k, dim=-1)
        probabilities = torch.softmax(values, dim=-1)
        sampled_index = torch.multinomial(
            probabilities, num_samples=1, generator=generator
        )
        return indices.gather(dim=-1, index=sampled_index)
    probabilities = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")
