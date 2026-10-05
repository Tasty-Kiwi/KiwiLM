"""Bidirectional masked evaluation; never reuse causal perplexity/retrieval scoring."""

from __future__ import annotations

import time

import torch

from kiwilm.comparison import generation_quality_metrics
from kiwilm.v3.experiments import GENERATION_CASES, NOISE_LEVELS, evaluation_contract
from kiwilm.v3.objectives import denoising_forward
from kiwilm.v3.sampling import SamplingConfig, generate_slots
from kiwilm.v3.trainer import DenoisingTrainer


def distributions(values: list[float]) -> dict:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "median": tensor.quantile(0.5).item(),
        "p90": tensor.quantile(0.9).item(),
        "p95": tensor.quantile(0.95).item(),
        "maximum": tensor.max().item(),
    }


def _rms(values, valid):
    selected = values.detach()[valid].double()
    return selected.square().mean().sqrt().item() if selected.numel() else 0.0


def health_audit(trainer: DenoisingTrainer, *, batches: int) -> dict:
    """Fixed validation, eval-mode gradients. No training/data/noise RNG advances.

    Hooks capture only scalar statistics. autograd.grad leaves training .grad
    buffers untouched. Amplification uses post-MLP / post-mixer residual RMS.
    1.5 is a diagnostic flag, not an automatic model promotion rule.
    """
    if type(batches) is not int or not 1 <= batches <= 100:
        raise ValueError("health audit needs 1-100 batches")
    model = trainer.model
    before = model.training
    data_rng = torch.Generator().manual_seed(trainer.config.validation_seed)
    noise_rng = torch.Generator().manual_seed(trainer.config.validation_seed + 1)
    records = [[] for _ in model.blocks]
    handles = []
    state = [{} for _ in model.blocks]

    def entry(index):
        def hook(module, args):
            state[index] = {"valid": args[1], "input_rms": _rms(args[0], args[1])}

        return hook

    def post_mixer(index):
        def hook(module, args):
            state[index]["post_mixer_rms"] = _rms(args[0], state[index]["valid"])

        return hook

    def mlp_update(index):
        def hook(module, args, output):
            state[index]["mlp_update_rms"] = _rms(output, state[index]["valid"])

        return hook

    def exit_block(index):
        def hook(module, args, output):
            state[index]["output_rms"] = _rms(output, state[index]["valid"])

        return hook

    for index, block in enumerate(model.blocks):
        handles.extend(
            (
                block.register_forward_pre_hook(entry(index)),
                block.mlp_norm.register_forward_pre_hook(post_mixer(index)),
                block.mlp.register_forward_hook(mlp_update(index)),
                block.register_forward_hook(exit_block(index)),
            )
        )
    parameters = list(model.named_parameters())
    finite, nonzero = True, True
    model.eval()
    try:
        with torch.enable_grad():
            for _ in range(batches):
                clean, _ = trainer.data.get_batch(
                    "validation",
                    batch_size=trainer.config.batch_size,
                    context_length=trainer.config.context_length,
                    generator=data_rng,
                )
                result, _ = denoising_forward(
                    model, clean, trainer.masking, generator=noise_rng, noise_level=1.0
                )
                gradients = torch.autograd.grad(result.loss, [p for _, p in parameters])
                finite &= torch.isfinite(result.loss).item() and all(
                    torch.isfinite(g).all().item() for g in gradients
                )
                for index in range(len(model.blocks)):
                    gradient = (
                        sum(
                            g.detach().double().square().sum().item()
                            for (name, _), g in zip(parameters, gradients, strict=True)
                            if name.startswith(f"blocks.{index}.mlp.")
                        )
                        ** 0.5
                    )
                    row = state[index]
                    denominator = row["post_mixer_rms"]
                    ratio = row["output_rms"] / denominator if denominator else None
                    contribution = row["mlp_update_rms"] / denominator if denominator else None
                    nonzero &= gradient > 0 and denominator > 0
                    finite &= all(
                        torch.isfinite(torch.tensor(row[key])).item()
                        for key in ("input_rms", "post_mixer_rms", "mlp_update_rms", "output_rms")
                    )
                    records[index].append(
                        {
                            "mlp_gradient_norm": gradient,
                            "residual_amplification": ratio,
                            "mlp_contribution": contribution,
                        }
                    )
    finally:
        for handle in handles:
            handle.remove()
        model.train(before)
    rows = []
    for index, samples in enumerate(records):
        rows.append(
            {
                "block": index,
                "mlp": "swiglu",
                "mixer": model.config.mixer_schedule[index],
                "samples": samples,
                "distributions": {
                    key: distributions([s[key] for s in samples])
                    if all(s[key] is not None for s in samples)
                    else None
                    for key in samples[0]
                },
                "amplification_above_1_5_rate": sum(
                    s["residual_amplification"] is None or s["residual_amplification"] > 1.5
                    for s in samples
                )
                / batches,
            }
        )
    first = rows[0]["distributions"]["mlp_gradient_norm"]["median"]
    last = rows[-1]["distributions"]["mlp_gradient_norm"]["median"]
    return {
        "batches": batches,
        "noise_level": 1.0,
        "all_finite": bool(finite),
        "all_mlp_gradients_nonzero": bool(nonzero),
        "blocks": rows,
        "swiglu_last_first_median_gradient_ratio": last / first if first else None,
    }


def generation_suite(trainer, controls) -> dict:
    """Fixed continuation/infilling/unconditional slots, five seeds by default."""
    rows = []
    for case in GENERATION_CASES:
        for seed in controls.generation_seeds:
            start = time.perf_counter()
            row = generate_slots(
                trainer.model,
                trainer.tokenizer,
                case["prompt"],
                suffix=case["suffix"],
                output_slots=controls.generation_slots,
                config=SamplingConfig(
                    controls.sampler_steps, controls.temperature, controls.top_k, seed
                ),
            )
            row.update(name=case["name"], seed=seed, seconds=time.perf_counter() - start)
            # Score the generated segment, never the protected prompt/suffix.
            row.update(generation_quality_metrics(row["infill"]))
            rows.append(row)
    return {
        "rows": rows,
        "semantic_consistency": None,
        "note": "Repetition is measured; semantic correctness needs human/task review.",
    }


@torch.no_grad()
def cloze_scores(model, tokenizer, *, prefix: str, suffix: str, choices: list[str]) -> dict:
    """Score equal-length answer candidates at protected-context MASK holes.

    Parallel masked pseudo-scores, NOT joint likelihood. Supports right context;
    different BPE answer lengths are refused to avoid a length-selection confound.
    Caller supplies the task/ground truth; no built-in causal retrieval reuse.
    """
    encoded = [tokenizer.encode(choice) for choice in choices]
    if len(choices) < 2 or not encoded[0] or len({len(ids) for ids in encoded}) != 1:
        raise ValueError("cloze choices need at least two nonempty equal-token-length answers")
    if len({tuple(ids) for ids in encoded}) != len(encoded):
        raise ValueError("cloze choices must have distinct token IDs")
    start = len(tokenizer.encode(prefix)) + 1
    ids = [
        tokenizer.bos_id,
        *tokenizer.encode(prefix),
        *([tokenizer.mask_id] * len(encoded[0])),
        *tokenizer.encode(suffix),
        tokenizer.eos_id,
    ]
    if len(ids) > model.config.context_length:
        raise ValueError("cloze task exceeds context; no truncation")
    before = model.training
    model.eval()
    try:
        values = torch.tensor([ids], device=model.token_embedding.weight.device)
        logits = model(values, noise_level=1.0)[
            0, start : start + len(encoded[0]), : tokenizer.mask_id
        ].float()
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError("nonfinite cloze logits")
        logp = logits.log_softmax(-1)
        scores = [
            logp.gather(1, torch.tensor(answer, device=logp.device)[:, None]).sum().item()
            for answer in encoded
        ]
    finally:
        model.train(before)
    return {
        "choices": choices,
        "scores": scores,
        "selected": max(range(len(scores)), key=scores.__getitem__),
        "sequence_length": len(ids),
        "context_limit": model.config.context_length,
        "scoring": "parallel masked pseudo-score, not autoregressive/joint likelihood",
    }


def evaluate_candidate(trainer, controls) -> dict:
    validation = trainer.evaluate(batches=controls.validation_batches, noise_levels=NOISE_LEVELS)
    validation["evaluation_contract"] = evaluation_contract(controls, trainer.masking.to_dict())
    return {
        "validation": validation,
        "health": health_audit(trainer, batches=controls.validation_batches),
        "generation": generation_suite(trainer, controls),
        "external_transfer": None,
        "contextual_retrieval": None,
        "missing_evidence": [
            "held-out contextual retrieval task",
            "external transfer/classification probes",
            "accelerator throughput/peak memory",
            "human semantic-quality review",
        ],
    }


def evaluate_transfer(trainer: DenoisingTrainer, data, *, batches: int = 20) -> dict:
    """External reconstruction transfer requires the same original BPE IDs.

    Not classification or AR perplexity. Separate evaluator RNGs; its unused
    optimizer never steps or changes the trained model.
    """
    evaluator = DenoisingTrainer(
        trainer.model, trainer.tokenizer, data, trainer.config, trainer.masking
    )
    result = evaluator.evaluate(batches=batches, noise_levels=NOISE_LEVELS)
    result["data_fingerprint"] = data.fingerprint
    result["tokenizer_sha256"] = trainer.tokenizer.fingerprint
    result["evaluation"] = "external fixed masked reconstruction; not AR perplexity/classification"
    return result
