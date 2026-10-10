"""Opt-in, fixed small collapse probes on the current CPU/CUDA training model.

No backward pass, optimizer update, model transfer, or training-RNG consumption.
This is investigation evidence, not an automatic model-promotion gate.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from kiwilm.v3.diagnostics import _rms, _token_cosine, context_probe
from kiwilm.v3.masking import corrupt_tokens

POLICY = {
    "schema": "kiwilm3-collapse-monitor-v1",
    "windows": 2,
    "batch_size": 1,
    "noise_levels": [0.15, 0.5, 0.9, 1.0],
    "unigram_train_prefix_tokens": 1_000_000,
    "unigram_smoothing": 1,
    "timing": "step-zero-and-every-validation",
    "health_window": 0,
    "health_noise_level": 0.15,
}


class CollapseMonitor:
    def __init__(self, trainer):
        if trainer.config.device not in {"cpu", "cuda"}:
            raise ValueError("collapse diagnostics support CPU/CUDA only, not XLA")
        self.trainer = trainer
        generator = torch.Generator().manual_seed(trainer.config.validation_seed)
        self.windows = [
            trainer.data.get_batch(
                "validation",
                batch_size=POLICY["batch_size"],
                context_length=trainer.model.config.context_length,
                generator=generator,
            )[0]
            for _ in range(POLICY["windows"])
        ]
        prefix = np.asarray(
            trainer.data.tokens("train")[: POLICY["unigram_train_prefix_tokens"]], dtype=np.int64
        )
        self.prefix_tokens = len(prefix)
        counts = np.bincount(prefix, minlength=trainer.tokenizer.mask_id).astype(np.float64)
        counts[list(trainer.masking.protected_token_ids)] = 0
        counts += POLICY["unigram_smoothing"]
        self.log_probabilities = torch.from_numpy(np.log(counts / counts.sum()))
        self.majority_id = int(counts.argmax())

    @torch.no_grad()
    def evaluate(self) -> dict:
        trainer = self.trainer
        model = trainer.model
        valid = (self.windows[0] != trainer.tokenizer.pad_id).to(trainer.runtime.device)
        blocks = [{"block": i, "mixer": kind} for i, kind in enumerate(model.config.mixer_schedule)]
        handles = []

        def capture(i, key, *, before=False):
            def hook(module, args, output=None):
                # Only the original-context first forward, not its reversed counterpart.
                if key in blocks[i]:
                    return
                values = args[0] if before else output
                blocks[i][key] = _rms(values, valid)
                if key == "output_rms":
                    blocks[i]["output_mean_token_cosine"] = _token_cosine(values, valid)

            return hook

        probes = []
        previous = model.training
        try:
            for i, block in enumerate(model.blocks):
                handles.extend(
                    [
                        block.register_forward_pre_hook(capture(i, "input_rms", before=True)),
                        block.mixer.register_forward_hook(capture(i, "mixer_update_rms")),
                        block.mlp_norm.register_forward_pre_hook(
                            capture(i, "post_mixer_rms", before=True)
                        ),
                        block.mlp.register_forward_hook(capture(i, "mlp_update_rms")),
                        block.register_forward_hook(capture(i, "output_rms")),
                    ]
                )
            health = corrupt_tokens(
                self.windows[0],
                trainer.masking,
                generator=torch.Generator().manual_seed(trainer.config.validation_seed + 1),
                noise_level=POLICY["health_noise_level"],
            )
            model.eval()
            with trainer.runtime.autocast():
                model(
                    health.input_ids.to(trainer.runtime.device),
                    attention_mask=valid,
                    noise_level=health.noise_level,
                )
            for handle in handles:
                handle.remove()
            handles.clear()
            for index, clean in enumerate(self.windows):
                for level in POLICY["noise_levels"]:
                    seed = trainer.config.validation_seed + 1 + index
                    corrupted = corrupt_tokens(
                        clean,
                        trainer.masking,
                        generator=torch.Generator().manual_seed(seed),
                        noise_level=level,
                    )
                    count = corrupted.masked_positions.sum().item()
                    if not count:
                        probes.append({"window": index, "noise_level": level, "masked_tokens": 0})
                        continue
                    row = context_probe(
                        model,
                        clean,
                        trainer.masking,
                        noise_level=level,
                        seed=seed,
                        runtime=trainer.runtime,
                    )
                    labels = clean[corrupted.masked_positions].long()
                    row.update(
                        window=index,
                        unigram_loss=-self.log_probabilities[labels].mean().item(),
                        unigram_accuracy=(labels == self.majority_id).float().mean().item(),
                    )
                    row["loss_minus_unigram"] = row["loss"] - row["unigram_loss"]
                    probes.append(row)
        finally:
            for handle in handles:
                handle.remove()
            model.train(previous)
        for row in blocks:
            for name, numerator, denominator in (
                ("mixer_contribution", "mixer_update_rms", "input_rms"),
                ("mixer_residual_amplification", "post_mixer_rms", "input_rms"),
                ("mlp_contribution", "mlp_update_rms", "post_mixer_rms"),
                ("mlp_residual_amplification", "output_rms", "post_mixer_rms"),
            ):
                bottom = row.get(denominator, 0)
                row[name] = row[numerator] / bottom if bottom else None
        return {
            "event": "v3_collapse_diagnostics",
            "step": trainer.step,
            "tokens_seen": trainer.tokens_seen,
            "policy": POLICY,
            "validation_seed": trainer.config.validation_seed,
            "runtime_precision": trainer.config.precision,
            "unigram_train_prefix_tokens": self.prefix_tokens,
            "unigram_majority_id": self.majority_id,
            "blocks": blocks,
            "context_probes": probes,
            "finite_block_statistics": all(
                math.isfinite(value)
                for row in blocks
                for value in row.values()
                if isinstance(value, float)
            ),
            "scope": "Fixed small investigation probes; no automatic promotion decision",
        }
