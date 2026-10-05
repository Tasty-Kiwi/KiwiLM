"""Bounded CPU M4 optimizer-step prototype; no cloud/production training launcher."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from kiwilm.data import PreparedTokenData
from kiwilm.models.kiwilm3 import KiwiLM3Encoder
from kiwilm.v3.masking import MaskingConfig
from kiwilm.v3.objectives import OBJECTIVE, denoising_forward
from kiwilm.v3.tokenizer import MaskBPETokenizer


@dataclass(frozen=True)
class DenoisingTrainConfig:
    max_steps: int = 100
    batch_size: int = 2
    context_length: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    data_seed: int = 42
    noise_seed: int = 142
    validation_seed: int = 242

    def __post_init__(self) -> None:
        for name in ("max_steps", "batch_size", "context_length"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_steps > 1000 or self.batch_size > 32 or self.context_length > 512:
            raise ValueError("M4 trainer is a bounded local prototype, not a long-run launcher")
        for name in ("data_seed", "noise_seed", "validation_seed"):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) < 2**32:
                raise ValueError(f"{name} must be an integer in [0,2**32)")
        for name in ("learning_rate", "weight_decay", "gradient_clip"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not self.learning_rate or not self.gradient_clip:
            raise ValueError("learning_rate and gradient_clip must be positive")


def _code_digest() -> str:
    # Include shared math/serialization/data/tokenizer helpers, not only the new objective.
    package = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class DenoisingTrainer:
    """Own separate data/noise RNGs; consume clean unshifted input windows only.

    Constant-LR AdamW/FP32/CPU deliberately bounds the acceptance claim. GPU/XLA
    execution, schedules/accumulation, throughput and Drive recovery are later gates.
    """

    def __init__(
        self,
        model: KiwiLM3Encoder,
        tokenizer: MaskBPETokenizer,
        data: PreparedTokenData,
        config: DenoisingTrainConfig,
        masking: MaskingConfig | None = None,
    ) -> None:
        tokenizer.assert_base_compatible(data.tokenizer)
        masking = masking or MaskingConfig(
            vocab_size=tokenizer.vocab_size, mask_id=tokenizer.mask_id
        )
        if (
            model.config.vocab_size != tokenizer.vocab_size
            or masking.vocab_size != tokenizer.vocab_size
            or masking.mask_id != tokenizer.mask_id
            or model.config.pad_token_id != masking.pad_id
            or masking.pad_id != tokenizer.pad_id
        ):
            raise ValueError(
                "model, tokenizer and masking policy must have identical vocabulary/padding"
            )
        if config.context_length > model.config.context_length:
            raise ValueError("training context exceeds the encoder window")
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError(
                "M4 local trainer requires CPU FP32; GPU/TPU training is not qualified"
            )
        if model.reconstruction_head.weight is not model.token_embedding.weight:
            raise ValueError("reconstruction head must remain tied before optimizer construction")
        for split in ("train", "validation"):
            if len(data.tokens(split)) <= config.context_length:
                raise ValueError("prepared splits must exceed training context length")
        self.model, self.tokenizer, self.data = model, tokenizer, data
        self.config, self.masking = config, masking
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.data_generator = torch.Generator().manual_seed(config.data_seed)
        self.noise_generator = torch.Generator().manual_seed(config.noise_seed)
        self.step = self.optimizer_steps = self.tokens_seen = 0
        self.at_step_boundary = True
        self.contract = {
            "engine": "m4-local-denoising-v1",
            "objective": OBJECTIVE,
            "model": model.config.to_dict(),
            "masking": masking.to_dict(),
            "training": asdict(config),
            "data_fingerprint": data.fingerprint,
            "tokenizer_sha256": tokenizer.fingerprint,
            "code_sha256": _code_digest(),
            "runtime": {
                "device": "cpu",
                "precision": "fp32",
                "torch": str(torch.__version__),
                "python": platform.python_version(),
            },
        }
        self.identity = hashlib.sha256(
            json.dumps(self.contract, sort_keys=True).encode()
        ).hexdigest()

    def train_step(self) -> dict:
        if self.step >= self.config.max_steps:
            raise ValueError("locked prototype step budget is exhausted")
        if not self.at_step_boundary:
            raise RuntimeError("previous step failed; restore the last verified checkpoint first")
        self.at_step_boundary = False
        clean, _unused_next_token_labels = self.data.get_batch(
            "train",
            batch_size=self.config.batch_size,
            context_length=self.config.context_length,
            generator=self.data_generator,
        )
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        result, corrupted = denoising_forward(
            self.model, clean, self.masking, generator=self.noise_generator
        )
        if not torch.isfinite(result.loss).item():
            raise FloatingPointError("nonfinite masked reconstruction loss")
        masked = result.masked_tokens.item()
        gradient_norm = None
        if masked:
            result.loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip, error_if_nonfinite=True
            ).item()
            self.optimizer.step()
            self.optimizer_steps += 1
        self.step += 1
        self.tokens_seen += corrupted.attention_mask.sum().item()
        self.at_step_boundary = True
        return {
            "event": "denoising_train",
            "step": self.step,
            "optimizer_steps": self.optimizer_steps,
            "tokens_seen": self.tokens_seen,
            "masked_tokens": masked,
            "eligible_tokens": corrupted.eligible_positions.sum().item(),
            "loss": result.loss.item() if masked else None,
            "masked_accuracy": result.correct.item() / masked if masked else None,
            "mean_noise": corrupted.noise_level.mean().item(),
            "gradient_norm": gradient_norm,
            "skipped_empty_mask": not bool(masked),
            "objective": OBJECTIVE,
        }

    @torch.no_grad()
    def evaluate(
        self, *, batches: int = 4, noise_levels: tuple[float, ...] = (0.15, 0.5, 0.9, 1.0)
    ) -> dict:
        if type(batches) is not int or not 1 <= batches <= 100 or not noise_levels:
            raise ValueError("validation needs 1-100 batches and nonempty noise levels")
        # Rebuilt each call: validation is fixed and cannot advance training RNGs.
        data_rng = torch.Generator().manual_seed(self.config.validation_seed)
        windows = [
            self.data.get_batch(
                "validation",
                batch_size=self.config.batch_size,
                context_length=self.config.context_length,
                generator=data_rng,
            )[0]
            for _ in range(batches)
        ]
        training = self.model.training
        self.model.eval()
        rows = []
        try:
            for level in noise_levels:
                noise_rng = torch.Generator().manual_seed(self.config.validation_seed + 1)
                total, correct, count, eligible = 0.0, 0, 0, 0
                for clean in windows:
                    result, corrupted = denoising_forward(
                        self.model, clean, self.masking, generator=noise_rng, noise_level=level
                    )
                    if not torch.isfinite(result.loss_sum).item():
                        raise FloatingPointError("nonfinite validation reconstruction loss")
                    total += result.loss_sum.item()
                    count += result.masked_tokens.item()
                    correct += result.correct.item()
                    eligible += corrupted.eligible_positions.sum().item()
                rows.append(
                    {
                        "noise_level": level,
                        "masked_tokens": count,
                        "eligible_tokens": eligible,
                        "loss": total / count if count else None,
                        "masked_accuracy": correct / count if count else None,
                    }
                )
        finally:
            self.model.train(training)
        return {
            "objective": OBJECTIVE,
            "fixed_validation_seed": self.config.validation_seed,
            "batches_per_level": batches,
            "rows": rows,
            "autoregressive_perplexity": None,
        }
