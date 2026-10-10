"""Single-device V3 training; CPU is the test backend, real TPU/XLA uses BF16 AMP.

No sessions are provisioned. This engine is separate from M4 local state and V2.
Corruption/validation stay on CPU; accelerator graphs retain fixed tensor shapes.
"""

from __future__ import annotations

import math
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version

import torch
from torch.nn import functional as F

from kiwilm.data import PreparedTokenData
from kiwilm.tpu_smoke import Runtime
from kiwilm.v3 import KiwiLM3Config, build_encoder
from kiwilm.v3.experiments import CANDIDATES, NOISE_LEVELS, canonical_digest
from kiwilm.v3.masking import MaskingConfig, corrupt_tokens
from kiwilm.v3.objectives import OBJECTIVE
from kiwilm.v3.tokenizer import MaskBPETokenizer
from kiwilm.v3.trainer import _code_digest

ENGINE = "kiwilm3-single-device-denoising-v1"


@dataclass(frozen=True)
class AcceleratorTrainConfig:
    device: str = "xla"
    precision: str = "bf16"
    max_tokens: int = 4096
    warmup_tokens: int = 512
    batch_size: int = 2
    grad_accum_steps: int = 2
    learning_rate: float = 0.001
    min_learning_rate: float = 0.0001
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 42
    data_seed: int = 42
    noise_seed: int = 142
    validation_seed: int = 242
    checkpoint_interval: int = 4
    eval_interval: int = 4
    eval_batches: int = 2

    def __post_init__(self):
        allowed = {"cpu": {"fp32"}, "cuda": {"fp32", "bf16"}, "xla": {"bf16"}}
        if self.device not in allowed or self.precision not in allowed[self.device]:
            raise ValueError("explicit CPU FP32, CUDA FP32/BF16 or single-TPU XLA BF16 required")
        for name in (
            "max_tokens",
            "batch_size",
            "grad_accum_steps",
            "checkpoint_interval",
            "eval_interval",
            "eval_batches",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive integer")
        if self.eval_batches > 100 or self.batch_size > 32 or self.grad_accum_steps > 32:
            raise ValueError("batch, accumulation or validation exceeds supported bounds")
        if type(self.warmup_tokens) is not int or not 0 <= self.warmup_tokens < self.max_tokens:
            raise ValueError("warmup_tokens must be nonnegative and below the full budget")
        for name in ("seed", "data_seed", "noise_seed", "validation_seed"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 2**32 - 1:
                raise ValueError(f"{name} must be a seed below 2**32-1")
        for name in ("learning_rate", "min_learning_rate", "weight_decay", "gradient_clip"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            self.learning_rate <= 0
            or self.gradient_clip <= 0
            or self.min_learning_rate > self.learning_rate
        ):
            raise ValueError("positive LR/clip and min_learning_rate <= learning_rate required")

    def to_dict(self):
        return asdict(self)


def learning_rate(config: AcceleratorTrainConfig, tokens: int) -> float:
    if not 0 <= tokens <= config.max_tokens:
        raise ValueError("schedule tokens outside the locked budget")
    if config.warmup_tokens and tokens <= config.warmup_tokens:
        return config.learning_rate * tokens / config.warmup_tokens
    progress = (tokens - config.warmup_tokens) / (config.max_tokens - config.warmup_tokens)
    return config.min_learning_rate + 0.5 * (config.learning_rate - config.min_learning_rate) * (
        1 + math.cos(math.pi * progress)
    )


def candidate_config(candidate: str, vocab_size: int, *, qualification: bool) -> KiwiLM3Config:
    if candidate not in CANDIDATES or type(qualification) is not bool:
        raise ValueError("select an explicit dense B/C candidate and boolean qualification mode")
    depth = int(candidate.split("-")[1])
    return KiwiLM3Config(
        vocab_size=vocab_size,
        num_blocks=depth,
        dropout=0.1,
        context_length=16 if qualification else 512,
        d_model=16 if qualification else 512,
        swiglu_dim=48 if qualification else 2048,
        num_heads=2 if qualification else 8,
        noise_embedding_dim=16 if qualification else 64,
        mixer_schedule=("attention",) * depth if candidate.startswith("attention") else None,
    )


def runtime_contract(runtime: Runtime) -> dict:
    result = {
        "device": runtime.device.type,
        "precision": runtime.precision,
        "torch": str(torch.__version__),
        "python": platform.python_version(),
    }
    if runtime.xm is not None:
        result["torch_xla"] = version("torch_xla")
        if result["torch"].split("+")[0] != result["torch_xla"].split("+")[0]:
            raise ValueError("torch and torch_xla versions must match")
        result["hardware"] = runtime.xm.xla_device_kind(runtime.device)
    elif runtime.device.type == "cuda":
        result["hardware"] = torch.cuda.get_device_name(runtime.device)
    else:
        result["cpu_threads"] = torch.get_num_threads()
    return result


def dense_reconstruction(logits, clean, selected, mask_id: int, *, accuracy: bool = False):
    """Same unweighted M4 CE, no dynamic selected-target indexing or host scalar reads.

    Trainer validates clean CPU IDs before transfer. Targets and selection retain
    B,T shape. Return a sum so all accumulated microbatches share one denominator.
    """
    if (
        logits.shape != (*clean.shape, mask_id + 1)
        or selected.shape != clean.shape
        or selected.dtype != torch.bool
    ):
        raise ValueError("dense loss shapes or boolean selection mismatch")
    labels = clean.long().masked_fill(~selected, -100)
    values = F.cross_entropy(
        logits[..., :mask_id].float().transpose(1, 2), labels, reduction="none", ignore_index=-100
    )
    correct = ((logits[..., :mask_id].argmax(-1) == clean) & selected).sum() if accuracy else None
    return values.sum(), correct


class AcceleratorTrainer:
    def __init__(
        self,
        model_config: KiwiLM3Config,
        config: AcceleratorTrainConfig,
        tokenizer: MaskBPETokenizer,
        data: PreparedTokenData,
        *,
        runtime: Runtime | None = None,
        masking: MaskingConfig | None = None,
    ):
        tokenizer.assert_base_compatible(data.tokenizer)
        masking = masking or MaskingConfig(
            vocab_size=tokenizer.vocab_size, mask_id=tokenizer.mask_id
        )
        if (
            model_config.vocab_size != tokenizer.vocab_size
            or model_config.pad_token_id != tokenizer.pad_id
            or masking.vocab_size != tokenizer.vocab_size
            or masking.mask_id != tokenizer.mask_id
            or masking.pad_id != tokenizer.pad_id
        ):
            raise ValueError("model, masking and tokenizer disagree")
        if any(
            len(data.tokens(split)) <= model_config.context_length
            for split in ("train", "validation")
        ):
            raise ValueError("prepared splits must exceed model context")
        self.runtime = runtime or Runtime(config.device, config.precision)
        if self.runtime.device.type != config.device or self.runtime.precision != config.precision:
            raise ValueError("runtime/config backend mismatch; no fallback")
        if (
            config.device == "cuda"
            and config.precision == "bf16"
            and not torch.cuda.is_bf16_supported(including_emulation=False)
        ):
            raise ValueError(
                "CUDA BF16 unsupported; explicitly use FP32 (FP16 is not implemented here)"
            )
        self.config, self.data, self.tokenizer, self.masking = config, data, tokenizer, masking
        torch.manual_seed(config.seed)
        if self.runtime.xm is not None:
            self.runtime.xm.set_rng_state(config.seed, self.runtime.device)
        self.model = build_encoder(model_config).to(self.runtime.device)
        if self.model.token_embedding.weight is not self.model.reconstruction_head.weight:
            raise RuntimeError("device transfer untied V3 reconstruction head")
        self.lr_tensor = torch.tensor(config.learning_rate, device=self.runtime.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr_tensor,
            weight_decay=config.weight_decay,
            foreach=False,
            fused=False,
            capturable=self.runtime.xla is not None,
        )
        self.data_generator = torch.Generator().manual_seed(config.data_seed)
        self.noise_generator = torch.Generator().manual_seed(config.noise_seed)
        self.step = self.optimizer_steps = self.tokens_seen = self.masked_tokens_seen = 0
        self.at_step_boundary = True
        self.contract = {
            "engine": ENGINE,
            "schema": 1,
            "objective": OBJECTIVE,
            "model": model_config.to_dict(),
            "training": config.to_dict(),
            "masking": masking.to_dict(),
            "tokenizer_sha256": tokenizer.fingerprint,
            "data_fingerprint": data.fingerprint,
            "code_sha256": _code_digest(),
            "runtime": runtime_contract(self.runtime),
            "optimizer": "torch-adamw-single-tensor",
            "token_budget": "nonpadding-input-tokens",
            "loss_normalization": "sum masked CE / total masks across accumulated microbatches",
        }
        self.identity = canonical_digest(self.contract)

    def _forward(self, clean_cpu, corrupted, *, accuracy=False):
        device = self.runtime.device
        clean = clean_cpu.to(device)
        selected = corrupted.masked_positions.to(device)
        with self.runtime.autocast():
            logits = self.model(
                corrupted.input_ids.to(device),
                attention_mask=corrupted.attention_mask.to(device),
                noise_level=corrupted.noise_level,
            )
            return dense_reconstruction(
                logits, clean, selected, self.tokenizer.mask_id, accuracy=accuracy
            )

    def train_step(self) -> dict:
        if not self.at_step_boundary:
            raise RuntimeError("failed step; restore a verified checkpoint before continuing")
        if self.tokens_seen >= self.config.max_tokens:
            raise ValueError("token budget exhausted")
        self.at_step_boundary = False
        remaining = self.config.max_tokens - self.tokens_seen
        microbatches, consumed, masks, eligible = [], 0, 0, 0
        for _ in range(self.config.grad_accum_steps):
            if consumed >= remaining:
                break
            clean, _unused_shifted_targets = self.data.get_batch(
                "train",
                batch_size=self.config.batch_size,
                context_length=self.model.config.context_length,
                generator=self.data_generator,
            )
            valid = clean != self.tokenizer.pad_id
            if not valid.any().item():
                raise ValueError("training window has no valid tokens; refuse a stalled budget")
            active = valid & (
                valid.flatten().long().cumsum(0).reshape_as(valid) <= remaining - consumed
            )
            corrupted = corrupt_tokens(
                clean, self.masking, generator=self.noise_generator, attention_mask=active
            )
            consumed += corrupted.attention_mask.sum().item()
            masks += corrupted.masked_positions.sum().item()
            eligible += corrupted.eligible_positions.sum().item()
            microbatches.append((clean, corrupted))
        lr = learning_rate(self.config, self.tokens_seen + consumed)
        # Copy host tensor values rather than inserting a changing scalar constant
        # into every XLA graph. Master parameters/moments remain FP32 under AMP.
        self.lr_tensor.copy_(torch.tensor(lr, dtype=self.lr_tensor.dtype))
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device=self.runtime.device)
        norm = None
        if masks:
            denominator = torch.tensor(float(masks)).to(self.runtime.device)
            for clean, corrupted in microbatches:
                loss_sum, _ = self._forward(clean, corrupted)
                (loss_sum / denominator).backward()
                total = total + loss_sum.detach()
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip, foreach=False
            )
            self.runtime.sync()
            if not torch.isfinite(total).item() or not torch.isfinite(norm).item():
                raise FloatingPointError("nonfinite loss/gradients; restore last committed state")
            self.optimizer.step()
            self.runtime.sync()
            self.optimizer_steps += 1
        self.step += 1
        self.tokens_seen += consumed
        self.masked_tokens_seen += masks
        self.at_step_boundary = True
        return {
            "event": "v3_train",
            "step": self.step,
            "optimizer_steps": self.optimizer_steps,
            "tokens_seen": self.tokens_seen,
            "masked_tokens_seen": self.masked_tokens_seen,
            "input_tokens": consumed,
            "masked_tokens": masks,
            "eligible_tokens": eligible,
            "loss": total.item() / masks if masks else None,
            "gradient_norm": norm.item() if norm is not None else None,
            "learning_rate": lr,
            "skipped_empty_mask": not bool(masks),
            "objective": OBJECTIVE,
        }

    @torch.no_grad()
    def evaluate(self) -> dict:
        rng = torch.Generator().manual_seed(self.config.validation_seed)
        windows = [
            self.data.get_batch(
                "validation",
                batch_size=self.config.batch_size,
                context_length=self.model.config.context_length,
                generator=rng,
            )[0]
            for _ in range(self.config.eval_batches)
        ]
        training = self.model.training
        self.model.eval()
        rows = []
        try:
            for level in NOISE_LEVELS:
                noise = torch.Generator().manual_seed(self.config.validation_seed + 1)
                total, correct = torch.zeros(2, device=self.runtime.device).unbind()
                masks = eligible = 0
                for clean in windows:
                    corrupted = corrupt_tokens(
                        clean, self.masking, generator=noise, noise_level=level
                    )
                    loss, hits = self._forward(clean, corrupted, accuracy=True)
                    total, correct = total + loss, correct + hits
                    masks += corrupted.masked_positions.sum().item()
                    eligible += corrupted.eligible_positions.sum().item()
                self.runtime.sync()
                if not torch.isfinite(total).item():
                    raise FloatingPointError("nonfinite fixed validation loss")
                rows.append(
                    {
                        "noise_level": level,
                        "masked_tokens": masks,
                        "eligible_tokens": eligible,
                        "loss": total.item() / masks if masks else None,
                        "masked_accuracy": correct.item() / masks if masks else None,
                    }
                )
        finally:
            self.model.train(training)
        return {
            "event": "v3_validation",
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "objective": OBJECTIVE,
            "validation_seed": self.config.validation_seed,
            "batches_per_level": self.config.eval_batches,
            "rows": rows,
            "autoregressive_perplexity": None,
        }
