"""Frozen B/C factorial specifications and provenance checks, never launch on import."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass

from kiwilm.v3.config import KiwiLM3Config
from kiwilm.v3.masking import MaskingConfig
from kiwilm.v3.objectives import OBJECTIVE
from kiwilm.v3.profile import profile_encoder
from kiwilm.v3.sampling import SamplingConfig
from kiwilm.v3.trainer import DenoisingTrainConfig

SUITE = "kiwilm3-bc-dense-denoising-v1"
CANDIDATES = ("hybrid-12", "attention-12", "hybrid-16", "attention-16")
COMPARISONS = {
    "B-12": ("hybrid-12", "attention-12"),
    "B-16": ("hybrid-16", "attention-16"),
    "C-hybrid": ("hybrid-12", "hybrid-16"),
    "C-attention": ("attention-12", "attention-16"),
}
NOISE_LEVELS = (0.15, 0.5, 0.9, 1.0)
GENERATION_CASES = (
    {"name": "unconditional", "prompt": "", "suffix": ""},
    {"name": "continuation", "prompt": "The cat ", "suffix": ""},
    {"name": "infilling", "prompt": "The ", "suffix": " cat."},
)


def canonical_digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ExperimentControls:
    initialization_seed: int = 42
    max_steps: int = 1000
    batch_size: int = 2
    context_length: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    dropout: float = 0.1
    data_seed: int = 42
    noise_seed: int = 142
    validation_seed: int = 242
    validation_batches: int = 20
    checkpoint_interval: int = 100
    generation_slots: int = 32
    generation_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    sampler_steps: int = 16
    temperature: float = 0.8
    top_k: int = 40

    def __post_init__(self) -> None:
        DenoisingTrainConfig(**self.training_values())
        SamplingConfig(self.sampler_steps, self.temperature, self.top_k)
        for name in ("initialization_seed",):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) < 2**32:
                raise ValueError(f"{name} must be an integer seed below 2**32")
        for name in ("validation_batches", "checkpoint_interval", "generation_slots"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.validation_batches > 100 or self.checkpoint_interval > self.max_steps:
            raise ValueError("validation/checkpoint intervals exceed the bounded prototype")
        if self.generation_slots + 2 > self.context_length:
            raise ValueError("generation slots need room for BOS/EOS inside the context")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("dropout must be in [0,1)")
        seeds = tuple(self.generation_seeds)
        if (
            not seeds
            or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        ):
            raise ValueError("generation seeds must be a nonempty unique sequence")
        object.__setattr__(self, "generation_seeds", seeds)

    def training_values(self) -> dict:
        return {
            key: getattr(self, key)
            for key in (
                "max_steps",
                "batch_size",
                "context_length",
                "learning_rate",
                "weight_decay",
                "data_seed",
                "noise_seed",
                "validation_seed",
            )
        }

    def to_dict(self) -> dict:
        result = asdict(self)
        result["generation_seeds"] = list(self.generation_seeds)
        return result


def build_suite(
    *,
    data_fingerprint: str,
    tokenizer_sha256: str,
    vocab_size: int,
    controls: ExperimentControls | None = None,
    qualification: bool = False,
) -> dict:
    """Define four dense candidates with matched controls; meta profiles allocate no weights.

    Qualification retains actual depths/kernels but uses width16, FFN48/context16.
    The current executable adapter is bounded CPU FP32, not a qualified TPU run.
    """
    for value in (data_fingerprint, tokenizer_sha256):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("data/tokenizer provenance requires lowercase SHA256 identities")
    if type(qualification) is not bool:
        raise ValueError("qualification must be explicit boolean")
    if controls is None:
        controls = (
            ExperimentControls(
                max_steps=20,
                context_length=16,
                validation_batches=2,
                checkpoint_interval=4,
                generation_slots=4,
                sampler_steps=4,
            )
            if qualification
            else ExperimentControls()
        )
    if qualification and controls.context_length > 64:
        raise ValueError("tiny qualification context must be at most 64")
    width, ffn, heads, noise = (16, 48, 2, 16) if qualification else (512, 2048, 8, 64)
    masking = MaskingConfig(vocab_size=vocab_size, mask_id=vocab_size - 1)
    candidates = {}
    for name in CANDIDATES:
        mixer, depth = name.split("-")
        config = KiwiLM3Config(
            vocab_size=vocab_size,
            d_model=width,
            swiglu_dim=ffn,
            num_heads=heads,
            noise_embedding_dim=noise,
            num_blocks=int(depth),
            context_length=controls.context_length,
            dropout=controls.dropout,
            mixer_schedule=("attention",) * int(depth) if mixer == "attention" else None,
        )
        candidates[name] = {"model": config.to_dict(), "profile": profile_encoder(config)}
    suite = {
        "schema_version": 1,
        "suite": SUITE,
        "objective": OBJECTIVE,
        "masking": masking.to_dict(),
        "data_fingerprint": data_fingerprint,
        "tokenizer_sha256": tokenizer_sha256,
        "controls": controls.to_dict(),
        "qualification": qualification,
        "candidates": candidates,
        "comparisons": {key: list(names) for key, names in COMPARISONS.items()},
        "dropped_experiments": ["A", "D"],
        "train_from_scratch": True,
        "runtime": {"device": "cpu", "precision": "fp32", "accelerator_qualified": False},
        "budget": {
            "step_attempts": controls.max_steps,
            "input_positions": controls.max_steps * controls.batch_size * controls.context_length,
            "nonpadding_tokens": "measured; depends on fixed sampled windows",
            "schedule": "constant-LR AdamW; no accumulation",
        },
        "training_started": False,
    }
    suite["identity"] = canonical_digest(suite)
    return suite


def validate_suite(suite: dict) -> ExperimentControls:
    """Rebuild the complete specification; reject edited profiles/settings or hidden candidates."""
    controls = ExperimentControls(**suite["controls"])
    rebuilt = build_suite(
        data_fingerprint=suite["data_fingerprint"],
        tokenizer_sha256=suite["tokenizer_sha256"],
        vocab_size=suite["masking"]["vocab_size"],
        controls=controls,
        qualification=suite["qualification"],
    )
    if suite != rebuilt:
        raise ValueError("experiment suite differs from its canonical frozen specification")
    return controls


def validate_comparison(reports: dict[str, dict], suite: dict, comparison: str) -> dict:
    """Refuse mismatched or incomplete provenance; never equate training-objective losses."""
    controls = validate_suite(suite)
    if comparison not in COMPARISONS:
        raise ValueError("only B/C comparisons are active")
    names = COMPARISONS[comparison]
    selected = []
    for name in names:
        if name not in reports:
            raise ValueError(f"comparison requires the completed {name} run")
        report = reports[name]
        expected = {
            "suite_identity": suite["identity"],
            "candidate": name,
            "data_fingerprint": suite["data_fingerprint"],
            "tokenizer_sha256": suite["tokenizer_sha256"],
            "objective": OBJECTIVE,
            "model": suite["candidates"][name]["model"],
            "controls": controls.to_dict(),
        }
        if (
            report.get("provenance") != expected
            or type(report.get("step")) is not int
            or report.get("step") != controls.max_steps
            or report.get("status") != "complete"
        ):
            raise ValueError(f"{name} has mismatched provenance or an incomplete budget")
        if not re.fullmatch(
            r"[0-9a-f]{64}", str(report.get("checkpoint_sha256", ""))
        ) or not report.get("validation"):
            raise ValueError("comparison requires checkpoint integrity and aligned validation")
        if report.get("profile") != suite["candidates"][name]["profile"]:
            raise ValueError("reported profile differs from the frozen candidate")
        for key in ("data_rng_sha256", "noise_rng_sha256", "code_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(report.get(key, ""))):
                raise ValueError(f"comparison requires valid {key}")
        runtime = report.get("runtime", {})
        if runtime.get("device") != "cpu" or runtime.get("precision") != "fp32":
            raise ValueError("comparison runtime differs from the qualified CPU FP32 adapter")
        if report["validation"].get("evaluation_contract") != evaluation_contract(
            controls, suite["masking"]
        ):
            raise ValueError("validation/generation evaluation controls are not aligned")
        if [row.get("noise_level") for row in report["validation"].get("rows", [])] != list(
            NOISE_LEVELS
        ):
            raise ValueError("required validation noise levels are missing")
        selected.append(report)
    for key in ("tokens_seen", "data_rng_sha256", "noise_rng_sha256", "runtime", "code_sha256"):
        if selected[0].get(key) is None or selected[0].get(key) != selected[1].get(key):
            raise ValueError(f"comparison has unmatched {key}")
    if not selected[0]["validation"].get("evaluation_contract") or selected[0]["validation"].get(
        "evaluation_contract"
    ) != selected[1]["validation"].get("evaluation_contract"):
        raise ValueError("validation/generation evaluation controls are not aligned")
    return {
        "comparison": comparison,
        "candidates": list(names),
        "provenance_matched": True,
        "canonical_winner": None,
        "reason": "Quality/health/backend evidence requires review; no automatic promotion.",
    }


def evaluation_contract(controls: ExperimentControls, masking: dict) -> dict:
    return {
        "validation_seed": controls.validation_seed,
        "batches": controls.validation_batches,
        "batch_size": controls.batch_size,
        "context_length": controls.context_length,
        "noise_levels": list(NOISE_LEVELS),
        "masking": masking,
        "generation_cases": list(GENERATION_CASES),
        "seeds": list(controls.generation_seeds),
        "slots": controls.generation_slots,
        "steps": controls.sampler_steps,
        "temperature": controls.temperature,
        "top_k": controls.top_k,
    }
