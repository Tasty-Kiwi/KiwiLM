"""Reproduce the local, read-only audit of the downloaded TPU 50M checkpoint.

Run from the repository root: uv run python examples/comparisons/kiwilm2-tpu-50m-smoke/audit.py
The untied model is diagnostic only: it does not repair or overwrite weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

import torch
from torch.nn import functional as F

from kiwilm.comparison import generation_quality_metrics
from kiwilm.config import ModelConfig
from kiwilm.data import PreparedTokenData
from kiwilm.diagnostics import cached_generation_parity_report
from kiwilm.generation import generate
from kiwilm.models import build_model
from kiwilm.optim import split_muon_parameters


def main() -> None:
    torch.set_num_threads(4)
    source = Path("runs/colab/tpu-v5e1-muon-smoke-50m")
    output = Path(__file__).parent
    manifest = json.loads((source / "artifact-manifest.json").read_text())
    integrity = {}
    for name, expected in manifest["files"].items():
        path = source / name
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        integrity[name] = {
            "sha256": digest,
            "passed": digest == expected["sha256"]
            and path.stat().st_size == expected["bytes"],
        }
    assert all(item["passed"] for item in integrity.values())
    remote = json.loads((source / "summary.json").read_text())
    payload = torch.load(source / "latest.pt", map_location="cpu", weights_only=True)
    config = ModelConfig.from_dict(payload["model_config"])
    data = PreparedTokenData("data/smollm-smoke")
    assert payload["data_fingerprint"] == remote["data_fingerprint"] == data.fingerprint
    assert remote["tokenizer_sha256"] == data.metadata["tokenizer"]["sha256"]
    assert payload["step"] == remote["step"] == 3052
    assert payload["training_state"]["tokens_seen"] == remote["tokens_seen"] == 50_000_000
    state = payload["model_state_dict"]
    embedding, head = state["token_embedding.weight"], state["lm_head.weight"]
    report = {
        "schema_version": 1,
        "decision": "do-not-promote: XLA trained untied embeddings/head",
        "source": str(source),
        "checkpoint": str(source / "latest.pt"),
        "integrity": integrity,
        "data_fingerprint": data.fingerprint,
        "tokenizer_sha256": remote["tokenizer_sha256"],
        "local_runtime": {"torch": torch.__version__, "device": "cpu", "threads": 4},
        "remote_summary": remote,
        "weight_tying": {
            "config_tie_embeddings": config.tie_embeddings,
            "checkpoint_weights_equal": torch.equal(embedding, head),
            "maximum_absolute_difference": float((embedding - head).abs().max()),
            "embedding_rms": float(embedding.square().mean().sqrt()),
            "head_rms": float(head.square().mean().sqrt()),
            "extra_parameters_if_untied": embedding.numel(),
        },
        "fixed_validation": {
            "seed": 43, "batch_size": 8, "context_length": 512,
            "batches": 50, "targets": 204800, "models": {},
        },
        "generation": [],
        "cache_parity": {},
    }
    metrics = [json.loads(line) for line in (source / "metrics.jsonl").read_text().splitlines()]
    training = [row for row in metrics if row.get("event") == "train"]
    steady = [row for row in training if not row["warmup"] and row["step"] < 3052]
    report["training_log_audit"] = {
        "rows": len(training),
        "unique_steps": len({row["step"] for row in training}),
        "all_losses_and_gradients_finite": all(
            math.isfinite(row["train_loss"]) and math.isfinite(row["gradient_norm"])
            and row["gradient_norm"] > 0 for row in training
        ),
        "steady_median_tokens_per_second": statistics.median(
            row["tokens_per_second"] for row in steady
        ),
        "validation": [row for row in metrics if row.get("event") == "validation"],
    }
    for mode in ("normal-tied-loader", "untied-diagnostic-reconstruction"):
        model = build_model(config)
        if mode.startswith("untied"):
            model.lm_head.weight = torch.nn.Parameter(torch.empty_like(model.lm_head.weight))
        model.load_state_dict(state)
        model.eval()
        muon, auxiliary = split_muon_parameters(model)
        report["weight_tying"][mode] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "muon_parameters": sum(parameter.numel() for parameter in muon),
            "adamw_parameters": sum(parameter.numel() for parameter in auxiliary),
            "head_in_muon": any(parameter is model.lm_head.weight for parameter in muon),
            "embedding_matches_checkpoint": torch.equal(model.token_embedding.weight, embedding),
            "head_matches_checkpoint": torch.equal(model.lm_head.weight, head),
        }
        for precision in ("fp32", "bf16"):
            generator = torch.Generator().manual_seed(43)
            losses = []
            batches = 50 if precision == "fp32" else 5
            for _ in range(batches):
                inputs, targets = data.get_batch(
                    "validation", batch_size=8, context_length=512, generator=generator,
                )
                with torch.no_grad(), torch.autocast(
                    "cpu", dtype=torch.bfloat16, enabled=precision == "bf16",
                ):
                    logits = model(inputs)
                    loss = F.cross_entropy(logits.float().reshape(-1, 32000), targets.reshape(-1))
                losses.append(float(loss))
            loss = statistics.mean(losses)
            result = {
                "validation_loss": loss, "perplexity": math.exp(loss),
                "batches": batches, "targets": batches * 8 * 512,
            }
            report["fixed_validation"]["models"][f"{mode}-{precision}"] = result
            print(mode, precision, result, flush=True)
            (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        for prompt in (
            "Once upon a time",
            "The capital of France is",
            "Explain why the sky is blue.",
        ):
            for cache in ("off", "auto"):
                text = generate(
                    model, data.tokenizer, prompt, max_new_tokens=64,
                    temperature=0.8, top_k=40, seed=42, cache=cache,
                )
                report["generation"].append({
                    "model": mode, "prompt": prompt, "cache": cache, "seed": 42,
                    "temperature": 0.8, "top_k": 40, "max_new_tokens": 64,
                    "text": text, "quality": generation_quality_metrics(text),
                })
        inputs, _ = data.get_batch(
            "validation", batch_size=1, context_length=512,
            generator=torch.Generator().manual_seed(141),
        )
        report["cache_parity"][mode] = cached_generation_parity_report(model, inputs)
        del model
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved", output / "audit.json", flush=True)


if __name__ == "__main__":
    main()
