"""Run from the repository root; cross-budget evaluation, never training."""

import gc
import hashlib
import itertools
import json
import statistics
from pathlib import Path

import torch

from kiwilm.comparison import generation_quality_metrics
from kiwilm.data import PreparedTokenData
from kiwilm.diagnostics import (
    aggregate_health_reports,
    cached_generation_parity_report,
    model_health_report,
)
from kiwilm.generation import generate
from kiwilm.inference import load_trained_model
from kiwilm.retrieval import (
    evaluate_retrieval_model,
    write_retrieval_artifacts,
)
from kiwilm.training import evaluate

OUT = Path(__file__).parent
RUNS = {
    "Dense AdamW 250M": "runs/kiwilm2-architecture/kiwilm2-adamw",
    "Dense Muon 0.01 250M": "runs/kiwilm2-muon-0.01",
    "Dense Muon 0.01 500M": "runs/kiwilm2-final-500m-muon",
}


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def main():
    torch.set_num_threads(4)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = PreparedTokenData("data/smollm-architecture", seed=143)
    job = json.loads(Path(RUNS["Dense Muon 0.01 500M"], "job.json").read_text())
    assert job["tokenizer_sha256"] == data.metadata["tokenizer"]["sha256"]
    # Whole-data fingerprints intentionally differ at different training budgets.
    # Evaluate every model against the same verified local validation artifact.
    suite = json.loads(Path("eval/story-consistency-prompts.json").read_text())
    write("generation-suite.json", suite)
    retrieval_suite = json.loads(
        Path(
            "examples/comparisons/kiwilm2-architecture-dense-adamw-vs-muon-0.01"
            "/retrieval/suite.json"
        ).read_text()
    )
    summary = {
        "device": str(device),
        "precision": "fp32",
        "validation_seed": 143,
        "validation_batches": 200,
        "validation_batch_size": 2,
        "evaluation_data": data.metadata,
        "models": {},
    }
    retrieval = []
    generation_rows = []
    for label, directory in RUNS.items():
        print(label, "loading", flush=True)
        path = Path(directory) / "latest.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        record = {
            key: payload[key]
            for key in ("step", "model_config", "train_config", "data_fingerprint", "metrics")
        }
        record["tokens_seen"] = payload["training_state"]["tokens_seen"]
        expected_tokens = 500_000_000 if label.endswith("500M") else 250_000_000
        assert record["tokens_seen"] == expected_tokens
        assert record["train_config"]["seed"] == 42
        if label.endswith("500M"):
            assert record["data_fingerprint"] == job["data_fingerprint"]
        else:
            assert record["data_fingerprint"] == data.fingerprint
        if summary["models"]:
            first = next(iter(summary["models"].values()))
            assert record["model_config"] == first["model_config"]
        record["checkpoint"] = str(path)
        with path.open("rb") as stream:
            record["checkpoint_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        del payload
        rows = [
            json.loads(line)
            for line in (Path(directory) / "metrics.jsonl").read_text().splitlines()
        ]
        record["validation_curve"] = [r for r in rows if r.get("event") == "validation"]
        train = [r for r in rows if r.get("event") == "train"]
        record["logged_steps_strictly_increasing"] = all(
            a["step"] < b["step"] for a, b in itertools.pairwise(train)
        )
        record["throughput_by_step_range"] = {}
        for low, high in ((500, 11000), (11000, 30518)):
            rates = [
                r["valid_tokens_per_second"]
                for r in train
                if low <= r["step"] < high and r.get("padding_fraction", 0) == 0
            ]
            record["throughput_by_step_range"][f"{low}-{high}"] = (
                statistics.median(rates) if rates else None
            )
        record["maximum_logged_memory_bytes"] = max(
            r.get("accelerator_memory_bytes", 0) for r in train
        )
        model, config = load_trained_model(path, data_fingerprint=None, device=device)
        record["parameters"] = sum(p.numel() for p in model.parameters())
        record["fixed_validation"] = evaluate(
            model,
            data,
            batch_size=2,
            context_length=512,
            num_batches=200,
            device=device,
            generator=torch.Generator().manual_seed(143),
            precision="fp32",
        )
        print(label, record["fixed_validation"], "health", flush=True)
        reports = []
        for seed in (141, 142):
            health_data = PreparedTokenData("data/smollm-architecture", seed=seed)
            for _ in range(50):
                inputs, targets = health_data.get_batch(
                    "validation", batch_size=2, context_length=512, device=device
                )
                reports.append(model_health_report(model, inputs, targets))
        record["health"] = aggregate_health_reports(reports)
        record["cached_generation"] = cached_generation_parity_report(model, inputs)
        write(label.replace(" ", "-") + "-health-batches.json", reports)
        summary["models"][label] = record
        write("summary.json", summary)
        print(label, "retrieval", flush=True)
        retrieval.append(
            evaluate_retrieval_model(
                model,
                retrieval_suite,
                label=label,
                architecture=config.architecture,
                checkpoint=path,
                device=device,
                batch_size=4,
            )
        )
        write_retrieval_artifacts(OUT / "retrieval", suite=retrieval_suite, evaluations=retrieval)
        print(label, "generation", flush=True)
        for case in suite["prompts"]:
            for profile in suite["sampling_profiles"]:
                text = generate(
                    model,
                    data.tokenizer,
                    case["prompt"],
                    max_new_tokens=suite["max_new_tokens"],
                    context_length=512,
                    temperature=profile["temperature"],
                    top_k=profile["top_k"],
                    seed=profile["seed"],
                    device=device,
                    cache="off",
                )
                continuation = (
                    text[len(case["prompt"]) :] if text.startswith(case["prompt"]) else text
                )
                generation_rows.append(
                    {
                        "model": label,
                        "case_id": case["id"],
                        "profile": profile,
                        "text": text,
                        **generation_quality_metrics(continuation),
                    }
                )
        (OUT / "generation-results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in generation_rows)
        )
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
    print("complete", flush=True)


if __name__ == "__main__":
    main()
