"""Matched 500M/1B checkpoint analysis. Run from the repository root; never trains."""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

import torch

from kiwilm.colab_artifacts import file_sha256
from kiwilm.comparison import generation_quality_metrics
from kiwilm.config import ModelConfig
from kiwilm.data import PreparedTokenData
from kiwilm.diagnostics import (
    aggregate_health_reports,
    cached_generation_parity_report,
    model_health_report,
)
from kiwilm.generation import generate
from kiwilm.model_profile import profile_kiwilm2
from kiwilm.models import build_model
from kiwilm.retrieval import evaluate_retrieval_model, write_retrieval_artifacts
from kiwilm.training import evaluate

OUT = Path(__file__).parent
PREVIOUS = OUT.parent / "kiwilm2-final-500m-muon"
RUNS = {
    "Dense Muon 0.01 500M": Path("runs/kiwilm2-final-500m-muon"),
    "Dense Muon 0.01 TPU 1B": Path("runs/colab/tpu-v6e1-final-1b-muon-resume1"),
}
FILES = {
    "Dense Muon 0.01 500M": "Dense-Muon-0.01-500M",
    "Dense Muon 0.01 TPU 1B": "Dense-Muon-0.01-TPU-1B",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    temporary.replace(path)


def row_key(row: dict) -> tuple:
    return row["model"], row["case_id"], row["profile"], row["seed"]


def generation_groups(rows: list[dict]) -> dict:
    groups = []
    for label in RUNS:
        for category in ("all", "story", "expository"):
            for profile in ("all", "focused", "creative"):
                selected = [r for r in rows if r["model"] == label
                            and (category == "all" or r["category"] == category)
                            and (profile == "all" or r["profile"] == profile)]
                if not selected:
                    continue
                groups.append({
                    "model": label, "category": category, "profile": profile,
                    "samples": len(selected),
                    "mean_repeated_four_gram_rate": statistics.mean(
                        r["repeated_four_gram_rate"] for r in selected),
                    "samples_repetition_over_0_5": sum(
                        r["repeated_four_gram_rate"] > 0.5 for r in selected),
                    "word_collapses_20_plus": sum(
                        r["maximum_consecutive_word_run"] >= 20 for r in selected),
                    "maximum_consecutive_word_run": max(
                        r["maximum_consecutive_word_run"] for r in selected),
                })
    labels = list(RUNS)
    indexed = {row_key(row): row for row in rows}
    differences = []
    for row in rows:
        if row["model"] != labels[0]:
            continue
        key = (labels[1], *row_key(row)[1:])
        if key in indexed:
            differences.append(indexed[key]["repeated_four_gram_rate"]
                               - row["repeated_four_gram_rate"])
    paired = ({"pairs": len(differences), "1b_less_repetition": sum(d < 0 for d in differences),
               "1b_more_repetition": sum(d > 0 for d in differences),
               "ties": sum(d == 0 for d in differences),
               "mean_change_1b_minus_500m": statistics.mean(differences),
               "median_change_1b_minus_500m": statistics.median(differences)}
              if differences else None)
    return {"groups": groups, "paired_repetition": paired}


def checked_payload(label: str, data: PreparedTokenData) -> tuple[dict, dict]:
    run = RUNS[label]
    checkpoint = run / "latest.pt"
    job_path = run / ("tpu-job.json" if label.endswith("1B") else "job.json")
    job = json.loads(job_path.read_text())
    require(job["tokenizer_sha256"] == data.metadata["tokenizer"]["sha256"],
            "checkpoint job tokenizer differs from the evaluation tokenizer")
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    expected = 1_000_000_000 if label.endswith("1B") else 500_000_000
    require(payload["training_state"]["tokens_seen"] == expected,
            "only exact-budget latest checkpoints are eligible")
    require(payload["data_fingerprint"] == job["data_fingerprint"],
            "checkpoint fingerprint differs from its original training job")
    config = payload["train_config"]
    require(config["seed"] == 42 and config["optimizer"] == "muon"
            and config["muon_lr"] == 0.01 and config["max_tokens"] == expected,
            "unexpected training seed/optimizer/budget")
    require(payload["model_config"]["architecture"] == "kiwilm2", "expected Dense architecture")
    require(torch.equal(payload["model_state_dict"]["token_embedding.weight"],
                        payload["model_state_dict"]["lm_head.weight"]), "untied saved head")
    sha = file_sha256(checkpoint)
    if label.endswith("1B"):
        remote = json.loads((run / "summary.json").read_text())
        require(remote["status"] == "final-complete" and remote["drive_backup"]["verified"]
                and sha == remote["drive_backup"]["checkpoint_sha256"],
                "1B checkpoint differs from the verified final Drive receipt")
        recipe = job["data_recipe"]
        require(recipe["resolved_revision"] == data.metadata["dataset"]["resolved_revision"]
                and recipe["seed"] == 42 and recipe["fineweb_probability"] == 0.7
                and recipe["validation_documents_per_source"] == 10_000,
                "1B validation recipe differs from the local frozen evaluation recipe")
    else:
        require(job["resolved_dataset_revision"] == data.metadata["dataset"]["resolved_revision"]
                and job["seed"] == 42 and job["fineweb_probability"] == 0.7
                and job["validation_documents_per_source"] == 10_000,
                "500M validation recipe differs from the local frozen evaluation recipe")
    return payload, {"checkpoint": str(checkpoint), "checkpoint_sha256": sha,
                     "training_job": job}


def reconstruct(payload: dict, device: torch.device):
    config = ModelConfig.from_dict(payload["model_config"])
    model = build_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    require(model.lm_head.weight is model.token_embedding.weight, "device transfer broke tying")
    return model


def training_record(label: str) -> dict:
    run = RUNS[label]
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    train = [r for r in rows if r.get("event") == "train"]
    result = {"logged_validation_curve": [r for r in rows if r.get("event") == "validation"],
              "logged_training_rows": len(train), "steps_strictly_increasing": all(
                  a["step"] < b["step"] for a, b in itertools.pairwise(train)),
              "all_losses_finite": all(math.isfinite(r["train_loss"]) for r in train)}
    if label.endswith("1B"):
        runtime = json.loads((run / "summary.json").read_text())
        steady = [r for r in train if not r["warmup"] and r["step"] < 61036]
        result.update(
            training_device="TPU v6e-1", precision="bf16", runtime_summary=runtime,
            weighted_full_run_steady_tokens_per_second=(
                16384 * len(steady) / sum(r["step_seconds"] for r in steady)),
            median_full_run_steady_tokens_per_second=statistics.median(
                r["tokens_per_second"] for r in steady),
            all_steps_present=all(r["step"] == i + 1 for i, r in enumerate(train)),
            tokens_exact_per_step=all(r["tokens_seen"] == min(r["step"] * 16384, 1_000_000_000)
                                      for r in train),
            peak_memory_bytes=runtime["memory"]["peak_bytes_used"],
        )
    else:
        rates = [r["valid_tokens_per_second"] for r in train
                 if r.get("padding_fraction", 0) == 0 and r["step"] >= 500
                 and "valid_tokens_per_second" in r and r["step"] < 30518]
        result.update(training_device="mixed GPU sessions; not a matched idle TPU benchmark",
                      precision="fp16", median_logged_tokens_per_second=statistics.median(rates),
                      peak_logged_memory_bytes=max(r.get("accelerator_memory_bytes", 0)
                                                   for r in train))
    return result


def core(data: PreparedTokenData, device: torch.device, suite: dict) -> dict:
    summary = {
        "schema_version": 1, "status": "running", "device": str(device), "precision": "fp32",
        "validation_seed": 143, "validation_batches": 200, "validation_batch_size": 2,
        "health_seeds": [141, 142], "health_batches_per_seed": 50,
        "context_length": 512, "evaluation_data": data.metadata,
        "retrieval_suite_source": (
            "examples/comparisons/kiwilm2-final-500m-muon/retrieval/suite.json"),
        "external_transfer": {
            "status": "not measured: no local prepared TinyStories/SimpleStories evaluation data",
            "generation_prompts_are_not_external_transfer_loss": True,
        },
        "models": {},
    }
    retrieval = []
    for label in RUNS:
        print(label, "provenance and loading", flush=True)
        payload, provenance = checked_payload(label, data)
        record = {key: payload[key] for key in ("step", "model_config", "train_config",
                                              "data_fingerprint", "metrics")}
        record.update(provenance)
        record["tokens_seen"] = payload["training_state"]["tokens_seen"]
        if summary["models"]:
            require(record["model_config"] == next(iter(summary["models"].values()))[
                "model_config"], "backbones differ")
        model = reconstruct(payload, device)
        del payload
        record["profile"] = profile_kiwilm2(model)
        record["training"] = training_record(label)
        print(label, "fixed validation: 200 batches", flush=True)
        record["fixed_validation"] = evaluate(
            model, data, batch_size=2, context_length=512, num_batches=200, device=device,
            generator=torch.Generator().manual_seed(143), precision="fp32")
        print(label, record["fixed_validation"], "health: 100 batches", flush=True)
        reports = []
        for seed in (141, 142):
            generator = torch.Generator().manual_seed(seed)
            for batch in range(50):
                inputs, targets = data.get_batch("validation", batch_size=2, context_length=512,
                                                generator=generator, device=device)
                reports.append({"data_seed": seed, "batch_index": batch,
                                **model_health_report(model, inputs, targets)})
                if (batch + 1) % 25 == 0:
                    print(label, "health", seed, batch + 1, "/ 50", flush=True)
        record["health"] = aggregate_health_reports(reports)
        record["cached_generation"] = cached_generation_parity_report(model, inputs)
        write(OUT / (FILES[label] + "-health-batches.json"), reports)
        summary["models"][label] = record
        write(OUT / "summary.json", summary)
        print(label, "full-context retrieval", flush=True)
        evaluation = evaluate_retrieval_model(
            model, suite, label=label, architecture="kiwilm2",
            checkpoint=Path(record["checkpoint"]),
            device=device, batch_size=4)
        retrieval.append(evaluation)
        write_retrieval_artifacts(OUT / "retrieval", suite=suite, evaluations=retrieval,
                                  title="Dense Muon scaling: 500M GPU versus 1B TPU")
        summary["retrieval"] = [r["summary"] for r in retrieval]
        write(OUT / "summary.json", summary)
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
    summary["status"] = "core-complete"
    write(OUT / "summary.json", summary)
    return summary


def generations(data: PreparedTokenData, device: torch.device, summary: dict, suite: dict) -> None:
    require(summary["device"] == str(device), "do not mix evaluation devices")
    path = OUT / "generation-results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    identities = {row_key(row) for row in rows}
    require(len(identities) == len(rows), "duplicate generation rows")
    for label in RUNS:
        payload, provenance = checked_payload(label, data)
        require(provenance["checkpoint_sha256"] == summary["models"][label]["checkpoint_sha256"],
                "checkpoint changed since core evaluation")
        model = reconstruct(payload, device)
        del payload
        for case in suite["prompts"]:
            for profile in suite["sampling_profiles"]:
                for seed in suite["seeds"]:
                    identity = (label, case["id"], profile["id"], seed)
                    if identity in identities:
                        existing = next(r for r in rows if row_key(r) == identity)
                        require(existing["checkpoint_sha256"] == provenance["checkpoint_sha256"]
                                and existing["prompt"] == case["prompt"]
                                and existing["temperature"] == profile["temperature"]
                                and existing["top_k"] == profile["top_k"],
                                "partial generation output has different provenance")
                        continue
                    text = generate(
                        model, data.tokenizer, case["prompt"],
                        max_new_tokens=160, context_length=512,
                        temperature=profile["temperature"], top_k=profile["top_k"], seed=seed,
                        device=device, cache="off")
                    continuation = text[len(case["prompt"]):] if text.startswith(
                        case["prompt"]) else text
                    rows.append({"model": label,
                                 "checkpoint_sha256": provenance["checkpoint_sha256"],
                                 "category": case["category"], "case_id": case["id"],
                                 "prompt": case["prompt"], "profile": profile["id"], "seed": seed,
                                 "temperature": profile["temperature"], "top_k": profile["top_k"],
                                 "text": text, **generation_quality_metrics(continuation)})
                    identities.add(identity)
                    write_rows(path, rows)
            print(label, case["id"], len(rows), "/ 240 samples", flush=True)
        del model
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
    expected = {(label, case["id"], profile["id"], seed) for label in RUNS
                for case in suite["prompts"] for profile in suite["sampling_profiles"]
                for seed in suite["seeds"]}
    require(identities == expected, "generation suite is incomplete or has unrequested rows")
    summary["generation"] = generation_groups(rows)
    summary["status"] = "complete"
    write(OUT / "summary.json", summary)
    write(OUT / "generation-summary.json", summary["generation"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("core", "generation", "all"), default="all")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; choose CPU explicitly, not a silent device change")
    device = torch.device(args.device)
    data = PreparedTokenData("data/smollm-architecture", seed=143)
    suite = json.loads((PREVIOUS / "retrieval/suite.json").read_text())
    compact = hashlib.sha256(data.tokenizer.to_json(pretty=False).encode()).hexdigest()
    require(suite["tokenizer"]["sha256"] == compact and suite["context_length"] == 512,
            "retrieval tokenizer or context mismatch")
    generation_suite = json.loads((PREVIOUS / "extended-generation/suite.json").read_text())
    write(OUT / "generation-suite.json", generation_suite)
    summary = (core(data, device, suite) if args.stage in {"core", "all"}
               else json.loads((OUT / "summary.json").read_text()))
    if args.stage in {"generation", "all"}:
        generations(data, device, summary, generation_suite)
    print("Analysis stage complete:", args.stage, flush=True)


if __name__ == "__main__":
    main()
