"""Reproducible multi-seed generation; run from the repository root."""

import hashlib
import json
import statistics
from pathlib import Path

import torch

from kiwilm.comparison import generation_quality_metrics
from kiwilm.data import PreparedTokenData
from kiwilm.generation import generate
from kiwilm.inference import load_trained_model

ROOT = Path(__file__).parent
OUT = ROOT / "extended-generation"


def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    source = json.loads((ROOT / "summary.json").read_text())
    suite = json.loads((ROOT / "generation-suite.json").read_text())
    for prompt in suite["prompts"]:
        prompt["category"] = "story"
    for name, prompt in (
        ("water_cycle", "Water evaporates from lakes and oceans when"),
        ("plant_science", "Plants need sunlight because"),
        ("history", "The printing press changed the way people shared information by"),
        ("technical", "A computer stores information in binary, which means"),
        ("procedure", "To make a cup of tea, first boil water. Next,"),
        ("explanation", "A map is useful when traveling to a new place because"),
    ):
        suite["prompts"].append({"id": name, "prompt": prompt, "category": "expository"})
    suite["seeds"] = list(range(42, 47))
    suite["cache"] = "off"
    suite["precision"] = "fp32"
    for profile in suite["sampling_profiles"]:
        profile.pop("seed", None)
    (OUT / "suite.json").write_text(json.dumps(suite, indent=2) + "\n")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = PreparedTokenData("data/smollm-architecture")
    assert data.metadata["tokenizer"] == source["evaluation_data"]["tokenizer"]
    rows = []
    for label, record in source["models"].items():
        checkpoint = Path(record["checkpoint"])
        with checkpoint.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == record["checkpoint_sha256"]
        model, _ = load_trained_model(checkpoint, data_fingerprint=None, device=device)
        for case in suite["prompts"]:
            for profile in suite["sampling_profiles"]:
                for seed in suite["seeds"]:
                    text = generate(
                        model,
                        data.tokenizer,
                        case["prompt"],
                        max_new_tokens=160,
                        context_length=512,
                        temperature=profile["temperature"],
                        top_k=profile["top_k"],
                        seed=seed,
                        device=device,
                        cache="off",
                    )
                    continuation = (
                        text[len(case["prompt"]) :] if text.startswith(case["prompt"]) else text
                    )
                    rows.append(
                        {
                            "model": label,
                            "checkpoint_sha256": record["checkpoint_sha256"],
                            "category": case["category"],
                            "case_id": case["id"],
                            "profile": profile["id"],
                            "seed": seed,
                            "text": text,
                            **generation_quality_metrics(continuation),
                        }
                    )
            (OUT / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            print(label, case["id"], len(rows), "/ 360", flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
    assert len(rows) == 360
    identities = {
        (row["model"], row["case_id"], row["profile"], row["seed"])
        for row in rows
    }
    assert len(identities) == len(rows)
    summaries = []
    for label in source["models"]:
        for category in ("all", "story", "expository"):
            for profile in ("all", "focused", "creative"):
                selected = [
                    r
                    for r in rows
                    if r["model"] == label
                    and (category == "all" or r["category"] == category)
                    and (profile == "all" or r["profile"] == profile)
                ]
                summaries.append(
                    {
                        "model": label,
                        "category": category,
                        "profile": profile,
                        "samples": len(selected),
                        "mean_repeated_four_gram_rate": statistics.mean(
                            r["repeated_four_gram_rate"] for r in selected
                        ),
                        "samples_repetition_over_0_5": sum(
                            r["repeated_four_gram_rate"] > 0.5 for r in selected
                        ),
                        "maximum_consecutive_word_run": max(
                            r["maximum_consecutive_word_run"] for r in selected
                        ),
                    }
                )
    (OUT / "summary.json").write_text(
        json.dumps({"device": str(device), "groups": summaries}, indent=2) + "\n"
    )
    report = [
        "# Extended generation samples",
        "",
        "12 prompts, two profiles, seeds 42-46; FP32, cache off, 160-token cap.",
        "",
    ]
    for row in rows:
        report += [
            f"## {row['model']} / {row['case_id']} / {row['profile']} / seed {row['seed']}",
            "",
            row["text"],
            "",
        ]
    (OUT / "samples.md").write_text("\n".join(report))
    print("Complete: 360 samples", flush=True)


if __name__ == "__main__":
    main()
