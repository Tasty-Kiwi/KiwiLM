"""Sweep decoding settings for the exact Dense Muon 500M checkpoint."""

import hashlib
import itertools
import json
import re
import statistics
from pathlib import Path

import torch

from kiwilm.comparison import generation_quality_metrics
from kiwilm.data import PreparedTokenData
from kiwilm.generation import generate
from kiwilm.inference import load_trained_model

ROOT = Path(__file__).parent
SOURCE = ROOT / "extended-generation"
OUT = ROOT / "decoding-sweep"
MODEL = "Dense Muon 0.01 500M"
PROFILES = (
    ("greedy", 0.0, None),
    ("t04-k20", 0.4, 20),
    ("t06-k20", 0.6, 20),
    ("t06-k40", 0.6, 40),
    ("t08-k40", 0.8, 40),
    ("t08-k80", 0.8, 80),
    ("t10-k40", 1.0, 40),
    ("t10-k80", 1.0, 80),
)


def diversity(text):
    words = re.findall(r"[\w']+", text.casefold())
    bigrams = list(itertools.pairwise(words))
    return {
        "word_count": len(words),
        "distinct_word_rate": len(set(words)) / len(words) if words else 0.0,
        "distinct_bigram_rate": len(set(bigrams)) / len(bigrams) if bigrams else 0.0,
    }


def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    suite = json.loads((SOURCE / "suite.json").read_text())
    base = [
        json.loads(line)
        for line in (SOURCE / "results.jsonl").read_text().splitlines()
        if json.loads(line)["model"] == MODEL
    ]
    source_profiles = {"t04-k20": "focused", "t08-k40": "creative"}
    rows = []
    for name, temperature, top_k in PROFILES:
        if name not in source_profiles:
            continue
        for row in base:
            if row["profile"] == source_profiles[name]:
                continuation = row["text"][
                    len(next(p["prompt"] for p in suite["prompts"] if p["id"] == row["case_id"])) :
                ]
                rows.append(
                    {
                        **row,
                        "profile": name,
                        "temperature": temperature,
                        "top_k": top_k,
                        "reused": True,
                        **diversity(continuation),
                    }
                )
    primary = json.loads((ROOT / "summary.json").read_text())
    record = primary["models"][MODEL]
    checkpoint = Path(record["checkpoint"])
    with checkpoint.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == record["checkpoint_sha256"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = PreparedTokenData("data/smollm-architecture")
    model, _ = load_trained_model(checkpoint, data_fingerprint=None, device=device)
    for name, temperature, top_k in PROFILES:
        if name in source_profiles:
            print(name, "reused", flush=True)
            continue
        for case in suite["prompts"]:
            for seed in suite["seeds"]:
                text = generate(
                    model,
                    data.tokenizer,
                    case["prompt"],
                    max_new_tokens=160,
                    context_length=512,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seed,
                    device=device,
                    cache="off",
                )
                continuation = (
                    text[len(case["prompt"]) :] if text.startswith(case["prompt"]) else text
                )
                rows.append(
                    {
                        "model": MODEL,
                        "checkpoint_sha256": record["checkpoint_sha256"],
                        "category": case["category"],
                        "case_id": case["id"],
                        "profile": name,
                        "temperature": temperature,
                        "top_k": top_k,
                        "seed": seed,
                        "text": text,
                        "reused": False,
                        **generation_quality_metrics(continuation),
                        **diversity(continuation),
                    }
                )
            (OUT / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            print(name, case["id"], len(rows), "/ 480", flush=True)
    assert len(rows) == 480
    assert len({(r["profile"], r["case_id"], r["seed"]) for r in rows}) == 480
    rows.sort(key=lambda r: (r["profile"], r["case_id"], r["seed"]))
    (OUT / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    summaries = []
    for name, temperature, top_k in PROFILES:
        for category in ("all", "story", "expository"):
            selected = [
                r
                for r in rows
                if r["profile"] == name and (category == "all" or r["category"] == category)
            ]
            summaries.append(
                {
                    "profile": name,
                    "temperature": temperature,
                    "top_k": top_k,
                    "category": category,
                    "samples": len(selected),
                    "mean_repeated_four_gram_rate": statistics.mean(
                        r["repeated_four_gram_rate"] for r in selected
                    ),
                    "median_repeated_four_gram_rate": statistics.median(
                        r["repeated_four_gram_rate"] for r in selected
                    ),
                    "samples_repetition_over_0_5": sum(
                        r["repeated_four_gram_rate"] > 0.5 for r in selected
                    ),
                    "mean_distinct_word_rate": statistics.mean(
                        r["distinct_word_rate"] for r in selected
                    ),
                    "mean_distinct_bigram_rate": statistics.mean(
                        r["distinct_bigram_rate"] for r in selected
                    ),
                    "maximum_consecutive_word_run": max(
                        r["maximum_consecutive_word_run"] for r in selected
                    ),
                }
            )
    metadata = {
        "model": MODEL,
        "device": str(device),
        "precision": "fp32",
        "cache": "off",
        "seeds": suite["seeds"],
        "max_new_tokens": 160,
        "prompt_count": len(suite["prompts"]),
        "sample_count": len(rows),
        "profiles": summaries,
    }
    (OUT / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (OUT / "suite.json").write_text(
        json.dumps(
            {
                "prompts": suite["prompts"],
                "seeds": suite["seeds"],
                "max_new_tokens": 160,
                "profiles": [{"id": n, "temperature": t, "top_k": k} for n, t, k in PROFILES],
            },
            indent=2,
        )
        + "\n"
    )
    print("Complete: 480 samples", flush=True)


if __name__ == "__main__":
    main()
