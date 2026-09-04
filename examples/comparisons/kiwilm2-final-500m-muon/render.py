"""Render measured comparison artifacts; run after evaluate.py finishes."""

import html
import json
import statistics
from pathlib import Path

from kiwilm.config import ModelConfig
from kiwilm.model_profile import profile_kiwilm2
from kiwilm.models import build_model

OUT = Path(__file__).parent
summary = json.loads((OUT / "summary.json").read_text())
rows = [json.loads(line) for line in (OUT / "generation-results.jsonl").read_text().splitlines()]
retrieval = [
    json.loads(line) for line in (OUT / "retrieval/results.jsonl").read_text().splitlines()
]
assert len(summary["models"]) == len(retrieval) == 3
assert len(rows) == 36
generation = {}
report = [
    "# Fixed generation comparison",
    "",
    "FP32, cache off, seed 42; six prompts x two sampling profiles per model.",
    "",
]
for label in summary["models"]:
    selected = [row for row in rows if row["model"] == label]
    generation[label] = {
        "samples": len(selected),
        "mean_repeated_four_gram_rate": statistics.mean(
            row["repeated_four_gram_rate"] for row in selected
        ),
        "maximum_consecutive_word_run": max(
            row["maximum_consecutive_word_run"] for row in selected
        ),
    }
    report += [f"## {label}", ""]
    for row in selected:
        report += [f"### {row['case_id']} / {row['profile']['id']}", "", row["text"], ""]
(OUT / "generation-report.md").write_text("\n".join(report))
summary["generation"] = generation
summary["retrieval"] = [item["summary"] for item in retrieval]
configs = [item["model_config"] for item in summary["models"].values()]
assert all(config == configs[0] for config in configs)
summary["static_profile"] = profile_kiwilm2(build_model(ModelConfig.from_dict(configs[0])))
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

# A dependency-free SVG shows the actual logged validation points, not a fitted curve.
svg = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="480" viewBox="0 0 920 480">',
    '<rect width="920" height="480" fill="white"/>',
    '<g font-family="sans-serif" font-size="13" fill="#222">',
    '<text x="70" y="25" font-size="18">Dense scaling: logged validation loss</text>',
]
for tick in range(0, 501, 100):
    x = 70 + tick * 1.6
    svg += [f'<path d="M{x} 50 V390" stroke="#eee"/>', f'<text x="{x}" y="410">{tick}M</text>']
for tick in (3.5, 4, 4.5, 5, 5.5, 6):
    y = 390 - (tick - 3.4) * 120
    svg += [f'<path d="M70 {y} H870" stroke="#eee"/>', f'<text x="30" y="{y}">{tick}</text>']
for index, ((label, model), color) in enumerate(
    zip(summary["models"].items(), ("#888", "#2471a3", "#c0392b"), strict=True)
):
    points = " ".join(
        f"{70 + min(row['step'] * 16384, model['tokens_seen']) / 1e6 * 1.6:.1f},"
        f"{390 - (row['validation_loss'] - 3.4) * 120:.1f}"
        for row in model["validation_curve"]
        if row["validation_loss"] <= 6.2
    )
    svg += [
        f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>',
        f'<text x="{70 + index * 270}" y="450" fill="{color}">{html.escape(label)}</text>',
    ]
svg += ["</g></svg>"]
(OUT / "validation-curve.svg").write_text("\n".join(svg))
print(json.dumps({"generation": generation, "retrieval": summary["retrieval"]}, indent=2))
