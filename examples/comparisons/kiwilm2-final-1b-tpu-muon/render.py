"""Render complete measured artifacts without training or reevaluating checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent


def main() -> None:
    summary = json.loads((OUT / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("all analysis stages must finish before rendering the final report")
    rows = [json.loads(line)
            for line in (OUT / "generation-results.jsonl").read_text().splitlines()]
    if len(rows) != 240 or len(summary["models"]) != 2:
        raise ValueError("expected two models and all 240 generation samples")
    report = ["# Matched five-seed generation samples", "",
              "Two exact-budget Dense checkpoints; 12 prompts x two profiles x seeds 42-46.",
              "FP32 MPS, 512 context, cache off, 160 new-token cap. All samples are retained.", ""]
    for row in rows:
        report += [f"## {row['model']} / {row['case_id']} / {row['profile']} / seed {row['seed']}",
                   "", row["text"], "",
                   f"Repeated four-gram rate: {row['repeated_four_gram_rate']:.4f}; "
                   f"maximum identical-word run: {row['maximum_consecutive_word_run']}.", ""]
    (OUT / "generation-report.md").write_text("\n".join(report))
    health = ["# Matched block-health distributions", "",
              "100 FP32 MPS batches per checkpoint, batch 2, context 512; "
              "50 batches each from data seeds 141 and 142.", "",
              "| Model | Block | Mixer | MLP | Amplification median | p90 | Maximum | "
              "Batches above 1.5 | Update/residual median |", "| --- | ---: | --- | --- | "
              "---: | ---: | ---: | ---: | ---: |"]
    for label, record in summary["models"].items():
        for block in record["health"]["blocks"]:
            amp = block["metrics"]["residual_amplification"]
            contribution = block["metrics"]["mlp_update_to_residual_rms"]
            health.append(f"| {label} | {block['index']} | {block['mixer']} | "
                          f"{block['mlp_type']} | "
                          f"{amp['median']:.3f} | {amp['p90']:.3f} | {amp['maximum']:.3f} | "
                          f"{amp['threshold_failure_count']} | {contribution['median']:.3f} |")
    (OUT / "health-report.md").write_text("\n".join(health) + "\n")

    print("Rendered generation-report.md and health-report.md; raw loss curves are in summary.json")


if __name__ == "__main__":
    main()
