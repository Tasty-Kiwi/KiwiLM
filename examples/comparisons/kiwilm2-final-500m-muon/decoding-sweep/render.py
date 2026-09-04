"""Render the decoding sweep's complete sample report and SVG plot."""

import html
import json
from pathlib import Path

ROOT = Path(__file__).parent
summary = json.loads((ROOT / "summary.json").read_text())
rows = [json.loads(line) for line in (ROOT / "results.jsonl").read_text().splitlines()]
report = [
    "# Dense Muon 500M decoding-sweep samples",
    "",
    "Every output is retained. The two existing profiles are exact reused samples.",
    "",
]
for row in rows:
    report += [
        f"## {row['profile']} / {row['case_id']} / seed {row['seed']}",
        "",
        (
            f"Temperature {row['temperature']}; top-k {row['top_k']}; "
            f"repeated 4-gram rate {row['repeated_four_gram_rate']:.4f}; "
            f"distinct bigram rate {row['distinct_bigram_rate']:.4f}."
        ),
        "",
        row["text"],
        "",
    ]
(ROOT / "samples.md").write_text("\n".join(report))

profiles = [item for item in summary["profiles"] if item["category"] == "all"]
svg = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="520" viewBox="0 0 980 520">',
    '<rect width="980" height="520" fill="white"/>',
    '<g font-family="sans-serif" fill="#222">',
    '<text x="70" y="30" font-size="19">Dense Muon 500M decoding sweep</text>',
    '<text x="70" y="52" font-size="12">Lower repetition and higher '
    "distinct-bigram rate are favorable metrics, but do not measure coherence.</text>",
]
for tick in range(0, 11, 2):
    y = 430 - tick * 34
    svg += [
        f'<path d="M70 {y} H930" stroke="#eee"/>',
        f'<text x="35" y="{y + 4}" font-size="11">{tick / 10:.1f}</text>',
    ]
for index, item in enumerate(profiles):
    x = 90 + index * 105
    rep_y = 430 - item["mean_repeated_four_gram_rate"] * 340
    div_y = 430 - item["mean_distinct_bigram_rate"] * 340
    label = html.escape(item["profile"])
    svg += [
        f'<circle cx="{x}" cy="{rep_y}" r="6" fill="#c0392b"/>',
        f'<circle cx="{x}" cy="{div_y}" r="6" fill="#2471a3"/>',
        f'<text x="{x}" y="455" text-anchor="middle" font-size="11">{label}</text>',
    ]
svg += [
    '<circle cx="710" cy="30" r="5" fill="#c0392b"/>'
    '<text x="720" y="34" font-size="12">repeated 4-grams</text>',
    '<circle cx="840" cy="30" r="5" fill="#2471a3"/>'
    '<text x="850" y="34" font-size="12">distinct bigrams</text>',
    "</g></svg>",
]
(ROOT / "metrics.svg").write_text("\n".join(svg))
