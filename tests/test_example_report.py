"""Single-checkpoint example report coverage."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from kiwilm.checkpoint import save_checkpoint
from kiwilm.config import GatedCNNConfig
from kiwilm.data import PreparedTokenData, prepare_from_stories
from kiwilm.example_report import generate_example_report
from kiwilm.models import build_model


def test_generate_example_report_writes_every_prompt_profile_pair(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    prepare_from_stories(
        data_dir,
        ["Once there was a bird.", "The bird found a friend."] * 3,
        ["They went home."] * 2,
        vocab_size=300,
        min_frequency=1,
    )
    data = PreparedTokenData(data_dir, seed=7)
    config = GatedCNNConfig(
        vocab_size=data.tokenizer.vocab_size,
        context_length=8,
        d_model=8,
        dropout=0.0,
        num_layers=1,
        dilations=(1,),
    )
    checkpoint = save_checkpoint(
        tmp_path / "model.pt",
        model=build_model(config),
        step=1,
        model_config=config,
        data_fingerprint=data.fingerprint,
    )
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "suite_version": 1,
                "max_new_tokens": 2,
                "sampling_profiles": [
                    {
                        "id": "focused",
                        "temperature": 0.4,
                        "top_k": 20,
                        "seed": 42,
                    },
                    {
                        "id": "creative",
                        "temperature": 0.8,
                        "top_k": 40,
                        "seed": 42,
                    },
                ],
                "prompts": [
                    {"id": "opening", "prompt": "Once"},
                    {"id": "bird", "prompt": "The bird"},
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "examples" / "model.md"

    summary = generate_example_report(
        checkpoint,
        data=data,
        suite_path=suite,
        output_path=output,
        device=torch.device("cpu"),
        title="Tiny model examples",
    )

    report = output.read_text(encoding="utf-8")
    assert summary["generation_count"] == 4
    assert report.startswith("# Tiny model examples\n")
    assert report.count("## ") == 4
    assert "## opening / focused" in report
    assert "## bird / creative" in report
