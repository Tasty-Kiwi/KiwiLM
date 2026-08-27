"""A/B generation report coverage."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from kiwilm.checkpoint import save_checkpoint
from kiwilm.comparison import compare_checkpoints
from kiwilm.config import KiwiLM2Config, KiwiLM2SlimConfig
from kiwilm.data import PreparedTokenData, prepare_from_stories
from kiwilm.models import build_model


def test_compare_checkpoints_writes_reproducible_machine_and_human_reports(
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
    common = {
        "vocab_size": data.tokenizer.vocab_size,
        "context_length": 8,
        "d_model": 8,
        "dropout": 0.0,
    }
    model_a_config = KiwiLM2Config(
        **common,
        num_query_heads=1,
        num_kv_heads=1,
        swiglu_dim=16,
        bigram_buckets=16,
        trigram_buckets=16,
    )
    model_b_config = KiwiLM2SlimConfig(
        **common,
        num_query_heads=1,
        num_kv_heads=1,
        swiglu_dim=16,
        bigram_buckets=16,
        trigram_buckets=16,
    )
    model_c_config = KiwiLM2Config(
        **common,
        num_query_heads=1,
        num_kv_heads=1,
        swiglu_dim=24,
        bigram_buckets=16,
        trigram_buckets=16,
    )
    checkpoint_a = save_checkpoint(
        tmp_path / "a.pt",
        model=build_model(model_a_config),
        step=1,
        model_config=model_a_config,
        data_fingerprint=data.fingerprint,
    )
    checkpoint_b = save_checkpoint(
        tmp_path / "b.pt",
        model=build_model(model_b_config),
        step=1,
        model_config=model_b_config,
        data_fingerprint=data.fingerprint,
    )
    checkpoint_c = save_checkpoint(
        tmp_path / "c.pt",
        model=build_model(model_c_config),
        step=1,
        model_config=model_c_config,
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
                        "id": "greedy",
                        "temperature": 0,
                        "top_k": 0,
                        "seed": 9,
                    }
                ],
                "prompts": [{"id": "opening", "prompt": "Once"}],
            }
        ),
        encoding="utf-8",
    )

    summary = compare_checkpoints(
        checkpoint_a,
        checkpoint_b,
        data=data,
        suite_path=suite,
        output_dir=tmp_path / "comparison",
        device=torch.device("cpu"),
        label_a="KiwiLM 2 Dense",
        label_b="KiwiLM 2 Slim",
    )

    rows = [
        json.loads(line)
        for line in Path(summary["results_path"]).read_text(encoding="utf-8").splitlines()
    ]
    report = Path(summary["report_path"]).read_text(encoding="utf-8")
    assert summary["generation_count"] == 2
    assert [row["architecture"] for row in rows] == [
        "kiwilm2",
        "kiwilm2_slim",
    ]
    assert "| KiwiLM 2 Dense | KiwiLM 2 Slim |" in report
    assert "opening / greedy" in report

    n_way_summary = compare_checkpoints(
        [checkpoint_a, checkpoint_b, checkpoint_c],
        data=data,
        suite_path=suite,
        output_dir=tmp_path / "n-way-comparison",
        device=torch.device("cpu"),
        labels=["Dense A", "Slim", "Dense B"],
    )
    n_way_rows = [
        json.loads(line)
        for line in Path(n_way_summary["results_path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    n_way_report = Path(n_way_summary["report_path"]).read_text(encoding="utf-8")

    assert n_way_summary["generation_count"] == 3
    assert [row["architecture"] for row in n_way_rows] == [
        "kiwilm2",
        "kiwilm2_slim",
        "kiwilm2",
    ]
    assert "| Dense A | Slim | Dense B |" in n_way_report
