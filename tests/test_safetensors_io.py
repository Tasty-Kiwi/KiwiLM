"""Portable Safetensors bundle coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from kiwilm.checkpoint import save_checkpoint
from kiwilm.config import KiwiLM2Config, KiwiLM2SlimConfig, KiwiLM2SlimV3Config
from kiwilm.data import PreparedTokenData, prepare_from_stories
from kiwilm.models import build_model
from kiwilm.safetensors_io import (
    MANIFEST_FILE,
    METADATA_FILE,
    MODEL_FILE,
    TOKENIZER_FILE,
    export_safetensors_bundle,
    load_safetensors_model,
    read_safetensors_metadata,
    sha256_file,
)


@pytest.mark.parametrize(
    ("config_type", "config_overrides"),
    [
        (KiwiLM2Config, {}),
        (KiwiLM2SlimConfig, {}),
        (KiwiLM2SlimV3Config, {"upper_swiglu_blocks": 3}),
        (KiwiLM2SlimV3Config, {"upper_swiglu_blocks": 4}),
        (
            KiwiLM2SlimV3Config,
            {"upper_swiglu_blocks": 4, "swiglu_residual_gate_init": 0.25},
        ),
        (
            KiwiLM2SlimV3Config,
            {"upper_swiglu_blocks": 4, "swiglu_residual_gate_init": 0.5},
        ),
    ],
)
def test_safetensors_export_round_trip_and_manifest(
    tmp_path: Path,
    config_type: type[KiwiLM2Config],
    config_overrides: dict[str, object],
) -> None:
    data_dir = tmp_path / "data"
    metadata = prepare_from_stories(
        data_dir,
        ["A small training story."],
        ["A small validation story."],
        vocab_size=300,
        min_frequency=1,
    )
    data = PreparedTokenData(data_dir)
    config = config_type(
        vocab_size=data.tokenizer.vocab_size,
        context_length=8,
        d_model=16,
        dropout=0.0,
        num_query_heads=2,
        num_kv_heads=1,
        swiglu_dim=32,
        bigram_buckets=16,
        trigram_buckets=16,
        **config_overrides,
    )
    model = build_model(config).eval()
    checkpoint = save_checkpoint(
        tmp_path / "best.pt",
        model=model,
        step=7,
        model_config=config,
        data_fingerprint=data.fingerprint,
        training_state={"tokens_seen": 123},
    )
    tokenizer_path = data_dir / metadata["tokenizer"]["file"]
    output_dir = tmp_path / "bundle"

    manifest = export_safetensors_bundle(
        checkpoint,
        output_dir,
        tokenizer_path=tokenizer_path,
        expected_data_fingerprint=data.fingerprint,
        expected_tokenizer_sha256=metadata["tokenizer"]["sha256"],
        variant="test-variant",
    )
    loaded, loaded_config = load_safetensors_model(
        output_dir,
        data_fingerprint=data.fingerprint,
        device=torch.device("cpu"),
    )

    assert loaded_config == config
    assert (output_dir / TOKENIZER_FILE).read_bytes() == tokenizer_path.read_bytes()
    assert json.loads((output_dir / METADATA_FILE).read_text())["tokens_seen"] == 123
    stored_manifest = json.loads((output_dir / MANIFEST_FILE).read_text())
    assert stored_manifest == manifest
    for name, details in manifest["files"].items():
        assert sha256_file(output_dir / name) == details["sha256"]
        assert (output_dir / name).stat().st_size == details["bytes"]
    embedded = read_safetensors_metadata(output_dir / MODEL_FILE)
    assert embedded["variant"] == "test-variant"
    assert embedded["checkpoint_sha256"] == sha256_file(checkpoint)
    inputs = torch.tensor([[2, 10, 11]], dtype=torch.long)
    with torch.no_grad():
        assert torch.equal(model(inputs), loaded(inputs))
    assert (
        loaded.token_embedding.weight.data_ptr()
        == loaded.lm_head.weight.data_ptr()
    )


def test_safetensors_export_and_load_reject_incompatible_data(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    metadata = prepare_from_stories(
        data_dir,
        ["training"],
        ["validation"],
        vocab_size=300,
        min_frequency=1,
    )
    data = PreparedTokenData(data_dir)
    config = KiwiLM2Config(
        vocab_size=data.tokenizer.vocab_size,
        context_length=8,
        d_model=16,
        dropout=0.0,
        num_query_heads=2,
        num_kv_heads=1,
        swiglu_dim=32,
        bigram_buckets=16,
        trigram_buckets=16,
    )
    checkpoint = save_checkpoint(
        tmp_path / "best.pt",
        model=build_model(config),
        step=1,
        model_config=config,
        data_fingerprint=data.fingerprint,
    )
    tokenizer_path = data_dir / metadata["tokenizer"]["file"]

    with pytest.raises(ValueError, match="fingerprint"):
        export_safetensors_bundle(
            checkpoint,
            tmp_path / "wrong",
            tokenizer_path=tokenizer_path,
            expected_data_fingerprint="f" * 64,
            variant="wrong",
        )

    bundle = tmp_path / "bundle"
    export_safetensors_bundle(
        checkpoint,
        bundle,
        tokenizer_path=tokenizer_path,
        expected_data_fingerprint=data.fingerprint,
        variant="valid",
    )
    with pytest.raises(ValueError, match="fingerprint"):
        load_safetensors_model(
            bundle,
            data_fingerprint="f" * 64,
            device=torch.device("cpu"),
        )
    with pytest.raises(FileExistsError):
        export_safetensors_bundle(
            checkpoint,
            bundle,
            tokenizer_path=tokenizer_path,
            variant="duplicate",
        )
