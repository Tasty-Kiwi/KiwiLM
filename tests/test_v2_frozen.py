"""Pre-cleanup V2 golden outputs; real artifact checks are offline and optional."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from kiwilm.checkpoint import save_checkpoint
from kiwilm.config import ModelConfig
from kiwilm.generation import generate_tokens
from kiwilm.inference import load_trained_model
from kiwilm.models import build_model

ROOT = Path(__file__).resolve().parents[1]
FROZEN = json.loads((ROOT / "tests/fixtures/kiwilm2_frozen.json").read_text())


@pytest.mark.parametrize("variant", FROZEN["variants"])
def test_v2_frozen_schema_outputs_checkpoint_and_rollover(tmp_path: Path, variant: dict) -> None:
    config = ModelConfig.from_dict(FROZEN["config"] | variant["config"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(FROZEN["seed"])
        model = build_model(config).eval()
    schema = [[name, list(value.shape)] for name, value in model.state_dict().items()]
    digest = hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode()).hexdigest()
    assert digest == variant["state_schema_sha256"]
    path = save_checkpoint(
        tmp_path / "v2.pt", model=model, model_config=config, step=7, data_fingerprint="frozen-v2"
    )
    restored, restored_config = load_trained_model(
        path,
        data_fingerprint="frozen-v2",
        device=torch.device("cpu"),
    )
    assert restored_config.to_dict() == config.to_dict()
    assert restored.lm_head.weight is restored.token_embedding.weight
    inputs = torch.tensor([FROZEN["input_ids"]])
    with torch.inference_mode():
        logits = restored(inputs)
        torch.testing.assert_close(logits, model(inputs), rtol=0, atol=0)
        torch.testing.assert_close(
            logits[0, -1, :8], torch.tensor(variant["last_logits"]), rtol=1e-5, atol=1e-7
        )
        for mode in ("auto", "off"):
            generated = generate_tokens(
                restored, inputs, max_new_tokens=3, temperature=0, cache=mode
            )
            assert generated[0].tolist() == variant["greedy_ids"]


def verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_v2",
        ROOT / "scripts/verify_kiwilm2_reference.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("kind", "variable", "default"),
    [
        (
            "training_checkpoint",
            "KIWILM_V2_CHECKPOINT",
            "runs/colab/tpu-v6e1-final-1b-muon-resume1/latest.pt",
        ),
        ("bf16_bundle", "KIWILM_V2_BUNDLE", "artifacts/huggingface/KiwiLM-2-bf16"),
    ],
)
def test_original_1b_artifact_against_pre_cleanup_reference(kind, variable, default) -> None:
    source = Path(os.environ.get(variable, str(ROOT / default)))
    if not source.exists():
        if variable in os.environ:
            pytest.fail(f"explicit {variable} path is missing")
        pytest.skip(f"original V2 artifact not present; set {variable} to verify offline")
    assert verifier().verify_reference(source, kind)["status"] == "passed"


def test_reference_verifier_rejects_different_weights(tmp_path: Path) -> None:
    source = tmp_path / "other.pt"
    source.write_bytes(b"not the frozen checkpoint")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verifier().verify_reference(source, "training_checkpoint")


@pytest.mark.skipif(shutil.which("bash") is None, reason="shell syntax requires Bash")
def test_all_archived_shell_launchers_have_valid_syntax() -> None:
    scripts = sorted((ROOT / "archive/kiwilm2/scripts").glob("*.sh"))
    assert len(scripts) == 11
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)
