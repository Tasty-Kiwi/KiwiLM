"""Independent export provenance and dataset-free, verified Hub loading."""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import kiwilm.cli as cli
import kiwilm.hub as hub
from kiwilm.checkpoint import save_checkpoint
from kiwilm.config import KiwiLM2Config
from kiwilm.data import prepare_from_stories
from kiwilm.models import build_model
from kiwilm.safetensors_io import (
    BUNDLE_FILES,
    export_provenance,
    export_safetensors_bundle,
    load_safetensors_model,
    sha256_file,
    verify_safetensors_bundle,
)


@pytest.fixture
def release_source(tmp_path: Path):
    data_dir = tmp_path / "old-budget"
    metadata = prepare_from_stories(
        data_dir, ["A small training story."], ["A validation story."],
        vocab_size=300, min_frequency=1,
    )
    config = KiwiLM2Config(
        vocab_size=metadata["tokenizer"]["vocab_size"], context_length=8,
        d_model=16, num_query_heads=2, num_kv_heads=1, swiglu_dim=32,
        bigram_buckets=16, trigram_buckets=16,
    )
    model = build_model(config).eval()
    checkpoint = save_checkpoint(
        tmp_path / "latest.pt", model=model, step=7, model_config=config,
        data_fingerprint="1b-job-fingerprint", training_state={"tokens_seen": 123},
    )
    job = tmp_path / "tpu-job.json"
    job.write_text(json.dumps({"data_fingerprint": "1b-job-fingerprint",
                               "tokenizer_sha256": metadata["tokenizer"]["sha256"]}))
    return data_dir, metadata, checkpoint, job, model


def export(source, output: Path) -> None:
    data_dir, metadata, checkpoint, job, _ = source
    provenance = export_provenance(metadata, job)
    export_safetensors_bundle(
        checkpoint, output, tokenizer_path=data_dir / metadata["tokenizer"]["file"],
        expected_data_fingerprint=provenance["checkpoint_data_fingerprint"],
        expected_tokenizer_sha256=provenance["tokenizer_sha256"],
        provenance=provenance, variant="test",
        dtype="fp32",
    )


def test_export_cli_separates_data_and_tokenizer_provenance(release_source, tmp_path):
    data_dir, metadata, checkpoint, job, _ = release_source
    base = ["export-safetensors", "--checkpoint", str(checkpoint),
            "--tokenizer-from", str(data_dir), "--output-dir", str(tmp_path / "bundle"),
            "--variant", "test"]
    with pytest.raises(ValueError, match="fingerprint"):
        cli.main(base)
    assert not (tmp_path / "bundle").exists()
    assert cli.main([*base, "--checkpoint-provenance", str(job)]) == 0
    stored = json.loads((tmp_path / "bundle/metadata.json").read_text())
    assert stored["data_fingerprint"] == "1b-job-fingerprint"
    assert stored["export_provenance"]["tokenizer_dataset_fingerprint"] == metadata["fingerprint"]
    assert stored["export_provenance"]["checkpoint_provenance_sha256"] == sha256_file(job)


def test_wrong_original_job_tokenizer_is_rejected(release_source):
    _, metadata, _, job, _ = release_source
    job.write_text(json.dumps({"data_fingerprint": "1b-job-fingerprint",
                               "tokenizer_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="tokenizer checksum"):
        export_provenance(metadata, job)


def test_bf16_is_export_default_and_native_loading_is_explicit(release_source, tmp_path):
    data_dir, _metadata, checkpoint, job, _ = release_source
    root = tmp_path / "bf16"
    assert cli.main([
        "export-safetensors", "--checkpoint", str(checkpoint), "--tokenizer-from", str(data_dir),
        "--checkpoint-provenance", str(job), "--output-dir", str(root), "--variant", "bf16",
    ]) == 0
    assert json.loads((root / "metadata.json").read_text())["weights_dtype"] == "bf16"
    fp32, _ = load_safetensors_model(root, data_fingerprint=None, device=torch.device("cpu"))
    native, _ = load_safetensors_model(
        root, data_fingerprint=None, device=torch.device("cpu"), dtype=torch.bfloat16,
    )
    assert next(fp32.parameters()).dtype == torch.float32
    assert next(native.parameters()).dtype == torch.bfloat16
    assert native.token_embedding.weight is native.lm_head.weight
    with torch.no_grad():
        assert torch.isfinite(native(torch.tensor([[2, 10, 11]]))).all()
    # Stored BF16 values survive both loading modes without extra rounding.
    for name, tensor in native.state_dict().items():
        assert torch.equal(tensor.float(), fp32.state_dict()[name])
    args = cli.build_parser().parse_args([
        "export-safetensors", "--checkpoint", "c.pt", "--tokenizer-from", "data",
        "--output-dir", "out", "--variant", "v", "--dtype", "fp32",
    ])
    assert args.dtype == "fp32"


@pytest.mark.parametrize("name", ("model.safetensors", "config.json", "tokenizer.json"))
def test_bundle_tampering_rejected_before_loading(release_source, tmp_path, name):
    root = tmp_path / "bundle"
    export(release_source, root)
    (root / name).write_bytes((root / name).read_bytes() + b" ")
    with pytest.raises(ValueError, match="integrity check"):
        load_safetensors_model(root, data_fingerprint=None, device=torch.device("cpu"))


def test_rehashed_config_still_must_agree_with_embedded_metadata(release_source, tmp_path):
    root = tmp_path / "bundle"
    export(release_source, root)
    config = json.loads((root / "config.json").read_text())
    config["context_length"] = 16
    (root / "config.json").write_text(json.dumps(config))
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"]["config.json"] = {
        "sha256": sha256_file(root / "config.json"), "bytes": (root / "config.json").stat().st_size,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="metadata disagrees"):
        verify_safetensors_bundle(root)


def test_hub_load_is_pinned_filtered_and_dataset_free(release_source, tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    export(release_source, root)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(root)

    monkeypatch.setattr(hub, "snapshot_download", download)
    model, tokenizer, config = hub.load_pretrained("owner/model", revision="a" * 40)
    assert calls == [{"repo_id": "owner/model", "repo_type": "model", "revision": "a" * 40,
                      "allow_patterns": list(BUNDLE_FILES), "cache_dir": None,
                      "local_files_only": False}]
    values = torch.tensor([[2, 10, 11]])
    with torch.no_grad():
        assert torch.equal(model(values), release_source[-1](values))
    assert tokenizer.vocab_size == config.vocab_size
    assert model.token_embedding.weight is model.lm_head.weight
    with pytest.raises(ValueError, match="commit SHA"):
        hub.load_pretrained("owner/model", revision="main")
    assert len(calls) == 1


def test_generate_from_bundle_does_not_load_data(release_source, tmp_path, monkeypatch, capsys):
    root = tmp_path / "bundle"
    export(release_source, root)

    def no_data(*args, **kwargs):
        raise AssertionError("dataset loading is forbidden for bundle inference")

    monkeypatch.setattr(cli, "load_prepared_data", no_data)
    assert cli.main(["generate", "--checkpoint", str(root), "--prompt", "Hi",
                     "--max-new-tokens", "4", "--top-k", "4", "--device", "cpu"]) == 0
    assert "Hi" in capsys.readouterr().out


@pytest.fixture
def publisher():
    path = Path(__file__).resolve().parents[1] / "scripts/publish_kiwilm2_release.py"
    spec = importlib.util.spec_from_file_location("release_publisher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("private", (True, False))
@pytest.mark.parametrize("draft", (True, False))
def test_publisher_creates_private_and_never_uploads_if_public(
    publisher, release_source, tmp_path, private, draft,
):
    root = tmp_path / "bundle"
    export(release_source, root)
    for name in publisher.RELEASE_FILES - set(BUNDLE_FILES) - {"release.json"}:
        (root / name).write_text("fixture")
    release = {
        "source_freeze_pending": draft, "source_tag": None if draft else "v2",
        "repo_id": "Tasty-Kiwi/KiwiLM-2",
        "files": {f.name: {"sha256": sha256_file(f), "bytes": f.stat().st_size}
                  for f in root.iterdir()},
    }
    (root / "release.json").write_text(json.dumps(release))
    calls = []
    api = SimpleNamespace(
        create_repo=lambda **kwargs: calls.append(("create", kwargs)),
        repo_info=lambda **kwargs: SimpleNamespace(private=private),
        upload_folder=lambda **kwargs: (calls.append(("upload", kwargs))
                                        or SimpleNamespace(oid="a" * 40)),
    )
    if private:
        assert publisher.publish(root, api, stage_private_draft=draft) == "a" * 40
        assert calls[1][1]["allow_patterns"] == sorted(publisher.RELEASE_FILES)
        assert ("draft" in calls[1][1]["commit_message"]) == draft
        assert json.loads((root / "release.json").read_text()) == release
    else:
        with pytest.raises(ValueError, match="non-private"):
            publisher.publish(root, api, stage_private_draft=draft)
        assert len(calls) == 1
    assert calls[0] == ("create", {"repo_id": "Tasty-Kiwi/KiwiLM-2", "repo_type": "model",
                                   "private": True, "exist_ok": False})


def test_publisher_refuses_unfrozen_draft_without_remote_calls(publisher, tmp_path):
    (tmp_path / "release.json").write_text(json.dumps({"source_freeze_pending": True}))
    with pytest.raises(ValueError, match="commit/tag"):
        publisher.publish(tmp_path, SimpleNamespace())


def test_release_wheel_must_match_source_not_just_filename(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_kiwilm2_release.py"
    spec = importlib.util.spec_from_file_location("release_preparer", path)
    preparer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparer)
    source = tmp_path / "kiwilm"
    source.mkdir()
    (source / "hub.py").write_text("# pinned source\n")
    wheel = tmp_path / "kiwilm.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("kiwilm/hub.py", "# pinned source\n")
    preparer.verify_package_wheel(wheel, source)
    (source / "hub.py").write_text("# changed source\n")
    with pytest.raises(ValueError, match="wheel source differs"):
        preparer.verify_package_wheel(wheel, source)
