"""1B preparation and command reconstruction, without live Colab or full training."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tokenizers import Tokenizer, models

from kiwilm.colab_artifacts import file_sha256
from kiwilm.config import KiwiLM2Config
from kiwilm.data import PreparedTokenData, metadata_fingerprint, prepare_from_stories
from kiwilm.tpu_final import (
    FINAL_SETTINGS,
    RECIPE,
    prepare_final_data,
    validate_final_data,
    write_final_job,
)
from kiwilm.tpu_setup import job_digest, job_training_options, validate_job
from kiwilm.tpu_smoke import Runtime, probe, probe_contract, probe_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def final_job(tmp_path: Path) -> tuple[dict, Path]:
    source = tmp_path / "source"
    source.mkdir()
    specials = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3}
    vocab = {**specials, **{f"token{i}": i for i in range(4, 32_000)}}
    tokenizer = source / "tokenizer.json"
    tokenizer.write_text(Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="[UNK]")).to_str())
    metadata = {
        "schema_version": 1, "dataset": {"name": "HuggingFaceTB/smollm-corpus",
                                         "requested_revision": "main",
                                         "resolved_revision": "a" * 40},
        "tokenizer": {"file": "tokenizer.json", "sha256": file_sha256(tokenizer),
                      "vocab_size": 32_000, "requested_vocab_size": 32_000,
                      "min_frequency": 2, "special_tokens": specials},
    }
    metadata["fingerprint"] = metadata_fingerprint(metadata)
    (source / "metadata.json").write_text(json.dumps(metadata))
    output = tmp_path / "results"
    output.mkdir()
    write_final_job(source, output / "tpu-job.json", tpu="v6e1")
    return json.loads((output / "tpu-job.json").read_text()), output


def test_final_job_uploads_only_tokenizer_and_freezes_full_budget(final_job) -> None:
    job, output = final_job
    assert all(job[k] == v for k, v in FINAL_SETTINGS.items())
    assert job["data_fingerprint"] is None
    assert job["data_recipe"] == {**RECIPE, "dataset_name": "HuggingFaceTB/smollm-corpus",
                                  "revision": "main", "resolved_revision": "a" * 40}
    assert {p.name for p in (output / "tokenizer").iterdir()} == {
        "tokenizer.json", "tokenizer-bundle.json"}
    assert not list(output.rglob("*.bin"))
    options = job_training_options(job)
    settings = probe_settings(KiwiLM2Config(vocab_size=32_000), precision="bf16", **options)
    assert settings.max_tokens == 1_000_000_000
    assert settings.warmup_tokens == 20_000_000
    assert settings.max_steps == 61_136
    assert settings.eval_batches == 200
    assert probe_contract(settings, "xla")["engine"] == "single-device-final-v1-tied"
    assert probe_contract(probe_settings(KiwiLM2Config(), precision="bf16"), "xla")[
        "engine"] == "single-device-smoke-v2-tied"


@pytest.mark.parametrize("field,value", [
    ("use_drive", False), ("max_tokens", 50_000_000), ("warmup_tokens", 1_000_000),
    ("eval_batches", 50), ("muon_lr", 0.02), ("precision", "fp16"),
    ("continuation_phase", "reference"), ("architecture", "kiwilm2_slim"),
])
def test_changed_long_run_controls_fail_before_allocation(final_job, field, value) -> None:
    job, _ = final_job
    job[field] = value
    with pytest.raises(ValueError):
        validate_job(job, Path("/content/drive"), check_cache_symlinks=False)


def test_resume_requires_ready_original_job_and_preserves_namespace(final_job, tmp_path) -> None:
    job, output = final_job
    with pytest.raises(ValueError, match="ready original"):
        write_final_job(Path("missing"), tmp_path / "resume.json", tpu="v6e1",
                        resume_job=output / "tpu-job.json")
    job["data_fingerprint"] = "b" * 64
    (output / "tpu-job.json").write_text(json.dumps(job))
    resumed = tmp_path / "resume.json"
    write_final_job(Path("missing"), resumed, tpu="v6e1", resume_job=output / "tpu-job.json")
    actual = json.loads(resumed.read_text())
    assert actual == {**job, "drive_resume_dir": job["drive_backup_dir"]}
    with pytest.raises(ValueError, match="cannot change"):
        write_final_job(Path("missing"), resumed, tpu="v6e1",
                        resume_job=output / "tpu-job.json", backup_dir="other")


def test_vm_prepares_data_once_then_locks_fingerprint(final_job, tmp_path, monkeypatch) -> None:
    job, output = final_job
    drive = tmp_path / "drive"
    job["drive_cache_dir"] = str(drive / "MyDrive/KiwiLM2/data/final1b")
    job["drive_backup_dir"] = str(drive / "MyDrive/KiwiLM2/checkpoints/final1b")
    monkeypatch.setattr(Path, "is_mount", lambda p: p == drive)
    preparation = Mock()
    monkeypatch.setattr("kiwilm.tpu_final.prepare_smollm_corpus", preparation)
    monkeypatch.setattr("kiwilm.tpu_final.validate_final_data",
                        Mock(return_value=SimpleNamespace(fingerprint="c" * 64)))
    paths = dict(data_dir=tmp_path / "vm-data", drive_root=drive,
                 tokenizer_dir=output / "tokenizer")
    before = job_digest(job)
    prepare_final_data(job, **paths)
    assert job["data_fingerprint"] == "c" * 64 and before != job_digest(job)
    assert preparation.call_args.args == (paths["data_dir"],)
    assert preparation.call_args.kwargs == {**job["data_recipe"],
                                           "tokenizer_from": output / "tokenizer"}
    prepare_final_data(job, **paths, require_ready=True)
    assert preparation.call_count == 1  # No network/preparation in the train action.


def test_required_resume_never_regenerates_missing_cache(final_job, tmp_path, monkeypatch) -> None:
    job, output = final_job
    drive = tmp_path / "drive"
    job.update(data_fingerprint="b" * 64,
               drive_cache_dir=str(drive / "MyDrive/KiwiLM2/data/final1b"),
               drive_backup_dir=str(drive / "MyDrive/KiwiLM2/checkpoints/final1b"))
    job["drive_resume_dir"] = job["drive_backup_dir"]
    monkeypatch.setattr(Path, "is_mount", lambda p: p == drive)
    preparation = Mock()
    monkeypatch.setattr("kiwilm.tpu_final.prepare_smollm_corpus", preparation)
    with pytest.raises(ValueError, match="refusing regeneration"):
        prepare_final_data(job, data_dir=tmp_path / "vm-data", drive_root=drive,
                           tokenizer_dir=output / "tokenizer")
    preparation.assert_not_called()


def test_fresh_job_cannot_overwrite_drive_progress(final_job, tmp_path, monkeypatch) -> None:
    job, output = final_job
    drive = tmp_path / "drive"
    job["drive_cache_dir"] = str(drive / "MyDrive/KiwiLM2/data/final1b")
    backup = drive / "MyDrive/KiwiLM2/checkpoints/final1b"
    job["drive_backup_dir"] = str(backup)
    backup.mkdir(parents=True)
    (backup / "latest.json").write_text("existing progress")
    monkeypatch.setattr(Path, "is_mount", lambda p: p == drive)
    with pytest.raises(ValueError, match="explicit resume"):
        prepare_final_data(job, data_dir=tmp_path / "vm-data", drive_root=drive,
                           tokenizer_dir=output / "tokenizer")
    assert (backup / "latest.json").read_text() == "existing progress"


def test_final_data_validation_rejects_provenance_mismatches(final_job, monkeypatch) -> None:
    job, _ = final_job
    metadata = {"config": {"seed": 42, "fineweb_probability": 0.7,
                           "validation_documents_per_source": 10_000,
                           "tokenizer_train_limit": 100_000},
                "dataset": {"name": job["data_recipe"]["dataset_name"],
                            "requested_revision": "main", "resolved_revision": "a" * 40},
                "splits": {"validation": {"tokens": 2_000_000}},
                "tokenizer": {"sha256": job["tokenizer_sha256"], "reused_from": {
                    "dataset_fingerprint": job["tokenizer_source_fingerprint"],
                    "tokenizer_sha256": job["tokenizer_sha256"]}}}
    monkeypatch.setattr("kiwilm.tpu_final.PreparedTokenData",
                        Mock(return_value=SimpleNamespace(metadata=metadata)))
    validate = Mock()
    monkeypatch.setattr("kiwilm.tpu_final.build_colab_job", validate)
    validate_final_data(Path("test-data"), job)
    validate.assert_called_once_with(Path("test-data"), phase="final-1b", architecture="kiwilm2")
    metadata["dataset"]["resolved_revision"] = "d" * 40
    with pytest.raises(ValueError, match="frozen recipe"):
        validate_final_data(Path("test-data"), job)


def test_final_worker_reconstructs_budget_timeout_and_required_resume(
    final_job, tmp_path, monkeypatch,
) -> None:
    job, _ = final_job
    job["data_fingerprint"] = "b" * 64
    job["drive_resume_dir"] = job["drive_backup_dir"]
    spec = importlib.util.spec_from_file_location(
        "final_bootstrap", ROOT / "scripts/colab_kiwilm2_tpu_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CONTENT", tmp_path)
    monkeypatch.setattr(module, "PYTHON", tmp_path / "python")
    (tmp_path / "python").touch()
    (tmp_path / "kiwilm-0.1.0-py3-none-any.whl").touch()
    (tmp_path / "kiwilm-tpu-preflight.json").write_text('{"vm_id":"new-vm"}')
    (tmp_path / "kiwilm-tpu-job.json").write_text(json.dumps(job))
    (tmp_path / "kiwilm-tpu-setup.json").write_text(json.dumps({
        "state": "ready", "job_digest": job_digest(job)}))
    monkeypatch.setenv("KIWILM2_TPU_ACTION", "train")
    monkeypatch.setattr(module, "run", Mock())
    worker = Mock()
    monkeypatch.setattr(module, "run_worker", worker)
    with pytest.raises(RuntimeError, match="refusing fresh"):
        module.main()
    worker.assert_not_called()
    resume = tmp_path / "kiwilm-tpu-resume"
    resume.mkdir()
    (resume / "latest.pt").write_bytes(b"fake checkpoint")
    module.main()
    command = worker.call_args.args[0]
    assert command[command.index("--phase") + 1] == "final-1b"
    assert command[command.index("--warmup-tokens") + 1] == "20000000"
    assert command[command.index("--eval-batches") + 1] == "200"
    assert "--final-artifacts-only" in command
    assert "--steps" not in command
    assert worker.call_args.kwargs["timeout"] == 79200
    assert command[command.index("--output-dir") + 1].endswith("kiwilm-tpu-final-1b")


def test_final_schedule_tiny_prefix_resume_and_no_periodic_repack(tmp_path, monkeypatch) -> None:
    prepare_from_stories(tmp_path / "data", ["A story. " * 30], ["Another story. " * 30],
                         vocab_size=300, min_frequency=1)
    data = PreparedTokenData(tmp_path / "data")
    config = KiwiLM2Config(vocab_size=data.tokenizer.vocab_size, d_model=8, context_length=8,
                          num_query_heads=2, num_kv_heads=1, swiglu_dim=12,
                          bigram_buckets=17, trigram_buckets=19,
                          conv_kernel_sizes=(3, 5, 3, 5, 3, 5))
    options = dict(config=config, runtime=Runtime("cpu", "fp32"), steps=2, warmup_steps=1,
                   batch_size=1, accumulation=1, eval_batches=1, checkpoint_interval=1,
                   max_tokens=1_000_000_000, warmup_tokens=20_000_000,
                   periodic_artifacts=False, artifact_dir=tmp_path / "artifacts")
    packaging = Mock()
    monkeypatch.setattr("kiwilm.tpu_smoke.create_colab_artifacts", packaging)
    probe(data, tmp_path / "run", **options)
    assert packaging.call_count == 1  # Final session packaging, not every save.
    result = probe(data, tmp_path / "run", resume=tmp_path / "run/latest.pt", **options)
    assert result["step"] == 4 and result["tokens_seen"] == 32
    saved = torch.load(tmp_path / "run/latest.pt", weights_only=True)
    assert saved["train_config"]["warmup_tokens"] == 20_000_000
    assert saved["training_state"]["smoke_contract"]["engine"] == "single-device-final-v1-tied"
    with pytest.raises(ValueError, match="matching hardware"):
        probe(data, tmp_path / "changed", resume=tmp_path / "run/latest.pt",
              **{**options, "warmup_tokens": 1_000_000})
