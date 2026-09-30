"""Exact input restoration and ownership locks, using local fake Drive folders."""

from __future__ import annotations

import errno
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from kiwilm.colab_artifacts import create_colab_artifacts, file_sha256
from kiwilm.colab_drive import CACHE_MARKER
from kiwilm.data import PreparedTokenData, prepare_from_stories
from kiwilm.tpu_setup import (
    cache_exists,
    job_digest,
    prepare_inputs,
    publish_data_cache,
    setup_owner,
    validate_job,
    write_setup_job,
)


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    source = tmp_path / "source"
    prepare_from_stories(
        source, ["A small training story. " * 8], ["A validation story. " * 8],
        vocab_size=300, min_frequency=1,
    )
    data = PreparedTokenData(source)
    drive = tmp_path / "drive"
    drive.mkdir()
    cache = drive / "MyDrive" / "KiwiLM2" / "data" / data.fingerprint
    monkeypatch.setattr(Path, "is_mount", lambda path: path == drive)
    return {
        "source": source, "cache": cache,
        "job": {
            "schema_version": 1, "use_drive": True, "data_fingerprint": data.fingerprint,
            "tokenizer_sha256": data.metadata["tokenizer"]["sha256"],
            "drive_cache_dir": str(cache), "resume_sha256": None,
        },
        "paths": {
            "data_dir": tmp_path / "vm-data", "resume_dir": tmp_path / "vm-resume",
            "drive_root": drive,
        },
    }


def test_drive_hit_restores_identical_data_without_upload(inputs: dict) -> None:
    (inputs["source"] / "artifact-manifest.json").write_text("not part of prepared data")
    publish_data_cache(inputs["source"], inputs["cache"], inputs["job"])
    before = {path.name: path.read_bytes() for path in inputs["cache"].iterdir()}
    result = prepare_inputs(inputs["job"], **inputs["paths"])
    assert result["state"] == "ready"
    assert result["data_source"] == "drive-cache"
    assert result["job_digest"] == job_digest(inputs["job"])
    local = inputs["paths"]["data_dir"]
    assert not (local / "artifact-manifest.json").exists()
    assert not (local / CACHE_MARKER).exists()
    for name, contents in before.items():
        assert (inputs["cache"] / name).read_bytes() == contents
        if name != CACHE_MARKER:
            assert (local / name).read_bytes() == contents


def test_cache_miss_upload_then_publish_new_cache(inputs: dict) -> None:
    first = prepare_inputs(inputs["job"], **inputs["paths"])
    assert first["state"] == "needs-data-upload"
    assert not inputs["cache"].exists()
    files = {path.name: path for path in inputs["source"].iterdir() if path.is_file()}
    create_colab_artifacts(
        files, inputs["paths"]["data_dir"], archive_name="data.tar.gz",
        compression="gzip", chunk_size=128,
    )
    result = prepare_inputs(inputs["job"], **inputs["paths"])
    assert result["state"] == "ready" and result["data_source"] == "uploaded-chunks"
    assert set(path.name for path in inputs["cache"].iterdir()) == {*files, CACHE_MARKER}
    assert json.loads((inputs["cache"] / CACHE_MARKER).read_text()) == {
        "fingerprint": inputs["job"]["data_fingerprint"],
    }


@pytest.mark.parametrize("defect", ["fingerprint", "tokenizer", "corrupt", "incomplete", "marker"])
def test_invalid_drive_cache_fails_without_modifying_it(inputs: dict, defect: str) -> None:
    publish_data_cache(inputs["source"], inputs["cache"], inputs["job"])
    job = dict(inputs["job"])
    if defect == "fingerprint":
        job["data_fingerprint"] = "f" * 64
    elif defect == "tokenizer":
        job["tokenizer_sha256"] = "f" * 64
    elif defect == "incomplete":
        (inputs["cache"] / CACHE_MARKER).unlink()
    elif defect == "marker":
        (inputs["cache"] / CACHE_MARKER).write_text(json.dumps({"fingerprint": "e" * 64}))
    else:
        metadata = json.loads((inputs["cache"] / "metadata.json").read_text())
        train = inputs["cache"] / metadata["splits"]["train"]["file"]
        contents = train.read_bytes()
        train.write_bytes(bytes([contents[0] ^ 1]) + contents[1:])
    before = {path.name: path.read_bytes() for path in inputs["cache"].iterdir()}
    with pytest.raises(ValueError):
        prepare_inputs(job, **inputs["paths"])
    assert before == {path.name: path.read_bytes() for path in inputs["cache"].iterdir()}
    assert not (inputs["paths"]["data_dir"] / "metadata.json").exists()


def test_publish_never_overwrites_existing_or_incomplete_cache(inputs: dict) -> None:
    inputs["cache"].mkdir(parents=True)
    (inputs["cache"] / "keep.txt").write_text("important existing content")
    with pytest.raises(FileExistsError):
        publish_data_cache(inputs["source"], inputs["cache"], inputs["job"])
    assert (inputs["cache"] / "keep.txt").read_text() == "important existing content"


def test_cache_marker_only_written_after_verified_copy(
    inputs: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("kiwilm.tpu_setup.atomic_copy_file", Mock(side_effect=OSError("Drive IO")))
    with pytest.raises(OSError, match="Drive IO"):
        publish_data_cache(inputs["source"], inputs["cache"], inputs["job"])
    assert not (inputs["cache"] / CACHE_MARKER).exists()


def test_unmounted_drive_fails_without_treating_it_as_cache_miss(
    inputs: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "is_mount", lambda _: False)
    with pytest.raises(RuntimeError, match="mount Google Drive"):
        prepare_inputs(inputs["job"], **inputs["paths"])
    assert not inputs["cache"].exists()


def test_transport_error_is_not_a_cache_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "stat", Mock(side_effect=OSError(errno.ENOTCONN, "disconnected")))
    with pytest.raises(OSError) as error:
        cache_exists(tmp_path / "cache")
    assert error.value.errno == errno.ENOTCONN


def test_explicit_training_validates_local_inputs_without_drive_io(
    inputs: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="not ready"):
        prepare_inputs(inputs["job"], **inputs["paths"], require_ready=True)
    paths = {**inputs["paths"], "data_dir": inputs["source"]}
    monkeypatch.setattr(Path, "is_mount", Mock(side_effect=AssertionError("no Drive IO")))
    result = prepare_inputs(inputs["job"], **paths, require_ready=True)
    assert result["state"] == "ready"
    assert not inputs["cache"].exists()
    train_file = PreparedTokenData(inputs["source"]).metadata["splits"]["train"]["file"]
    (inputs["source"] / train_file).write_bytes(b"broken")
    with pytest.raises(ValueError, match="size mismatch"):
        prepare_inputs(inputs["job"], **paths, require_ready=True)


def test_drive_hit_needs_only_resume_upload_and_verifies_it(inputs: dict, tmp_path: Path) -> None:
    publish_data_cache(inputs["source"], inputs["cache"], inputs["job"])
    checkpoint = tmp_path / "latest.pt"
    weights = torch.ones(3, 2)
    torch.save({
        "model_config": {"tie_embeddings": True},
        "model_state_dict": {"token_embedding.weight": weights, "lm_head.weight": weights},
    }, checkpoint)
    job = {**inputs["job"], "resume_sha256": file_sha256(checkpoint)}
    assert prepare_inputs(job, **inputs["paths"])["state"] == "needs-resume-upload"
    create_colab_artifacts({"latest.pt": checkpoint}, inputs["paths"]["resume_dir"], chunk_size=128)
    assert prepare_inputs(job, **inputs["paths"])["state"] == "ready"
    bad_job = {**job, "resume_sha256": "f" * 64}
    with pytest.raises(ValueError, match="resume checkpoint checksum"):
        prepare_inputs(bad_job, **inputs["paths"])


@pytest.mark.parametrize("cache_name", ["", "/content/drive", "/tmp/elsewhere", "../data/cache"])
def test_rejects_unsafe_drive_cache_targets(inputs: dict, cache_name: str) -> None:
    job = {**inputs["job"], "drive_cache_dir": cache_name}
    with pytest.raises(ValueError):
        validate_job(job, inputs["paths"]["drive_root"])


def test_rejects_symlink_cache_outside_drive(inputs: dict, tmp_path: Path) -> None:
    inputs["cache"].parent.mkdir(parents=True)
    inputs["cache"].symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(ValueError, match="inside the mounted"):
        validate_job(inputs["job"], inputs["paths"]["drive_root"])


def test_setup_ownership_requires_exact_remote_job(tmp_path: Path) -> None:
    job = {"schema_version": 1, "use_drive": False, "data_fingerprint": "a" * 64,
           "tokenizer_sha256": "b" * 64, "resume_sha256": None}
    (tmp_path / "tpu-job.json").write_text(json.dumps(job))
    report = tmp_path / "setup.json"
    report.write_text(json.dumps({"state": "ready", "job_digest": job_digest(job)}))
    options = dict(session="mine", tpu="v6e1")
    setup_owner(tmp_path, report, claim=True, **options)
    setup_owner(tmp_path, report, **options)
    assert job_digest(job) == job_digest(dict(reversed(list(job.items()))))
    with pytest.raises(FileExistsError):
        setup_owner(tmp_path, report, claim=True, **options)
    with pytest.raises(ValueError, match="ownership differs"):
        setup_owner(tmp_path, report, session="other", tpu="v6e1")
    report.write_text(json.dumps({"state": "ready", "job_digest": "stale"}))
    with pytest.raises(ValueError, match="job changed"):
        setup_owner(tmp_path, report, **options)


def test_local_job_validation_runs_before_allocation(
    inputs: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    build = Mock(return_value={"data_fingerprint": inputs["job"]["data_fingerprint"]})
    monkeypatch.setattr("kiwilm.tpu_setup.build_colab_job", build)
    path = tmp_path / "job.json"
    write_setup_job(inputs["source"], path, use_drive=True, cache_dir="")
    job = json.loads(path.read_text())
    assert job["drive_cache_dir"].endswith(inputs["job"]["data_fingerprint"])
    assert job["tokenizer_sha256"] == inputs["job"]["tokenizer_sha256"]
    build.assert_called_once_with(inputs["source"], phase="smoke", architecture="kiwilm2")
    path.unlink()
    with pytest.raises(ValueError):
        write_setup_job(inputs["source"], path, use_drive=True, cache_dir="/content/drive")
    assert not path.exists()
