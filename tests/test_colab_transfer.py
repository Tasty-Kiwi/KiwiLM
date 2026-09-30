"""Uploads are mocked: no Colab allocations, file transfers or training."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

from kiwilm.colab_artifacts import create_colab_artifacts, reassemble_colab_artifacts
from kiwilm.colab_transfer import upload_artifacts


def artifact_fixture(tmp_path: Path) -> Path:
    source = tmp_path / "source.bin"
    source.write_bytes(b"some compressed tokens" * 500)
    directory = tmp_path / "uploads"
    create_colab_artifacts(
        {source.name: source}, directory, compression="gzip", archive_name="data.tar.gz",
        chunk_size=32,
    )
    return directory


def test_parallel_upload_is_bounded_and_manifest_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = artifact_fixture(tmp_path)
    remote = tmp_path / "remote"
    remote.mkdir()
    barrier = threading.Barrier(3, timeout=5)
    lock = threading.Lock()
    active = peak = started = 0
    completed = []

    def fake_run(command, **kwargs):
        nonlocal active, peak, started
        assert command[:4] == ["fake-colab", "upload", "-s", "test-session"]
        assert kwargs["timeout"] == 120 and kwargs["check"] is True
        with lock:
            active += 1
            peak = max(peak, active)
            started += 1
            initial = started <= 3
        if initial:
            barrier.wait()
        source = Path(command[-2])
        shutil.copyfile(source, remote / source.name)
        with lock:
            active -= 1
            completed.append(source.name)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("kiwilm.colab_transfer.subprocess.run", fake_run)
    upload_artifacts(directory, "/content/data", session="test-session", colab_bin="fake-colab")
    assert peak == 3
    assert completed[-1] == "artifact-manifest.json"
    assert set(completed) == {path.name for path in directory.iterdir()}
    reassemble_colab_artifacts(remote / "artifact-manifest.json", remote)
    assert (remote / "source.bin").read_bytes() == (tmp_path / "source.bin").read_bytes()


def test_upload_retries_transient_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = artifact_fixture(tmp_path)
    failed_name = json.loads((directory / "artifact-manifest.json").read_text())["parts"][0]["name"]
    tries = []

    def fake_run(command, **kwargs):
        tries.append(Path(command[-2]).name)
        if tries[-1] == failed_name and tries.count(failed_name) < 3:
            raise subprocess.TimeoutExpired(command, 120)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("kiwilm.colab_transfer.subprocess.run", fake_run)
    upload_artifacts(directory, "/content/data", session="test", workers=1, retry_delay=0)
    assert tries.count(failed_name) == 3
    assert tries[-1] == "artifact-manifest.json"


def test_failed_part_never_publishes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = artifact_fixture(tmp_path)
    run = Mock(side_effect=subprocess.CalledProcessError(1, ["fake-colab"]))
    monkeypatch.setattr("kiwilm.colab_transfer.subprocess.run", run)
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        upload_artifacts(
            directory, "/content/data", session="test", workers=2, attempts=2, retry_delay=0,
        )
    assert run.call_count >= 2
    assert all(Path(call.args[0][-2]).name != "artifact-manifest.json"
               for call in run.call_args_list)


@pytest.mark.parametrize("defect", ["corrupt", "duplicate", "traversal", "malformed"])
def test_invalid_parts_rejected_before_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str,
) -> None:
    directory = artifact_fixture(tmp_path)
    manifest_path = directory / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if defect == "corrupt":
        part = directory / manifest["parts"][0]["name"]
        part.write_bytes(b"corrupt")
    elif defect == "duplicate":
        manifest["parts"].append(manifest["parts"][0])
    elif defect == "traversal":
        manifest["parts"][0]["name"] = "../tokens"
    else:
        manifest["parts"][0] = None
    manifest_path.write_text(json.dumps(manifest))
    run = Mock()
    monkeypatch.setattr("kiwilm.colab_transfer.subprocess.run", run)
    with pytest.raises(ValueError):
        upload_artifacts(directory, "/content/data", session="test")
    run.assert_not_called()


@pytest.mark.parametrize("workers", [0, 5, True])
def test_upload_worker_bounds_before_subprocess(workers: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="1-4 workers"):
        upload_artifacts(tmp_path, "/content/data", session="test", workers=workers)
