from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from kiwilm.colab_artifacts import create_colab_artifacts, reassemble_colab_artifacts


def build_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote"
    local = tmp_path / "local"
    remote.mkdir()
    local.mkdir()
    files = {
        "best.pt": b"best checkpoint",
        "latest.pt": b"latest checkpoint",
        "metrics.jsonl": b'{"loss": 1.0}\n',
        "summary.json": b'{"step": 10}\n',
        "examples.md": b"# Examples\n",
    }
    archive_path = remote / "artifacts.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        for name, contents in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))
    archive_bytes = archive_path.read_bytes()
    parts = []
    for index, offset in enumerate(range(0, len(archive_bytes), 97)):
        contents = archive_bytes[offset : offset + 97]
        name = f"artifacts.tar.part-{index:04d}"
        (local / name).write_bytes(contents)
        parts.append(
            {
                "name": name,
                "bytes": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "archive": {
            "name": archive_path.name,
            "bytes": len(archive_bytes),
            "sha256": hashlib.sha256(archive_bytes).hexdigest(),
        },
        "files": {
            name: {
                "bytes": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
            for name, contents in files.items()
        },
        "parts": parts,
    }
    manifest_path = local / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, local


def test_reassembles_and_verifies_colab_artifacts(tmp_path: Path) -> None:
    manifest_path, output_dir = build_artifacts(tmp_path)

    extracted = reassemble_colab_artifacts(manifest_path, output_dir)

    assert {path.name for path in extracted} == {
        "best.pt",
        "latest.pt",
        "metrics.jsonl",
        "summary.json",
        "examples.md",
    }
    assert (output_dir / "best.pt").read_bytes() == b"best checkpoint"
    assert (output_dir / "examples.md").read_text() == "# Examples\n"


def test_rejects_corrupt_colab_artifact_part(tmp_path: Path) -> None:
    manifest_path, output_dir = build_artifacts(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    first_part = output_dir / manifest["parts"][0]["name"]
    first_part.write_bytes(first_part.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="part size mismatch"):
        reassemble_colab_artifacts(manifest_path, output_dir)


def test_creates_and_reassembles_chunked_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    remote = tmp_path / "remote"
    local = tmp_path / "local"
    source.mkdir()
    local.mkdir()
    (source / "latest.pt").write_bytes(b"checkpoint" * 100)
    (source / "summary.json").write_text('{"ok": true}\n', encoding="utf-8")

    remote_manifest = create_colab_artifacts(
        {
            "latest.pt": source / "latest.pt",
            "summary.json": source / "summary.json",
        },
        remote,
        chunk_size=97,
    )
    manifest = json.loads(remote_manifest.read_text(encoding="utf-8"))
    local_manifest = local / remote_manifest.name
    local_manifest.write_bytes(remote_manifest.read_bytes())
    for part in manifest["parts"]:
        (local / part["name"]).write_bytes((remote / part["name"]).read_bytes())

    extracted = reassemble_colab_artifacts(local_manifest, local)

    assert {path.name for path in extracted} == {"latest.pt", "summary.json"}
    assert (local / "latest.pt").read_bytes() == b"checkpoint" * 100
