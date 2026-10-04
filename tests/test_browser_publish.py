"""Static playground deployment is allowlisted and cannot allocate hardware."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kiwilm.safetensors_io import sha256_file


@pytest.fixture
def deployment(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/publish_browser_playground.py"
    spec = importlib.util.spec_from_file_location("browser_publisher", path)
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    dist = tmp_path / "dist"
    model = dist / "models/kiwilm2"
    model.mkdir(parents=True)
    (dist / "index.html").write_text("fixture")
    for name in ("model.onnx", "tokenizer.json", "parity.json"):
        (model / name).write_text("fixture")
    manifest = {"model_repo": "Tasty-Kiwi/KiwiLM-2",
                "model_revision": "53d6abcb4d11c8a98936c19d55d2b741a99d7602",
                "verification": {"direct_cached_rollover": "passed"},
                "files": {file.name: {"sha256": sha256_file(file),
                                      "bytes": file.stat().st_size} for file in model.iterdir()}}
    (model / "browser-manifest.json").write_text(json.dumps(manifest))
    readme = tmp_path / "README.md"
    readme.write_text("---\nsdk: static\n---\n")
    calls = []
    legacy = json.loads(publisher.LEGACY_MANIFEST.read_text())
    siblings = [SimpleNamespace(rfilename=f"models/{model_id}/{name}", size=details["bytes"],
                                lfs=SimpleNamespace(sha256=details["sha256"]))
                for model_id, bundle in legacy["models"].items()
                for name, details in bundle["files"].items()]
    api = SimpleNamespace(repo_info=lambda *args, **kwargs:
                          SimpleNamespace(sdk="static", sha="b" * 40, siblings=siblings),
                          create_commit=lambda **kwargs: calls.append(kwargs)
                          or SimpleNamespace(oid="c" * 40))
    return publisher, dist, readme, api, calls


def test_publish_preserves_history_uses_space_and_parent_commit(deployment):
    publisher, dist, readme, api, calls = deployment
    assert publisher.publish(dist, readme, api) == "c" * 40
    assert calls[0]["repo_id"] == "Tasty-Kiwi/KiwiLM-Playground"
    assert calls[0]["repo_type"] == "space"
    assert calls[0]["parent_commit"] == "b" * 40
    assert len(calls[0]["operations"]) == 6
    assert all(type(operation).__name__ == "CommitOperationAdd"
               for operation in calls[0]["operations"])


def test_refuse_nonstatic_space(deployment):
    publisher, dist, readme, api, calls = deployment
    api.repo_info = lambda *args, **kwargs: SimpleNamespace(sdk="docker")
    with pytest.raises(ValueError, match="SDK"):
        publisher.publish(dist, readme, api)
    assert not calls


@pytest.mark.parametrize("failure", ("tampered", "unexpected"))
def test_integrity_and_allowlist_fail_before_upload(deployment, failure):
    publisher, dist, readme, api, calls = deployment
    if failure == "tampered":
        (dist / "models/kiwilm2/model.onnx").write_text("changed")
    else:
        (dist / "secret.env").write_text("not a deployment asset")
    with pytest.raises(ValueError):
        publisher.publish(dist, readme, api)
    assert not calls


@pytest.mark.parametrize("failure", ("missing", "tampered"))
def test_legacy_asset_integrity_checked_before_commit(deployment, failure):
    publisher, dist, readme, api, calls = deployment
    info = api.repo_info()
    if failure == "missing":
        info.siblings.pop()
    else:
        info.siblings[0].lfs.sha256 = "0" * 64
    api.repo_info = lambda *args, **kwargs: info
    with pytest.raises(ValueError, match="legacy asset"):
        publisher.publish(dist, readme, api)
    assert not calls
