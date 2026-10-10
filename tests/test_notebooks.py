"""Notebook generation/default safety plus mocked environment installation."""

from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pytest

from kiwilm.notebook_setup import prepare_environment

ROOT = Path(__file__).parents[1]


def generated_notebooks():
    pytest.importorskip("nbformat", reason="install the notebooks extra for notebook validation")
    spec = importlib.util.spec_from_file_location(
        "build_books", ROOT / "scripts/build_kiwilm3_notebooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.notebooks()


@pytest.mark.parametrize("kind", ["smoke", "train", "evaluate"])
def test_notebook_generated_safe_and_executable(kind, tmp_path):
    nbformat = pytest.importorskip("nbformat")
    generated = generated_notebooks()[f"kiwilm3-{kind}.ipynb"]
    saved = nbformat.read(ROOT / "notebooks" / f"kiwilm3-{kind}.ipynb", as_version=4)
    nbformat.validate(saved)
    # Outputs/execution counts are intentionally saved after safe-default execution.
    assert [cell.source for cell in saved.cells] == [cell.source for cell in generated.cells]
    state = {"__name__": "__notebook_test__"}
    for cell in saved.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, f"{kind}.ipynb", "exec"), state)
    for name in (
        "MOUNT_DRIVE",
        "SETUP_ENVIRONMENT",
        "PREPARE_DATA",
        "RUN_PREFLIGHT",
        "START_TRAINING",
        "RUN_EVALUATION",
        "RESTORE_FOR_EVALUATION",
        "EXPORT_SAFETENSORS",
    ):
        assert state[name] is False
    assert not (tmp_path / "latest.pt").exists()
    text = "\n".join(cell.source for cell in saved.cells)
    assert "tiny V2 causal model" in text
    assert "colab new" not in text and "colab run" not in text


def test_worker_pins_match_lock():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    versions = {(package["name"], package["version"]) for package in lock["package"]}
    for line in (ROOT / "src/kiwilm/notebook_requirements.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, version = line.split(";")[0].strip().split("==")
        assert (name, version) in versions


def test_install_is_isolated_pinned_and_idempotent(tmp_path, monkeypatch):
    wheel = tmp_path / "kiwilm.whl"
    wheel.write_bytes(b"mock wheel")
    directory = tmp_path / "isolated"
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        assert kwargs["check"] is True
        if "venv" in command:
            directory.mkdir()

    monkeypatch.setattr("kiwilm.notebook_setup.subprocess.run", execute)
    worker = prepare_environment(wheel, directory, "xla")
    assert worker == directory / "bin/python"
    assert len(calls) == 5
    assert calls[0] == [sys.executable, "-m", "pip", "install", "uv==0.11.7"]
    assert "torch==2.9.0" in calls[2] and "torch_xla[tpu]==2.9.0" in calls[3]
    assert "--no-deps" in calls[4]
    assert all("--system" not in command for command in calls)
    assert prepare_environment(wheel, directory, "xla") == worker
    assert len(calls) == 5
    with pytest.raises(FileExistsError, match="differs"):
        prepare_environment(wheel, directory, "cpu")


def test_interrupted_install_resumes_only_same_contract(tmp_path, monkeypatch):
    wheel = tmp_path / "kiwilm.whl"
    wheel.write_bytes(b"mock wheel")
    directory = tmp_path / "isolated"
    failed = False

    def execute(command, **_kwargs):
        nonlocal failed
        if "venv" in command:
            directory.mkdir()
        if "torch==2.9.0" in command and not failed:
            failed = True
            raise RuntimeError("interrupted mock installation")

    monkeypatch.setattr("kiwilm.notebook_setup.subprocess.run", execute)
    with pytest.raises(RuntimeError):
        prepare_environment(wheel, directory, "xla")
    assert not (directory / "kiwilm-environment.json").exists()
    prepare_environment(wheel, directory, "xla")
    assert json.loads((directory / "kiwilm-environment.json").read_text())["torch_xla"] == "2.9.0"


def test_cuda_environment_pins_torch_without_cpu_or_xla(tmp_path, monkeypatch):
    wheel = tmp_path / "kiwilm.whl"
    wheel.write_bytes(b"mock wheel")
    directory = tmp_path / "isolated-cuda"
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        if "venv" in command:
            directory.mkdir()

    monkeypatch.setattr("kiwilm.notebook_setup.subprocess.run", execute)
    prepare_environment(wheel, directory, "cuda")
    assert "torch==2.13.0" in calls[2] and "torch==2.13.0" in calls[3]
    assert not any("torch_xla" in str(c) or "/whl/cpu" in str(c) for c in calls)
    marker = json.loads((directory / "kiwilm-environment.json").read_text())
    assert marker["device"] == "cuda" and marker["torch_xla"] is None
    assert "--no-deps" in calls[-1]
