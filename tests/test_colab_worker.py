"""Exercise worker recovery without installing packages or renting a Colab VM."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from kiwilm.colab_drive import restore_checkpoint_backup, restore_prepared_data
from kiwilm.colab_kiwilm2 import build_colab_job, checkpoint_backup_key

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "colab_kiwilm2_train.py"


def _worker_functions(tmp_path: Path, *, required: bool = True) -> dict:
    tree = ast.parse(WORKER.read_text())
    functions = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef)],
        type_ignores=[],
    )
    namespace = {
        "Path": Path,
        "shutil": shutil,
        "CONTENT": tmp_path,
        "RUN_DIR": tmp_path / "run",
        "DATA_DIR": tmp_path / "data",
        "job": {
            "require_resume": required, "drive_root": str(tmp_path),
            "data_cache_key": "test-cache",
        },
        "restore_checkpoint_backup": restore_checkpoint_backup,
        "restore_prepared_data": restore_prepared_data,
    }
    # Postpone annotations exactly as the real worker does.
    functions.body.insert(0, ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0,
    ))
    exec(compile(ast.fix_missing_locations(functions), str(WORKER), "exec"), namespace)
    return namespace


def test_worker_required_resume_without_any_source_fails(tmp_path: Path) -> None:
    namespace = _worker_functions(tmp_path)
    with pytest.raises(RuntimeError, match="refusing to train from scratch"):
        namespace["stage_resume"](None)


def test_worker_explicit_resume_takes_precedence(tmp_path: Path) -> None:
    namespace = _worker_functions(tmp_path)
    (tmp_path / "resume.pt").write_bytes(b"explicit")
    (tmp_path / "resume-metrics.jsonl").write_text('{"step":5000}\n')
    restore = Mock(side_effect=AssertionError("must not read Drive"))
    namespace["restore_checkpoint_backup"] = restore
    assert namespace["stage_resume"](tmp_path / "backup") == tmp_path / "resume.pt"
    assert (tmp_path / "run" / "metrics.jsonl").read_text() == '{"step":5000}\n'
    restore.assert_not_called()


def test_worker_passes_resume_requirement_and_storage_root(tmp_path: Path) -> None:
    namespace = _worker_functions(tmp_path)
    restore = Mock(return_value=tmp_path / "run" / "resume.pt")
    namespace["restore_checkpoint_backup"] = restore
    namespace["stage_resume"](tmp_path / "backup")
    restore.assert_called_once_with(
        tmp_path / "backup", tmp_path / "run", required=True, storage_root=tmp_path,
    )


def test_worker_does_not_prepare_data_when_required_resume_fails(tmp_path: Path) -> None:
    namespace = _worker_functions(tmp_path)
    namespace.update(backup_dir=None, drive_root=None)
    prepare = Mock(side_effect=AssertionError("must not prepare data"))
    namespace["prepare_data"] = prepare
    tree = ast.parse(WORKER.read_text())
    startup = ast.Module(body=[
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in {"resume_path", "data"}
                for target in node.targets)
    ], type_ignores=[])
    with pytest.raises(RuntimeError, match="refusing to train from scratch"):
        exec(compile(startup, str(WORKER), "exec"), namespace)
    prepare.assert_not_called()


def test_worker_does_not_regenerate_data_when_recovery_cache_is_unavailable(tmp_path: Path) -> None:
    namespace = _worker_functions(tmp_path)
    restore = Mock(side_effect=RuntimeError("Drive disconnected"))
    prepare = Mock(side_effect=AssertionError("must not regenerate data"))
    namespace.update(restore_prepared_data=restore, prepare_smollm_corpus=prepare)
    with pytest.raises(RuntimeError, match="Drive disconnected"):
        namespace["prepare_data"](tmp_path)
    restore.assert_called_once_with(
        tmp_path / "data" / "test-cache", tmp_path / "data",
        required=True, storage_root=tmp_path,
    )
    prepare.assert_not_called()


def test_old_worker_job_defaults_to_optional_resume(tmp_path: Path) -> None:
    namespace = _worker_functions(tmp_path)
    del namespace["job"]["require_resume"]
    assert namespace["stage_resume"](None) is None


def test_recovery_policy_preserves_existing_500m_backup_key() -> None:
    options = dict(
        phase="final-500m", architecture="kiwilm2", optimizer="muon", muon_lr=0.01,
        precision="fp16", compile_policy="eager",
    )
    normal = build_colab_job(**options)
    recovery = build_colab_job(**options, require_resume=True)
    legacy = {key: value for key, value in normal.items() if key != "require_resume"}
    assert recovery["require_resume"] is True
    assert checkpoint_backup_key(normal) == checkpoint_backup_key(recovery)
    assert checkpoint_backup_key(recovery) == checkpoint_backup_key(legacy)
    assert checkpoint_backup_key(recovery) == "final-500m-kiwilm2-muon-0p01-4c73fc982a62"
    with pytest.raises(TypeError, match="require_resume"):
        build_colab_job(**options, require_resume="true")


def test_job_cli_serializes_require_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location(
        "prepare_colab_job", ROOT / "scripts" / "prepare_kiwilm2_colab_job.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "job.json"
    monkeypatch.setattr("sys.argv", [
        "prepare_colab_job", "--output", str(output), "--phase", "final-500m",
        "--architecture", "kiwilm2", "--require-resume",
    ])
    assert module.main() == 0
    assert json.loads(output.read_text())["require_resume"] is True


@pytest.mark.skipif(shutil.which("bash") is None, reason="launcher requires Bash")
@pytest.mark.parametrize(("required", "explicit"), [(True, False), (False, True), (False, False)])
def test_launcher_passes_recovery_policy_before_allocating(
    tmp_path: Path, required: bool, explicit: bool,
) -> None:
    # Fake both external tools: the job-builder call deliberately stops the
    # launcher before wheel building, Drive access or any VM allocation.
    arguments = tmp_path / "arguments.json"
    uv = tmp_path / "uv"
    uv.write_text(
        f"#!{sys.executable}\nimport json,sys\n"
        f"with open({str(arguments)!r}, 'w') as f: json.dump(sys.argv[1:], f)\n"
        "sys.exit(9)\n"
    )
    colab = tmp_path / "colab"
    colab.write_text(
        '#!/bin/sh\nif [ "$1" = status ]; then echo "not found"; exit 0; fi\nexit 99\n'
    )
    uv.chmod(0o755)
    colab.chmod(0o755)
    resume = tmp_path / "latest.pt"
    if explicit:
        resume.write_bytes(b"checkpoint")
    env = {
        name: value for name, value in os.environ.items()
        if not name.startswith(("KIWILM", "COLAB"))
    }
    env.update(
        PATH=f"{tmp_path}{os.pathsep}{env.get('PATH', '')}",
        TMPDIR=str(tmp_path), COLAB_BIN=str(colab),
        KIWILM_RESULT_DIR=str(tmp_path / "results"),
        KIWILM2_REQUIRE_RESUME="1" if required else "0",
        KIWILM2_RESUME_FROM=str(resume) if explicit else "",
    )
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_colab_kiwilm2.sh")],
        env=env, cwd=ROOT, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 9, result.stdout + result.stderr
    assert ("--require-resume" in json.loads(arguments.read_text())) == (required or explicit)
