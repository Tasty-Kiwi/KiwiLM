"""Live-streaming proofs with tiny local print-only children, never training."""

from __future__ import annotations

import importlib.util
import io
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def bootstrap():
    root = Path(__file__).resolve().parents[1]
    path = root / "archive/kiwilm2/scripts/colab_kiwilm2_tpu_smoke.py"
    spec = importlib.util.spec_from_file_location("stream_bootstrap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_progress_reaches_console_before_child_exits_and_full_log_is_preserved(
    bootstrap, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = tmp_path / "progress-seen"

    class Console(io.StringIO):
        def write(self, value: str) -> int:
            if '"event": "train"' in value:
                gate.touch()
            return super().write(value)

    console = Console()
    monkeypatch.setattr(sys, "stdout", console)
    log = tmp_path / "worker.log"
    log.write_text('{"event":"historical"}\n')
    code = (
        "import json,sys,time; from pathlib import Path; "
        "print('verbose XLA diagnostic'); print('native stderr diagnostic',file=sys.stderr); "
        "print(json.dumps(dict(event='train',step=10,tokens_per_second=80000)),flush=True); "
        f"gate=Path({str(gate)!r}); deadline=time.monotonic()+3\n"
        "while not gate.exists() and time.monotonic()<deadline: time.sleep(0.01)\n"
        "if not gate.exists(): sys.exit(17)\n"
        "print(json.dumps(dict(event='finished',step=10)),end='',flush=True)"
    )
    bootstrap.run_worker(
        [sys.executable, "-u", "-c", code], env=os.environ.copy(), log_path=log, timeout=5,
    )
    assert gate.exists()  # Child cannot complete successfully until progress is streamed.
    output = console.getvalue()
    assert '"event": "train"' in output
    assert '"event": "finished"' in output  # Final unterminated line is drained.
    assert "verbose XLA diagnostic" not in output
    assert "native stderr diagnostic" not in output
    assert "historical" not in output
    assert "Initial XLA compilation" in output
    contents = log.read_text()
    assert contents.startswith('{"event":"historical"}')
    assert "verbose XLA diagnostic" in contents and "native stderr diagnostic" in contents
    assert '"event": "train"' in contents and '"event": "finished"' in contents


def test_quiet_worker_emits_heartbeat(bootstrap, tmp_path: Path, capsys) -> None:
    bootstrap.run_worker(
        [sys.executable, "-u", "-c", "import time; time.sleep(0.15)"],
        env=os.environ.copy(), log_path=tmp_path / "worker.log", timeout=3,
        heartbeat_interval=0.03,
    )
    output = capsys.readouterr().out
    assert "TPU worker still running" in output
    assert "compilation/evaluation may be quiet" in output


def test_nonzero_worker_shows_failure_tail_and_propagates_exit(
    bootstrap, tmp_path: Path, capsys,
) -> None:
    log = tmp_path / "worker.log"
    with pytest.raises(RuntimeError, match="exit code 7"):
        bootstrap.run_worker(
            [sys.executable, "-u", "-c", "import sys; print('deliberate failure'); sys.exit(7)"],
            env=os.environ.copy(), log_path=log, timeout=3,
        )
    assert "deliberate failure" in capsys.readouterr().err
    assert "deliberate failure" in log.read_text()


def test_timeout_reaps_child_and_keeps_log(bootstrap, tmp_path: Path, monkeypatch) -> None:
    original = bootstrap.subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(bootstrap.subprocess, "Popen", spawn)
    log = tmp_path / "worker.log"
    with pytest.raises(RuntimeError, match="worker limit"):
        bootstrap.run_worker(
            [sys.executable, "-u", "-c", "import time; print('started'); time.sleep(10)"],
            env=os.environ.copy(), log_path=log, timeout=0.25, heartbeat_interval=0.05,
        )
    assert len(children) == 1 and children[0].poll() is not None
    assert "started" in log.read_text()


def test_console_failure_reaps_child(bootstrap, tmp_path: Path, monkeypatch) -> None:
    original = bootstrap.subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        children.append(process)
        return process

    class BrokenConsole(io.StringIO):
        def write(self, value):
            if '"event"' in value:
                raise BrokenPipeError("console disconnected")
            return super().write(value)

    monkeypatch.setattr(bootstrap.subprocess, "Popen", spawn)
    monkeypatch.setattr(sys, "stdout", BrokenConsole())
    with pytest.raises(BrokenPipeError):
        bootstrap.run_worker(
            [sys.executable, "-u", "-c",
             "import time; print('{\"event\":\"train\"}',flush=True); time.sleep(10)"],
            env=os.environ.copy(), log_path=tmp_path / "worker.log", timeout=3,
        )
    assert len(children) == 1 and children[0].poll() is not None


@pytest.mark.parametrize("timeout,heartbeat", [(0, 30), (1, 0), (-1, 30)])
def test_invalid_limits_never_spawn(bootstrap, tmp_path, monkeypatch, timeout, heartbeat) -> None:
    spawn = Mock()
    monkeypatch.setattr(bootstrap.subprocess, "Popen", spawn)
    with pytest.raises(ValueError, match="must be positive"):
        bootstrap.run_worker(
            ["not-a-real-worker"], env={}, log_path=tmp_path / "worker.log",
            timeout=timeout, heartbeat_interval=heartbeat,
        )
    spawn.assert_not_called()
