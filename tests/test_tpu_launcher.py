"""Exercise shell ownership/setup with fake CLIs; never contact Colab or train."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kiwilm.data import prepare_from_stories

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_colab_kiwilm2_tpu_smoke.sh"

# This fake CLI deliberately cannot allocate hardware or execute training.
FAKE_COLAB = r'''
import json, os, shutil, sys
from pathlib import Path
root = Path(os.environ["FAKE_REMOTE"])
args = sys.argv[1:]
with Path(os.environ["FAKE_CALLS"]).open("a") as stream:
    stream.write(json.dumps(args) + "\n")
command = args[0]
def mapped(value):
    return root / value.lstrip("/")
if command == "status":
    print("Session not found")
    sys.exit(1)
if command in {"new", "stop", "log"}:
    sys.exit(0)
if command == "upload":
    target = mapped(args[-1])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args[-2], target)
elif command == "download":
    shutil.copyfile(mapped(args[-2]), args[-1])
elif command == "exec":
    if "--env" not in args:
        sys.stdin.read()
        mapped("/content/kiwilm-data-artifacts").mkdir(parents=True, exist_ok=True)
        mapped("/content/kiwilm-tpu-resume").mkdir(parents=True, exist_ok=True)
        sys.exit(0)
    action = args[args.index("--env") + 1].split("=", 1)[1]
    if action == "preflight":
        mapped("/content/kiwilm-tpu-preflight.json").write_text('{"simulated":true}')
    elif action == "prepare":
        from kiwilm.tpu_setup import prepare_inputs
        if os.environ.get("FAKE_FAIL_SETUP"):
            sys.exit(2)
        data = mapped("/content/kiwilm-data-artifacts")
        if os.environ.get("FAKE_CACHE_HIT") and not (data / "metadata.json").exists():
            shutil.copytree(os.environ["KIWILM2_DATA_DIR"], data)
        job = json.loads(mapped("/content/kiwilm-tpu-job.json").read_text())
        result = prepare_inputs(job, data_dir=data,
            resume_dir=mapped("/content/kiwilm-tpu-resume"), drive_root=mapped("/content/drive"))
        mapped("/content/kiwilm-tpu-setup.json").write_text(json.dumps(result))
    elif action == "train":
        from kiwilm.colab_artifacts import create_colab_artifacts
        # Only simulated artifact production. No model or training loop exists here.
        output = mapped("/content/kiwilm-tpu-smoke")
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text('{"status":"simulated-complete"}')
        (output / "latest.pt").write_bytes(b"simulated checkpoint")
        (output / "worker.log").write_text("simulated, not actual training")
        create_colab_artifacts({p.name:p for p in output.iterdir()},
            mapped("/content/kiwilm-tpu-artifacts"), chunk_size=4096)
'''

FAKE_UV = r'''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == "build":
    directory = Path(args[args.index("--out-dir") + 1])
    (directory / "kiwilm-0.1.0-py3-none-any.whl").write_bytes(b"fake wheel")
else:
    assert args[:3] == ["run", "--locked", "python"]
    command = args[3:]
    if command[0] == "-c" and "write_setup_job" in command[1]:
        # Only freeze validation is stubbed for the tiny data, never real 50M data.
        prefix = ("import json; from pathlib import Path; import kiwilm.tpu_setup as ts; "
            "ts.build_colab_job=lambda p,**kw: "
            "{'data_fingerprint':json.loads((p/'metadata.json').read_text())['fingerprint']}; ")
        command[1] = prefix + command[1]
    os.execv(sys.executable, [sys.executable, *command])
'''


class FixtureEnvironment(dict):
    def __repr__(self) -> str:
        # Failure diagnostics must not dump inherited credentials/environment values.
        return "<isolated fake-CLI environment>"


@pytest.fixture
def launcher(tmp_path: Path) -> tuple[dict[str, str], Path]:
    data = tmp_path / "data"
    prepare_from_stories(
        data, ["A training story. " * 8], ["A validation story. " * 8],
        vocab_size=300, min_frequency=1,
    )
    binary = tmp_path / "bin"
    binary.mkdir()
    for name, contents in [("colab", FAKE_COLAB), ("uv", FAKE_UV)]:
        path = binary / name
        path.write_text(f"#!{sys.executable}\n" + contents)
        path.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    environment = FixtureEnvironment({
        **os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
        "COLAB_BIN": str(binary / "colab"), "COLAB_TPU": "v6e1", "KIWILM2_USE_DRIVE": "0",
        "COLAB_SESSION_NAME": "fake-owned-session", "KIWILM2_DATA_DIR": str(data),
        "KIWILM_RESULT_DIR": str(tmp_path / "results"), "KIWILM2_UPLOAD_WORKERS": "3",
        "FAKE_CALLS": str(log), "FAKE_REMOTE": str(tmp_path / "remote"),
        "KIWILM2_RESUME_FROM": "", "KIWILM2_TPU_DRIVE_CACHE": "",
    })
    return environment, log


def calls(log: Path) -> list[list[str]]:
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


@pytest.mark.parametrize("cache_hit", [False, True])
def test_setup_only_and_explicit_train_lifecycle(launcher, cache_hit: bool) -> None:
    env, log = launcher
    if cache_hit:
        env["FAKE_CACHE_HIT"] = "1"
    setup = subprocess.run(
        ["bash", str(SCRIPT)], env=env, cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert setup.returncode == 0, setup.stdout + setup.stderr
    assert "NO TRAINING STARTED" in setup.stdout
    assert "remains billable" in setup.stdout
    setup_calls = calls(log)
    assert any(command[0] == "new" for command in setup_calls)
    assert not any(command[0] == "stop" for command in setup_calls)
    assert not any("KIWILM2_TPU_ACTION=train" in command for command in setup_calls)
    data_uploads = [command for command in setup_calls if command[0] == "upload"
                    and "/kiwilm-data-artifacts/" in command[-1]]
    assert bool(data_uploads) is not cache_hit
    if data_uploads:
        assert data_uploads[-1][-1].endswith("artifact-manifest.json")
        assert all("data.tar.gz.part-" in command[-1] for command in data_uploads[:-1])
    result = Path(env["KIWILM_RESULT_DIR"])
    assert (result / "setup-owner.json").is_file()
    train = subprocess.run(
        ["bash", str(SCRIPT), "train"], env=env, cwd=ROOT,
        capture_output=True, text=True, timeout=60,
    )
    assert train.returncode == 0, train.stdout + train.stderr
    later = calls(log)[len(setup_calls):]
    assert any("KIWILM2_TPU_ACTION=train" in command for command in later)
    assert later[-1] == ["stop", "-s", "fake-owned-session"]
    assert (result / "latest.pt").read_bytes() == b"simulated checkpoint"


def test_failed_setup_stops_only_its_new_session(launcher) -> None:
    env, log = launcher
    env["FAKE_FAIL_SETUP"] = "1"
    result = subprocess.run(
        ["bash", str(SCRIPT), "setup"], env=env, cwd=ROOT,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    recorded = calls(log)
    assert recorded[-1] == ["stop", "-s", "fake-owned-session"]
    assert not any("KIWILM2_TPU_ACTION=train" in command for command in recorded)
    assert not (Path(env["KIWILM_RESULT_DIR"]) / "setup-owner.json").exists()


def test_train_without_owner_never_calls_colab(launcher) -> None:
    env, log = launcher
    result = subprocess.run(
        ["bash", str(SCRIPT), "train"], env=env, cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "run setup first" in result.stderr
    assert calls(log) == []


def test_invalid_upload_limit_rejected_before_allocation(launcher) -> None:
    env, log = launcher
    env["KIWILM2_UPLOAD_WORKERS"] = "5"
    result = subprocess.run(
        ["bash", str(SCRIPT), "setup"], env=env, cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0 and "must be 1-4" in result.stderr
    assert calls(log) == []


def test_mismatched_remote_lock_never_trains_or_stops_foreign_session(launcher) -> None:
    env, log = launcher
    result_dir = Path(env["KIWILM_RESULT_DIR"])
    result_dir.mkdir()
    job = {"schema_version": 1, "use_drive": False, "data_fingerprint": "a" * 64,
           "tokenizer_sha256": "b" * 64}
    (result_dir / "tpu-job.json").write_text(json.dumps(job))
    (result_dir / "setup-owner.json").write_text(json.dumps({
        "session": "foreign-session", "tpu": "v6e1", "job_digest": "stale",
    }))
    remote = Path(env["FAKE_REMOTE"]) / "content"
    remote.mkdir(parents=True)
    (remote / "kiwilm-tpu-setup.json").write_text('{"state":"ready","job_digest":"stale"}')
    result = subprocess.run(
        ["bash", str(SCRIPT), "train"], env=env, cwd=ROOT,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert [command[0] for command in calls(log)] == ["download"]
