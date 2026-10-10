"""Fake Colab control plane + real tiny CPU workers. Never rent cloud hardware."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from kiwilm.colab_artifacts import create_colab_artifacts, file_sha256
from kiwilm.v3 import colab_cli as cli
from kiwilm.v3.accelerator import AcceleratorTrainConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bootstrap():
    spec = importlib.util.spec_from_file_location("v3_colab_bootstrap", cli.BOOTSTRAP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=[None, "L4"])
def frozen(tmp_path, request):
    wheel = tmp_path / "kiwilm-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"fake reviewed wheel; never installed")
    args = cli.parser().parse_args(
        [
            "setup",
            "--wheel",
            str(wheel),
            "--drive-mount",
            "existing",
            "--state-dir",
            str(tmp_path / "first"),
            *(["--gpu", request.param] if request.param else []),
        ]
    )
    spec = cli.make_spec(args, args.state_dir)
    return args.state_dir, spec


class FakeColab(cli.Launcher):
    def __init__(self, directory):
        super().__init__(directory)
        self.calls = []
        self.files = {}
        self.stage = "prepared"
        self.alive = False
        self.wrong_owner = False
        self.fail_action = None

    def command(self, args, *, timeout=120):
        args = list(map(str, args))
        self.calls.append(args)
        _owner, spec = cli.load_state(self.directory)
        token, identity = spec["token"], cli.canonical_digest(spec)
        if args[0] == self.fail_action:
            raise TimeoutError("simulated socket loss")
        if args[0] == "status":
            return f"[colab] Session '{spec['session']}' not found."
        if args[0] == "download":
            Path(args[-1]).write_bytes(self.files[Path(args[-2]).name])
        if args[0] == "log" and "-o" in args:
            Path(args[-1]).write_text('{"event":"fake_history"}\n')
        if args[0] != "exec":
            return ""
        env = dict(args[i + 1].split("=", 1) for i, v in enumerate(args) if v == "--env")
        action = env["KIWILM3_REMOTE_ACTION"]
        if action == self.fail_action:
            raise TimeoutError("simulated socket loss")
        value = {"token": "wrong" if self.wrong_owner else token, "spec_sha256": identity}
        if action == "setup":
            value.update(stage="prepared", training_started=False)
        elif action == "preflight":
            self.stage = "ready"
            value.update(stage="ready", training_started=False)
            self.files["preflight.json"] = json.dumps({"identity": "a" * 64}).encode()
            self.files["inputs.json"] = json.dumps(
                {"data_fingerprint": "b" * 64, "tokenizer_sha256": "c" * 64}
            ).encode()
        elif action == "start":
            self.stage, self.alive = "training", True
            value.update(stage="submitted", training_started=True)
        elif action in {"status", "logs"}:
            value.update(state={"stage": self.stage}, worker_alive=self.alive, logs={})
        return cli.PREFIX + json.dumps(value) + "\n"


def test_default_plan_and_shell_are_nonallocating(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("cloud command"))
    assert cli.main([]) == 0
    assert not list(tmp_path.iterdir())
    # Shell syntax only: never execute setup/new/drivemount.
    assert os.system(f"bash -n {ROOT / 'scripts/run_colab_kiwilm3_tpu.sh'}") == 0
    assert os.system(f"bash -n {ROOT / 'scripts/run_colab_kiwilm3_gpu.sh'}") == 0
    assert cli.main([], default_gpu="L4") == 0


def test_lr_diagnostic_launcher_plan_and_locked_defaults(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location(
        "lr_diagnostic_launcher", ROOT / "scripts/run_colab_kiwilm3_lr_diagnostic.py"
    )
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    assert (
        subprocess.run(
            ["bash", "-n", str(ROOT / "scripts/run_colab_kiwilm3_lr_diagnostic.sh")], check=False
        ).returncode
        == 0
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("cloud command"))
    assert launcher.main([]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert not plan["training_started"] and not plan["cloud_actions"]
    assert plan["config"]["max_tokens"] == 5_000_000
    assert plan["config"]["learning_rate"] == 0.0003
    assert plan["config"]["min_learning_rate"] == 0.00003
    assert plan["config"]["warmup_tokens"] == 1_000_000
    assert not list(tmp_path.iterdir())
    calls = []
    monkeypatch.setattr(launcher, "colab_main", lambda args, **kwargs: calls.append(args) or 0)
    assert launcher.main(["run", "--state-dir", "my-state"]) == 0
    parsed = cli.parser(default_gpu="L4").parse_args(calls[-1])
    assert parsed.state_dir == Path("my-state") and parsed.collapse_diagnostics
    assert parsed.profile == "smoke" and parsed.candidate == "hybrid-12"
    assert parsed.expected_data_fingerprint == launcher.DATA_FINGERPRINT
    assert parsed.expected_tokenizer_sha256 == launcher.TOKENIZER_SHA256
    wheel = tmp_path / "kiwilm-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"fake wheel")
    parsed.wheel, parsed.drive_mount = wheel, "existing"
    frozen = cli.make_spec(parsed, parsed.state_dir)
    assert frozen["request"]["config"] == plan["config"]
    assert frozen["request"]["collapse_diagnostics"] is True
    assert frozen["request"]["qualification"] is False
    assert frozen["request"]["stop_after_step"] is None
    assert frozen["request"]["run_name"] == "lr3e4-diagnostic-5m"
    assert frozen["expected_inputs"]["data_fingerprint"] == launcher.DATA_FINGERPRINT
    for option in (
        "--precision=fp32",
        "--controls=other.json",
        "--candidate=attention-12",
        "--wheel=old.whl",
    ):
        with pytest.raises(ValueError, match="locked"):
            launcher.main(["run", option])


def test_monitor_flag_and_input_provenance_survive_spec_resume(frozen, tmp_path):
    directory, original = frozen
    if original.get("gpu") is None:
        malformed = json.loads(json.dumps(original))
        malformed["request"]["collapse_diagnostics"] = True
        with pytest.raises(ValueError, match="GPU"):
            cli.validate_spec(malformed)
        return
    wheel = directory / "package" / original["wheel_name"]
    args = cli.parser(default_gpu="L4").parse_args(
        [
            "setup",
            "--wheel",
            str(wheel),
            "--drive-mount",
            "existing",
            "--state-dir",
            str(tmp_path / "diagnostic"),
            "--collapse-diagnostics",
            "--expected-data-fingerprint",
            "b" * 64,
            "--expected-tokenizer-sha256",
            "c" * 64,
        ]
    )
    spec = cli.make_spec(args, args.state_dir)
    assert spec["request"]["collapse_diagnostics"] is True
    launcher = FakeColab(args.state_dir)
    launcher.setup(spec, mount="existing")
    launcher.preflight()
    resume = cli.parser(default_gpu="L4").parse_args(
        [
            "resume-run",
            "--from-state",
            str(args.state_dir),
            "--state-dir",
            str(tmp_path / "resume-diag"),
            "--drive-mount",
            "existing",
        ]
    )
    restored = cli.make_spec(resume, resume.state_dir)
    assert restored["expected_inputs"] == spec["expected_inputs"]
    assert restored["request"]["collapse_diagnostics"] is True
    assert restored["request"]["config"] == spec["request"]["config"]
    malformed = json.loads(json.dumps(spec))
    malformed["expected_inputs"].pop("tokenizer_sha256")
    with pytest.raises(ValueError, match="expected inputs"):
        cli.validate_spec(malformed)


def test_fresh_input_mismatch_refused_before_training(bootstrap, tmp_path):
    root = tmp_path / "vm"
    root.mkdir()
    spec = {
        "request": {
            "qualification": True,
            "data_dir": str(root / "data"),
            "tokenizer_path": str(root / "tokenizer.json"),
        },
        "expected_inputs": {"data_fingerprint": "b" * 64, "tokenizer_sha256": "c" * 64},
    }
    with pytest.raises(ValueError, match="requested provenance"):
        bootstrap.prepare_inputs(root, spec)
    assert not (root / "inputs.json").exists()
    assert not (root / "run").exists()


@pytest.mark.parametrize(
    "options",
    [
        ["--profile", "smoke"],
        ["--backup-root", "/content/drive/MyDrive/../outside"],
        ["--candidate", "hadamard-12"],
        ["--tpu", "A100"],
        ["--gpu", "V100"],
        ["--gpu", "T4"],
        ["--gpu", "L4", "--tpu", "v5e1"],
        ["--tpu", "v5e1", "--precision", "fp32"],
    ],
)
def test_invalid_plan_never_allocates(tmp_path, monkeypatch, options):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("cloud command"))
    with pytest.raises((ValueError, SystemExit)):
        cli.main(
            ["setup", "--state-dir", str(tmp_path / "state"), "--drive-mount", "existing", *options]
        )
    assert not (tmp_path / "state").exists()


def test_staged_setup_preflight_and_explicit_detached_training(frozen):
    path, spec = frozen
    fake = FakeColab(path)
    assert not fake.setup(spec, mount="existing")["training_started"]
    assert [c[0] for c in fake.calls].count("new") == 1
    kind, name = cli.hardware(spec)
    assert [c for c in fake.calls if c[0] == "new"] == [
        ["new", "-s", spec["session"], f"--{kind}", name]
    ]
    assert len([c for c in fake.calls if c[0] == "upload"]) == 3
    assert not any("KIWILM3_REMOTE_ACTION=start" in c for c in fake.calls)
    with pytest.raises(FileNotFoundError):
        fake.train()
    fake.preflight()
    assert fake.train()["training_started"]
    with pytest.raises(RuntimeError, match="not ready"):
        fake.train()
    assert not any(c[0] == "stop" for c in fake.calls)


def test_missing_or_stale_owner_refuses_cloud_stop(frozen):
    path, _spec = frozen
    fake = FakeColab(path)
    fake.wrong_owner = True
    with pytest.raises(ValueError, match="ownership"):
        fake.stop()
    assert not any(c[0] == "stop" for c in fake.calls)
    (path / "bootstrap.py").write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        fake.remote("status")


def test_unknown_allocation_vs_owned_setup_failure_cleanup(frozen):
    path, spec = frozen
    fake = FakeColab(path)
    fake.fail_action = "new"
    with pytest.raises(TimeoutError):
        fake.setup(spec, mount="existing")
    assert not any(c[0] == "stop" for c in fake.calls)
    fake.calls.clear()
    fake.fail_action = "setup"
    with pytest.raises(TimeoutError):
        fake.setup(spec, mount="existing")
    assert any(c[0] == "stop" for c in fake.calls)


def test_unknown_training_launch_never_stops_or_relaunches(frozen):
    path, _spec = frozen
    fake = FakeColab(path)
    fake.preflight()
    fake.fail_action = "start"
    with pytest.raises(TimeoutError):
        fake.train()
    assert (path / "train-intent.json").is_file()
    assert not any(c[0] == "stop" for c in fake.calls)
    with pytest.raises(FileExistsError):
        fake.train()
    fake.fail_action = "logs"
    with pytest.raises(TimeoutError):
        fake.watch(stop_when_done=True)
    assert not any(c[0] == "stop" for c in fake.calls)


def test_control_plane_diagnostics_work_without_kernel(frozen):
    path, _spec = frozen
    fake = FakeColab(path)
    fake.fail_action = "status"
    result = fake.diagnose()
    assert "error" in result["checks"]["session"]
    assert "output" in result["checks"]["history"]
    assert [c[0] for c in fake.calls] == ["status", "log"]
    assert not result["training_started"]


def test_resume_reuses_exact_artifacts_controls_and_job_identity(frozen, tmp_path):
    source, original = frozen
    FakeColab(source).preflight()
    args = cli.parser().parse_args(
        [
            "resume",
            "--from-state",
            str(source),
            "--state-dir",
            str(tmp_path / "second"),
            "--drive-mount",
            "existing",
        ]
    )
    resumed = cli.make_spec(args, args.state_dir)
    assert resumed["wheel_sha256"] == original["wheel_sha256"]
    assert resumed["bootstrap_sha256"] == original["bootstrap_sha256"]
    assert resumed["expected_job_identity"] == "a" * 64
    assert resumed["expected_inputs"]["data_fingerprint"] == "b" * 64
    assert resumed["request"]["config"] == original["request"]["config"]
    assert cli.hardware(resumed) == cli.hardware(original)
    assert resumed["request"]["run_name"] == original["request"]["run_name"]
    assert resumed["request"]["mode"] == "resume"
    assert resumed["request"]["require_new_vm"] and resumed["request"]["stop_after_step"] is None
    assert resumed["token"] != original["token"] and resumed["session"] != original["session"]
    assert resumed["request"]["data_dir"] != original["request"]["data_dir"]
    with pytest.raises(FileExistsError):
        cli.make_spec(args, args.state_dir)


def test_resume_rejects_corrupt_saved_wheel(frozen, tmp_path):
    source, spec = frozen
    (source / "package" / spec["wheel_name"]).write_bytes(b"bad")
    args = cli.parser().parse_args(
        [
            "resume",
            "--from-state",
            str(source),
            "--state-dir",
            str(tmp_path / "second"),
            "--drive-mount",
            "existing",
        ]
    )
    with pytest.raises(ValueError, match="changed"):
        cli.make_spec(args, args.state_dir)
    assert not args.state_dir.exists()


def test_chunk_collection_retry_and_verified_stop(frozen, tmp_path):
    path, spec = frozen
    fake = FakeColab(path)
    fake.stage = "paused"
    source = tmp_path / "spec.json"
    source.write_text(json.dumps(spec))
    manifest = create_colab_artifacts({"spec.json": source}, tmp_path / "parts", chunk_size=128)
    fake.files = {p.name: p.read_bytes() for p in manifest.parent.iterdir() if p.is_file()}
    fake.fail_action = "download"
    with pytest.raises(TimeoutError):
        fake.collect()
    assert (path / "downloads/download-owner.json").exists()
    fake.fail_action = None
    fake.collect()
    assert (path / "downloads/download-complete.json").is_file()
    assert json.loads((path / "downloads/spec.json").read_text()) == spec
    with pytest.raises(FileExistsError, match="already collected"):
        fake.collect()
    fake.stop()
    assert fake.calls[-1][0] == "stop"


def collection_fixture(path, spec, tmp_path):
    fake = FakeColab(path)
    fake.stage = "complete"
    source = tmp_path / "spec.json"
    source.write_text(json.dumps(spec))
    manifest = create_colab_artifacts({"spec.json": source}, tmp_path / "parts", chunk_size=128)
    fake.files = {p.name: p.read_bytes() for p in manifest.parent.iterdir() if p.is_file()}
    return fake, json.loads(manifest.read_text())["parts"]


def test_collection_survives_cwd_change(frozen, tmp_path, monkeypatch):
    path, spec = frozen
    monkeypatch.chdir(tmp_path)
    fake, _ = collection_fixture(Path(path.name), spec, tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    command = fake.command

    def changed_cwd(args, **kwargs):
        result = command(args, **kwargs)
        if args[0] == "download":
            monkeypatch.chdir(other)
        return result

    monkeypatch.setattr(fake, "command", changed_cwd)
    fake.collect()
    assert (path / "downloads/download-complete.json").exists()
    assert not list(other.iterdir())


def test_moved_state_mid_collection_keeps_chunks_and_resumes(frozen, tmp_path, monkeypatch):
    path, spec = frozen
    fake, parts = collection_fixture(path, spec, tmp_path)
    command = fake.command
    moved = tmp_path / "moved-state"

    def move_state(args, **kwargs):
        result = command(args, **kwargs)
        if args[0] == "download" and Path(args[-2]).name == parts[1]["name"]:
            path.rename(moved)
        return result

    monkeypatch.setattr(fake, "command", move_state)
    with pytest.raises(RuntimeError, match="folder disappeared or changed"):
        fake.collect()
    assert not path.exists()  # Never recreate a deleted/moved owner directory.
    assert (moved / "downloads" / parts[0]["name"]).exists()
    assert not (moved / "downloads" / parts[1]["name"]).exists()
    moved.rename(path)
    monkeypatch.setattr(fake, "command", command)
    first_count = sum(
        c[0] == "download" and Path(c[-2]).name == parts[0]["name"] for c in fake.calls
    )
    fake.collect()
    assert (
        sum(c[0] == "download" and Path(c[-2]).name == parts[0]["name"] for c in fake.calls)
        == first_count
    )
    assert (path / "downloads/download-complete.json").exists()
    assert not any(c[0] in {"new", "stop"} for c in fake.calls)


def test_partial_chunk_copy_never_publishes_corrupt_final_file(frozen, tmp_path, monkeypatch):
    path, spec = frozen
    fake, parts = collection_fixture(path, spec, tmp_path)
    copy = cli.shutil.copyfileobj

    def interrupted(source, target, *args, **kwargs):
        target.write(source.read(10))
        raise OSError("simulated local disk interruption")

    monkeypatch.setattr(cli.shutil, "copyfileobj", interrupted)
    with pytest.raises(OSError, match="disk interruption"):
        fake.collect()
    assert not (path / "downloads" / parts[0]["name"]).exists()
    assert not list((path / "downloads").glob("*.pending"))
    monkeypatch.setattr(cli.shutil, "copyfileobj", copy)
    fake.collect()
    assert (path / "downloads/download-complete.json").exists()


def test_command_never_recreates_missing_state_directory(tmp_path, monkeypatch):
    path = tmp_path / "missing"
    launcher = cli.Launcher(path)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("cloud command"))
    with pytest.raises(RuntimeError, match="state folder is missing"):
        launcher.command(["status"])
    assert not path.exists()


def test_remote_start_is_detached_claimed_once(bootstrap, frozen, tmp_path, monkeypatch):
    _path, spec = frozen
    root = tmp_path / "remote"
    root.mkdir()
    monkeypatch.setattr(bootstrap, "boot_id", lambda: "vm-test")
    bootstrap.stage(root, spec, "ready")
    commands = []

    def popen(*args, **kwargs):
        commands.append((args, kwargs))
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(bootstrap.subprocess, "Popen", popen)
    result = bootstrap.start(root, spec)
    assert result["training_started"] and commands[0][1]["start_new_session"]
    assert "supervise" in commands[0][0][0]
    assert (root / "training.claim").is_file()
    with pytest.raises(RuntimeError, match="preflight"):
        bootstrap.start(root, spec)
    inspected = bootstrap.inspect(root, spec)
    assert inspected["state"]["stage"] == "interrupted"


def test_native_supervisor_uses_v3_cpu_engine_and_preserves_state(bootstrap, tmp_path, monkeypatch):
    # Bounded synthetic CPU optimization only, not simulated TPU acceptance.
    root = tmp_path / "vm"
    (root / "environment/bin").mkdir(parents=True)
    (root / "environment/bin/python").symlink_to(sys.executable)
    config = AcceleratorTrainConfig(
        device="cpu", precision="fp32", max_tokens=129, warmup_tokens=64, eval_batches=1
    )
    request = {
        "config": config.to_dict(),
        "candidate": "hybrid-12",
        "qualification": True,
        "run_name": "cli-local-test",
        "data_dir": str(root / "data"),
        "tokenizer_path": str(root / "tokenizer.json"),
        "run_dir": str(root / "run"),
        "backup_root": str(tmp_path / "drive"),
        "storage_root": None,
        "mode": "fresh",
        "stop_after_step": 1,
        "require_new_vm": False,
        "start_training": False,
    }
    spec = {"request": request, "token": "f" * 32}
    monkeypatch.setattr(bootstrap, "boot_id", lambda: "local-api-test")
    monkeypatch.setattr(
        bootstrap,
        "environment",
        lambda root: (
            Path(sys.executable),
            {"PATH": os.environ["PATH"], "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"},
        ),
    )
    inputs = bootstrap.prepare_inputs(root, spec)
    assert inputs == bootstrap.prepare_inputs(root, spec)
    result = bootstrap.native(root, "preflight", request)
    bootstrap.write(root / "preflight.json", result)
    bootstrap.supervise(root, spec)
    summary = json.loads((root / "summary.json").read_text())
    assert summary["status"] == "paused" and summary["step"] == 1
    assert (Path(summary["backup_dir"]) / "latest.json").exists()
    assert (root / "run/latest.pt").is_file()
    assert "v3_train" in (root / "worker.log").read_text()
    assert file_sha256(root / "run/latest.pt")


def test_notebooks_unchanged_and_bootstrap_has_no_session_allocation():
    source = cli.BOOTSTRAP.read_text()
    assert "colab new" not in source and "drivemount" not in source
    assert (ROOT / "notebooks/kiwilm3-tpu.ipynb").is_file()


def test_kernel_argv_is_not_mistaken_for_requested_action(bootstrap, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ipykernel_launcher.py", "-f", "kernel.json"])
    monkeypatch.setenv("KIWILM3_REMOTE_ACTION", "status")
    monkeypatch.setenv("KIWILM3_REMOTE_ROOT", str(tmp_path))
    calls = []
    monkeypatch.setattr(bootstrap, "dispatch", lambda action, root: calls.append(action) or {})
    bootstrap.main()
    assert calls == ["status"] and cli.PREFIX in capsys.readouterr().out


def test_remote_artifact_and_spec_locks_before_installer(bootstrap, frozen, tmp_path, monkeypatch):
    local, spec = frozen
    remote = tmp_path / f"kiwilm3-cli-{spec['token']}"
    remote.mkdir()
    for source in (
        local / "spec.json",
        local / "bootstrap.py",
        local / "package" / spec["wheel_name"],
    ):
        (remote / source.name).write_bytes(source.read_bytes())
    monkeypatch.setattr(
        bootstrap, "Path", lambda value: tmp_path if value == "/content" else Path(value)
    )
    monkeypatch.setenv("KIWILM3_EXPECTED_SPEC_SHA", cli.canonical_digest(spec))
    assert bootstrap.load(remote) == spec
    (remote / spec["wheel_name"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        bootstrap.load(remote)
    monkeypatch.setenv("KIWILM3_EXPECTED_SPEC_SHA", "0" * 64)
    with pytest.raises(ValueError, match="host ownership"):
        bootstrap.load(remote)


@pytest.mark.parametrize("gpu,precision", [("L4", "bf16"), ("A100", "bf16"), ("H100", "bf16")])
def test_gpu_specs_lock_device_precision_and_session(gpu, precision, tmp_path):
    wheel = tmp_path / "kiwilm-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"fake")
    args = cli.parser(default_gpu="L4").parse_args(
        ["setup", "--gpu", gpu, "--drive-mount", "existing", "--wheel", str(wheel)]
    )
    assert args.state_dir == Path("runs/colab/kiwilm3-gpu")
    spec = cli.make_spec(args, tmp_path / "state")
    assert spec["gpu"] == gpu and spec["tpu"] is None
    assert spec["request"]["config"]["device"] == "cuda"
    assert spec["request"]["config"]["precision"] == precision
    assert f"-{gpu.lower()}-" in spec["session"]
    assert spec["request"]["stop_after_step"] == 8
    assert spec["request"]["start_training"] is False
    # Both explicit modes are accepted on a native-BF16 card; never inferred on resume.
    if gpu == "L4":
        args.precision = "fp32"
        other = cli.make_spec(args, tmp_path / "fp32")
        assert other["request"]["config"]["precision"] == "fp32"


def test_gpu_defaults_and_explicit_tpu_override(tmp_path):
    wheel = tmp_path / "kiwilm-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"fake")
    for options, expected in [([], ("gpu", "L4")), (["--tpu", "v5e1"], ("tpu", "v5e1"))]:
        args = cli.parser(default_gpu="L4").parse_args(
            ["setup", "--wheel", str(wheel), "--drive-mount", "existing", *options]
        )
        spec = cli.make_spec(args, tmp_path / expected[0])
        assert cli.hardware(spec) == expected


@pytest.mark.parametrize(
    "values,options",
    [
        ({"device": "cpu", "precision": "fp32"}, []),
        ({"precision": "fp32"}, ["--precision", "bf16"]),
    ],
)
def test_gpu_conflicting_controls_refused_before_allocation(tmp_path, monkeypatch, values, options):
    controls = tmp_path / "controls.json"
    controls.write_text(json.dumps(values))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("cloud command"))
    with pytest.raises(ValueError):
        cli.main(
            [
                "setup",
                "--gpu",
                "L4",
                "--controls",
                str(controls),
                "--state-dir",
                str(tmp_path / "state"),
                "--drive-mount",
                "existing",
                *options,
            ]
        )
    assert not (tmp_path / "state").exists()


def test_bootstrap_environment_selects_backend_not_inherited_tpu(bootstrap, frozen, monkeypatch):
    path, spec = frozen
    monkeypatch.setenv("PJRT_DEVICE", "TPU")
    python, env = bootstrap.environment(path)
    assert python == path / "environment/bin/python"
    if spec["request"]["config"]["device"] == "xla":
        assert env["PJRT_DEVICE"] == "TPU"
    else:
        assert "PJRT_DEVICE" not in env
    assert env["PYTHONUNBUFFERED"] == "1"


def test_bootstrap_setup_passes_locked_device_to_isolated_installer(
    bootstrap, frozen, tmp_path, monkeypatch
):
    _path, spec = frozen
    wheel = tmp_path / spec["wheel_name"]
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "kiwilm/notebook_setup.py",
            "import json\n"
            "def prepare_environment(wheel, directory, device):\n"
            "    (wheel.parent / 'installer.json').write_text(json.dumps(device))\n",
        )
        archive.writestr("kiwilm/notebook_requirements.txt", "# fake installer only\n")
    monkeypatch.setattr(Path, "is_mount", lambda self: True)
    monkeypatch.setattr(bootstrap, "boot_id", lambda: "fake-vm")
    monkeypatch.setattr(bootstrap, "environment", lambda root: (Path("/fake/python"), {}))
    calls = []
    monkeypatch.setattr(bootstrap, "run_logged", lambda *a, **kw: calls.append((a, kw)))
    result = bootstrap.setup(tmp_path, spec)
    assert (
        json.loads((tmp_path / "installer.json").read_text()) == spec["request"]["config"]["device"]
    )
    assert result["stage"] == "prepared" and not result["training_started"]
    assert calls[0][0][1][-1] == "inputs"  # input preparation only, never optimizer steps


def test_gpu_preflight_refuses_wrong_allocated_card(bootstrap, tmp_path, monkeypatch):
    (tmp_path / "inputs.json").write_text("{}")
    monkeypatch.setattr(bootstrap, "boot_id", lambda: "vm")
    runtime = {"device": "cuda", "hardware": "Tesla T4"}
    result = {"identity": "a" * 64, "job": {"contract": {"runtime": runtime}}}
    monkeypatch.setattr(bootstrap, "native", lambda *a: result)
    spec = {"gpu": "L4", "token": "a" * 32, "request": {}}
    with pytest.raises(ValueError, match="allocated GPU"):
        bootstrap.preflight(tmp_path, spec)
    assert not (tmp_path / "preflight.json").exists()
    runtime["hardware"] = "NVIDIA L4"
    assert bootstrap.preflight(tmp_path, spec)["stage"] == "ready"


def test_combined_run_orders_stages_and_collects_before_stop(frozen, monkeypatch):
    path, spec = frozen
    fake = FakeColab(path)
    collected = []
    original = fake.train

    def finish():
        original()
        fake.stage, fake.alive = "paused", False

    monkeypatch.setattr(fake, "train", finish)
    monkeypatch.setattr(fake, "collect", lambda: collected.append("verified"))
    original_stop = fake.stop

    def stop():
        assert collected == ["verified"]
        return original_stop()

    monkeypatch.setattr(fake, "stop", stop)
    result = fake.run(spec, mount="existing", interval=5)
    assert result["state"]["stage"] == "paused"
    actions = [
        next(c.split("=", 1)[1] for c in call if c.startswith("KIWILM3_REMOTE_ACTION="))
        for call in fake.calls
        if call[0] == "exec"
    ]
    assert actions.index("setup") < actions.index("preflight") < actions.index("start")
    assert fake.calls[-1] == ["stop", "-s", spec["session"]]


@pytest.mark.parametrize(
    "failure,stops", [("new", False), ("preflight", True), ("start", False), ("logs", False)]
)
def test_combined_run_failure_boundaries(frozen, monkeypatch, failure, stops):
    path, spec = frozen
    fake = FakeColab(path)
    fake.fail_action = failure
    with pytest.raises(TimeoutError):
        fake.run(spec, mount="existing")
    assert any(c[0] == "stop" for c in fake.calls) == stops
    if failure in {"new", "preflight"}:
        assert not (path / "train-intent.json").exists()
    else:
        assert (path / "train-intent.json").exists()
    assert sum("KIWILM3_REMOTE_ACTION=start" in c for c in fake.calls) <= 1


def test_combined_run_collect_failure_keeps_vm_for_recovery(frozen, monkeypatch):
    path, spec = frozen
    fake = FakeColab(path)
    original = fake.train

    def finish():
        original()
        fake.stage, fake.alive = "complete", False

    def fail():
        raise TimeoutError("download failed")

    monkeypatch.setattr(fake, "train", finish)
    monkeypatch.setattr(fake, "collect", fail)
    with pytest.raises(TimeoutError, match="download failed"):
        fake.run(spec, mount="existing")
    assert not any(c[0] == "stop" for c in fake.calls)


@pytest.mark.parametrize("action", ["run", "resume-run"])
def test_combined_run_invalid_interval_never_creates_state_or_allocates(
    tmp_path, monkeypatch, action
):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("cloud command"))
    args = [action, "--interval", "1", "--state-dir", str(tmp_path / "state")]
    if action == "resume-run":
        args += ["--from-state", str(tmp_path / "missing")]
    with pytest.raises(ValueError, match="interval"):
        cli.main(args, default_gpu="L4")
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("action", ["run", "resume-run"])
def test_combined_commands_dispatch_once_with_exact_resume_artifacts(
    frozen, tmp_path, monkeypatch, action
):
    source, spec = frozen
    FakeColab(source).preflight()
    calls = []
    monkeypatch.setattr(
        cli.Launcher, "run", lambda self, value, **kw: calls.append((value, kw)) or {}
    )
    args = [
        action,
        "--state-dir",
        str(tmp_path / "combined"),
        "--drive-mount",
        "existing",
        "--interval",
        "5",
    ]
    if action == "resume-run":
        args += ["--from-state", str(source)]
    else:
        args += ["--wheel", str(source / "package" / spec["wheel_name"])]
        kind, name = cli.hardware(spec)
        args += [f"--{kind}", name]
    assert cli.main(args, default_gpu="L4") == 0
    assert len(calls) == 1 and calls[0][1] == {"mount": "existing", "interval": 5}
    saved = calls[0][0]
    assert cli.hardware(saved) == cli.hardware(spec)
    if action == "resume-run":
        assert saved["request"]["mode"] == "resume" and saved["request"]["stop_after_step"] is None
        assert saved["wheel_sha256"] == spec["wheel_sha256"]
        assert saved["bootstrap_sha256"] == spec["bootstrap_sha256"]
        assert saved["request"]["config"] == spec["request"]["config"]
