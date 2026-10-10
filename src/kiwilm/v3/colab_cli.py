"""User-run Colab CLI orchestration for V3; importing/planning never allocates.

Notebooks remain supported. The reviewed wheel/bootstrap are retained locally
and reused byte-for-byte on a fresh-VM resume. No V2 trainer/schema is used.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import suppress
from pathlib import Path, PurePosixPath

from kiwilm.colab_artifacts import file_sha256, reassemble_colab_artifacts
from kiwilm.v3.accelerator import AcceleratorTrainConfig
from kiwilm.v3.experiments import CANDIDATES, canonical_digest

ROOT = Path(__file__).resolve().parents[3]
BOOTSTRAP = ROOT / "scripts/colab_kiwilm3_tpu.py"
SCHEMA = "kiwilm3-colab-cli-v1"
PREFIX = "KIWILM3_RESULT="
GPUS = ("L4", "A100", "H100")


def hardware(spec):
    """Old TPU-only ownership records remain valid, without rewriting their digests."""
    gpu, tpu = spec.get("gpu"), spec.get("tpu")
    if gpu is not None:
        if gpu not in GPUS or tpu is not None:
            raise ValueError("select exactly one supported GPU or TPU; no fallback")
        return "gpu", gpu
    if tpu not in {"v5e1", "v6e1"}:
        raise ValueError("TPU must be v5e1 or v6e1; no fallback")
    return "tpu", tpu


def validate_backend(config, kind, name):
    if kind == "tpu":
        if config.device != "xla" or config.precision != "bf16":
            raise ValueError("TPU launcher requires explicit XLA BF16")
    elif config.device != "cuda":
        raise ValueError(f"GPU {name} launcher requires CUDA; no fallback")


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def drive_path(value):
    path = PurePosixPath(value)
    if ".." in path.parts or not path.is_relative_to("/content/drive/MyDrive"):
        raise ValueError("Drive paths must be beneath /content/drive/MyDrive without traversal")
    return str(path)


def validate_spec(spec):
    if spec.get("schema") != SCHEMA:
        raise ValueError("not a V3 CLI session specification")
    if not re.fullmatch(r"[0-9a-f]{32}", spec["token"]):
        raise ValueError("invalid session ownership token")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,100}", spec["session"]):
        raise ValueError("invalid session name")
    kind, name = hardware(spec)
    request = spec["request"]
    config = AcceleratorTrainConfig(**request["config"])
    validate_backend(config, kind, name)
    diagnostic = request.get("collapse_diagnostics", False)
    if type(diagnostic) is not bool or (diagnostic and kind != "gpu"):
        raise ValueError("collapse diagnostics require a GPU and a boolean policy flag")
    if request["candidate"] not in CANDIDATES or type(request["qualification"]) is not bool:
        raise ValueError("only the four dense B/C candidates are supported")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", request["run_name"]):
        raise ValueError("invalid experiment run name")
    if request["mode"] not in {"fresh", "resume"} or request.get("start_training") is not False:
        raise ValueError("spec must lock fresh/resume and keep training disabled")
    if request["storage_root"] != "/content/drive":
        raise ValueError("launcher uses the checked /content/drive mount")
    base = f"/content/kiwilm3-cli-{spec['token']}"
    for key, suffix in (
        ("data_dir", "data"),
        ("tokenizer_path", "v3-tokenizer.json"),
        ("run_dir", "run"),
    ):
        if request[key] != f"{base}/{suffix}":
            raise ValueError("VM paths differ from the isolated session root")
    drive_path(request["backup_root"])
    if not request["qualification"]:
        drive_path(request["data_cache"])
    if not re.fullmatch(r"kiwilm-[A-Za-z0-9_.+-]+\.whl", spec["wheel_name"]):
        raise ValueError("invalid wheel filename")
    for key in ("wheel_sha256", "bootstrap_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", spec[key]):
            raise ValueError("missing reviewed artifact checksum")
    stop = request["stop_after_step"]
    if stop is not None and (type(stop) is not int or stop <= 0):
        raise ValueError("pause step must be a positive integer")
    inputs = spec.get("expected_inputs")
    if inputs is not None and (
        not isinstance(inputs, dict)
        or set(inputs) != {"data_fingerprint", "tokenizer_sha256"}
        or any(
            not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in inputs.values()
        )
    ):
        raise ValueError("expected inputs must lock both data fingerprint and tokenizer SHA256")


def load_state(directory):
    owner = json.loads((directory / "owner.json").read_text())
    spec = json.loads((directory / "spec.json").read_text())
    validate_spec(spec)
    if owner["spec_sha256"] != canonical_digest(spec) or owner["token"] != spec["token"]:
        raise ValueError("local session owner/spec lock changed")
    if (
        file_sha256(directory / "package" / spec["wheel_name"]) != spec["wheel_sha256"]
        or file_sha256(directory / "bootstrap.py") != spec["bootstrap_sha256"]
    ):
        raise ValueError("saved reviewed wheel/bootstrap changed; do not rebuild for resume")
    return owner, spec


class Launcher:
    def __init__(self, directory, *, colab_bin="colab", colab_config=None):
        # Anchor once: a later cwd change must not redirect ownership/downloads.
        self.directory = Path(directory).absolute()
        self.cli = [colab_bin] + (["--config", str(colab_config)] if colab_config else [])

    def command(self, args, *, timeout=120):
        command = [*self.cli, *map(str, args)]
        print("[kiwilm3] " + " ".join(command), flush=True)
        if not (self.directory / "owner.json").is_file():
            raise RuntimeError(
                "Local Colab state folder is missing or was moved/deleted. "
                "Restore its original owner/spec/wheel/bootstrap before recovery; "
                "do not restart training or recreate empty state."
            )
        output = []
        log_path = self.directory / "launcher.log"
        started = last_heartbeat = time.monotonic()
        with (
            log_path.open("a", encoding="utf-8") as log,
            log_path.open(encoding="utf-8", errors="replace") as reader,
        ):
            log.write(json.dumps({"command": command, "time": time.time()}) + "\n")
            log.flush()
            reader.seek(0, os.SEEK_END)
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            try:
                while process.poll() is None:
                    if time.monotonic() - started > timeout:
                        raise TimeoutError(
                            f"Colab command timed out after {timeout}s; outcome may be unknown"
                        )
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=1)
                    chunk = reader.read()
                    output.append(chunk)
                    print(chunk, end="", flush=True)
                    if time.monotonic() - last_heartbeat >= 30:
                        print(
                            f"[kiwilm3] Waiting ({time.monotonic() - started:.0f}s); {log_path}",
                            flush=True,
                        )
                        last_heartbeat = time.monotonic()
                chunk = reader.read()
                output.append(chunk)
                print(chunk, end="", flush=True)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
            if process.returncode:
                raise RuntimeError(f"Colab command failed ({process.returncode}); see {log_path}")
        return "".join(output)

    def remote(self, action, *, timeout=120):
        _owner, spec = load_state(self.directory)
        output = self.command(
            [
                "exec",
                "-s",
                spec["session"],
                "--timeout",
                timeout - 10,
                "--env",
                f"KIWILM3_REMOTE_ROOT=/content/kiwilm3-cli-{spec['token']}",
                "--env",
                f"KIWILM3_REMOTE_ACTION={action}",
                "--env",
                f"KIWILM3_EXPECTED_SPEC_SHA={canonical_digest(spec)}",
                "-f",
                self.directory / "bootstrap.py",
            ],
            timeout=timeout,
        )
        results = []
        for line in output.splitlines():
            if PREFIX in line:
                results.append(json.loads(line.split(PREFIX, 1)[1]))
        if not results:
            raise RuntimeError("CLI returned no V3 receipt; connection/worker outcome is unknown")
        result = results[-1]
        if result.get("token") != spec["token"] or result.get("spec_sha256") != canonical_digest(
            spec
        ):
            raise ValueError("remote session ownership/spec mismatch; refusing action")
        return result

    def download_json(self, name):
        _owner, spec = load_state(self.directory)
        # Download to a new temporary path, then validate; retain history locally.
        with tempfile.TemporaryDirectory(prefix="v3-receipt-") as folder:
            target = Path(folder) / name
            self.command(
                [
                    "download",
                    "-s",
                    spec["session"],
                    f"/content/kiwilm3-cli-{spec['token']}/{name}",
                    target,
                ]
            )
            value = json.loads(target.read_text())
        if (
            name == "preflight.json"
            and spec.get("expected_job_identity")
            and (value["identity"] != spec["expected_job_identity"])
        ):
            raise ValueError("resume preflight differs from original native V3 job")
        target = self.directory / name
        if target.exists() and json.loads(target.read_text()) != value:
            raise ValueError("saved session receipt changed; retain originals and investigate")
        if not target.exists():
            write_new(target, value)
        return value

    def setup(self, spec, *, mount):
        # Only allocate after all local files/controls are verified.
        _owner, saved = load_state(self.directory)
        if saved != spec:
            raise ValueError("setup spec mismatch")
        session = spec["session"]
        kind, name = hardware(spec)
        status = self.command(["status", "-s", session])
        if f"Session '{session}' not found." not in status:
            raise RuntimeError("session already exists or status is ambiguous; never reuse it")
        allocated = False
        try:
            print(
                f"[kiwilm3] SETUP ONLY: allocating billable {kind.upper()} {name}. "
                "No optimizer steps.",
                flush=True,
            )
            self.command(["new", "-s", session, f"--{kind}", name], timeout=360)
            allocated = True
            base = f"/content/kiwilm3-cli-{spec['token']}"
            self.remote("initialize")
            # Upload only the small wheel/spec/bootstrap, never datasets or .venv.
            for source, name in (
                (self.directory / "package" / spec["wheel_name"], spec["wheel_name"]),
                (self.directory / "spec.json", "spec.json"),
                (self.directory / "bootstrap.py", "bootstrap.py"),
            ):
                self.command(["upload", "-s", session, source, f"{base}/{name}"])
            if mount == "cli":
                self.command(["drivemount", "-s", session], timeout=600)
            elif mount == "manual":
                self.command(["url", "-s", session])
                input("Mount Drive at /content/drive yourself, then press Enter: ")
            result = self.remote("setup", timeout=2400)
            write_new(self.directory / "setup.json", result)
        except BaseException:
            if allocated:
                # Setup never launches training. Stop only the session we just created.
                for command in (
                    ["log", "-s", session, "-o", self.directory / "setup-failure.jsonl"],
                    ["stop", "-s", session],
                ):
                    try:
                        self.command(command, timeout=60)
                    except Exception as cleanup_error:
                        print(
                            f"[kiwilm3] Cleanup failed: {cleanup_error}. "
                            f"Check billing/session and stop {session} manually.",
                            file=sys.stderr,
                        )
            else:
                print(
                    "[kiwilm3] Allocation may have timed out after success. "
                    "Check colab sessions/website for an orphan; no blind retry.",
                    file=sys.stderr,
                )
            raise
        print(
            f"[kiwilm3] Setup complete; {kind.upper()} remains billable. Next: preflight. "
            f"Stop: colab stop -s {session}"
        )
        return result

    def preflight(self):
        result = self.remote("preflight", timeout=1200)
        self.download_json("preflight.json")
        self.download_json("inputs.json")
        if not (self.directory / "ready.json").exists():
            write_new(self.directory / "ready.json", result)
        return result

    def run(self, spec, *, mount, interval=30):
        """Explicit one-command workflow; never retry an ambiguous training submission."""
        if not 5 <= interval <= 60:
            raise ValueError("watch interval must be 5-60 seconds")
        self.setup(spec, mount=mount)
        try:
            self.preflight()
        except BaseException:
            # No optimizer steps have been requested. Release only this owned VM.
            try:
                self.stop()
            except Exception as error:
                print(
                    f"[kiwilm3] Preflight failed; could not verify/stop the VM: {error}. "
                    f"Inspect billing/session and stop {spec['session']} manually when safe.",
                    file=sys.stderr,
                )
            raise
        # Once submission is attempted, a lost connection must leave the VM intact.
        self.train()
        return self.watch(interval=interval, stop_when_done=True)

    def train(self):
        _owner, spec = load_state(self.directory)
        ready = json.loads((self.directory / "ready.json").read_text())
        if ready.get("spec_sha256") != canonical_digest(spec) or ready.get("stage") != "ready":
            raise ValueError("matching local preflight receipt required")
        current = self.remote("status")
        if current["state"].get("stage") != "ready":
            raise RuntimeError("remote not ready; do not duplicate or implicitly restart training")
        write_new(self.directory / "train-intent.json", {"spec_sha256": canonical_digest(spec)})
        try:
            result = self.remote("start")
        except BaseException:
            print(
                "[kiwilm3] Launch outcome unknown. Worker may be running; do not stop/relaunch. "
                "Use status/logs. Training intent retained.",
                file=sys.stderr,
            )
            raise
        write_new(self.directory / "submitted.json", result)
        print(
            "[kiwilm3] Detached worker submitted. Closing this terminal does not stop it. "
            "Use watch/status/logs; stop the billable VM explicitly when done."
        )
        return result

    def collect(self):
        state = self.remote("status")
        if state["worker_alive"] or state["state"].get("stage") in {"launching", "training"}:
            raise RuntimeError("worker still running; use periodic Drive checkpoints/status/logs")
        _owner, spec = load_state(self.directory)
        base = f"/content/kiwilm3-cli-{spec['token']}"
        output = self.directory / "downloads"
        owner_path = output / "download-owner.json"
        expected_owner = {"spec_sha256": canonical_digest(spec)}
        if output.exists() and (
            not owner_path.is_file() or json.loads(owner_path.read_text()) != expected_owner
        ):
            raise FileExistsError("download directory is unmanaged or belongs to another session")
        if (output / "download-complete.json").exists():
            raise FileExistsError("verified artifacts already collected; never overwrite them")
        # Packaging is idempotent from the host: if a prior packaging attempt
        # succeeded but the socket dropped, use its immutable existing manifest.
        try:
            self.remote("collect", timeout=600)
        except (RuntimeError, TimeoutError):
            print("[kiwilm3] Packaging receipt missing; trying an existing artifact manifest.")
        output.mkdir(exist_ok=True)
        if not owner_path.exists():
            write_new(owner_path, expected_owner)
        output_identity = (output.stat().st_dev, output.stat().st_ino)

        def check_destination():
            try:
                load_state(self.directory)
                stat = output.stat()
                valid = (
                    not output.is_symlink()
                    and (stat.st_dev, stat.st_ino) == output_identity
                    and json.loads(owner_path.read_text()) == expected_owner
                )
            except (OSError, ValueError, KeyError) as error:
                raise RuntimeError(
                    "Local download/state folder disappeared or changed during collection. "
                    "Restore the original state directory and rerun collect; "
                    "do not rerun training. Existing verified chunks are reusable."
                ) from error
            if not valid:
                raise RuntimeError(
                    "Local download directory ownership/identity changed; stop recovery"
                )

        check_destination()
        manifest = output / "artifact-manifest.json"
        if not manifest.exists():
            with tempfile.TemporaryDirectory(prefix="v3-download-") as folder:
                pending = Path(folder) / "artifact-manifest.json"
                self.command(
                    [
                        "download",
                        "-s",
                        spec["session"],
                        f"{base}/artifacts/artifact-manifest.json",
                        pending,
                    ]
                )
                check_destination()
                write_new(manifest, json.loads(pending.read_text()))
        description = json.loads(manifest.read_text())
        if description.get("schema_version") != 1 or not description.get("parts"):
            raise ValueError("invalid download manifest")
        for part in description["parts"]:
            check_destination()
            name = part["name"]
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or "\\" in name
                or name in {".", ".."}
            ):
                raise ValueError("unsafe artifact part name")
            target = output / name
            if target.exists():
                if target.stat().st_size != part["bytes"] or file_sha256(target) != part["sha256"]:
                    raise ValueError(
                        "existing downloaded part is corrupt; preserve and investigate"
                    )
                continue
            with tempfile.TemporaryDirectory(prefix="v3-part-") as folder:
                pending = Path(folder) / name
                self.command(
                    ["download", "-s", spec["session"], f"{base}/artifacts/{name}", pending]
                )
                if (
                    pending.stat().st_size != part["bytes"]
                    or file_sha256(pending) != part["sha256"]
                ):
                    raise ValueError("downloaded part checksum/size mismatch")
                check_destination()
                # A failed copy must not leave a final-named, corrupt chunk that
                # prevents the next collect from resuming. Publish without overwrite.
                descriptor, staged_name = tempfile.mkstemp(
                    prefix=f".{name}.", suffix=".pending", dir=output
                )
                staged = Path(staged_name)
                try:
                    with os.fdopen(descriptor, "wb") as destination, pending.open("rb") as source:
                        shutil.copyfileobj(source, destination)
                        destination.flush()
                        os.fsync(destination.fileno())
                    check_destination()
                    os.link(staged, target)
                finally:
                    staged.unlink(missing_ok=True)
        check_destination()
        for name, details in description.get("files", {}).items():
            if Path(name).name != name or "\\" in name or name in {".", ".."}:
                raise ValueError("unsafe artifact file name")
            if (output / name).exists() and file_sha256(output / name) != details["sha256"]:
                raise FileExistsError("refuse overwriting unrelated file in download directory")
        files = reassemble_colab_artifacts(manifest, output)
        if json.loads((output / "spec.json").read_text()) != spec:
            raise ValueError("downloaded run belongs to a different session")
        self.command(["log", "-s", spec["session"], "-o", output / "session.jsonl"])
        write_new(output / "download-complete.json", expected_owner)
        return {"files": [str(path) for path in files], "training_started": False}

    def stop(self):
        # A stale local owner cannot stop a new unrelated session with the same name.
        self.remote("status")
        _owner, spec = load_state(self.directory)
        return self.command(["stop", "-s", spec["session"]])

    def diagnose(self):
        """Control-plane diagnostics that do not require a responsive notebook kernel."""
        _owner, spec = load_state(self.directory)
        checks = {}
        for name, command in (
            ("session", ["status", "-s", spec["session"]]),
            ("history", ["log", "-s", spec["session"], "-n", "30"]),
        ):
            try:
                checks[name] = {"output": self.command(command, timeout=60)}
            except Exception as error:
                checks[name] = {"error": str(error)}
        return {
            "session": spec["session"],
            "tpu": spec["tpu"],
            "gpu": spec.get("gpu"),
            "checks": checks,
            "training_started": False,
            "note": "No kernel exec, allocation, training, restart or stop requested.",
        }

    def watch(self, *, interval=30, stop_when_done=False):
        if not 5 <= interval <= 60:
            raise ValueError("watch interval must be 5-60 seconds")
        try:
            while True:
                result = self.remote("logs")
                print(json.dumps(result, indent=2), flush=True)
                state = result["state"].get("stage")
                if (
                    state in {"complete", "paused", "failed", "interrupted"}
                    and not result["worker_alive"]
                ):
                    self.collect()
                    if stop_when_done:
                        self.stop()
                    if state in {"failed", "interrupted"}:
                        raise RuntimeError(
                            "worker failed/interrupted; preserve Drive head and inspect logs"
                        )
                    return result
                time.sleep(interval)
        except BaseException:
            print(
                "[kiwilm3] Monitoring ended; no automatic VM stop on connection/interruption. "
                "Reattach with status/watch, or stop explicitly; it may still be billable.",
                file=sys.stderr,
            )
            raise


def make_spec(args, directory):
    """Freeze artifacts before allocation. Resume never rebuilds the reviewed package."""
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("state directory exists; use its actions or a new empty directory")
    if args.drive_mount in {"cli", "manual"} and not sys.stdin.isatty():
        raise ValueError("Drive authorization is interactive; run setup/resume in your terminal")
    source = None
    if args.action in {"resume", "resume-run"}:
        _owner, source = load_state(args.from_state)
        prior = json.loads((args.from_state / "preflight.json").read_text())
        inputs = json.loads((args.from_state / "inputs.json").read_text())
        wheel = args.from_state / "package" / source["wheel_name"]
        bootstrap = args.from_state / "bootstrap.py"
        request = dict(source["request"], mode="resume", stop_after_step=None, require_new_vm=True)
        expected_inputs = {key: inputs[key] for key in ("data_fingerprint", "tokenizer_sha256")}
        expected_job = prior["identity"]
        kind, name = hardware(source)
    else:
        if args.profile == "smoke" and not args.data_cache:
            raise ValueError("full-width smoke requires --data-cache; no hidden dataset download")
        values = json.loads(args.controls.read_text()) if args.controls else {}
        gpu = args.gpu or (args.default_gpu if not args.tpu else None)
        kind, name = ("gpu", gpu) if gpu else ("tpu", args.tpu or "v6e1")
        backend = {
            "device": "cuda" if gpu else "xla",
            "precision": "bf16",
        }
        if args.precision is not None:
            if "precision" in values and values["precision"] != args.precision:
                raise ValueError("--precision conflicts with the frozen controls")
            values["precision"] = args.precision
        config = AcceleratorTrainConfig(
            **(
                {
                    "max_tokens": 50_000_000,
                    "warmup_tokens": 1_000_000,
                    "batch_size": 8,
                    "grad_accum_steps": 4,
                    "checkpoint_interval": 500,
                    "eval_interval": 500,
                    "eval_batches": 20,
                }
                if args.profile == "smoke"
                else {}
            )
            | backend
            | values
        )
        request = {
            "config": config.to_dict(),
            "qualification": args.profile == "qualification",
            "candidate": args.candidate,
            "run_name": args.run_name
            or ("bc-smoke-50m" if args.profile == "smoke" else "recovery-qualification"),
            "storage_root": "/content/drive",
            "backup_root": drive_path(args.backup_root),
            "data_cache": drive_path(args.data_cache) if args.data_cache else "",
            "mode": "fresh",
            "require_new_vm": False,
            "start_training": False,
            "stop_after_step": args.stop_after_step
            if args.stop_after_step is not None
            else (8 if args.profile == "qualification" else None),
        }
        expected_inputs = expected_job = None
        if args.expected_data_fingerprint or args.expected_tokenizer_sha256:
            expected_inputs = {
                "data_fingerprint": args.expected_data_fingerprint,
                "tokenizer_sha256": args.expected_tokenizer_sha256,
            }
            if any(
                v is None or not re.fullmatch(r"[0-9a-f]{64}", v) for v in expected_inputs.values()
            ):
                raise ValueError("supply both expected data fingerprint and tokenizer SHA256")
        if args.collapse_diagnostics:
            if kind != "gpu":
                raise ValueError("collapse diagnostics require a GPU")
            request["collapse_diagnostics"] = True
        bootstrap = BOOTSTRAP
        wheel = args.wheel
    # Fail malformed controls before creating files or invoking any cloud CLI.
    config = AcceleratorTrainConfig(**request["config"])
    validate_backend(config, kind, name)
    directory.mkdir(parents=True, exist_ok=True)
    package = directory / "package"
    package.mkdir()
    if wheel is None:
        subprocess.run(["uv", "build", "--wheel", "--out-dir", str(package)], check=True, cwd=ROOT)
        wheels = list(package.glob("kiwilm-*.whl"))
        if len(wheels) != 1:
            raise ValueError("build must produce exactly one KiwiLM wheel")
        wheel = wheels[0]
    else:
        if not wheel.is_file():
            raise FileNotFoundError(wheel)
        shutil.copyfile(wheel, package / wheel.name)
        wheel = package / wheel.name
    shutil.copyfile(bootstrap, directory / "bootstrap.py")
    token = uuid.uuid4().hex
    session = args.session or (
        f"kiwilm3-{name.lower()}-{request['candidate']}-{request['mode']}-{token[:8]}"
    )
    base = f"/content/kiwilm3-cli-{token}"
    request.update(
        data_dir=f"{base}/data", tokenizer_path=f"{base}/v3-tokenizer.json", run_dir=f"{base}/run"
    )
    spec = {
        "schema": SCHEMA,
        "token": token,
        "session": session,
        "tpu": name if kind == "tpu" else None,
        "wheel_name": wheel.name,
        "wheel_sha256": file_sha256(wheel),
        "bootstrap_sha256": file_sha256(directory / "bootstrap.py"),
        "request": request,
        "expected_inputs": expected_inputs,
        "expected_job_identity": expected_job,
    }
    if kind == "gpu":
        spec["gpu"] = name
    validate_spec(spec)
    write_new(directory / "spec.json", spec)
    write_new(directory / "owner.json", {"token": token, "spec_sha256": canonical_digest(spec)})
    return spec


def parser(*, default_gpu=None):
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = result.add_subparsers(dest="action", required=True)
    for action in (
        "plan",
        "setup",
        "resume",
        "run",
        "resume-run",
        "preflight",
        "train",
        "status",
        "logs",
        "diagnose",
        "watch",
        "collect",
        "stop",
    ):
        item = sub.add_parser(action, allow_abbrev=False)
        item.set_defaults(default_gpu=default_gpu)
        item.add_argument(
            "--state-dir",
            type=Path,
            default=Path("runs/colab/kiwilm3-gpu" if default_gpu else "runs/colab/kiwilm3-tpu"),
        )
        item.add_argument("--colab-bin", default=os.environ.get("COLAB_BIN", "colab"))
        item.add_argument("--colab-config", type=Path)
        if action in {"setup", "resume", "run", "resume-run"}:
            item.add_argument("--session")
            item.add_argument("--drive-mount", choices=("cli", "manual", "existing"), default="cli")
        if action in {"setup", "run"}:
            item.add_argument(
                "--profile", choices=("qualification", "smoke"), default="qualification"
            )
            item.add_argument("--candidate", choices=CANDIDATES, default="hybrid-12")
            selection = item.add_mutually_exclusive_group()
            selection.add_argument("--tpu", choices=("v5e1", "v6e1"))
            selection.add_argument("--gpu", choices=GPUS)
            item.add_argument("--precision", choices=("fp32", "bf16"))
            item.add_argument("--run-name")
            item.add_argument("--controls", type=Path)
            item.add_argument("--collapse-diagnostics", action="store_true")
            item.add_argument("--expected-data-fingerprint")
            item.add_argument("--expected-tokenizer-sha256")
            item.add_argument("--wheel", type=Path)
            item.add_argument("--data-cache")
            item.add_argument("--backup-root", default="/content/drive/MyDrive/KiwiLM3")
            item.add_argument("--stop-after-step", type=int)
        if action in {"resume", "resume-run"}:
            item.add_argument("--from-state", required=True, type=Path)
        if action in {"watch", "run", "resume-run"}:
            item.add_argument("--interval", type=int, default=30)
        if action == "watch":
            item.add_argument("--stop-when-done", action="store_true")
    return result


def main(argv=None, *, default_gpu=None):
    arguments = sys.argv[1:] if argv is None else argv
    args = parser(default_gpu=default_gpu).parse_args(arguments or ["plan"])
    if args.action == "plan":
        print(
            "No cloud actions. Order: setup -> preflight -> train -> watch --stop-when-done.\n"
            "New-VM resume: resume --from-state ORIGINAL --state-dir NEW -> preflight -> train.\n"
            "Explicit combined workflows: run [setup options], or resume-run --from-state ORIGINAL "
            "--state-dir NEW. Both collect and stop when done.\n"
            "Notebooks supported. Default: tiny hybrid-12 recovery qualification on "
            + (f"NVIDIA {default_gpu}/CUDA BF16." if default_gpu else "TPU/XLA BF16.")
        )
        return 0
    launcher = Launcher(args.state_dir, colab_bin=args.colab_bin, colab_config=args.colab_config)
    if args.action in {"run", "resume-run"} and not 5 <= args.interval <= 60:
        raise ValueError("watch interval must be 5-60 seconds")
    if args.action in {"setup", "resume", "run", "resume-run"}:
        spec = make_spec(args, args.state_dir)
        if args.action in {"run", "resume-run"}:
            result = launcher.run(spec, mount=args.drive_mount, interval=args.interval)
        else:
            result = launcher.setup(spec, mount=args.drive_mount)
    elif args.action == "watch":
        result = launcher.watch(interval=args.interval, stop_when_done=args.stop_when_done)
    elif args.action in {"status", "logs"}:
        result = launcher.remote(args.action)
    else:
        result = getattr(launcher, args.action)()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
