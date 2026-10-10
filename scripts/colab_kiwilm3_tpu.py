"""Standalone CLI bootstrap on Colab; stdlib only until the isolated worker is ready.

Never allocate/mount here. The human-run host launcher owns those actions.
Training is detached from the kernel websocket, with append-only worker logs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback
import zipfile
from collections import deque
from contextlib import suppress
from pathlib import Path

PREFIX = "KIWILM3_RESULT="


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def checksum(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load(root):
    spec = json.loads((root / "spec.json").read_text())
    if digest(spec) != os.environ.get("KIWILM3_EXPECTED_SPEC_SHA"):
        raise ValueError("remote spec differs from the host ownership lock")
    token = spec["token"]
    if (
        spec["schema"] != "kiwilm3-colab-cli-v1"
        or root != Path("/content") / f"kiwilm3-cli-{token}"
        or len(token) != 32
        or any(c not in "0123456789abcdef" for c in token)
    ):
        raise ValueError("invalid V3 CLI root/token/schema")
    if checksum(root / spec["wheel_name"]) != spec["wheel_sha256"]:
        raise ValueError("reviewed wheel checksum changed")
    if checksum(root / "bootstrap.py") != spec["bootstrap_sha256"]:
        raise ValueError("reviewed CLI bootstrap checksum changed")
    return spec


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def stage(root, spec, name, **extra):
    state = {
        "stage": name,
        "spec_sha256": digest(spec),
        "token": spec["token"],
        "vm_id": boot_id(),
        "updated_at": time.time(),
        **extra,
    }
    write(root / "state.json", state)
    print(json.dumps({"event": "v3_colab_stage", **state}), flush=True)
    return state


def environment(root):
    python = root / "environment/bin/python"
    device = json.loads((root / "spec.json").read_text())["request"]["config"]["device"]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if device == "xla":
        env["PJRT_DEVICE"] = "TPU"
    else:
        env.pop("PJRT_DEVICE", None)
    env["LD_LIBRARY_PATH"] = (
        str(python.resolve().parent.parent / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    )
    return python, env


def run_logged(root, command, *, env=None, timeout=1800):
    """Retain all stderr, stream output, and emit heartbeats during quiet compilation."""
    log_path = root / "worker.log"
    started = last_heartbeat = time.monotonic()
    with (
        log_path.open("a", encoding="utf-8") as log,
        log_path.open(encoding="utf-8", errors="replace") as reader,
    ):
        reader.seek(0, os.SEEK_END)
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"worker exceeded {timeout}s; inspect {log_path}")
                with suppress(subprocess.TimeoutExpired):
                    child.wait(timeout=1)
                output = reader.read()
                if output:
                    print(output, end="", flush=True)
                if time.monotonic() - last_heartbeat >= 30:
                    print(
                        json.dumps(
                            {
                                "event": "v3_colab_heartbeat",
                                "pid": child.pid,
                                "elapsed_seconds": round(time.monotonic() - started),
                            }
                        ),
                        flush=True,
                    )
                    last_heartbeat = time.monotonic()
            print(reader.read(), end="", flush=True)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
        if child.returncode:
            raise RuntimeError(f"worker exited {child.returncode}; inspect {log_path}")


def native(root, action, request, *, timeout=1800):
    python, env = environment(root)
    path = root / f"request-{action}.json"
    write(path, request)
    # Results are captured in a per-action file, not parsed from historical logs.
    result_path = root / f"result-{action}.json"
    code = (
        "import json,sys; from pathlib import Path; "
        "from kiwilm.v3.accelerator_worker import execute; "
        "r=execute(sys.argv[1],json.loads(Path(sys.argv[2]).read_text())); "
        "Path(sys.argv[3]).write_text(json.dumps(r,sort_keys=True)+'\\n')"
    )
    run_logged(
        root,
        [str(python), "-c", code, action, str(path), str(result_path)],
        env=env,
        timeout=timeout,
    )
    return json.loads(result_path.read_text())


def prepare_inputs(root, spec):
    """Called only in the isolated Python worker; never on the notebook kernel."""
    from kiwilm.data import PreparedTokenData
    from kiwilm.v3.accelerator_worker import execute
    from kiwilm.v3.tokenizer import MaskBPETokenizer

    request = spec["request"]
    data, token = Path(request["data_dir"]), Path(request["tokenizer_path"])
    marker = root / "inputs.json"
    if not marker.exists():
        if data.exists() or token.exists():
            raise FileExistsError("partial input preparation exists; retain it and use a new VM")
        if request["qualification"]:
            execute("prepare-demo", request)
        else:
            execute("restore-data", request)
            MaskBPETokenizer.from_base(PreparedTokenData(data).tokenizer).save(token)
    prepared, tokenizer = PreparedTokenData(data), MaskBPETokenizer.load(token)
    tokenizer.assert_base_compatible(prepared.tokenizer)
    receipt = {
        "spec_sha256": digest(spec),
        "data_fingerprint": prepared.fingerprint,
        "tokenizer_sha256": tokenizer.fingerprint,
    }
    if marker.exists() and json.loads(marker.read_text()) != receipt:
        raise ValueError("prepared inputs changed since setup")
    if spec.get("expected_inputs") and spec["expected_inputs"] != {
        k: receipt[k] for k in ("data_fingerprint", "tokenizer_sha256")
    }:
        raise ValueError("prepared data/tokenizer differ from requested provenance")
    write(marker, receipt)
    return receipt


def setup(root, spec):
    if (root / "training.claim").exists():
        raise RuntimeError("training was already requested; setup cannot reset it")
    if not Path(spec["request"]["storage_root"]).is_mount():
        raise RuntimeError("Drive mount unavailable; no installation or training started")
    stage(root, spec, "installing")
    wheel = root / spec["wheel_name"]
    # Reuse the exact notebook setup from the reviewed wheel without importing
    # kiwilm/model dependencies in Colab's potentially different kernel Python.
    with zipfile.ZipFile(wheel) as archive:
        for name in ("notebook_setup.py", "notebook_requirements.txt"):
            (root / name).write_bytes(archive.read(f"kiwilm/{name}"))
    module_spec = importlib.util.spec_from_file_location(
        "v3_environment", root / "notebook_setup.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.prepare_environment(wheel, root / "environment", spec["request"]["config"]["device"])
    stage(root, spec, "preparing-inputs")
    python, env = environment(root)
    run_logged(root, [str(python), str(root / "bootstrap.py"), "inputs"], env=env)
    return stage(root, spec, "prepared", training_started=False)


def preflight(root, spec):
    if (root / "training.claim").exists():
        raise RuntimeError("training already requested; cannot rerun preflight on this VM")
    if not (root / "inputs.json").is_file():
        raise RuntimeError("finish setup first")
    stage(root, spec, "preflight")
    result = native(root, "preflight", spec["request"])
    if spec.get("gpu"):
        runtime = result["job"]["contract"]["runtime"]
        if runtime["device"] != "cuda" or not re.search(
            rf"\b{re.escape(spec['gpu'])}\b", runtime["hardware"], re.IGNORECASE
        ):
            raise ValueError("allocated GPU differs from requested hardware; no fallback")
    if spec.get("expected_job_identity") and result["identity"] != spec["expected_job_identity"]:
        raise ValueError("resume native job identity changed; original wheel/data/runtime required")
    write(root / "preflight.json", result)
    return stage(root, spec, "ready", training_started=False, job_identity=result["identity"])


def supervise(root, spec):
    try:
        stage(root, spec, "training", pid=os.getpid(), training_started=True)
        expected = json.loads((root / "preflight.json").read_text())["identity"]
        check = native(root, "preflight", spec["request"])
        if check["identity"] != expected:
            raise ValueError("job changed since preflight; refusing optimizer steps")
        result = native(root, "train", {**spec["request"], "start_training": True}, timeout=86_400)
        write(root / "summary.json", result)
        stage(
            root,
            spec,
            result["status"],
            pid=os.getpid(),
            training_started=True,
            step=result["step"],
            tokens_seen=result["tokens_seen"],
            backup_dir=result["backup_dir"],
        )
    except BaseException as error:
        traceback.print_exc()
        stage(root, spec, "failed", error=str(error), training_started=True)
        raise


def start(root, spec):
    ready = json.loads((root / "state.json").read_text())
    if ready["stage"] != "ready" or ready["spec_sha256"] != digest(spec):
        raise RuntimeError("matching ready preflight required before explicit train")
    # Never remove this claim automatically, even if the launch outcome is unknown.
    with (root / "training.claim").open("x") as claim:
        claim.write(digest(spec) + "\n")
    stage(root, spec, "launching", training_started=True)
    with (root / "supervisor.log").open("a") as log:
        process = subprocess.Popen(
            [sys.executable, str(root / "bootstrap.py"), "supervise"],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "KIWILM3_REMOTE_ROOT": str(root), "PYTHONUNBUFFERED": "1"},
        )
    # Supervisor owns subsequent state writes; don't race its training/failed update.
    write(root / "process.json", {"pid": process.pid, "vm_id": boot_id()})
    return {
        "stage": "submitted",
        "pid": process.pid,
        "token": spec["token"],
        "spec_sha256": digest(spec),
        "training_started": True,
    }


def inspect(root, spec, *, logs=False):
    state = json.loads((root / "state.json").read_text()) if (root / "state.json").exists() else {}
    if state and state["spec_sha256"] != digest(spec):
        raise ValueError("remote owner/spec mismatch")
    process = (
        json.loads((root / "process.json").read_text())
        if (root / "process.json").exists()
        else None
    )
    alive = False
    if process and process["vm_id"] == boot_id():
        command = Path(f"/proc/{process['pid']}/cmdline")
        alive = command.is_file() and str(root / "bootstrap.py").encode() in command.read_bytes()
    if state.get("stage") in {"training", "launching"} and not alive:
        state = {**state, "stage": "interrupted", "note": "worker gone; never restart implicitly"}
    tail = {}
    progress = None
    metrics = root / "run/metrics.jsonl"
    if metrics.is_file():
        with metrics.open(errors="replace") as stream:
            for line in deque(stream, maxlen=80):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a concurrently written partial line is not a committed step
                if row.get("event") == "v3_train":
                    progress = {
                        key: row.get(key)
                        for key in (
                            "step",
                            "tokens_seen",
                            "loss",
                            "learning_rate",
                            "tokens_per_second",
                        )
                    }
    if logs:
        for name in ("worker.log", "supervisor.log"):
            if (root / name).exists():
                with (root / name).open(errors="replace") as stream:
                    tail[name] = "".join(deque(stream, maxlen=80))
    return {
        "token": spec["token"],
        "spec_sha256": digest(spec),
        "state": state,
        "worker_alive": alive,
        "progress": progress,
        "logs": tail,
    }


def collect(root, spec):
    state = inspect(root, spec)
    if state["worker_alive"] or state["state"].get("stage") in {"launching", "training"}:
        raise RuntimeError("worker still running; use logs/status and periodic Drive commits")
    from kiwilm.colab_artifacts import create_colab_artifacts

    artifacts = root / "artifacts"
    if artifacts.exists():
        raise FileExistsError("artifacts already collected; download the existing manifest")
    names = {p.name: p for p in (root / "run").glob("*") if p.is_file()}
    for name in (
        "spec.json",
        "state.json",
        "inputs.json",
        "preflight.json",
        "summary.json",
        "worker.log",
        "supervisor.log",
    ):
        if (root / name).is_file():
            names[name] = root / name
    manifest = create_colab_artifacts(names, artifacts, chunk_size=4 * 1024 * 1024)
    return {"token": spec["token"], "spec_sha256": digest(spec), "manifest": str(manifest)}


def dispatch(action, root):
    spec = load(root)
    if action == "setup":
        return setup(root, spec)
    if action == "inputs":
        return prepare_inputs(root, spec)
    if action == "preflight":
        return preflight(root, spec)
    if action == "start":
        return start(root, spec)
    if action == "supervise":
        return supervise(root, spec)
    if action in {"status", "logs"}:
        return inspect(root, spec, logs=action == "logs")
    if action == "collect":
        return collect(root, spec)
    raise ValueError("unknown V3 CLI action")


def main():
    actions = {"inputs", "supervise", "collect"}
    action = (
        sys.argv[1]
        if len(sys.argv) > 1 and sys.argv[1] in actions
        else os.environ.get("KIWILM3_REMOTE_ACTION", "status")
    )
    root = Path(os.environ["KIWILM3_REMOTE_ROOT"])
    if action == "initialize":
        if root.parent != Path("/content") or not re.fullmatch(
            r"kiwilm3-cli-[0-9a-f]{32}", root.name
        ):
            raise ValueError("invalid initialization root")
        root.mkdir(exist_ok=False)
        print(
            PREFIX
            + json.dumps(
                {
                    "token": root.name.removeprefix("kiwilm3-cli-"),
                    "spec_sha256": os.environ["KIWILM3_EXPECTED_SPEC_SHA"],
                }
            )
        )
        return
    # collect imports the package, so run that action inside the isolated worker.
    if action == "collect" and sys.executable != str(root / "environment/bin/python"):
        python, env = environment(root)
        subprocess.run([str(python), str(root / "bootstrap.py"), "collect"], env=env, check=True)
        return
    try:
        result = dispatch(action, root)
    except Exception:
        traceback.print_exc()
        raise
    print(PREFIX + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
