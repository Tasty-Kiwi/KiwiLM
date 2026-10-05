"""Notebook-local environment and subprocess control; never allocate a Colab VM."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

UV_VERSION = "0.11.7"
TORCH_VERSION = "2.13.0"
XLA_VERSION = "2.9.0"


def environment_plan(wheel: Path, directory: Path, device: str) -> dict[str, Any]:
    if device not in {"cpu", "cuda", "xla"}:
        raise ValueError("explicitly select cpu, cuda or xla")
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError("upload a reviewed KiwiLM wheel before environment setup")
    requirements = Path(__file__).with_name("notebook_requirements.txt")
    return {
        "schema": 1,
        "python": "3.12",
        "device": device,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
        "uv": UV_VERSION,
        "torch": XLA_VERSION if device == "xla" else TORCH_VERSION,
        "torch_xla": XLA_VERSION if device == "xla" else None,
        "environment": str(directory),
    }


def prepare_environment(wheel: Path, directory: Path, device: str) -> Path:
    """Install into an isolated Python 3.12 worker, not Colab's torch/kernel environment."""
    plan = environment_plan(wheel, directory, device)
    marker = directory / "kiwilm-environment.json"
    intent = directory / "kiwilm-install-intent.json"
    if directory.exists():
        recorded = marker if marker.exists() else intent
        if not recorded.exists() or json.loads(recorded.read_text()) != plan:
            raise FileExistsError(
                "environment differs or is unmanaged; choose a new environment directory"
            )
        if marker.exists():
            return directory / "bin" / "python"
    subprocess.run([sys.executable, "-m", "pip", "install", f"uv=={UV_VERSION}"], check=True)
    uv = [sys.executable, "-m", "uv"]
    if not directory.exists():
        subprocess.run([*uv, "venv", "--python", "3.12", str(directory)], check=True)
    intent.write_text(json.dumps(plan, sort_keys=True) + "\n")
    python = directory / "bin" / "python"
    install = [*uv, "pip", "install", "--python", str(python)]
    if device in {"cpu", "xla"}:
        subprocess.run(
            [
                *install,
                f"torch=={plan['torch']}",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ],
            check=True,
        )
    else:
        subprocess.run([*install, f"torch=={TORCH_VERSION}"], check=True)
    requirements = Path(__file__).with_name("notebook_requirements.txt")
    additions = [f"torch_xla[tpu]=={XLA_VERSION}"] if device == "xla" else []
    # Repeat the torch constraint so resolving XLA/core dependencies cannot upgrade it.
    subprocess.run(
        [*install, f"torch=={plan['torch']}", *additions, "-r", str(requirements)], check=True
    )
    # Probe uses eager Dense only: Triton/browser packages are intentionally unnecessary.
    subprocess.run([*install, "--no-deps", str(wheel)], check=True)
    marker.write_text(json.dumps(plan, sort_keys=True) + "\n")
    return python


def worker_environment(python: Path, device: str) -> dict[str, str]:
    environment = os.environ.copy()
    library = python.resolve().parent.parent / "lib"
    environment["LD_LIBRARY_PATH"] = (
        str(library) + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
    )
    environment["PYTHONUNBUFFERED"] = "1"
    if device == "xla":
        environment["PJRT_DEVICE"] = "TPU"
    return environment


def run_notebook_action(python: Path, action: str, request: dict[str, Any]) -> dict[str, Any]:
    """Stream a worker inside this notebook VM. No CLI session lifecycle or remote execution."""
    if action not in {"prepare", "preflight", "train", "evaluate", "restore"}:
        raise ValueError("unknown notebook action")
    device = request.get("config", {}).get("device", "cpu")
    with tempfile.TemporaryDirectory(prefix="kiwilm-notebook-request-") as temporary:
        request_path = Path(temporary) / "request.json"
        request_path.write_text(json.dumps(request) + "\n")
        process = subprocess.Popen(
            [str(python), "-m", "kiwilm.notebook_worker", action, str(request_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=worker_environment(python, device),
        )
        result = None
        try:
            assert process.stdout is not None
            for line in process.stdout:
                if line.startswith('{"notebook_result":'):
                    result = json.loads(line)["notebook_result"]
                    print(
                        json.dumps({"event": "notebook_action_complete", "action": action}),
                        flush=True,
                    )
                else:
                    print(line, end="", flush=True)
            if process.wait() != 0:
                raise RuntimeError(
                    "notebook worker failed; retain the last committed checkpoint and read the log"
                )
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            if process.stdout is not None:
                process.stdout.close()
        if result is None:
            raise RuntimeError("worker finished without a result")
        return result
