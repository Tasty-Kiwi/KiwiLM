"""Bootstrap an isolated, version-matched PyTorch/XLA TPU smoke environment."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

CONTENT = Path("/content")
ENV = CONTENT / "kiwilm-tpu-env"
PYTHON = ENV / "bin" / "python"
WORKER_TIMEOUT = 7200


def run(command: list[str], *, timeout: int = 300, env: dict | None = None) -> None:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env)
    print(result.stdout, end="", flush=True)
    print(result.stderr, end="", file=sys.stderr, flush=True)
    result.check_returncode()


def run_worker(
    command: list[str], *, env: dict, log_path: Path,
    timeout: float = WORKER_TIMEOUT, heartbeat_interval: float = 30,
) -> None:
    """Stream structured progress while retaining all output in an append-only log."""
    if timeout <= 0 or heartbeat_interval <= 0:
        raise ValueError("worker timeout and heartbeat interval must be positive")
    print(
        "Starting TPU worker. Initial XLA compilation may take a few minutes. "
        f"Full output: {log_path}", flush=True,
    )
    started = last_progress = time.monotonic()
    pending = ""
    with log_path.open("a", encoding="utf-8") as log, log_path.open(
        "r", encoding="utf-8", errors="replace",
    ) as reader:
        # On resume, stream only new output; keep historical output in the file.
        reader.seek(0, os.SEEK_END)
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)

        def drain(*, final: bool = False) -> None:
            nonlocal pending, last_progress
            pending += reader.read()
            lines = pending.split("\n")
            pending = lines.pop()
            if final and pending:
                lines.append(pending)
                pending = ""
            for line in lines:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and "event" in event:
                    print(line, flush=True)
                    last_progress = time.monotonic()

        try:
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise RuntimeError(f"TPU worker exceeded its {timeout:g}-second worker limit")
                try:
                    process.wait(timeout=min(1, remaining, heartbeat_interval))
                except subprocess.TimeoutExpired:
                    drain()
                    now = time.monotonic()
                    if now - last_progress >= heartbeat_interval:
                        print(
                            f"TPU worker still running ({now - started:.0f}s elapsed); "
                            f"compilation/evaluation may be quiet. Full output: {log_path}",
                            flush=True,
                        )
                        last_progress = now
                else:
                    drain(final=True)
                    break
        finally:
            # Preserve the existing bounded lifetime, including interrupted/error exits.
            if process.poll() is None:
                process.kill()
                process.wait()
        if process.returncode:
            with log_path.open(encoding="utf-8", errors="replace") as failure_log:
                tail = "".join(deque(failure_log, maxlen=40))
            print(f"TPU worker failed; last log lines:\n{tail}", file=sys.stderr, flush=True)
            raise RuntimeError(f"TPU probe failed with exit code {process.returncode}")


def data_restore_command(python: Path, data_dir: Path) -> list[str]:
    # The artifact API resolves chunks relative to output_dir, not the manifest.
    return [str(python), "-c",
            "from pathlib import Path; "
            "from kiwilm.colab_artifacts import reassemble_colab_artifacts; "
            f"reassemble_colab_artifacts(Path({str(data_dir / 'artifact-manifest.json')!r}), "
            f"Path({str(data_dir)!r}))"]


def main() -> None:
    action = os.environ.get("KIWILM2_TPU_ACTION", "preflight")
    if action not in {"preflight", "prepare", "train"}:
        raise ValueError("KIWILM2_TPU_ACTION must be preflight, prepare or train")
    wheels = list(CONTENT.glob("kiwilm-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("upload exactly one KiwiLM wheel")
    # Colab's system Python may lack ensurepip. uv supplies a standalone
    # Python and installs into its environment without touching system Torch.
    if not PYTHON.is_file():
        run([sys.executable, "-m", "pip", "install", "uv"], timeout=120)
        uv = [sys.executable, "-m", "uv"]
        run([*uv, "venv", "--python", "3.12", str(ENV)], timeout=180)
        run([
            *uv, "pip", "install", "--python", str(PYTHON), "torch==2.9.0",
            "--index-url", "https://download.pytorch.org/whl/cpu",
        ])
        run([
            *uv, "pip", "install", "--python", str(PYTHON),
            "torch==2.9.0", "torch_xla[tpu]==2.9.0", str(wheels[0]),
        ])
    # _XLAC links libpython explicitly. Standalone Python's lib directory is
    # not in the system dynamic-linker search path on Colab.
    python_lib = PYTHON.resolve().parent.parent / "lib"
    env = {
        **os.environ, "PJRT_DEVICE": "TPU", "PT_XLA_DEBUG_LEVEL": "2",
        "LD_LIBRARY_PATH": str(python_lib) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""),
    }
    preflight = CONTENT / "kiwilm-tpu-preflight.json"
    if not preflight.is_file():
        run([
            str(PYTHON), "-c",
            "import json,torch,uuid; from pathlib import Path; "
            "from kiwilm.tpu_smoke import Runtime; "
            "r=Runtime('xla','bf16'); x=torch.ones((128,128),device=r.device); "
            "y=x@x; r.sync(); value=float(y[0,0].cpu()); assert value==128; "
            f"Path({str(preflight)!r}).write_text(json.dumps(dict("
            "device=str(r.device),torch=torch.__version__,"
            "torch_xla=r.xla.__version__,matmul_result=value,vm_id=uuid.uuid4().hex)))",
        ], env=env)
    if action == "preflight":
        print("TPU preflight complete; ready for dataset upload.", flush=True)
        return
    job_path = CONTENT / "kiwilm-tpu-job.json"
    setup_path = CONTENT / "kiwilm-tpu-setup.json"
    if not job_path.is_file():
        raise RuntimeError("upload the frozen TPU setup job first")
    job = json.loads(job_path.read_text())
    final = job.get("phase") == "final-1b"
    if action == "train":
        if not setup_path.is_file() or json.loads(setup_path.read_text()).get("state") != "ready":
            raise RuntimeError("TPU setup must finish before explicit training")
        from hashlib import sha256

        job = json.loads(job_path.read_text())
        digest = sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
        if json.loads(setup_path.read_text())["job_digest"] != digest:
            raise RuntimeError("TPU job changed since setup; refusing training")
    command = [
        str(PYTHON), "-m", "kiwilm.tpu_setup", "--job", str(job_path),
        "--data-dir", str(CONTENT / "kiwilm-data-artifacts"),
        "--resume-dir", str(CONTENT / "kiwilm-tpu-resume"),
        "--drive-root", str(CONTENT / "drive"), "--report", str(setup_path),
    ]
    if action == "train":
        command.append("--require-ready")
    if final and action == "prepare":
        run_worker(command, env=env, log_path=CONTENT / "kiwilm-tpu-prepare.log", timeout=21600)
    else:
        run(command, timeout=840)
    if action == "prepare":
        print("Input preparation finished; no training was started.", flush=True)
        return
    data_dir = CONTENT / "kiwilm-data-artifacts"
    output = CONTENT / ("kiwilm-tpu-final-1b" if final else "kiwilm-tpu-smoke")
    output.mkdir(exist_ok=True)
    summary = output / "summary.json"
    if summary.is_file() and json.loads(summary.read_text()).get("status") in {
        "smoke-complete", "final-complete",
    }:
        raise RuntimeError("this TPU run is already complete; refusing another training launch")
    resume_dir = CONTENT / "kiwilm-tpu-resume"
    resume_args = []
    if (output / "latest.pt").is_file():
        resume_args = ["--resume", str(output / "latest.pt")]
    elif (resume_dir / "latest.pt").is_file():
        resume_args = ["--resume", str(resume_dir / "latest.pt")]
    if job.get("drive_resume_dir") and not resume_args:
        raise RuntimeError("required Drive checkpoint is missing; refusing fresh training")
    if resume_args and not (output / "metrics.jsonl").exists():
        sidecar = Path(resume_args[1]).parent / "metrics.jsonl"
        if sidecar.is_file():
            shutil.copyfile(sidecar, output / "metrics.jsonl")
    phase = job.get("continuation_phase")
    test_args = []
    if phase is not None:
        test_args = ["--steps", "40" if phase == "reference" else "20"]
        if phase == "resume":
            test_args += ["--require-new-vm", "--compare-reference",
                          str(resume_dir / "reference" / "latest.pt")]
    vm_id = json.loads(preflight.read_text()).get("vm_id")
    if phase is not None and not vm_id:
        raise RuntimeError("continuation requires a fresh preflight VM identity")
    persistence_args = ["--vm-id", vm_id] if vm_id else []
    if job.get("drive_backup_dir"):
        persistence_args += ["--drive-backup-dir", job["drive_backup_dir"],
                             "--drive-root", str(CONTENT / "drive"), "--job", str(job_path)]
    elif job.get("use_drive"):
        raise RuntimeError("Drive-enabled training requires a locked checkpoint backup directory")
    interval = "20" if phase is not None else "500"
    final_args = (["--phase", "final-1b", "--warmup-tokens", "20000000",
                   "--final-artifacts-only"] if final else [])
    print(f"First periodic checkpoint is at step {interval}; Ctrl+C stops this VM.", flush=True)
    if job.get("drive_backup_dir"):
        print(f"Verified checkpoint backups: {job['drive_backup_dir']}", flush=True)
    run_worker([
            str(PYTHON), "-u", "-m", "kiwilm.tpu_smoke",
            "--data-dir", str(data_dir),
            "--output-dir", str(output), "--warmup-steps", "10" if phase is not None else "20",
            "--eval-batches", "200" if final else ("5" if phase is not None else "50"),
            "--eval-interval", interval,
            "--checkpoint-interval", interval,
            "--artifact-dir", str(CONTENT / "kiwilm-tpu-artifacts"),
            *resume_args, *test_args, *persistence_args, *final_args,
        ], env=env, log_path=output / "worker.log", timeout=79200 if final else WORKER_TIMEOUT)
    print("TPU worker completed; packaging downloadable artifacts.", flush=True)
    run([
        str(PYTHON), "-c",
        "from pathlib import Path; from kiwilm.colab_artifacts import create_colab_artifacts; "
        f"p=Path({str(output)!r}); "
        "create_colab_artifacts({f.name:f for f in p.iterdir() if f.is_file()}, "
        "Path('/content/kiwilm-tpu-artifacts'), chunk_size=4*1024*1024)",
    ])


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"TPU smoke failed: {error}", file=sys.stderr, flush=True)
        raise
