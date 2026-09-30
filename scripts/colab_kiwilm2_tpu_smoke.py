"""Bootstrap an isolated, version-matched PyTorch/XLA TPU smoke environment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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
            "import json,torch; from pathlib import Path; from kiwilm.tpu_smoke import Runtime; "
            "r=Runtime('xla','bf16'); x=torch.ones((128,128),device=r.device); "
            "y=x@x; r.sync(); value=float(y[0,0].cpu()); assert value==128; "
            f"Path({str(preflight)!r}).write_text(json.dumps(dict("
            "device=str(r.device),torch=torch.__version__,"
            "torch_xla=r.xla.__version__,matmul_result=value)))",
        ], env=env)
    if action == "preflight":
        print("TPU preflight complete; ready for dataset upload.", flush=True)
        return
    job_path = CONTENT / "kiwilm-tpu-job.json"
    setup_path = CONTENT / "kiwilm-tpu-setup.json"
    if not job_path.is_file():
        raise RuntimeError("upload the frozen TPU setup job first")
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
    run(command, timeout=840)
    if action == "prepare":
        print("Input preparation finished; no training was started.", flush=True)
        return
    data_dir = CONTENT / "kiwilm-data-artifacts"
    output = CONTENT / "kiwilm-tpu-smoke"
    output.mkdir(exist_ok=True)
    summary = output / "summary.json"
    if summary.is_file() and json.loads(summary.read_text()).get("status") == "smoke-complete":
        raise RuntimeError("this TPU smoke is already complete; refusing another training launch")
    resume_dir = CONTENT / "kiwilm-tpu-resume"
    resume_args = []
    if (output / "latest.pt").is_file():
        resume_args = ["--resume", str(output / "latest.pt")]
    elif (resume_dir / "latest.pt").is_file():
        resume_args = ["--resume", str(resume_dir / "latest.pt")]
    with (output / "worker.log").open("a") as log:
        process = subprocess.Popen([
            str(PYTHON), "-u", "-m", "kiwilm.tpu_smoke",
            "--data-dir", str(data_dir),
            "--output-dir", str(output), "--warmup-steps", "20",
            "--eval-batches", "50", "--eval-interval", "500",
            "--checkpoint-interval", "500",
            "--artifact-dir", str(CONTENT / "kiwilm-tpu-artifacts"),
            *resume_args,
        ], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            process.wait(timeout=WORKER_TIMEOUT)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise RuntimeError("TPU smoke exceeded its two-hour worker limit") from None
    print((output / "worker.log").read_text(), flush=True)
    if process.returncode:
        raise RuntimeError(f"TPU probe failed with exit code {process.returncode}")
    run([
        str(PYTHON), "-c",
        "from pathlib import Path; from kiwilm.colab_artifacts import create_colab_artifacts; "
        "p=Path('/content/kiwilm-tpu-smoke'); "
        "create_colab_artifacts({f.name:f for f in p.iterdir() if f.is_file()}, "
        "Path('/content/kiwilm-tpu-artifacts'), chunk_size=4*1024*1024)",
    ])


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"TPU smoke failed: {error}", file=sys.stderr, flush=True)
        raise
