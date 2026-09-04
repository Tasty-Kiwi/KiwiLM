"""Bootstrap an isolated, version-matched PyTorch/XLA TPU smoke environment."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CONTENT = Path("/content")
ENV = CONTENT / "kiwilm-tpu-env"
PYTHON = ENV / "bin" / "python"


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
    data_dir = CONTENT / "kiwilm-data-artifacts"
    if not (data_dir / "artifact-manifest.json").is_file():
        print("TPU preflight complete; ready for dataset upload.", flush=True)
        return
    run(data_restore_command(PYTHON, data_dir))
    output = CONTENT / "kiwilm-tpu-smoke"
    output.mkdir(exist_ok=True)
    with (output / "worker.log").open("w") as log:
        process = subprocess.Popen([
            str(PYTHON), "-u", "-m", "kiwilm.tpu_smoke",
            "--data-dir", str(data_dir),
            "--output-dir", str(output), "--steps", "200", "--warmup-steps", "20",
        ], env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise RuntimeError("TPU probe exceeded its 15-minute worker limit") from None
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
