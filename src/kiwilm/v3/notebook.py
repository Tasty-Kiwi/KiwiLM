"""Stream the V3 worker inside the selected notebook VM, never a remote session."""

import json
import subprocess
import tempfile
from pathlib import Path

from kiwilm.notebook_setup import worker_environment


def run_action(python: Path, action: str, request: dict) -> dict:
    if action not in {"prepare-demo", "restore-data", "preflight", "train"}:
        raise ValueError("unknown V3 notebook action")
    with tempfile.TemporaryDirectory(prefix="kiwilm-v3-request-") as directory:
        path = Path(directory) / "request.json"
        path.write_text(json.dumps(request) + "\n")
        process = subprocess.Popen(
            [str(python), "-m", "kiwilm.v3.accelerator_worker", action, str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=worker_environment(python, request.get("config", {}).get("device", "cpu")),
        )
        result = None
        try:
            assert process.stdout is not None
            for line in process.stdout:
                if line.startswith('{"notebook_result":'):
                    result = json.loads(line)["notebook_result"]
                    print(
                        json.dumps({"event": "v3_notebook_action_complete", "action": action}),
                        flush=True,
                    )
                else:
                    print(line, end="", flush=True)
            if process.wait() != 0:
                raise RuntimeError(
                    "V3 worker failed; retain local/committed state and inspect the log"
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
            raise RuntimeError("V3 worker finished without a result")
        return result
