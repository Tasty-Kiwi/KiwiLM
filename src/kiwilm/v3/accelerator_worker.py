"""Notebook-local subprocess adapter for V3. Never provision a Colab session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kiwilm.colab_drive import restore_prepared_data
from kiwilm.v3.accelerator import AcceleratorTrainConfig
from kiwilm.v3.accelerator_workflow import preflight, prepare_demo, train


def execute(action: str, request: dict) -> dict:
    if action == "prepare-demo":
        return prepare_demo(Path(request["data_dir"]), Path(request["tokenizer_path"]))
    if action == "restore-data":
        destination = Path(request["data_dir"])
        if destination.exists():
            raise FileExistsError("restore prepared data into a new local directory")
        mount = Path(request["storage_root"])
        if not mount.is_mount():
            raise RuntimeError("Drive mount unavailable; data preparation has not started")
        if not restore_prepared_data(Path(request["data_cache"]), destination, storage_root=mount):
            raise ValueError("no complete prepared cache; do not silently download/rebuild data")
        return {"data_dir": str(destination), "training_started": False}
    config = AcceleratorTrainConfig(**request["config"])
    common = {
        "data_dir": Path(request["data_dir"]),
        "tokenizer_path": Path(request["tokenizer_path"]),
        "candidate": request["candidate"],
        "qualification": request["qualification"],
        "run_name": request["run_name"],
    }
    if "collapse_diagnostics" in request:
        common["collapse_diagnostics"] = request["collapse_diagnostics"]
    if action == "preflight":
        return preflight(config, **common)
    if action == "train":
        return train(
            config,
            **common,
            run_dir=Path(request["run_dir"]),
            backup_root=Path(request["backup_root"]),
            mode=request["mode"],
            start_training=request.get("start_training") is True,
            storage_root=Path(request["storage_root"]) if request.get("storage_root") else None,
            stop_after_step=request.get("stop_after_step"),
            require_new_vm=request.get("require_new_vm") is True,
        )
    raise ValueError("unknown V3 accelerator action")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare-demo", "restore-data", "preflight", "train"))
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    print(
        json.dumps({"notebook_result": execute(args.action, json.loads(args.request.read_text()))}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
