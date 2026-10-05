"""Small local-VM adapter for thin notebook cells, not a Colab session launcher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kiwilm.data import PreparedTokenData
from kiwilm.notebook_workflow import (
    NotebookConfig,
    backup_namespace,
    contract_identity,
    evaluate_probe,
    experiment_contract,
    preflight,
    prepare_probe_data,
    train_probe,
)
from kiwilm.tpu_checkpoint import DriveCheckpointStore


def execute(action: str, request: dict) -> dict:
    config = NotebookConfig(**request.get("config", {}))
    if config.device == "cpu":
        # Tiny CPU qualification: fixed reduction order, no thread overhead.
        torch.set_num_threads(1)
    data_dir = Path(request["data_dir"])
    if action == "prepare":
        return {"data": prepare_probe_data(data_dir), "training_started": False}
    if action == "preflight":
        return preflight(config, data_dir)
    if action == "train":
        return train_probe(
            config,
            data_dir=data_dir,
            run_dir=Path(request["run_dir"]),
            backup_root=Path(request["backup_root"]),
            mode=request["mode"],
            start_training=request.get("start_training") is True,
            stop_after_step=request.get("stop_after_step"),
            storage_root=Path(request["storage_root"]) if request.get("storage_root") else None,
        )
    if action == "restore":
        destination = Path(request["run_dir"])
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError("restore requires a new empty local directory")
        data = PreparedTokenData(data_dir)
        identity = contract_identity(experiment_contract(config, data))
        store = DriveCheckpointStore(
            backup_namespace(Path(request["backup_root"]), config, identity),
            identity=identity,
            storage_root=Path(request["storage_root"]) if request.get("storage_root") else None,
        )
        return store.restore(destination)
    if action == "evaluate":
        return evaluate_probe(
            config,
            data_dir=data_dir,
            checkpoint=Path(request["checkpoint"]),
            batches=request.get("eval_batches", config.eval_batches),
            export_dir=Path(request["export_dir"]) if request.get("export_dir") else None,
        )
    raise ValueError("unknown notebook action")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", "train", "evaluate", "restore"))
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    result = execute(args.action, json.loads(args.request.read_text()))
    print(json.dumps({"notebook_result": result}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
