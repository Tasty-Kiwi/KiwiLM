"""V3 native training state over generic verified latest/previous transport.

V2/M4 payloads cannot be interpreted here. Publication waits for read-back;
no training step may run ahead of a failed Drive commit.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from kiwilm.checkpoint import capture_rng_state, restore_rng_state
from kiwilm.colab_artifacts import file_sha256
from kiwilm.tpu_checkpoint import DriveCheckpointStore
from kiwilm.v3.accelerator import AcceleratorTrainer, learning_rate
from kiwilm.v3.experiments import canonical_digest

FORMAT = "kiwilm3-accelerator-training-v1"


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def numpy_state():
    name, keys, position, gaussian, cached = np.random.get_state()
    return {
        "name": name,
        "keys": keys.tolist(),
        "position": position,
        "gaussian": gaussian,
        "cached": cached,
    }


def restore_numpy(state):
    np.random.set_state(
        (
            state["name"],
            np.array(state["keys"], dtype=np.uint32),
            state["position"],
            state["gaussian"],
            state["cached"],
        )
    )


def validate_payload(payload, trainer: AcceleratorTrainer, job: dict) -> None:
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("not a V3 accelerator checkpoint; V2/M4 use separate loaders")
    if (
        payload.get("contract") != trainer.contract
        or payload.get("identity") != trainer.identity
        or payload.get("job") != job
        or payload.get("job_identity") != canonical_digest(job)
    ):
        raise ValueError(
            "V3 resume contract differs (model/objective/data/tokenizer/code/runtime/job)"
        )
    if payload.get("completed_boundary") is not True:
        raise ValueError("checkpoint is not a completed optimizer boundary")
    step, updates, tokens, masks = [
        payload.get(key) for key in ("step", "optimizer_steps", "tokens_seen", "masked_tokens_seen")
    ]
    if (
        any(type(value) is not int for value in (step, updates, tokens, masks))
        or not 0 <= updates <= step <= tokens <= trainer.config.max_tokens
        or not 0 <= masks <= tokens
        or (step == 0 and (tokens != 0 or masks != 0))
    ):
        raise ValueError("checkpoint progress is inconsistent with token budget")
    if payload.get("schedule") != {"tokens_seen": tokens, "policy": "linear-warmup-cosine-v1"}:
        raise ValueError("checkpoint schedule/token progress mismatch")
    if not isinstance(payload.get("optimizer"), dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("checkpoint missing model/optimizer state")
    expected = trainer.model.state_dict()
    state = payload["model"]
    if state.keys() != expected.keys() or any(
        not isinstance(state[key], torch.Tensor)
        or state[key].shape != expected[key].shape
        or state[key].dtype != expected[key].dtype
        or not torch.isfinite(state[key]).all().item()
        for key in expected
    ):
        raise ValueError("invalid or nonfinite model tensors")
    if not torch.equal(state["token_embedding.weight"], state["reconstruction_head.weight"]):
        raise ValueError("checkpoint has unequal tied head/embedding values")
    optimizer = payload["optimizer"]
    groups = optimizer.get("param_groups", [])
    expected_group = trainer.optimizer.state_dict()["param_groups"][0]
    if (
        len(groups) != 1
        or set(groups[0]) != set(expected_group)
        or any(groups[0][key] != value for key, value in expected_group.items() if key != "lr")
    ):
        raise ValueError("checkpoint optimizer groups differ from the frozen AdamW policy")
    expected_lr = learning_rate(trainer.config, tokens) if step else trainer.config.learning_rate
    lr = groups[0]["lr"]
    if (
        not isinstance(lr, torch.Tensor)
        or lr.shape != ()
        or not torch.equal(lr, torch.tensor(expected_lr, dtype=lr.dtype))
    ):
        raise ValueError("checkpoint optimizer LR differs from its token schedule")
    moments = optimizer.get("state", {})
    ids = groups[0]["params"]
    if set(moments) != (set(ids) if updates else set()):
        raise ValueError("checkpoint missing AdamW moment state")
    for index, parameter in zip(ids, trainer.model.parameters(), strict=True):
        if not updates:
            break
        moment = moments[index]
        if set(moment) != {"step", "exp_avg", "exp_avg_sq"} or any(
            not isinstance(moment[key], torch.Tensor)
            or moment[key].shape != parameter.shape
            or moment[key].dtype != parameter.dtype
            or not torch.isfinite(moment[key]).all().item()
            for key in ("exp_avg", "exp_avg_sq")
        ):
            raise ValueError("checkpoint has invalid/nonfinite AdamW moments")
        if (
            not isinstance(moment["step"], torch.Tensor)
            or moment["step"].shape != ()
            or moment["step"].item() != updates
            or (moment["exp_avg_sq"] < 0).any().item()
        ):
            raise ValueError("checkpoint optimizer progress or variance is invalid")
    streams = payload.get("generators", {})
    if set(streams) != {"data", "noise"}:
        raise ValueError("checkpoint missing data/noise RNG recovery state")
    for value in streams.values():
        torch.Generator().set_state(value)
    rng = payload.get("rng", {})
    if not {"global", "numpy", "xla"}.issubset(rng) or not {"torch", "python"}.issubset(
        rng["global"]
    ):
        raise ValueError("checkpoint missing global/NumPy/backend RNG recovery state")
    if trainer.config.device == "xla" and type(rng["xla"]) is not int:
        raise ValueError("checkpoint missing XLA RNG state")
    if trainer.config.device == "cuda" and "cuda" not in rng["global"]:
        raise ValueError("checkpoint missing CUDA RNG state")
    # Validate recovery states without mutating live global generators/model.
    random.Random().setstate(rng["global"]["python"])
    torch.Generator().set_state(rng["global"]["torch"])
    isolated_numpy = np.random.RandomState()
    np_state = rng["numpy"]
    isolated_numpy.set_state(
        (
            np_state["name"],
            np.array(np_state["keys"], dtype=np.uint32),
            np_state["position"],
            np_state["gaussian"],
            np_state["cached"],
        )
    )
    if not isinstance(payload.get("metrics"), dict) or set(payload["metrics"]) != {
        "bytes",
        "sha256",
    }:
        raise ValueError("checkpoint missing verified metrics prefix")


def save_state(
    trainer: AcceleratorTrainer,
    destination: Path,
    *,
    job: dict,
    metrics: Path,
    vm_id: str | None = None,
) -> Path:
    if not trainer.at_step_boundary:
        raise RuntimeError("cannot save a failed/incomplete V3 step")
    trainer.runtime.sync()
    if (
        trainer.contract["model"] != trainer.model.config.to_dict()
        or trainer.contract["training"] != trainer.config.to_dict()
        or trainer.contract["masking"] != trainer.masking.to_dict()
        or trainer.contract["data_fingerprint"] != trainer.data.fingerprint
        or trainer.contract["tokenizer_sha256"] != trainer.tokenizer.fingerprint
        or canonical_digest(trainer.contract) != trainer.identity
    ):
        raise ValueError("live training settings changed; cannot reinterpret this run")
    payload = {
        "format": FORMAT,
        "contract": trainer.contract,
        "identity": trainer.identity,
        "job": job,
        "job_identity": canonical_digest(job),
        "completed_boundary": True,
        "step": trainer.step,
        "optimizer_steps": trainer.optimizer_steps,
        "tokens_seen": trainer.tokens_seen,
        "masked_tokens_seen": trainer.masked_tokens_seen,
        "schedule": {"tokens_seen": trainer.tokens_seen, "policy": "linear-warmup-cosine-v1"},
        "model": cpu_tree(trainer.model.state_dict()),
        "optimizer": cpu_tree(trainer.optimizer.state_dict()),
        "generators": {
            "data": trainer.data_generator.get_state(),
            "noise": trainer.noise_generator.get_state(),
        },
        "rng": {
            "global": capture_rng_state(),
            "numpy": numpy_state(),
            "xla": trainer.runtime.xm.get_rng_state(trainer.runtime.device)
            if trainer.runtime.xm is not None
            else None,
        },
        "metrics": {"bytes": metrics.stat().st_size, "sha256": file_sha256(metrics)},
        "vm_id": vm_id,
    }
    validate_payload(payload, trainer, job)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".v3-state-", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def restore_state(
    trainer: AcceleratorTrainer,
    source: Path,
    *,
    job: dict,
    expected_sha256: str,
    metrics: Path,
    require_new_vm: bool = False,
    vm_id: str | None = None,
) -> dict:
    if file_sha256(source) != expected_sha256:
        raise ValueError("V3 checkpoint checksum mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    validate_payload(payload, trainer, job)
    if payload["metrics"] != {"bytes": metrics.stat().st_size, "sha256": file_sha256(metrics)}:
        raise ValueError("metrics differ from the committed checkpoint prefix")
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    train_steps = [row["step"] for row in rows if row.get("event") == "v3_train"]
    if train_steps != list(range(1, payload["step"] + 1)) or any(
        row.get("step", 0) > payload["step"] for row in rows
    ):
        raise ValueError("metrics progress differs from checkpoint; no silent truncation")
    if require_new_vm and (not vm_id or not payload.get("vm_id") or payload["vm_id"] == vm_id):
        raise ValueError("fresh-runtime qualification requires distinct verified VM identifiers")
    trainer.model.load_state_dict(payload["model"], strict=True)
    trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.lr_tensor.copy_(payload["optimizer"]["param_groups"][0]["lr"])
    trainer.optimizer.param_groups[0]["lr"] = trainer.lr_tensor
    trainer.data_generator.set_state(payload["generators"]["data"])
    trainer.noise_generator.set_state(payload["generators"]["noise"])
    restore_rng_state(payload["rng"]["global"])
    restore_numpy(payload["rng"]["numpy"])
    if trainer.runtime.xm is not None:
        trainer.runtime.xm.set_rng_state(payload["rng"]["xla"], trainer.runtime.device)
    for key in ("step", "optimizer_steps", "tokens_seen", "masked_tokens_seen"):
        setattr(trainer, key, payload[key])
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.at_step_boundary = True
    return {
        "step": trainer.step,
        "tokens_seen": trainer.tokens_seen,
        "source_vm_id": payload.get("vm_id"),
        "vm_id": vm_id,
        "distinct_vm_ids": bool(vm_id and payload.get("vm_id") and vm_id != payload["vm_id"]),
        "learning_rate": learning_rate(trainer.config, trainer.tokens_seen),
    }


class V3CheckpointStore:
    """Isolated V3 namespace; generic transport sees bytes, not a V2 training schema."""

    def __init__(
        self,
        root: Path,
        *,
        job: dict,
        storage_root: Path | None = None,
        attempts: int = 3,
        retry_delay: float = 5,
    ):
        self.job, self.identity = job, canonical_digest(job)
        self.transport = DriveCheckpointStore(
            root,
            identity=self.identity,
            storage_root=storage_root,
            attempts=attempts,
            retry_delay=retry_delay,
        )

    @property
    def root(self):
        return self.transport.root

    def publish(self, trainer: AcceleratorTrainer, run: Path, *, vm_id=None) -> dict:
        job_path = run / "job.json"
        expected = {
            "job": self.job,
            "identity": self.identity,
            "tokenizer_json": trainer.tokenizer.to_json(),
        }
        if json.loads(job_path.read_text()) != expected:
            raise ValueError("local job/tokenizer file differs from the frozen job")
        save_state(
            trainer, run / "latest.pt", job=self.job, metrics=run / "metrics.jsonl", vm_id=vm_id
        )
        return self.transport.publish(
            run, step=trainer.step, tokens=trainer.tokens_seen, job=job_path
        )

    def restore(
        self,
        trainer: AcceleratorTrainer,
        run: Path,
        *,
        vm_id=None,
        require_new_vm=False,
        expected_step=None,
    ) -> dict:
        if run.exists() and any(run.iterdir()):
            raise FileExistsError("V3 restore requires a new empty local directory")
        receipt = self.transport.restore(run, expected_step=expected_step)
        expected = {
            "job": self.job,
            "identity": self.identity,
            "tokenizer_json": trainer.tokenizer.to_json(),
        }
        if (
            not (run / "metrics.jsonl").is_file()
            or not (run / "job.json").is_file()
            or json.loads((run / "job.json").read_text()) != expected
        ):
            raise ValueError("committed V3 job, tokenizer or metrics are missing/incompatible")
        progress = restore_state(
            trainer,
            run / "latest.pt",
            job=self.job,
            expected_sha256=receipt["checkpoint_sha256"],
            metrics=run / "metrics.jsonl",
            vm_id=vm_id,
            require_new_vm=require_new_vm,
        )
        if progress["step"] != receipt["step"]:
            raise ValueError("Drive manifest and V3 payload step disagree")
        # A previous-generation restore is inspectable but cannot silently train
        # behind a newer committed head; choose an explicit new recovery namespace.
        self.transport.validate_progress(trainer.step)
        return {**receipt, **progress}
