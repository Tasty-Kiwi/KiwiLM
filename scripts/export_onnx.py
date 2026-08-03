"""Export a KiwiLM Safetensors bundle to browser-compatible ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from kiwilm.inference import load_trained_model
from kiwilm.safetensors_io import sha256_file


class LogitsWrapper(nn.Module):
    """Expose only logits so the ONNX graph has a stable public contract."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        model = self.model
        values = model.token_embedding(input_ids)
        if model.config.architecture == "model_x":
            values = model.feedforward_blocks[0](model.cnn_blocks[0](values))
            values = model.feedforward_blocks[1](model.attention_blocks[0](values))
            values = model.feedforward_blocks[2](model.cnn_blocks[1](values))
            values = model.feedforward_blocks[3](model.attention_blocks[1](values))
        elif model.config.architecture == "model_y":
            for block in model.blocks:
                values = block(values)
        else:
            raise ValueError(f"unsupported browser architecture: {model.config.architecture}")
        final_value = model.final_norm(values[:, -1, :])
        return model.lm_head(final_value)


class ONNXRMSNorm(nn.Module):
    """RMSNorm expressed with primitive operators supported by ONNX Runtime."""

    def __init__(self, source: nn.RMSNorm) -> None:
        super().__init__()
        self.eps = float(source.eps or torch.finfo(source.weight.dtype).eps)
        self.weight = source.weight

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        variance = values.square().mean(dim=-1, keepdim=True)
        return values * torch.rsqrt(variance + self.eps) * self.weight


def replace_rms_norms(module: nn.Module) -> None:
    """Replace native RMSNorm nodes without changing any learned parameters."""

    for name, child in tuple(module.named_children()):
        if isinstance(child, nn.RMSNorm):
            setattr(module, name, ONNXRMSNorm(child))
        else:
            replace_rms_norms(child)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model, config = load_trained_model(
        args.bundle,
        data_fingerprint=None,
        device=torch.device("cpu"),
    )
    replace_rms_norms(model)
    wrapper = LogitsWrapper(model).eval()
    example = torch.tensor([[2, 100, 200, 300]], dtype=torch.long)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (example,),
        args.output,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    graph = onnx.load(args.output)
    onnx.checker.check_model(graph)

    session = ort.InferenceSession(
        str(args.output),
        providers=["CPUExecutionProvider"],
    )
    generator = torch.Generator().manual_seed(42)
    maximum_error = 0.0
    for length in (1, 16, config.context_length):
        token_ids = torch.randint(
            0,
            config.vocab_size,
            (1, length),
            generator=generator,
        )
        with torch.inference_mode():
            expected = wrapper(token_ids).numpy()
        actual = session.run(None, {"input_ids": token_ids.numpy()})[0]
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(expected - actual))),
        )
    if maximum_error > 1e-4:
        raise RuntimeError(f"ONNX parity error is too large: {maximum_error}")

    result = {
        "architecture": config.architecture,
        "bytes": args.output.stat().st_size,
        "context_length": config.context_length,
        "maximum_logit_error": maximum_error,
        "output": str(args.output.resolve()),
        "sha256": sha256_file(args.output),
        "vocab_size": config.vocab_size,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
