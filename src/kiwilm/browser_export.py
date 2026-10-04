"""Dense KiwiLM 2 cached ONNX graph for entirely browser-local inference.

Token/n-gram/position IDs and a causal mask are supplied by the client. Keeping
integer hashing outside ONNX avoids GPU integer-overflow/provider differences.
One graph handles both prefill (empty KV state) and single-token decoding.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from kiwilm.models.kiwilm2 import KiwiLM2GQA, KiwiLM2LM
from kiwilm.safetensors_io import load_safetensors_model, sha256_file
from kiwilm.tokenizer import ByteBPETokenizer


class BrowserCachedGraph(nn.Module):
    """Primitive-operator implementation sharing the original learned modules."""

    def __init__(self, model: KiwiLM2LM) -> None:
        super().__init__()
        if model.config.architecture != "kiwilm2":
            raise ValueError("browser export currently supports Dense kiwilm2 only")
        self.model = model.eval()

    def cache_specs(self) -> list[dict]:
        specs = []
        for index, block in enumerate(self.model.blocks):
            mixer = block.mixer
            if isinstance(mixer, KiwiLM2GQA):
                for kind in ("key", "value"):
                    specs.append({"name": f"cache_{index}_{kind}", "axis": 2,
                                  "shape": [1, mixer.num_kv_heads, 0, mixer.head_dim]})
            else:
                specs.append({"name": f"cache_{index}_conv", "axis": None,
                              "shape": [1, mixer.kernel_size - 1, self.model.config.d_model]})
        return specs

    @staticmethod
    def _rope(values: Tensor, mixer: KiwiLM2GQA, positions: Tensor) -> Tensor:
        cosines = mixer.rope.cosines[positions][None, None]
        sines = mixer.rope.sines[positions][None, None]
        even, odd = values[..., 0::2], values[..., 1::2]
        return torch.stack((even * cosines - odd * sines,
                            odd * cosines + even * sines), dim=-1).flatten(start_dim=-2)

    def forward(
        self, input_ids: Tensor, bigram_ids: Tensor, trigram_ids: Tensor,
        position_ids: Tensor, attention_bias: Tensor, *caches: Tensor,
    ) -> tuple[Tensor, ...]:
        model = self.model
        values = (model.token_embedding(input_ids) + model.ngram_embedding.bigram(bigram_ids)
                  + model.ngram_embedding.trigram(trigram_ids)) * model.embedding_scale
        updated = []
        cache_index = 0
        for block in model.blocks:
            normalized = block.mixer_norm(values)
            mixer = block.mixer
            if isinstance(mixer, KiwiLM2GQA):
                query = self._rope(mixer._split(mixer.query(normalized), mixer.num_query_heads),
                                   mixer, position_ids)
                key = self._rope(mixer._split(mixer.key(normalized), mixer.num_kv_heads),
                                 mixer, position_ids)
                value = mixer._split(mixer.value(normalized), mixer.num_kv_heads)
                key = torch.cat((caches[cache_index], key), dim=2)
                value = torch.cat((caches[cache_index + 1], value), dim=2)
                updated.extend((key, value))
                cache_index += 2
                repeats = mixer.num_query_heads // mixer.num_kv_heads
                expanded_key = key.repeat_interleave(repeats, dim=1)
                expanded_value = value.repeat_interleave(repeats, dim=1)
                scores = query @ expanded_key.transpose(-1, -2) / math.sqrt(mixer.head_dim)
                attended = torch.softmax(scores + attention_bias, dim=-1) @ expanded_value
                update = mixer.output(mixer._merge(attended))
            else:
                projected, gate = mixer.input(normalized).chunk(2, dim=-1)
                window = torch.cat((caches[cache_index], projected), dim=1)
                cache_index += 1
                updated.append(window[:, -(mixer.kernel_size - 1):])
                convolved = mixer.depthwise(window.transpose(1, 2)).transpose(1, 2)
                update = mixer.output(convolved * torch.nn.functional.silu(gate))
            values = values + update
            values = values + block.mlp(block.mlp_norm(values))
        return (model.lm_head(model.final_norm(values[:, -1:, :])), *updated)


def graph_inputs(
    graph: BrowserCachedGraph, token_ids: Tensor, caches: tuple[Tensor, ...] | None = None,
    *, previous_ids: Tensor | None = None,
) -> tuple[Tensor, ...]:
    """Reference client input construction, including shifted n-gram history."""

    if caches is None:
        caches = tuple(torch.zeros(spec["shape"]) for spec in graph.cache_specs())
        history = token_ids
        position = 0
    else:
        if previous_ids is None:
            raise ValueError("cached input needs previous token IDs")
        history = torch.cat((previous_ids, token_ids), dim=1)
        position = previous_ids.shape[1]
    length = token_ids.shape[1]
    bigram, trigram = graph.model.ngram_embedding.indices(history)
    positions = torch.arange(position, position + length, dtype=torch.int32)
    keys = torch.arange(position + length)
    bias = torch.where(keys[None] <= positions[:, None], 0.0, -1e9)[None, None]
    return (token_ids.int(), bigram[:, -length:].int(), trigram[:, -length:].int(),
            positions, bias, *caches)


def verify_graph(graph: BrowserCachedGraph, session, *, tolerance: float = 0.002) -> dict:
    """Check full-window, incremental, and rollover logits against original model."""

    model = graph.model
    generator = torch.Generator().manual_seed(143)
    rows = []

    def run(ids, caches=None, previous=None):
        inputs = graph_inputs(graph, ids, caches, previous_ids=previous)
        outputs = session.run(None, {entry.name: value.numpy()
                                    for entry, value in zip(session.get_inputs(), inputs,
                                                            strict=True)})
        with torch.inference_mode():
            expected = model(ids if previous is None else torch.cat((previous, ids), dim=1))
        error = float(np.max(np.abs(outputs[0] - expected[:, -1:].numpy())))
        assert np.isfinite(outputs[0]).all()
        if error > tolerance:
            raise ValueError(f"ONNX logit parity failed: {error} > {tolerance}")
        rows.append({"length": ids.shape[1], "past_length": 0 if previous is None
                     else previous.shape[1], "maximum_logit_error": error})
        return tuple(torch.from_numpy(value) for value in outputs[1:])

    with torch.inference_mode():
        lengths = {1, min(31, model.config.context_length), model.config.context_length}
        for length in sorted(lengths):
            ids = torch.randint(model.config.vocab_size, (1, length), generator=generator)
            run(ids)
        history = torch.randint(model.config.vocab_size, (1, 3), generator=generator)
        state = run(history)
        for _ in range(min(8, model.config.context_length - 3)):
            ids = torch.randint(model.config.vocab_size, (1, 1), generator=generator)
            state = run(ids, state, history)
            history = torch.cat((history, ids), dim=1)
        # Exact rollover policy: discard all caches, re-prefill the newest window.
        history = torch.randint(model.config.vocab_size,
                                (1, model.config.context_length + 2), generator=generator)
        run(history[:, -model.config.context_length:])
    return {"tolerance": tolerance, "checks": rows, "direct_cached_rollover": "passed"}


def export_browser_bundle(bundle: Path, output: Path, *, revision: str) -> dict:
    """Export verified released weights; never overwrite an existing browser bundle."""

    import onnx
    import onnxruntime as ort

    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("revision must be a full Hub commit SHA")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    model, config = load_safetensors_model(bundle, data_fingerprint=None,
                                           device=torch.device("cpu"))
    graph = BrowserCachedGraph(model).eval()
    specs = graph.cache_specs()
    input_names = ["input_ids", "bigram_ids", "trigram_ids", "position_ids", "attention_bias",
                   *(spec["name"] for spec in specs)]
    output_names = ["logits", *(spec["name"].replace("cache_", "present_") for spec in specs)]
    dynamic_axes = {name: {1: "sequence"} for name in input_names[:3]}
    dynamic_axes["position_ids"] = {0: "sequence"}
    dynamic_axes["attention_bias"] = {2: "sequence", 3: "total_sequence"}
    for spec, name in zip(specs, output_names[1:], strict=True):
        if spec["axis"] is not None:
            dynamic_axes[spec["name"]] = {2: "past_sequence"}
            dynamic_axes[name] = {2: "total_sequence"}
    output.mkdir(parents=True)
    file = output / "model.onnx"
    inputs = graph_inputs(graph, torch.tensor([[2, 100, 200]], dtype=torch.int32))
    torch.onnx.export(graph, inputs, file, input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes, opset_version=18, dynamo=False)
    onnx.checker.check_model(str(file))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(file), options, providers=["CPUExecutionProvider"])
    verification = verify_graph(graph, session)
    tokenizer = json.loads((bundle / "tokenizer.json").read_text())
    special_tokens = {token["content"]: token["id"] for token in tokenizer["added_tokens"]
                      if token["special"]}
    shutil.copyfile(bundle / "tokenizer.json", output / "tokenizer.json")
    native_tokenizer = ByteBPETokenizer.load(bundle / "tokenizer.json")
    prompts = ["", "Once upon a time, a fox found a box.", "café 🥝\n雨"]
    reference = {"tokenizer": [{"text": text, "ids": native_tokenizer.encode(text)}
                                for text in prompts], "logits": []}
    full_ids = [(index * 131 + 17) % config.vocab_size
                for index in range(config.context_length)]
    with torch.inference_mode():
        for ids in ([2, 100], [2, 100, 200], full_ids, [*full_ids[1:], 201]):
            values = model(torch.tensor([ids]))[0, -1].tolist()
            reference["logits"].append({"ids": ids, "values": values})
    (output / "parity.json").write_text(json.dumps(reference) + "\n")
    manifest = {
        "schema_version": 1, "architecture": config.architecture,
        "model_repo": "Tasty-Kiwi/KiwiLM-2", "model_revision": revision,
        "source_weights_sha256": sha256_file(bundle / "model.safetensors"),
        "dtype": "float32", "context_length": config.context_length,
        "vocab_size": config.vocab_size, "d_model": config.d_model,
        "bigram_buckets": config.bigram_buckets, "trigram_buckets": config.trigram_buckets,
        "special_tokens": special_tokens, "cache_specs": specs,
        "verification": verification,
        "files": {name: {"sha256": sha256_file(output / name),
                          "bytes": (output / name).stat().st_size}
                  for name in ("model.onnx", "tokenizer.json", "parity.json")},
    }
    (output / "browser-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
