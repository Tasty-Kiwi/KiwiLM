"""Optional browser export checks without touching release weights or the Hub."""

import json
from dataclasses import replace

import pytest
import torch

from kiwilm.browser_export import BrowserCachedGraph, export_browser_bundle, graph_inputs
from kiwilm.checkpoint import save_checkpoint
from kiwilm.config import KiwiLM2Config, KiwiLM2SlimConfig
from kiwilm.models import build_model
from kiwilm.safetensors_io import export_safetensors_bundle
from kiwilm.tokenizer import ByteBPETokenizer

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


@pytest.fixture
def model():
    torch.manual_seed(42)
    return build_model(KiwiLM2Config(vocab_size=300, context_length=16, d_model=16,
                                    num_query_heads=2, num_kv_heads=1, swiglu_dim=32,
                                    bigram_buckets=16, trigram_buckets=16)).eval()


def test_graph_matches_native_prefill_caches_and_incremental_logits(model):
    graph = BrowserCachedGraph(model)
    history = torch.tensor([[2, 299, 37]])
    with torch.inference_mode():
        expected, native_cache = model.prefill(history)
        outputs = graph(*graph_inputs(graph, history))
        torch.testing.assert_close(outputs[0], expected[:, -1:])
        flattened = []
        for state in native_cache.mixers:
            flattened.extend(state if isinstance(state, tuple) else [state])
        for actual, expected in zip(outputs[1:], flattened, strict=True):
            torch.testing.assert_close(actual, expected)
        for token in (13, 29, 100):
            ids = torch.tensor([[token]])
            outputs = graph(*graph_inputs(graph, ids, outputs[1:], previous_ids=history))
            history = torch.cat((history, ids), dim=1)
            torch.testing.assert_close(outputs[0], model(history)[:, -1:])


def test_hash_inputs_use_cached_predecessors_and_mask(model):
    graph = BrowserCachedGraph(model)
    ids = torch.tensor([[2, 299, 37]])
    inputs = graph_inputs(graph, ids)
    bigram, trigram = model.ngram_embedding.indices(ids)
    assert torch.equal(inputs[1], bigram.int())
    assert torch.equal(inputs[2], trigram.int())
    assert inputs[4][0, 0, 0, 1] == -1e9
    outputs = graph(*inputs)
    with pytest.raises(ValueError, match="previous token"):
        graph_inputs(graph, torch.tensor([[1]]), outputs[1:])


def test_export_round_trip_dynamic_caches_and_no_overwrite(model, tmp_path):
    tokenizer = ByteBPETokenizer.train(["The fox found a box. A short story."], vocab_size=300)
    model = build_model(replace(model.config, vocab_size=tokenizer.vocab_size)).eval()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    checkpoint = save_checkpoint(tmp_path / "model.pt", model=model, model_config=model.config,
                                 step=1, data_fingerprint="browser-test")
    bundle = tmp_path / "bundle"
    export_safetensors_bundle(checkpoint, bundle, tokenizer_path=tokenizer_path, variant="test")
    output = tmp_path / "browser"
    manifest = export_browser_bundle(bundle, output, revision="a" * 40)
    assert manifest["verification"]["direct_cached_rollover"] == "passed"
    assert manifest["special_tokens"]["[BOS]"] == 2
    assert len(manifest["cache_specs"]) == 14
    assert json.loads((output / "browser-manifest.json").read_text()) == manifest
    with pytest.raises(FileExistsError):
        export_browser_bundle(bundle, output, revision="a" * 40)
    with pytest.raises(ValueError, match="commit SHA"):
        export_browser_bundle(bundle, tmp_path / "invalid", revision="main")


def test_slim_browser_export_rejected():
    with pytest.raises(ValueError, match="Dense"):
        BrowserCachedGraph(build_model(KiwiLM2SlimConfig(d_model=16, num_query_heads=2,
                                                        num_kv_heads=1)))
