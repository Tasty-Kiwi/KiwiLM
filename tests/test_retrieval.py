"""Counterfactual context-retrieval suite and metric coverage."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from kiwilm.retrieval import (
    build_retrieval_suite,
    evaluate_retrieval_model,
    write_retrieval_artifacts,
)


class TinyTokenizer:
    """Deterministic word tokenizer sufficient for retrieval-unit tests."""

    def __init__(self) -> None:
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self._tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
        self._ids = {token: index for index, token in enumerate(self._tokens)}

    def encode(self, text: str) -> list[int]:
        pieces = re.findall(r" ?[A-Za-z']+| ?[^A-Za-z\s]", text)
        ids = []
        for piece in pieces:
            if piece not in self._ids:
                self._ids[piece] = len(self._tokens)
                self._tokens.append(piece)
            ids.append(self._ids[piece])
        return ids

    def decode(
        self,
        ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        pieces = []
        for token_id in ids:
            if skip_special_tokens and token_id < 4:
                continue
            pieces.append(self._tokens[token_id])
        return "".join(pieces)

    @property
    def vocab_size(self) -> int:
        return len(self._tokens)


class ContextCopyModel(nn.Module):
    """Predict the final candidate token mentioned in the context."""

    def __init__(self, vocab_size: int, candidate_ids: list[int]) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.candidate_ids = candidate_ids

    def forward(self, input_ids: Tensor) -> Tensor:
        batch, sequence = input_ids.shape
        logits = torch.zeros(batch, sequence, self.vocab_size, device=input_ids.device)
        for batch_index in range(batch):
            mentioned = [
                int(token_id)
                for token_id in input_ids[batch_index].tolist()
                if int(token_id) in self.candidate_ids
            ]
            if mentioned:
                logits[batch_index, -1, mentioned[-1]] = 8.0
        return logits


class ConstantModel(nn.Module):
    def __init__(self, vocab_size: int, preferred_id: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.preferred_id = preferred_id

    def forward(self, input_ids: Tensor) -> Tensor:
        batch, sequence = input_ids.shape
        logits = torch.zeros(batch, sequence, self.vocab_size, device=input_ids.device)
        logits[:, -1, self.preferred_id] = 1.0
        return logits


def _suite(*, seed: int = 42) -> tuple[TinyTokenizer, dict[str, object]]:
    tokenizer = TinyTokenizer()
    suite = build_retrieval_suite(
        tokenizer,
        context_length=256,
        distances=(32, 64, 128, 192),
        pairs_per_distance=8,
        seed=seed,
    )
    return tokenizer, suite


def test_retrieval_suite_is_deterministic_balanced_and_token_exact() -> None:
    tokenizer_a, suite_a = _suite(seed=17)
    tokenizer_b, suite_b = _suite(seed=17)
    _, suite_c = _suite(seed=18)

    assert suite_a == suite_b
    assert suite_a != suite_c
    assert len(suite_a["candidates"]) == 4
    for candidate in suite_a["candidates"]:
        assert tokenizer_a.encode(candidate["text"]) == [candidate["token_id"]]

    candidate_ids = {candidate["token_id"] for candidate in suite_a["candidates"]}
    for distance in suite_a["distances"]:
        pairs = [pair for pair in suite_a["pairs"] if pair["measured_distance"] == distance]
        assert len(pairs) == 8
        assert {pair["template"] for pair in pairs} == {
            "lantern",
            "ribbon",
            "gate",
            "blanket",
        }
        targets = [
            pair["variants"][variant]["target_index"] for pair in pairs for variant in ("a", "b")
        ]
        assert [targets.count(index) for index in range(4)] == [4, 4, 4, 4]
        for pair in pairs:
            assert pair["requested_distance"] == pair["measured_distance"]
            lengths = {
                len(pair["variants"][variant]["input_ids"]) for variant in ("a", "b", "control")
            }
            assert len(lengths) == 1
            assert lengths.pop() <= suite_a["context_length"]
            assert not candidate_ids.intersection(pair["variants"]["control"]["input_ids"])
            for variant in ("a", "b"):
                values = pair["variants"][variant]
                measured = len(values["input_ids"]) - 1 - values["answer_position"]
                assert measured == distance

    assert tokenizer_a.vocab_size == tokenizer_b.vocab_size


def test_default_retrieval_suite_exercises_the_512_token_window() -> None:
    suite = build_retrieval_suite(TinyTokenizer(), pairs_per_distance=4)
    assert suite["context_length"] == 512
    assert suite["distances"] == [32, 128, 256, 384, 448]
    assert all(
        pair["measured_distance"] == pair["requested_distance"]
        for pair in suite["pairs"]
    )


def test_retrieval_metrics_reward_counterfactual_context_use() -> None:
    tokenizer, suite = _suite()
    candidate_ids = [candidate["token_id"] for candidate in suite["candidates"]]
    evaluation = evaluate_retrieval_model(
        ContextCopyModel(tokenizer.vocab_size, candidate_ids),
        suite,
        label="copy",
        device=torch.device("cpu"),
        batch_size=7,
    )

    summary = evaluation["summary"]
    assert summary["candidate_accuracy"] == 1.0
    assert summary["paired_flip_accuracy"] == 1.0
    assert summary["mean_target_vs_best_distractor_margin"] == 8.0
    assert summary["mean_contextual_logit_lift"] == 8.0
    assert all(metrics["candidate_accuracy"] == 1.0 for metrics in summary["by_distance"].values())
    assert all(
        metrics["paired_flip_accuracy"] == 1.0 for metrics in summary["by_template"].values()
    )


def test_balanced_suite_exposes_context_free_candidate_prior() -> None:
    tokenizer, suite = _suite()
    candidate_ids = [candidate["token_id"] for candidate in suite["candidates"]]
    evaluation = evaluate_retrieval_model(
        ConstantModel(tokenizer.vocab_size, candidate_ids[0]),
        suite,
        label="constant",
        device=torch.device("cpu"),
    )

    summary = evaluation["summary"]
    assert summary["candidate_accuracy"] == 0.25
    assert summary["paired_flip_accuracy"] == 0.0
    assert summary["mean_contextual_logit_lift"] == 0.0


def test_retrieval_artifacts_are_machine_and_human_readable(tmp_path: Path) -> None:
    tokenizer, suite = _suite()
    candidate_ids = [candidate["token_id"] for candidate in suite["candidates"]]
    evaluation = evaluate_retrieval_model(
        ContextCopyModel(tokenizer.vocab_size, candidate_ids),
        suite,
        label="Tiny SAN",
        architecture="kiwilm2",
        checkpoint="best.pt",
        device=torch.device("cpu"),
    )

    artifacts = write_retrieval_artifacts(
        tmp_path / "retrieval",
        suite=suite,
        evaluations=[evaluation],
    )

    written_suite = json.loads(Path(artifacts["suite_path"]).read_text())
    written_result = json.loads(Path(artifacts["results_path"]).read_text().strip())
    report = Path(artifacts["report_path"]).read_text()
    assert written_suite == suite
    assert written_result["summary"]["candidate_accuracy"] == 1.0
    assert "| Tiny SAN | 100.00% | 100.00% | 8.0000 | 8.0000 |" in report
    assert "### By distance" in report
    assert artifacts["model_count"] == 1


def test_retrieval_suite_rejects_incompatible_candidates_and_window() -> None:
    tokenizer = TinyTokenizer()
    with pytest.raises(ValueError, match="exactly four"):
        build_retrieval_suite(tokenizer, candidate_values=("red", "blue"))
    with pytest.raises(ValueError, match="exceeding context_length"):
        build_retrieval_suite(
            tokenizer,
            context_length=64,
            distances=(64,),
            pairs_per_distance=4,
        )
