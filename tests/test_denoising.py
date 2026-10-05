"""M4 tokenizer/mask/loss correctness and bounded local training/recovery evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from kiwilm.data import PreparedTokenData, prepare_from_stories
from kiwilm.tokenizer import ByteBPETokenizer, ReservedTokenError
from kiwilm.v3 import KiwiLM3Config, build_encoder
from kiwilm.v3.checkpoints import load_training_state, save_training_state
from kiwilm.v3.masking import MaskingConfig, corrupt_tokens
from kiwilm.v3.objectives import denoising_forward, masked_reconstruction_loss
from kiwilm.v3.tokenizer import MASK_TOKEN, MaskBPETokenizer
from kiwilm.v3.trainer import DenoisingTrainConfig, DenoisingTrainer
from kiwilm.v3.weights import load_encoder_weights, save_encoder_weights


@pytest.fixture(autouse=True)
def fixed_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def data(tmp_path):
    path = tmp_path / "data"
    prepare_from_stories(
        path,
        ["the cat saw the red ball. " * 30] * 3,
        ["the cat saw the red ball. " * 30],
        vocab_size=300,
        min_frequency=1,
        show_progress=False,
    )
    return PreparedTokenData(path)


def make_trainer(data, **overrides):
    tokenizer = MaskBPETokenizer.from_base(data.tokenizer)
    config = KiwiLM3Config(
        vocab_size=tokenizer.vocab_size,
        context_length=16,
        d_model=16,
        num_heads=2,
        swiglu_dim=48,
        noise_embedding_dim=16,
        num_blocks=3,
        dropout=0.1,
    )
    return DenoisingTrainer(
        build_encoder(config), tokenizer, data, DenoisingTrainConfig(max_steps=40, **overrides)
    )


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_mask_token_json_preserves_bpe_ids_and_reserved_text(data, tmp_path):
    base = data.tokenizer
    original = base.to_json()
    v3 = MaskBPETokenizer.from_base(base)
    assert v3.mask_id == base.vocab_size and v3.vocab_size == base.vocab_size + 1
    assert base.to_json() == original
    assert v3._tokenizer.get_vocab() == {**base._tokenizer.get_vocab(), MASK_TOKEN: v3.mask_id}
    for text in ("the cat saw a ball", "Привет 👋 €", "unknown string"):
        assert v3.encode(text) == base.encode(text)
        assert v3.decode(v3.encode(text)) == base.decode(base.encode(text))
    with pytest.raises(ReservedTokenError, match="MASK"):
        v3.encode("malicious [MASK] control string")
    with pytest.raises(ReservedTokenError, match="MASK"):
        v3.encode_with_offsets("[MASK]")
    assert v3._tokenizer.encode(MASK_TOKEN).ids == [v3.mask_id]
    assert v3.decode([v3.mask_id]) == ""
    assert v3.decode([v3.mask_id], skip_special_tokens=False) == MASK_TOKEN
    path = tmp_path / "v3-tokenizer.json"
    v3.save(path)
    loaded = MaskBPETokenizer.load(path)
    assert loaded.fingerprint == v3.fingerprint
    loaded.assert_base_compatible(base)
    assert loaded.mask_id == v3.mask_id
    with pytest.raises(FileExistsError):
        v3.save(path)
    with pytest.raises(ValueError, match="already contains MASK"):
        MaskBPETokenizer.from_base(loaded)
    with pytest.raises(ValueError):
        MaskBPETokenizer.from_json(base.to_json())
    mismatched = ByteBPETokenizer.train(["different words and BPE merges"], vocab_size=300)
    with pytest.raises(ValueError, match="ID-preserving"):
        v3.assert_base_compatible(mismatched)


def test_tokenizer_rejects_non_special_mask(data):
    raw = json.loads(MaskBPETokenizer.from_base(data.tokenizer).to_json())
    raw["added_tokens"][-1]["special"] = False
    with pytest.raises(ValueError, match="special token"):
        MaskBPETokenizer(Tokenizer.from_str(json.dumps(raw)))


@pytest.mark.parametrize("noise", [0.0, 0.15, 0.5, 1.0])
def test_padding_control_tokens_and_prompt_are_never_corrupted(noise):
    ids = torch.tensor([[2, 4, 5, 6, 3, 0], [2, 8, 7, 6, 3, 0]])
    original = ids.clone()
    protected = torch.zeros_like(ids, dtype=torch.bool)
    protected[:, 1] = True
    result = corrupt_tokens(
        ids,
        MaskingConfig(vocab_size=10, mask_id=9),
        generator=torch.Generator().manual_seed(42),
        noise_level=noise,
        protected_mask=protected,
    )
    expected_eligible = torch.tensor([[False, False, True, True, False, False]]).expand_as(ids)
    assert torch.equal(result.eligible_positions, expected_eligible)
    assert torch.equal(ids, original)
    assert torch.equal(result.input_ids[~result.masked_positions], ids[~result.masked_positions])
    assert (result.input_ids[result.masked_positions] == 9).all()
    assert not (result.masked_positions & ~expected_eligible).any()
    if noise == 0:
        assert not result.masked_positions.any()
    if noise == 1:
        assert torch.equal(result.masked_positions, expected_eligible)
    valid = torch.ones_like(ids, dtype=torch.bool)
    forced_valid = corrupt_tokens(
        ids,
        MaskingConfig(vocab_size=10, mask_id=9),
        generator=torch.Generator(),
        noise_level=1,
        attention_mask=valid,
    )
    assert not forced_valid.attention_mask[:, -1].any()


def test_mask_noise_rng_recovery_independent_of_global_rng():
    config = MaskingConfig(vocab_size=10, mask_id=9)
    ids = torch.full((2000, 10), 4)
    generator = torch.Generator().manual_seed(142)
    state = generator.get_state()
    first = corrupt_tokens(ids, config, generator=generator)
    generator.set_state(state)
    torch.manual_seed(666)
    torch.rand(17)
    second = corrupt_tokens(ids, config, generator=generator)
    assert torch.equal(first.noise_level, second.noise_level)
    assert torch.equal(first.input_ids, second.input_ids)
    full_rate = (first.noise_level == 1).float().mean().item()
    assert full_rate == pytest.approx(0.1, abs=0.025)
    assert first.noise_level.min() >= 0.01
    assert (first.noise_level < 0.1).any() and (first.noise_level > 0.9).any()
    fixed = corrupt_tokens(ids, config, generator=generator, noise_level=0.5)
    assert fixed.masked_positions.float().mean().item() == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize(
    "overrides",
    [
        {"vocab_size": 1},
        {"vocab_size": True},
        {"mask_id": 4},
        {"pad_id": -1},
        {"protected_token_ids": (0, 0)},
        {"protected_token_ids": (2, 3)},
        {"protected_token_ids": (0, 9)},
        {"min_noise": 0},
        {"min_noise": 1},
        {"min_noise": float("nan")},
        {"full_mask_probability": -1},
        {"full_mask_probability": float("inf")},
    ],
)
def test_mask_config_validation(overrides):
    values = {"vocab_size": 10, "mask_id": 9}
    values.update(overrides)
    with pytest.raises(ValueError):
        MaskingConfig(**values)


@pytest.mark.parametrize(
    "noise", [True, -1, 1.1, float("nan"), torch.zeros(3), torch.ones(2, dtype=torch.long)]
)
def test_mask_noise_validation(noise):
    with pytest.raises(ValueError, match="noise_level"):
        corrupt_tokens(
            torch.full((2, 5), 4),
            MaskingConfig(vocab_size=10, mask_id=9),
            generator=torch.Generator(),
            noise_level=noise,
        )


@pytest.mark.parametrize(
    "ids",
    [
        torch.tensor([[9]]),
        torch.tensor([[-1]]),
        torch.tensor([[10]]),
        torch.zeros(1, 0, dtype=torch.long),
        torch.ones(2, 3),
        torch.ones(3, dtype=torch.long),
    ],
)
def test_clean_ids_validation(ids):
    with pytest.raises(ValueError):
        corrupt_tokens(ids, MaskingConfig(vocab_size=10, mask_id=9), generator=torch.Generator())


def test_mask_shape_validation():
    for name in ("attention_mask", "protected_mask"):
        with pytest.raises(ValueError, match=name):
            corrupt_tokens(
                torch.full((2, 3), 4),
                MaskingConfig(vocab_size=10, mask_id=9),
                generator=torch.Generator(),
                **{name: torch.ones(2, 3)},
            )


def test_loss_exact_normalization_ignored_targets_and_gradients():
    logits = torch.randn(2, 4, 10, requires_grad=True)
    targets = torch.tensor([[2, 4, 5, 0], [2, 6, 7, 3]])
    selected = torch.tensor([[False, True, False, False], [False, True, True, False]])
    result = masked_reconstruction_loss(logits, targets, selected, mask_id=9)
    reference = torch.nn.functional.cross_entropy(logits[selected][:, :9], targets[selected])
    torch.testing.assert_close(result.loss, reference)
    assert result.masked_tokens == 3
    result.loss.backward()
    assert logits.grad[~selected].count_nonzero() == 0
    assert logits.grad[..., 9].count_nonzero() == 0
    assert logits.grad[selected][:, :9].norm() > 0
    changed = targets.clone()
    changed[~selected] = 999  # ignored, even when not a legal vocabulary ID
    torch.testing.assert_close(
        masked_reconstruction_loss(logits, changed, selected, mask_id=9).loss, reference
    )


def test_empty_mask_loss_is_differentiable_zero():
    logits = torch.randn(2, 3, 10, requires_grad=True)
    result = masked_reconstruction_loss(
        logits, torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3, dtype=torch.bool), mask_id=9
    )
    assert result.loss == 0 and result.masked_tokens == 0
    result.loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.count_nonzero() == 0


def test_clean_targets_have_no_input_side_channel():
    config = MaskingConfig(vocab_size=10, mask_id=9)
    model = build_encoder(
        KiwiLM3Config(
            vocab_size=10,
            d_model=16,
            num_heads=2,
            swiglu_dim=48,
            noise_embedding_dim=16,
            num_blocks=3,
        )
    ).eval()
    observed = []
    hook = model.register_forward_pre_hook(lambda _module, args: observed.append(args[0].clone()))
    first = torch.tensor([[2, 4, 5, 3]])
    second = torch.tensor([[2, 6, 7, 3]])
    try:
        result, batch = denoising_forward(
            model, first, config, generator=torch.Generator(), noise_level=1
        )
        denoising_forward(model, second, config, generator=torch.Generator(), noise_level=1)
    finally:
        hook.remove()
    assert torch.equal(observed[0], observed[1])
    assert torch.equal(batch.input_ids, torch.tensor([[2, 9, 9, 3]]))
    result.loss.backward()
    assert model.token_embedding.weight.grad[9].norm() > 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_steps": 0},
        {"max_steps": 1001},
        {"batch_size": True},
        {"context_length": 0},
        {"learning_rate": 0},
        {"learning_rate": float("nan")},
        {"noise_seed": -1},
        {"gradient_clip": 0},
    ],
)
def test_train_config_validation(overrides):
    with pytest.raises(ValueError):
        DenoisingTrainConfig(**overrides)


def test_validation_does_not_change_training_rng_or_mode(data):
    trainer = make_trainer(data)
    trainer.model.train()
    noise, order, global_rng = (
        trainer.noise_generator.get_state().clone(),
        trainer.data_generator.get_state().clone(),
        torch.get_rng_state().clone(),
    )
    first = trainer.evaluate(batches=2)
    assert first == trainer.evaluate(batches=2)
    assert trainer.model.training
    assert torch.equal(noise, trainer.noise_generator.get_state())
    assert torch.equal(order, trainer.data_generator.get_state())
    assert torch.equal(global_rng, torch.get_rng_state())
    assert [row["noise_level"] for row in first["rows"]] == [0.15, 0.5, 0.9, 1.0]
    assert first["autoregressive_perplexity"] is None


def test_training_is_finite_and_learns_toy_reconstruction(data):
    torch.manual_seed(42)
    trainer = make_trainer(data)
    before = trainer.evaluate(batches=4)
    rows = [trainer.train_step() for _ in range(40)]
    after = trainer.evaluate(batches=4)
    assert all(row["loss"] is None or torch.isfinite(torch.tensor(row["loss"])) for row in rows)
    assert trainer.optimizer_steps > 30
    assert sum(row["loss"] for row in after["rows"]) < sum(row["loss"] for row in before["rows"])
    with pytest.raises(ValueError, match="exhausted"):
        trainer.train_step()


def test_zero_mask_step_does_not_apply_adamw_or_decay(data):
    trainer = make_trainer(data)
    trainer.masking = replace(
        trainer.masking, protected_token_ids=tuple(range(trainer.tokenizer.mask_id))
    )
    before = {name: tensor.clone() for name, tensor in trainer.model.state_dict().items()}
    row = trainer.train_step()
    assert row["skipped_empty_mask"] and row["loss"] is None
    assert trainer.step == 1 and trainer.optimizer_steps == 0
    assert not trainer.optimizer.state
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(before[name], value)


def test_checkpoint_resume_exact_with_dropout_and_noise(data, tmp_path):
    torch.manual_seed(141)
    first = make_trainer(data)
    for _ in range(3):
        first.train_step()
    path = save_training_state(first, tmp_path / "step3.pt")
    sha = checksum(path)
    expected = [first.train_step() for _ in range(3)]
    resumed = make_trainer(data)
    load_training_state(resumed, path, expected_sha256=sha)
    assert [resumed.train_step() for _ in range(3)] == expected
    for name, value in first.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
    assert first.evaluate() == resumed.evaluate()
    with pytest.raises(FileExistsError):
        save_training_state(first, path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_training_state(resumed, path, expected_sha256="b" * 64)
    different = make_trainer(data, noise_seed=143)
    before = different.model.token_embedding.weight.detach().clone()
    with pytest.raises(ValueError, match="contract mismatch"):
        load_training_state(different, path, expected_sha256=sha)
    assert torch.equal(before, different.model.token_embedding.weight)


def test_checkpoint_rejects_missing_state_and_inference_artifacts(data, tmp_path):
    trainer = make_trainer(data)
    path = save_training_state(trainer, tmp_path / "state.pt")
    payload = torch.load(path, weights_only=True)
    del payload["noise_rng"]
    malformed = tmp_path / "malformed.pt"
    torch.save(payload, malformed)
    with pytest.raises(ValueError, match="missing required noise_rng"):
        load_training_state(trainer, malformed, expected_sha256=checksum(malformed))
    torch.save({"format": "kiwilm3-encoder-weights-v1"}, malformed)
    with pytest.raises(ValueError, match="different loaders"):
        load_training_state(trainer, malformed, expected_sha256=checksum(malformed))


def test_failed_step_cannot_be_saved_or_silently_continued(data, tmp_path):
    trainer = make_trainer(data)
    path = save_training_state(trainer, tmp_path / "before.pt")
    with torch.no_grad():
        trainer.model.token_embedding.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        trainer.train_step()
    with pytest.raises(RuntimeError, match="incomplete"):
        save_training_state(trainer, tmp_path / "failed.pt")
    with pytest.raises(RuntimeError, match="restore"):
        trainer.train_step()
    load_training_state(trainer, path, expected_sha256=checksum(path))
    assert trainer.at_step_boundary and trainer.train_step()["step"] == 1


def test_resume_configuration_cannot_be_mutated_after_locking(data, tmp_path):
    trainer = make_trainer(data)
    trainer.masking = replace(trainer.masking, min_noise=0.02)
    with pytest.raises(ValueError, match="live trainer settings changed"):
        save_training_state(trainer, tmp_path / "changed.pt")


def test_fresh_process_restores_noise_data_dropout_and_progress(data, tmp_path):
    torch.manual_seed(42)
    trainer = make_trainer(data)
    for _ in range(2):
        trainer.train_step()
    path = save_training_state(trainer, tmp_path / "step2.pt")
    digest = checksum(path)
    expected = [trainer.train_step() for _ in range(2)]
    program = """
import json, sys, torch
from kiwilm.data import PreparedTokenData
from kiwilm.v3 import KiwiLM3Config, build_encoder
from kiwilm.v3.tokenizer import MaskBPETokenizer
from kiwilm.v3.trainer import DenoisingTrainer, DenoisingTrainConfig
from kiwilm.v3.masking import MaskingConfig
from kiwilm.v3.checkpoints import load_training_state
torch.set_num_threads(1)
data_path, checkpoint, digest = sys.argv[1:]
contract = torch.load(checkpoint, weights_only=True)['contract']
data = PreparedTokenData(data_path)
tokenizer = MaskBPETokenizer.from_base(data.tokenizer)
torch.manual_seed(999)
trainer = DenoisingTrainer(build_encoder(KiwiLM3Config.from_dict(contract['model'])),
    tokenizer, data, DenoisingTrainConfig(**contract['training']),
    MaskingConfig(**contract['masking']))
load_training_state(trainer, checkpoint, expected_sha256=digest)
print(json.dumps([trainer.train_step() for _ in range(2)]))
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(data.data_dir), str(path), digest],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == expected


def test_tokenizer_conversion_command_preserves_original(data, tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = data.data_dir / data.metadata["tokenizer"]["file"]
    before = source.read_bytes()
    output = tmp_path / "mask-tokenizer.json"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/prepare_kiwilm3_tokenizer.py"),
            "--source",
            str(source),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["existing_ids_preserved"] and not report["training_started"]
    assert report["mask_id"] == data.tokenizer.vocab_size
    assert source.read_bytes() == before
    MaskBPETokenizer.load(output).assert_base_compatible(data.tokenizer)


def test_mask_tokenizer_matches_inference_weight_roundtrip(data, tmp_path):
    trainer = make_trainer(data)
    trainer.train_step()
    trainer.model.eval()
    path = save_encoder_weights(
        trainer.model,
        tmp_path / "denoiser.safetensors",
        dtype="fp32",
        tokenizer_sha256=trainer.tokenizer.fingerprint,
    )
    loaded, config = load_encoder_weights(
        path,
        expected_config=trainer.model.config,
        expected_tokenizer_sha256=trainer.tokenizer.fingerprint,
    )
    assert config.vocab_size == trainer.tokenizer.mask_id + 1
    clean, _ = data.get_batch("validation", batch_size=2, context_length=16)
    expected, _ = denoising_forward(
        trainer.model,
        clean,
        trainer.masking,
        generator=torch.Generator().manual_seed(42),
        noise_level=1,
    )
    actual, _ = denoising_forward(
        loaded, clean, trainer.masking, generator=torch.Generator().manual_seed(42), noise_level=1
    )
    torch.testing.assert_close(expected.loss, actual.loss, rtol=0, atol=0)


def test_denoising_notebook_generated_sources_and_safe_default_execution(tmp_path, monkeypatch):
    import nbformat

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "m4_builder", root / "scripts/build_kiwilm3_denoising_notebook.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    saved = nbformat.read(root / "notebooks/kiwilm3-denoising.ipynb", as_version=4)
    nbformat.validate(saved)
    assert [cell.source for cell in saved.cells] == [
        cell.source for cell in builder.notebook().cells
    ]
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for cell in saved.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, "m4-notebook-cell", "exec"), namespace)
    assert namespace["trainer"] is None
    assert not namespace["START_TRAINING"] and not namespace["INSTALL_PACKAGE"]
    assert list(tmp_path.iterdir()) == []
