"""Gradio playground for the final KiwiLM Model X and Model Y checkpoints."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import torch

from kiwilm.generation import generate_stream
from kiwilm.inference import load_trained_model
from kiwilm.tokenizer import ByteBPETokenizer


@dataclass(frozen=True)
class ModelOption:
    path: Path
    description: str


MODEL_X_ROOT = Path(os.getenv("KIWILM_MODEL_X_ROOT", "/models/x"))
MODEL_Y_ROOT = Path(os.getenv("KIWILM_MODEL_Y_ROOT", "/models/y"))
MODELS = {
    "Model X — direct SFT v2": ModelOption(
        MODEL_X_ROOT,
        "Hybrid gated CNN + attention · 5.39M parameters · throughput finalist",
    ),
    "Model Y — direct SFT v2": ModelOption(
        MODEL_Y_ROOT / "direct-sft-v2",
        "Four-block Transformer · 5.37M parameters · best in-domain quality",
    ),
    "Model Y — CPT → SFT v2": ModelOption(
        MODEL_Y_ROOT / "cpt-sft-v2",
        "SimpleStories CPT + instruction tuning · best focused adherence",
    ),
}

DEFAULT_PROMPT = """Instruction: Write a story that follows every provided condition. \
Use every requested word exactly as written.
Features: Dialogue
Words: oak, gloomy, kind
Summary: Two friends help each other get home before dark.
Story:
"""

_MODEL_CACHE: dict[str, tuple[torch.nn.Module, ByteBPETokenizer]] = {}
_MODEL_LOCK = threading.Lock()


def load_model(label: str) -> tuple[torch.nn.Module, ByteBPETokenizer]:
    """Load a mounted model once and retain it for subsequent requests."""

    if label not in MODELS:
        raise gr.Error(f"Unknown model: {label}")
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(label)
        if cached is not None:
            return cached
        option = MODELS[label]
        model_file = option.path / "model.safetensors"
        tokenizer_file = option.path / "tokenizer.json"
        if not model_file.is_file() or not tokenizer_file.is_file():
            raise gr.Error(
                f"Model files are unavailable at {option.path}. "
                "The private Hub volume may still be mounting."
            )
        model, _ = load_trained_model(
            option.path,
            data_fingerprint=None,
            device=torch.device("cpu"),
        )
        tokenizer = ByteBPETokenizer.load(tokenizer_file)
        _MODEL_CACHE[label] = (model, tokenizer)
        return model, tokenizer


def describe_model(label: str) -> str:
    """Return a short description for the selected model."""

    option = MODELS.get(label)
    return option.description if option else "Select a model."


def generate_story(
    label: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> Iterator[str]:
    """Stream an accumulated completion into the Gradio output box."""

    prompt = prompt.strip("\r\n") + "\n"
    if not prompt.strip():
        raise gr.Error("Enter a prompt before generating.")
    model, tokenizer = load_model(label)
    emitted = ""
    resolved_top_k = None if top_k == 0 else int(top_k)
    for chunk in generate_stream(
        model,
        tokenizer,
        prompt,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_k=resolved_top_k,
        seed=int(seed),
        device="cpu",
        include_prompt=False,
        cache="auto",
    ):
        emitted += chunk
        yield emitted


with gr.Blocks(title="KiwiLM Playground", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🥝 KiwiLM Playground

        Compare the final **Model X** hybrid and **Model Y** Transformer on
        instruction-conditioned TinyStories. Generation is streamed from a
        CPU runtime and models are loaded lazily on first use.
        """
    )
    with gr.Row():
        with gr.Column(scale=3):
            model_choice = gr.Dropdown(
                choices=list(MODELS),
                value="Model Y — CPT → SFT v2",
                label="Model",
            )
            model_description = gr.Markdown(
                describe_model("Model Y — CPT → SFT v2")
            )
            prompt = gr.Textbox(
                value=DEFAULT_PROMPT,
                label="Prompt",
                lines=9,
            )
        with gr.Column(scale=2):
            max_new_tokens = gr.Slider(
                minimum=16,
                maximum=256,
                value=160,
                step=8,
                label="Maximum new tokens",
            )
            temperature = gr.Slider(
                minimum=0,
                maximum=1.5,
                value=0.4,
                step=0.05,
                label="Temperature (0 = greedy)",
            )
            top_k = gr.Slider(
                minimum=0,
                maximum=100,
                value=20,
                step=1,
                label="Top-k (0 = disabled)",
            )
            seed = gr.Number(value=42, precision=0, label="Seed")
            generate_button = gr.Button("Generate story", variant="primary")

    output = gr.Textbox(label="Completion", lines=16, show_copy_button=True)
    gr.Markdown(
        "Tiny research models can repeat, forget constraints, and produce "
        "inconsistent stories. Outputs require review."
    )

    model_choice.change(
        describe_model,
        inputs=model_choice,
        outputs=model_description,
    )
    generate_button.click(
        generate_story,
        inputs=[
            model_choice,
            prompt,
            max_new_tokens,
            temperature,
            top_k,
            seed,
        ],
        outputs=output,
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=16).launch()
