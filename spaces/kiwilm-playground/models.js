export const MODELS = {
  kiwilm2: {
    label: "KiwiLM 2 · Dense · 1B tokens", contextLength: 512, vocabSize: 32000,
    description: "64.25M parameters · Dense GQA + gated convolution · 512-token context. First use downloads about 323 MB. Cached decoding; WebGPU or WebAssembly. Desktop browsers recommended.",
    card: "https://huggingface.co/Tasty-Kiwi/KiwiLM-2",
    caveat: "Experimental base model, not a chatbot. It can loop, drift off topic, and invent facts. 41/120 evaluated samples had severe repetition.",
  },
  x: {
    label: "KiwiLM 1 · Model X · Direct SFT v2", contextLength: 256, vocabSize: 8192,
    description: "5.39M parameters · hybrid gated CNN + attention · 256-token context. About 22 MB; original uncached ONNX decoding. Auto uses compatible WebAssembly.",
    card: "https://huggingface.co/Tasty-Kiwi/KiwiLM-X",
    caveat: "Historical instruction-tuned story model. It can repeat, forget constraints and produce inconsistent stories.",
  },
  "y-direct": {
    label: "KiwiLM 1 · Model Y · Direct SFT v2", contextLength: 256, vocabSize: 8192,
    description: "5.37M parameters · Transformer · direct instruction tuning · 256-token context. About 22 MB; original uncached ONNX decoding. Auto uses compatible WebAssembly.",
    card: "https://huggingface.co/Tasty-Kiwi/KiwiLM",
    caveat: "Historical instruction-tuned story model. It can repeat, forget constraints and produce inconsistent stories.",
  },
  "y-cpt": {
    label: "KiwiLM 1 · Model Y · CPT → SFT v2", contextLength: 256, vocabSize: 8192,
    description: "5.37M parameters · Transformer · SimpleStories CPT then instruction tuning · 256-token context. About 22 MB; original uncached ONNX decoding. Auto uses compatible WebAssembly.",
    card: "https://huggingface.co/Tasty-Kiwi/KiwiLM",
    caveat: "Historical instruction-tuned story model. It can repeat, forget constraints and produce inconsistent stories.",
  },
};

export function getModel(id) {
  if (!Object.hasOwn(MODELS, id)) throw new Error("Unknown model.");
  return MODELS[id];
}

export function engineFor(modelId, backend, hasGPU) {
  getModel(modelId);
  return backend === "auto" ? (modelId === "kiwilm2" && hasGPU ? "webgpu" : "wasm") : backend;
}

export function examplePrompts(modelId) {
  getModel(modelId);
  if (modelId === "kiwilm2") return {
    story: "Once upon a time, a fox found a box.",
    explanation: "Rain falls when",
    dialogue: "\"Where are we going?\" asked the rabbit.\n\"",
  };
  const instruction = (features, words, summary) => "Instruction: Write a story that follows every provided condition. Use every requested word exactly as written.\n"
    + `Features: ${features}\nWords: ${words}\nSummary: ${summary}\nStory:\n`;
  return {
    story: instruction("Dialogue", "oak, gloomy, kind", "Two friends help each other get home before dark."),
    explanation: instruction("", "rain, cloud, garden", "A child learns why rain helps the garden."),
    dialogue: instruction("Dialogue", "rabbit, path, friend", "A rabbit asks a friend where they are going."),
  };
}
