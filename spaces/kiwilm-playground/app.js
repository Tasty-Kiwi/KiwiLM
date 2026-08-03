import { Tokenizer } from "@huggingface/tokenizers";
import * as ort from "onnxruntime-web/wasm";
import "./style.css";

const CONTEXT_LENGTH = 256;
const VOCAB_SIZE = 8192;
const BOS_ID = 2;
const EOS_ID = 3;
const MODEL_ROOT = `${import.meta.env.BASE_URL}models/`;

const MODEL_OPTIONS = {
  x: {
    description: "Hybrid gated CNN + attention · 5.39M parameters · throughput finalist",
  },
  "y-direct": {
    description: "Four-block Transformer · 5.37M parameters · best in-domain quality",
  },
  "y-cpt": {
    description: "SimpleStories CPT + instruction tuning · best focused adherence",
  },
};

const DEFAULT_PROMPT = `Instruction: Write a story that follows every provided condition. Use every requested word exactly as written.
Features: Dialogue
Words: oak, gloomy, kind
Summary: Two friends help each other get home before dark.
Story:
`;

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

const cache = new Map();
let stopRequested = false;

const elements = {
  form: document.querySelector("#controls"),
  model: document.querySelector("#model"),
  description: document.querySelector("#model-description"),
  prompt: document.querySelector("#prompt"),
  tokens: document.querySelector("#tokens"),
  tokensValue: document.querySelector("#tokens-value"),
  temperature: document.querySelector("#temperature"),
  temperatureValue: document.querySelector("#temperature-value"),
  topK: document.querySelector("#top-k"),
  topKValue: document.querySelector("#top-k-value"),
  seed: document.querySelector("#seed"),
  generate: document.querySelector("#generate"),
  stop: document.querySelector("#stop"),
  copy: document.querySelector("#copy"),
  output: document.querySelector("#output"),
  status: document.querySelector("#status"),
};

function setStatus(message, kind = "") {
  elements.status.textContent = message;
  elements.status.dataset.kind = kind;
}

function updateModelDescription() {
  elements.description.textContent = MODEL_OPTIONS[elements.model.value].description;
}

function setBusy(busy) {
  elements.generate.disabled = busy;
  elements.stop.disabled = !busy;
  elements.model.disabled = busy;
}

async function loadModel(modelId) {
  if (cache.has(modelId)) {
    return cache.get(modelId);
  }
  setStatus(`Loading ${modelId} tokenizer and 22 MB ONNX model…`, "working");
  const [tokenizerJson, tokenizerConfig, session] = await Promise.all([
    fetch(`${MODEL_ROOT}${modelId}/tokenizer.json`).then((response) => response.json()),
    fetch(`${MODEL_ROOT}${modelId}/tokenizer_config.json`).then((response) => response.json()),
    ort.InferenceSession.create(`${MODEL_ROOT}${modelId}/model.onnx`, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    }),
  ]);
  const tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
  const loaded = { tokenizer, session };
  cache.set(modelId, loaded);
  return loaded;
}

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function selectToken(logits, temperature, topK, random) {
  if (temperature === 0) {
    let bestIndex = 0;
    for (let index = 1; index < logits.length; index += 1) {
      if (logits[index] > logits[bestIndex]) bestIndex = index;
    }
    return bestIndex;
  }

  const candidates = Array.from(logits, (value, index) => ({ index, value }));
  candidates.sort((left, right) => right.value - left.value);
  const selected = topK > 0 ? candidates.slice(0, topK) : candidates;
  const maximum = selected[0].value / temperature;
  let total = 0;
  for (const candidate of selected) {
    candidate.weight = Math.exp(candidate.value / temperature - maximum);
    total += candidate.weight;
  }
  let threshold = random() * total;
  for (const candidate of selected) {
    threshold -= candidate.weight;
    if (threshold <= 0) return candidate.index;
  }
  return selected.at(-1).index;
}

async function generate(event) {
  event.preventDefault();
  stopRequested = false;
  setBusy(true);
  elements.output.textContent = "";

  try {
    const modelId = elements.model.value;
    const { tokenizer, session } = await loadModel(modelId);
    const encoded = tokenizer.encode(elements.prompt.value, { add_special_tokens: false });
    const allTokens = [BOS_ID, ...encoded.ids];
    const promptLength = allTokens.length;
    const maximumTokens = Number(elements.tokens.value);
    const temperature = Number(elements.temperature.value);
    const topK = Number(elements.topK.value);
    const random = mulberry32(Number(elements.seed.value));
    const startedAt = performance.now();

    for (let step = 0; step < maximumTokens && !stopRequested; step += 1) {
      const context = allTokens.slice(-CONTEXT_LENGTH);
      const input = new ort.Tensor(
        "int64",
        BigInt64Array.from(context, BigInt),
        [1, context.length],
      );
      const result = await session.run({ input_ids: input });
      const logits = result.logits.data;
      if (logits.length !== VOCAB_SIZE) {
        throw new Error(`Unexpected logits size: ${logits.length}`);
      }
      const token = selectToken(logits, temperature, topK, random);
      allTokens.push(token);
      const completion = tokenizer.decode(allTokens.slice(promptLength), {
        skip_special_tokens: true,
      });
      elements.output.textContent = completion;
      const elapsed = (performance.now() - startedAt) / 1000;
      setStatus(
        `Generated ${step + 1}/${maximumTokens} tokens · ${((step + 1) / elapsed).toFixed(1)} tok/s`,
        "working",
      );
      if (token === EOS_ID) break;
      await new Promise(requestAnimationFrame);
    }
    setStatus(stopRequested ? "Generation stopped." : "Generation complete.", "success");
  } catch (error) {
    console.error(error);
    setStatus(`Generation failed: ${error.message}`, "error");
  } finally {
    setBusy(false);
  }
}

elements.prompt.value = DEFAULT_PROMPT;
updateModelDescription();
elements.model.addEventListener("change", updateModelDescription);
elements.tokens.addEventListener("input", () => {
  elements.tokensValue.textContent = elements.tokens.value;
});
elements.temperature.addEventListener("input", () => {
  elements.temperatureValue.textContent = Number(elements.temperature.value).toFixed(2);
});
elements.topK.addEventListener("input", () => {
  elements.topKValue.textContent = elements.topK.value;
});
elements.form.addEventListener("submit", generate);
elements.stop.addEventListener("click", () => {
  stopRequested = true;
  setStatus("Stopping after the current token…", "working");
});
elements.copy.addEventListener("click", async () => {
  await navigator.clipboard.writeText(elements.output.textContent);
  elements.copy.textContent = "Copied";
  setTimeout(() => {
    elements.copy.textContent = "Copy";
  }, 1200);
});
