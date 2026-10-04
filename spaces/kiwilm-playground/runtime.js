// Pure client-side hashing, causal-mask, context and sampling helpers.
import { getModel } from "./models.js";
export function ngramIndices(tokens, manifest, start = 0) {
  const bigram = new Int32Array(tokens.length - start);
  const trigram = new Int32Array(tokens.length - start);
  for (let i = start; i < tokens.length; i++) {
    const current = tokens[i];
    if (!Number.isInteger(current) || current < 0 || current >= manifest.vocab_size) {
      throw new Error("Invalid tokenizer ID.");
    }
    const previous = tokens[i - 1] ?? 0;
    const previous2 = tokens[i - 2] ?? 0;
    // JS numbers are exact here (< 2^53); never use 32-bit bitwise multiplication.
    bigram[i - start] = (previous * 1000003 + current * 9176 + 17) % manifest.bigram_buckets;
    trigram[i - start] = (previous2 * 1000003 + previous * 9176 + current * 131 + 29)
      % manifest.trigram_buckets;
  }
  return { bigram, trigram };
}

export function contextPlan(tokens, cachedLength, contextLength) {
  const window = tokens.slice(-contextLength);
  const reset = cachedLength === 0 || cachedLength >= contextLength;
  return { window, reset, start: reset ? 0 : window.length - 1,
    past: reset ? 0 : cachedLength };
}

export function attentionBias(length, past) {
  const total = past + length;
  const bias = new Float32Array(length * total);
  for (let query = 0; query < length; query++) {
    bias.fill(-1e9, query * total + past + query + 1, (query + 1) * total);
  }
  return bias;
}

export function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function selectToken(logits, temperature, topK, random) {
  if (!logits.length || !logits.every(Number.isFinite)) throw new Error("Non-finite model logits.");
  if (temperature === 0) {
    return logits.reduce((best, value, i) => value > logits[best] ? i : best, 0);
  }
  const candidates = Array.from(logits, (value, index) => ({ index, value }));
  candidates.sort((a, b) => b.value - a.value);
  const selected = topK > 0 ? candidates.slice(0, topK) : candidates;
  let total = 0;
  for (const candidate of selected) {
    candidate.weight = Math.exp((candidate.value - selected[0].value) / temperature);
    total += candidate.weight;
  }
  let threshold = random() * total;
  for (const candidate of selected) {
    threshold -= candidate.weight;
    if (threshold <= 0) return candidate.index;
  }
  return selected.at(-1).index;
}

export function validateSettings(settings) {
  getModel(settings.modelId);
  if (typeof settings.prompt !== "string" || settings.prompt.length > 32000) {
    throw new Error("Prompt must contain at most 32,000 characters.");
  }
  if (["[PAD]", "[UNK]", "[BOS]", "[EOS]"].some(t => settings.prompt.includes(t))) {
    throw new Error("Prompt contains a reserved tokenizer control token.");
  }
  for (const [key, minimum, maximum] of [["tokens", 1, 256], ["topK", 0, 100],
    ["seed", 0, 4294967295]]) {
    if (!Number.isInteger(settings[key]) || settings[key] < minimum || settings[key] > maximum) {
      throw new Error(`Invalid ${key}.`);
    }
  }
  if (!Number.isFinite(settings.temperature) || settings.temperature < 0 || settings.temperature > 1.5) {
    throw new Error("Invalid temperature.");
  }
  if (!["auto", "webgpu", "wasm"].includes(settings.backend)) throw new Error("Invalid engine.");
}
