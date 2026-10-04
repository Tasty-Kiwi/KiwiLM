import { test } from "node:test";
import assert from "node:assert/strict";
import { ngramIndices, attentionBias, contextPlan, mulberry32, selectToken, validateSettings } from "./runtime.js";
import { MODELS, getModel, engineFor, examplePrompts } from "./models.js";
import legacyManifest from "./legacy-manifest.json" with { type: "json" };

const manifest = { vocab_size: 32000, bigram_buckets: 16384, trigram_buckets: 16384 };
test("hashes preserve integer precision and cached predecessors", () => {
  const ids = [2, 31999, 17001, 0];
  const expectedBigram = ids.map((id, i) => Number((BigInt(ids[i - 1] ?? 0) * 1000003n
    + BigInt(id) * 9176n + 17n) % 16384n));
  const expectedTrigram = ids.map((id, i) => Number((BigInt(ids[i - 2] ?? 0) * 1000003n
    + BigInt(ids[i - 1] ?? 0) * 9176n + BigInt(id) * 131n + 29n) % 16384n));
  assert.deepEqual(Array.from(ngramIndices(ids, manifest).bigram), expectedBigram);
  assert.deepEqual(Array.from(ngramIndices(ids, manifest).trigram), expectedTrigram);
  assert.deepEqual(Array.from(ngramIndices(ids, manifest, 3).trigram), expectedTrigram.slice(3));
  assert.throws(() => ngramIndices([-1], manifest));
});
test("causal mask covers both prefill and cached decode", () => {
  assert.deepEqual(Array.from(attentionBias(3, 0)), [0, -1e9, -1e9, 0, 0, -1e9, 0, 0, 0]);
  assert.deepEqual(Array.from(attentionBias(1, 3)), [0, 0, 0, 0]);
});
test("rollover resets cache and n-gram boundaries", () => {
  assert.deepEqual(contextPlan([1, 2, 3], 0, 4), { window: [1, 2, 3], reset: true, start: 0, past: 0 });
  assert.deepEqual(contextPlan([1, 2, 3, 4], 3, 4), { window: [1, 2, 3, 4], reset: false, start: 3, past: 3 });
  assert.deepEqual(contextPlan([1, 2, 3, 4, 5], 4, 4), { window: [2, 3, 4, 5], reset: true, start: 0, past: 0 });
});
test("seeded sampling, stable large logits, top-k and greedy", () => {
  const logits = Float32Array.from([10001, 10003, 10002]);
  const sample = () => Array.from({ length: 10 }, () => selectToken(logits, 0.8, 2, random));
  let random = mulberry32(42);
  const first = sample();
  random = mulberry32(42);
  assert.deepEqual(first, sample());
  assert.ok(first.every(id => id === 1 || id === 2));
  assert.equal(selectToken(logits, 0, 0, random), 1);
  assert.throws(() => selectToken([NaN], 0, 0, random));
});
test("reject control tokens, invalid seeds and unbounded requests", () => {
  const valid = { modelId: "kiwilm2", prompt: "Hello", tokens: 160, temperature: 0.8, topK: 40, seed: 42, backend: "auto" };
  validateSettings(valid);
  for (const invalid of [{ prompt: "[BOS]" }, { tokens: 257 }, { seed: -1 },
    { temperature: NaN }, { backend: "remote" }, { modelId: "unknown" }]) {
    assert.throws(() => validateSettings({ ...valid, ...invalid }));
  }
});
test("model selection isolates tokenizer dimensions and engine defaults", () => {
  assert.deepEqual(Object.keys(MODELS), ["kiwilm2", "x", "y-direct", "y-cpt"]);
  for (const id of Object.keys(MODELS)) {
    const model = getModel(id);
    assert.equal(model.contextLength, id === "kiwilm2" ? 512 : 256);
    assert.equal(model.vocabSize, id === "kiwilm2" ? 32000 : 8192);
    assert.equal(engineFor(id, "auto", true), id === "kiwilm2" ? "webgpu" : "wasm");
    assert.equal(engineFor(id, "wasm", true), "wasm");
    assert.equal(engineFor(id, "webgpu", true), "webgpu");
    assert.equal(examplePrompts(id).story.includes("Instruction:"), id !== "kiwilm2");
  }
  assert.throws(() => getModel("__proto__"));
});
test("all legacy bundles have independent pinned model checksums", () => {
  assert.equal(legacyManifest.source_space_revision.length, 40);
  const checksums = new Set();
  for (const id of ["x", "y-direct", "y-cpt"]) {
    const files = legacyManifest.models[id].files;
    assert.deepEqual(Object.keys(files), ["model.onnx", "tokenizer.json", "tokenizer_config.json"]);
    for (const details of Object.values(files)) {
      assert.match(details.sha256, /^[a-f0-9]{64}$/);
      assert.ok(details.bytes > 0);
    }
    checksums.add(files["model.onnx"].sha256);
  }
  assert.equal(checksums.size, 3);
});
