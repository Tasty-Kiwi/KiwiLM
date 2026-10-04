import { Tokenizer } from "@huggingface/tokenizers";
import * as ort from "onnxruntime-web/webgpu";
import { ngramIndices, attentionBias, contextPlan, mulberry32, selectToken, validateSettings } from "./runtime.js";
import { getModel, engineFor } from "./models.js";
import legacyManifest from "./legacy-manifest.json";

ort.env.wasm.numThreads = 1; // Works without cross-origin isolation on static Spaces.
const modelRoot = new URL("./models/kiwilm2/", import.meta.env.BASE_URL
  ? new URL(import.meta.env.BASE_URL, self.location.origin) : self.location.href);
const modelsRoot = new URL("../", modelRoot);
let loaded = null;
let active = false;
let cancelled = false;
let controller = null;
const notify = (status, extra = {}) => self.postMessage({ type: "progress", status, ...extra });

async function verifiedFile(name, details, root = modelRoot) {
  const url = new URL(name, root).href;
  let storage = null;
  try { storage = await caches.open("kiwilm2-browser-v1"); } catch { /* Cache may be disabled. */ }
  let response = storage ? await storage.match(url) : null;
  let buffer;
  if (response) {
    buffer = await response.arrayBuffer();
  } else {
    response = await fetch(url, { signal: controller.signal });
    if (!response.ok) throw new Error(`Download failed: ${name} (HTTP ${response.status}).`);
    const reader = response.body.getReader();
    const chunks = [];
    let total = 0;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      chunks.push(value);
      total += value.length;
      notify(`Downloading ${name}: ${(total / 1e6).toFixed(1)} / ${(details.bytes / 1e6).toFixed(1)} MB…`);
    }
    buffer = new ArrayBuffer(total);
    const bytes = new Uint8Array(buffer);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  }
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", buffer)),
    b => b.toString(16).padStart(2, "0")).join("");
  if (buffer.byteLength !== details.bytes || digest !== details.sha256) {
    if (storage) await storage.delete(url);
    throw new Error(`Integrity check failed for ${name}; retry to download again.`);
  }
  if (storage) {
    try { await storage.put(url, new Response(buffer)); } catch { /* Quota is not fatal. */ }
  }
  return buffer;
}

async function loadModel(backend, modelId) {
  if (loaded && loaded.requested === backend && loaded.modelId === modelId) return loaded;
  if (loaded) { await loaded.session.release(); loaded = null; }
  if (modelId !== "kiwilm2") return loadLegacyModel(backend, modelId);
  const response = await fetch(new URL("browser-manifest.json", modelRoot), { cache: "no-store", signal: controller.signal });
  if (!response.ok) throw new Error("Browser model manifest unavailable.");
  const manifest = await response.json();
  if (manifest.schema_version !== 1 || manifest.architecture !== "kiwilm2") throw new Error("Unsupported browser bundle.");
  const tokenizerBytes = await verifiedFile("tokenizer.json", manifest.files["tokenizer.json"]);
  const tokenizer = new Tokenizer(JSON.parse(new TextDecoder().decode(tokenizerBytes)), {
    bos_token: "[BOS]", eos_token: "[EOS]", pad_token: "[PAD]", unk_token: "[UNK]",
  });
  const weights = await verifiedFile("model.onnx", manifest.files["model.onnx"]);
  const reference = JSON.parse(new TextDecoder().decode(
    await verifiedFile("parity.json", manifest.files["parity.json"])));
  if (cancelled) throw new DOMException("Stopped", "AbortError");
  let engine = engineFor(modelId, backend, !!navigator.gpu);
  let fallback = "";
  let session;
  notify(`Initializing ${engine} and checking cached/full parity…`);
  try {
    session = await ort.InferenceSession.create(weights, { executionProviders: [engine], graphOptimizationLevel: "all" });
    await checkSession(session, manifest, reference, tokenizer);
  } catch (error) {
    if (session) await session.release();
    if (backend !== "auto" || engine !== "webgpu") throw error;
    fallback = `WebGPU unavailable (${error.message}); using WebAssembly. `;
    engine = "wasm";
    notify(`${fallback}Initializing…`);
    session = await ort.InferenceSession.create(weights, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    try { await checkSession(session, manifest, reference, tokenizer); } catch (error) { await session.release(); throw error; }
  }
  loaded = { session, tokenizer, manifest, requested: backend, engine, fallback, modelId };
  return loaded;
}

async function legacyStep(session, manifest, tokens) {
  const context = tokens.slice(-manifest.context_length);
  const result = await session.run({ input_ids: new ort.Tensor("int64",
    BigInt64Array.from(context, BigInt), [1, context.length]) });
  if (result.logits.data.length !== manifest.vocab_size || !result.logits.data.every(Number.isFinite)) {
    throw new Error("Invalid legacy model logits.");
  }
  return { logits: result.logits.data, state: null, past: 0 };
}

async function loadLegacyModel(backend, modelId) {
  const definition = getModel(modelId);
  const files = legacyManifest.models[modelId].files;
  const root = new URL(`${modelId}/`, modelsRoot);
  const tokenizerBytes = await verifiedFile("tokenizer.json", files["tokenizer.json"], root);
  const configBytes = await verifiedFile("tokenizer_config.json", files["tokenizer_config.json"], root);
  const tokenizer = new Tokenizer(JSON.parse(new TextDecoder().decode(tokenizerBytes)),
    JSON.parse(new TextDecoder().decode(configBytes)));
  const weights = await verifiedFile("model.onnx", files["model.onnx"], root);
  if (cancelled) throw new DOMException("Stopped", "AbortError");
  const manifest = { context_length: definition.contextLength, vocab_size: definition.vocabSize,
    special_tokens: { "[BOS]": 2, "[EOS]": 3 } };
  let engine = engineFor(modelId, backend, !!navigator.gpu);
  let fallback = "";
  let session;
  notify(`Initializing ${definition.label} · ${engine}…`);
  try {
    session = await ort.InferenceSession.create(weights, { executionProviders: [engine], graphOptimizationLevel: "all" });
    await legacyStep(session, manifest, [2, 100, 200]);
  } catch (error) {
    if (session) await session.release();
    if (backend !== "auto" || engine !== "webgpu") throw error;
    fallback = `WebGPU unavailable (${error.message}); using WebAssembly. `;
    engine = "wasm";
    session = await ort.InferenceSession.create(weights, { executionProviders: [engine], graphOptimizationLevel: "all" });
    try { await legacyStep(session, manifest, [2, 100, 200]); }
    catch (error) { await session.release(); throw error; }
  }
  loaded = { session, tokenizer, manifest, requested: backend, engine, fallback, modelId };
  return loaded;
}

function initialCaches(manifest) {
  return Object.fromEntries(manifest.cache_specs.map(spec => [spec.name,
    new ort.Tensor("float32", new Float32Array(spec.shape.reduce((a, b) => a * b, 1)), spec.shape)]));
}

async function step(session, manifest, tokens, state, past) {
  const plan = contextPlan(tokens, past, manifest.context_length);
  if (plan.reset) state = initialCaches(manifest);
  const ids = plan.window.slice(plan.start);
  const { bigram, trigram } = ngramIndices(plan.window, manifest, plan.start);
  const positions = Int32Array.from(ids, (_, i) => plan.past + i);
  const feeds = { ...state,
    input_ids: new ort.Tensor("int32", Int32Array.from(ids), [1, ids.length]),
    bigram_ids: new ort.Tensor("int32", bigram, [1, ids.length]),
    trigram_ids: new ort.Tensor("int32", trigram, [1, ids.length]),
    position_ids: new ort.Tensor("int32", positions, [ids.length]),
    attention_bias: new ort.Tensor("float32", attentionBias(ids.length, plan.past), [1, 1, ids.length, plan.past + ids.length]),
  };
  const result = await session.run(feeds);
  const nextState = Object.fromEntries(manifest.cache_specs.map(spec => [spec.name,
    result[spec.name.replace("cache_", "present_")]]));
  if (result.logits.data.length !== manifest.vocab_size) throw new Error("Unexpected logits shape.");
  return { logits: result.logits.data, state: nextState, past: plan.window.length };
}

async function checkSession(session, manifest, reference, tokenizer) {
  for (const fixture of reference.tokenizer) {
    const actual = tokenizer.encode(fixture.text, { add_special_tokens: false }).ids;
    if (JSON.stringify(actual) !== JSON.stringify(fixture.ids)) throw new Error("Tokenizer parity failed.");
  }
  const first = await step(session, manifest, [2, 100], null, 0);
  const cached = await step(session, manifest, [2, 100, 200], first.state, first.past);
  const full = await step(session, manifest, [2, 100, 200], null, 0);
  let error = 0;
  const compare = (actual, expected) => {
    for (let i = 0; i < actual.length; i++) {
      if (!Number.isFinite(actual[i])) throw new Error("Non-finite startup logits.");
      error = Math.max(error, Math.abs(actual[i] - expected[i]));
    }
  };
  compare(first.logits, reference.logits[0].values);
  compare(cached.logits, reference.logits[1].values);
  compare(full.logits, cached.logits);
  notify("Checking full-window and rollover parity against PyTorch…");
  const window = reference.logits[2].ids;
  const atLimit = await step(session, manifest, window, null, 0);
  compare(atLimit.logits, reference.logits[2].values);
  const rollover = await step(session, manifest, [...window, 201], atLimit.state, atLimit.past);
  compare(rollover.logits, reference.logits[3].values);
  if (error > 0.002) throw new Error(`PyTorch/cached/rollover parity failed: ${error.toFixed(6)}.`);
  notify(`Tokenizer, PyTorch and cached rollover parity passed (max error ${error.toFixed(6)}).`);
}

async function generate(settings) {
  validateSettings(settings);
  const { session, tokenizer, manifest, engine, fallback } = await loadModel(settings.backend, settings.modelId);
  if (cancelled) {
    self.postMessage({ type: "done", status: "Stopped." });
    return;
  }
  const ids = [manifest.special_tokens["[BOS]"], ...tokenizer.encode(settings.prompt, { add_special_tokens: false }).ids];
  const promptLength = ids.length;
  const random = mulberry32(settings.seed);
  let state = null;
  let past = 0;
  let text = "";
  let count = 0;
  const started = performance.now();
  const truncated = ids.length > manifest.context_length
    ? `Prompt cropped to the newest ${manifest.context_length} tokens. ` : "";
  for (; count < settings.tokens && !cancelled; count++) {
    const result = settings.modelId === "kiwilm2"
      ? await step(session, manifest, ids, state, past)
      : await legacyStep(session, manifest, ids);
    state = result.state;
    past = result.past;
    if (cancelled) break;
    const token = selectToken(result.logits, settings.temperature, settings.topK, random);
    ids.push(token);
    text = tokenizer.decode(ids.slice(promptLength), { skip_special_tokens: true });
    notify(`${fallback}${truncated}${engine} · ${count + 1}/${settings.tokens} tokens · ${((count + 1) * 1000 / (performance.now() - started)).toFixed(1)} tok/s`, { text });
    if (token === manifest.special_tokens["[EOS]"]) { count++; break; }
    await new Promise(resolve => setTimeout(resolve, 0)); // Let Stop messages through.
  }
  self.postMessage({ type: "done", text, status: `${cancelled ? "Stopped" : "Complete"} · ${engine} · ${count} tokens · ${(count * 1000 / (performance.now() - started)).toFixed(1)} tok/s. ${fallback}` });
}

self.addEventListener("message", async ({ data }) => {
  if (data.type === "stop") { cancelled = true; controller?.abort(); return; }
  if (data.type !== "generate" || active) return;
  active = true;
  cancelled = false;
  controller = new AbortController();
  try { await generate(data.settings); }
  catch (error) {
    self.postMessage({ type: error.name === "AbortError" ? "done" : "error",
      status: error.name === "AbortError" ? "Stopped." : `Generation failed: ${error.message}` });
  } finally {
    active = false;
    controller = null;
  }
});
