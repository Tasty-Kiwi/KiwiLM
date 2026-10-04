import "./style.css";
import { getModel, examplePrompts } from "./models.js";

const modelSelector = document.querySelector("#model");
let prompts = examplePrompts(modelSelector.value);
const form = document.querySelector("#controls");
const output = document.querySelector("#output");
const status = document.querySelector("#status");
const stop = document.querySelector("#stop");
const prompt = document.querySelector("#prompt");
const example = document.querySelector("#example");
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
let busy = false;

function setBusy(value) {
  busy = value;
  for (const control of form.elements) control.disabled = value;
  stop.disabled = !value;
}
function setStatus(message, kind = "") {
  status.textContent = message;
  status.dataset.kind = kind;
}
prompt.value = prompts.story;
function updateModel() {
  const model = getModel(modelSelector.value);
  document.querySelector("#model-description").textContent = model.description;
  document.querySelector("#model-caveat").textContent = model.caveat;
  document.querySelector("#model-card").href = model.card;
}
updateModel();
modelSelector.addEventListener("change", () => {
  const wasExample = Object.values(prompts).includes(prompt.value);
  prompts = examplePrompts(modelSelector.value);
  if (wasExample) prompt.value = prompts[example.value];
  updateModel();
  output.textContent = "Your continuation will appear here.";
  setStatus("Ready. Only the selected model downloads when you generate.");
});
example.addEventListener("change", () => { prompt.value = prompts[example.value]; });
for (const id of ["tokens", "temperature", "top-k"]) {
  document.querySelector(`#${id}`).addEventListener("input", event => {
    document.querySelector(`#${id}-value`).textContent = id === "temperature"
      ? Number(event.target.value).toFixed(2) : event.target.value;
  });
}
form.addEventListener("submit", event => {
  event.preventDefault();
  if (busy) return;
  setBusy(true);
  output.textContent = "";
  setStatus("Preparing browser-local inference…", "working");
  worker.postMessage({ type: "generate", settings: {
    prompt: prompt.value, modelId: modelSelector.value,
    backend: document.querySelector("#backend").value,
    tokens: Number(document.querySelector("#tokens").value),
    temperature: Number(document.querySelector("#temperature").value),
    topK: Number(document.querySelector("#top-k").value),
    seed: Number(document.querySelector("#seed").value),
  } });
});
stop.addEventListener("click", () => {
  worker.postMessage({ type: "stop" });
  stop.disabled = true;
  setStatus("Stopping after the current browser operation…", "working");
});
worker.addEventListener("message", ({ data }) => {
  if (data.text !== undefined) output.textContent = data.text;
  setStatus(data.status, data.type === "error" ? "error" : data.type === "done" ? "success" : "working");
  if (["done", "error"].includes(data.type)) setBusy(false);
});
worker.addEventListener("error", event => {
  setBusy(false);
  setStatus(`Browser worker failed: ${event.message}. Reload to retry.`, "error");
});
document.querySelector("#copy").addEventListener("click", async event => {
  try {
    await navigator.clipboard.writeText(output.textContent);
    event.target.textContent = "Copied";
    setTimeout(() => { event.target.textContent = "Copy"; }, 1200);
  } catch { setStatus("Clipboard unavailable; select the output to copy it.", "error"); }
});
