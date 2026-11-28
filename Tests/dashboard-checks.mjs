// Model-free checks of the dashboard page's own logic: its script runs in a stub page with a fake API, no browser and
// no model. Usage: node Tests/dashboard-checks.mjs page.html (the page as `GET /dashboard` serves it).
import { readFileSync } from "node:fs";
import vm from "node:vm";

const script = /<script>\n([\s\S]*?)<\/script>/.exec(readFileSync(process.argv[2], "utf8"))[1];
const noop = () => {};
// Anything the art and layout code asks of a canvas or a style answers harmlessly.
const deep = new Proxy(function () {}, { get: (t, k) => k === Symbol.toPrimitive ? () => 1 : k === "then" ? undefined : deep, apply: () => deep, set: () => true });
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, { id, hidden: false, disabled: false, value: "", textContent: "", innerHTML: "", max: "", dataset: {},
    style: { setProperty: noop }, classList: { toggle: noop, add: noop, remove: noop, contains: () => false },
    listeners: {}, addEventListener(type, fn) { this.listeners[type] = fn; }, click() { return this.listeners.click && this.listeners.click({ target: this }); }, open: false, showModal() { this.open = true; }, close() { this.open = false; }, querySelector: () => null, querySelectorAll: () => [], focus: noop,
    attrs: {}, setAttribute(k, v) { this.attrs[k] = String(v); }, toggleAttribute(k, on) { if (on) this.attrs[k] = ""; else delete this.attrs[k]; },
    getAttribute(k) { return this.attrs[k] ?? null; }, hasAttribute(k) { return k in this.attrs; }, removeAttribute(k) { delete this.attrs[k]; }, getContext: () => deep, getBoundingClientRect: () => ({ width: 375, height: 200, left: 0, top: 0 }), clientWidth: 375, clientHeight: 200 });
  return elements.get(id);
}
// The API: each path's answer, and every request the page sent.
const routes = {}, sent = [];
const answer = (body, status = 200) => ({ ok: status < 300, status, json: async () => body });
let unreachable = false;
async function fetch(path, options = {}) {
  if (unreachable) throw new TypeError("Failed to fetch");
  sent.push({ path, method: options.method || "GET", body: options.body, headers: options.headers || {}, signal: options.signal });
  const key = Object.keys(routes).find((k) => path === k || path.startsWith(k + "?"));
  if (key && typeof routes[key] === "function") return routes[key](path, options);
  return key ? answer(...[].concat(routes[key])) : answer({ error: { message: "not here" } }, 404);
}
let confirmAnswer = false, confirmed = [];
const context = vm.createContext({
  document: { getElementById: element, documentElement: {}, querySelectorAll: () => [], addEventListener: noop, hidden: false, body: {}, activeElement: null },
  MutationObserver: class { observe() {} },
  window: { addEventListener: noop, devicePixelRatio: 2, innerWidth: 375 }, location: { hash: "", pathname: "/dashboard", hostname: "127.0.0.1", origin: "http://127.0.0.1", reloads: 0, reload() { this.reloads += 1; } },
  history: { replaceState: noop }, sessionStorage: { getItem: () => null, setItem: noop }, localStorage: { items: {}, getItem(k) { return this.items[k] ?? null; }, setItem(k, v) { this.items[k] = String(v); } }, matchMedia: () => ({ matches: false, addEventListener: noop }),
  getComputedStyle: () => ({ getPropertyValue: () => "0, 0, 0" }), requestAnimationFrame: noop, cancelAnimationFrame: noop, setTimeout: () => 0, clearTimeout: noop,
  setInterval: () => 0, clearInterval: noop, performance: { now: () => 0 }, URLSearchParams, CSS: { escape: (s) => s }, fetch,
  AbortSignal: { timeout: (milliseconds) => ({ milliseconds }) },
  confirm: (text) => { confirmed.push(text); return confirmAnswer; },
});
const run = (code) => vm.runInContext(code, context);
const settle = async () => { for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r)); };
let failures = 0;
const check = (ok, what) => { console.log((ok ? "ok    " : "FAIL  ") + what); if (!ok) failures += 1; };

// GLM runs (its engine has no plan previews); Qwen was added earlier. `v1/jobs` still reports that finished add.
routes["v1/status"] = { status: "ready", version: "2.3.1-dev", model: { name: "GLM-5.3-Flash", fingerprint: "glm" }, requests: { recent: [] },
  settings: { context: { value: "32k", source: "config" }, quality: { value: "exact", source: "config" } }, quality_levels: ["exact", "dynamic", "fast"], mac: { gpu: "Apple M2 Pro", memory_bytes: 34359738368 } };
routes["v1/models"] = { data: [{ id: "glm-5.3-flash", origami: { model: "glm-5.3-flash", setting: "glm-5.3-flash", default: true } },
  { id: "qwen3.8-flash-next-4bit", origami: { model: "qwen3.8-flash-next-4bit", setting: "qwen3.8-flash-next-4bit" } }] };
routes["v1/models/found"] = { models: [], unreadable: [] };
const addJob = (state, started) => ({ job: { kind: "add", title: "Adding qwen3.8-flash-next-4bit", state, stage: state === "running" ? "Starting" : "done",
  lines: [], seconds: 0, started, ...(state === "done" ? { percent: 1, added: "qwen3.8-flash-next-4bit" } : {}) } });
routes["v1/jobs"] = addJob("done", 1000);
run(script);
await settle(); await run("refresh()"); await settle();
check(run("modelChoice") === "glm-5.3-flash", "a job that ended before the page loaded does not choose its model");
check(/it is in the Model menu\.$/.test(element("job-stage").textContent), "it still shows its result line, without claiming a choice");
run('$("quality").value = "1"; dialMoved()');
check(run("JSON.stringify(changes())") === '{"quality":"dynamic"}', "Apply after a quality move sends only the quality");
// The quality dial offers only the running model's levels; a level it does not offer shows as the one it runs.
check(run("LEVELS.map((l) => l[0]).join()") === "exact,dynamic,fast" && element("quality").max === "2"
  && element("st-floor").innerHTML === '<option value="dynamic">Dynamic</option><option value="fast">Fast</option>',
  "GLM's quality dial and comparison floor offer exact, dynamic and fast");
check(run('levelAt("fastest") + "" + levelAt("dynamic") + levelAt("exact") + levelAt("custom")') === "2100",
  "fastest, which GLM does not offer, shows as fast, as GLM runs it");

// An add this page sees run and finish chooses the model, and Apply names the switch.
routes["v1/jobs"] = addJob("running", 2000); await run("refresh()"); await settle();
routes["v1/jobs"] = addJob("done", 2000); await run("refresh()"); await settle();
check(run("modelChoice") === "qwen3.8-flash-next-4bit", "an add seen running chooses its model when it ends");
check(run("effects()[0][0]") === "Model: glm-5.3-flash → qwen3.8-flash-next-4bit", "the change summary names the model switch");
await run("refresh()"); await settle();
run('modelChoice = "glm-5.3-flash"'); await run("refresh()"); await settle();
check(run("modelChoice") === "glm-5.3-flash", "a later poll of the same finished job does not choose it again");

// A model a job chose goes with a dial change only once the user confirms both.
run('modelChoice = "qwen3.8-flash-next-4bit"; $("quality").value = "1"; dialMoved()');
const posted = () => sent.filter((r) => r.path === "v1/config").length;
confirmAnswer = false; await run("apply()"); await settle();
check(confirmed.length === 1 && /Model: glm-5.3-flash → qwen3.8-flash-next-4bit/.test(confirmed[0]) && posted() === 0, "a declined confirm saves nothing");
confirmAnswer = true; routes["v1/config"] = { saved: true }; routes["v1/reload"] = {}; await run("apply()"); await settle();
const body = JSON.parse((sent.find((r) => r.path === "v1/config") || {}).body || "{}");
check(body.model === "qwen3.8-flash-next-4bit" && body.quality === "dynamic", "a confirmed Apply saves both");
// A model the user picks from the menu is their own choice: no extra question.
confirmed = []; run('modelByJob = false; $("quality").value = "1"'); await run("apply()"); await settle();
const last = JSON.parse(sent.filter((r) => r.path === "v1/config").pop().body);
check(confirmed.length === 0 && last.model === "qwen3.8-flash-next-4bit" && last.quality === "dynamic", "a model picked from the menu needs no extra confirm");

// The folders section and its browser: a folder chosen for expert copies joins the stores, saved comma-separated.
routes["v1/fs"] = { roots: [{ name: "Home", path: "/Users/me", drive: "internal drive", free_bytes: 5e11 }],
  settings: { stores: [{ path: "/Volumes/A/Origami", drive: "A", free_bytes: 1e12 }, { path: "/Users/me/o/store", drive: "internal drive", free_bytes: 5e11 }], hot_copy_dir: null, data_dir: "/Users/me/o", models_dirs: [],
              context_dir: { path: "/Users/me/pc", drive: "internal drive", free_bytes: 5e11 } } };
await run("loadFolders()"); await settle();
check(!/Expert folders|Hot copy/.test(element("folders").innerHTML), "a GLM or DeepSeek model has no expert folders, and the hot copy is not a folder here");
run('kind = "qwen"; ready = true; showFolders()');
const folders = () => element("folders").innerHTML.replace(/<wbr>/g, "");
check(/Expert folders/.test(folders()) && /\/Volumes\/A\/Origami/.test(folders()) && !/\/Users\/me\/o\/store/.test(folders()) && /\/<wbr>Volumes\/<wbr>A/.test(element("folders").innerHTML),
  "Qwen's 4-bit build lists its expert folders, never the data folder's own; paths wrap after their slashes");
run('noModel = true; showFolders()');
check(!/Expert folders/.test(element("folders").innerHTML), "with no model running the expert folders are hidden");
run('noModel = false; kind = "fixed"; xFrom = { sources: [] }; showFolders()');
check(/Expert folders/.test(element("folders").innerHTML), "a one-width Qwen build lists them too: its pack and copies go there");
run('xFrom = { no_hot_copy: "x", sources: [] }; showFolders()');
check(!/Expert folders/.test(element("folders").innerHTML), "a GGUF, whose experts stay in their file, does not");
run('kind = "qwen"; xFrom = null; showFolders()');
run('browse = async () => ({ path: "/Users/me/o/store", drive: "internal drive", free_bytes: 5e11 })');
const configs = () => sent.filter((r) => r.path === "v1/config").length, configsBefore = configs();
await run('folderAction("store-add", "", null)'); await settle();
check(configs() === configsBefore && element("folders-note").textContent === "Origami already uses its data folder.", "the data folder cannot be added as an expert folder");
run('browse = async () => ({ path: "/Volumes/B/Origami", drive: "B", free_bytes: 2e12 })');
await run('folderAction("store-add", "", null)'); await settle();
check(JSON.parse(JSON.parse(sent.filter((r) => r.path === "v1/config").pop().body).stores).join(",") === "/Volumes/A/Origami,/Volumes/B/Origami", "a folder chosen for expert copies joins the stores");

// Saved prompts: every entry is listed and opens in the dialog; delete asks in the page first; an edited system prompt is
// saved as a new one that replaces it.
const fire = (id) => element(id).listeners.click({ target: element(id) });
const used = "2026-10-31T00:00:00Z", system = { id: "8-def", kind: "system", name: "coder", tokens: 600, bytes: 5e7, pinned: true, last_used: used };
routes["v1/status"].prompt_cache = { entries: [{ id: "4-abc", kind: "prompt", tokens: 900, bytes: 1e8, pinned: false, last_used: used }, system] };
await run("refresh()"); await settle();
check(/data-open="8-def"[\s\S]*data-open="4-abc"/.test(element("prompts").innerHTML), "every saved prompt is listed, system prompts first");
routes["v1/prompts/8-def"] = { ...system, text: "<|im_start|>system\nYou code.", system_text: "You code." };
await run('pvOpen("8-def")'); await settle();
check(element("pv").open && element("pv-text").textContent.endsWith("You code.") && /600 tokens/.test(element("pv-meta").textContent)
  && !element("pv-edit-start").hidden && element("pv-name").value === "coder", "an entry opens with its text, tokens and name");
fire("pv-delete");
check(!element("pv-confirm").hidden && !sent.some((r) => r.method === "DELETE"), "delete asks in the page before anything is removed");
await fire("pv-delete-yes"); await settle();
check(sent.some((r) => r.method === "DELETE" && r.path === "v1/prompts/8-def") && !element("pv").open, "a confirmed delete removes it and closes the dialog");
await run('pvOpen("8-def")'); await settle();
fire("pv-edit-start");
check(!element("pv-edit").hidden && element("pv-edit").value === "You code." && !/ends early/.test(element("pv-note").textContent),
  "editing starts from the system prompt's own text");
element("pv-edit").value = "You code well.";
routes["v1/prompts"] = { id: "9-123", message: "saved the system prompt as coder" };
routes["v1/prompts/9-123"] = { ...system, id: "9-123", text: "You code well.", system_text: "You code well." };
await fire("pv-edit-start"); await settle();
const saved = JSON.parse(sent.filter((r) => r.path === "v1/prompts" && r.method === "POST").pop().body);
check(saved.text === "You code well." && saved.name === "coder" && saved.replace === "8-def" && run("pvOpened") === "9-123",
  "an edited system prompt is saved as a new one replacing the old, and opens");
routes["v1/prompts/7-aaa"] = { ...system, id: "7-aaa", text: "<|im_start|>system\nYou co", system_text: "You co", system_whole: false };
await run('pvOpen("7-aaa")'); await settle(); fire("pv-edit-start");
check(/longer system prompt/.test(element("pv-meta").textContent) && /ends early/.test(element("pv-note").textContent),
  "a save holding only the prompt's start says so before it is edited");

// The model menu shows each model's state from the front: a drive asleep, unplugged or not allowed is named, nothing is
// called missing that the front did not find missing, and an entry still being checked stays listed.
routes["v1/models"] = { data: [{ id: "glm-5.3-flash", origami: { model: "glm-5.3-flash", setting: "glm-5.3-flash", default: true, availability: "available", drive: "internal drive" } },
  { id: "deepseek", origami: { model: "deepseek", setting: "deepseek", availability: "not_connected", drive: "ExternalSSD" } },
  { id: "far", origami: { model: "far", setting: "far", availability: "slow", drive: "ExternalSSD" } },
  { id: "far-fast", origami: { model: "far", setting: "far" } }] };
routes["v1/models/found"] = { models: [{ path: "/Volumes/B/Models/x.gguf", name: "x", bytes: 4e9, verdict: "runs", drive: "B", state: "no_permission" }], unreadable: [], searching: false };
await run("loadModels()"); await settle(); await run("loadFound(false)"); await settle();
const menu = element("model").innerHTML;
check(/deepseek · ExternalSSD not connected/.test(menu) && /far · ExternalSSD slow to answer/.test(menu) && !/not found/.test(menu)
  && /x · needs permission to read B · 4 GB/.test(menu) && (menu.match(/<option/g) || []).length === 5, "the model menu labels each model's state and keeps every entry");
routes["v1/models"].data[1].origami.availability = "no_permission"; await run("loadModels()"); await settle();
check(run("noticeKind") === "found" && element("notice-title").textContent === "Origami can't read ExternalSSD yet"
  && /deepseek · needs permission to read ExternalSSD/.test(element("model").innerHTML), "an added model on a drive macOS refuses asks for the permission");
element("model").innerHTML += "<!-- open -->"; await run("loadModels()"); await settle();
check(/<!-- open -->/.test(element("model").innerHTML), "a poll that brings nothing new leaves the menu being chosen from alone");

// Connect an app: Pi's block names the listed model, maps every Pi thinking level and takes the context the listing gives.
routes["v1/models"] = { data: [{ id: "origami", context_length: 32768, capabilities: ["completion", "tools", "thinking"], reasoning_efforts: ["none", "low", "high", "max"], origami: { model: "glm-5.3-flash" } }] };
await run("loadConnect()"); await settle();
const pi = JSON.parse(element("c-pi").textContent).providers.origami, piModel = pi.models[0];
check(pi.baseUrl === "http://127.0.0.1/v1" && element("c-model").textContent === "origami" && piModel.reasoning && piModel.contextWindow === 32768
  && Object.keys(piModel.thinkingLevelMap).join() === "off,low,high,max" && piModel.thinkingLevelMap.off === "none"
  && piModel.input.join() === "text" && pi.compat.thinkingFormat === "openai", "Connect an app gives Pi the listed model's distinct thinking levels");
// The Thinking dial offers the same levels, each once, and is not disabled.
check(element("reasoning").max === "3" && /None/.test(element("t-reasoning").innerHTML) && /Max/.test(element("t-reasoning").innerHTML)
  && !/Medium/.test(element("t-reasoning").innerHTML) && !element("reasoning").disabled, "the Thinking dial lists the model's distinct levels");

// The page follows Origami's state: a reload turns the control that started it and holds the dials and the model menu
// until the new model answers (a ready answer from the old one before it went down does not count); a failed start
// frees them again; a running job turns its own button and holds the other; no answer at all disables the page.
const busy = (id) => element(id).attrs["aria-busy"] === "true";
const glm = routes["v1/status"], loading = { status: "loading", version: "2.3.1-dev", model: "glm-5.3-flash" };
routes["v1/models"] = { data: [{ id: "glm-5.3-flash", origami: { model: "glm-5.3-flash", setting: "glm-5.3-flash", default: true } }] };
routes["v1/jobs"] = { job: null }; routes["v1/config"] = { saved: true }; routes["v1/reload"] = {};
// The Applies above end their reload first.
routes["v1/status"] = loading; await run("refresh()"); await settle();
routes["v1/status"] = glm; await run("refresh()"); await settle();
run('$("quality").value = "1"; dialMoved()');
check(!element("apply").disabled && !busy("apply") && !element("dials").disabled, "a moved dial can be applied");
await run("apply()"); await settle();
check(element("apply").disabled && busy("apply") && element("dials").disabled && element("model").disabled && element("chip-text").textContent === "Loading",
  "Apply turns until the reload is through, the dials and the model menu wait");
await run("refresh()"); await settle();
check(busy("apply") && run("reloading"), "the old model still answering ready does not end the reload");
// While the reload waits for the running reply, the model that still serves answers and says so: a banner, not a
// loading screen; saves paused for lack of disk say why with their numbers.
const paused = "conversation saves are paused: 4.0 GB free, and a save keeps 10.0 GB free plus its image (6.8 GB); free disk space or lower the context";
routes["v1/status"] = { ...glm, pending: "reload after the current reply to apply the saved settings", prompt_cache: { entries: [], saves_paused: paused },
  requests: { recent: [], current: { phase: "decode", completion_tokens: 40, output_pieces: 40 } } };
await run("refresh()"); await settle();
check(!element("banner").hidden && element("banner-text").textContent.startsWith("The model will reload after the current reply to apply the saved settings. It keeps serving")
  && element("banner-text").textContent.endsWith("Conversation saves are paused: 4.0 GB free, and a save keeps 10.0 GB free plus its image (6.8 GB); free disk space or lower the context.")
  && element("chip-text").textContent === "Working" && busy("apply") && run("reloading"), "a pending reload shows as a banner over the serving model, which the page follows");
routes["v1/status"] = loading; await run("refresh()"); await settle();
routes["v1/status"] = glm; await run("refresh()"); await settle();
check(!run("reloading") && !busy("apply") && !element("dials").disabled && !element("model").disabled && element("chip-text").textContent === "Ready"
  && element("banner").hidden, "the new model running ends the reload, frees the controls and drops the banner");
routes["v1/status"] = { status: "failed", failure: "No model at /x", version: "2.3.1-dev", model: "glm-5.3-flash" };
await run("refresh()"); await settle();
check(/data-mload="glm-5\.3-flash"[^>]*>Load glm-5\.3-flash/.test(element("first").innerHTML) && element("picker").hidden && element("use-model").hidden,
  "with no model loaded, each added model has its Load button");
check(run("tuneBlock(false, false, false)") === "No model is running", "with no model found, tuning says no model is running");
run('modelChoice = "qwen"; showStart()');
await run("apply()"); await settle();
check(element("use-model").disabled && busy("use-model") && element("model").disabled, "Start with this model turns and cannot be pressed again");
await run("refresh()"); await settle();
check(busy("use-model"), "the failed answer from before the reload does not end it");
routes["v1/status"] = loading; await run("refresh()"); await settle();
routes["v1/status"] = { status: "failed", failure: "glm-5.3-flash did not start: out of memory", version: "2.3.1-dev", model: "glm-5.3-flash" };
await run("refresh()"); await settle();
check(!run("reloading") && !busy("use-model") && !element("model").disabled && element("chip-text").textContent === "Failed" && "data-failed" in element("use-model").attrs,
  "a failed start frees the controls and marks the one that started it");
check(element("power").innerHTML === "Try again" && element("power").attrs["aria-label"] === "Try again" && element("power").dataset.starts === "true",
  "after a failed start the header button offers Try again, not Stop");
check(run("tuneBlock(false, false, false)") === "The model failed to start: Try again first", "after a failed start, tuning points to the header's Try again");
routes["v1/status"] = glm; routes["v1/jobs"] = { job: { kind: "add", title: "Adding glm-q2", state: "running", stage: "Converting", seconds: 30, started: 4000, percent: 0.2, lines: [] } };
await run("refresh()"); await settle();
check(busy("add-go") && element("add-go").disabled && element("job").attrs["aria-busy"] === "true", "a running conversion turns Add");
check(element("live-text").textContent === "Adding glm-q2 · 20%", "the header names it in the same poll");
routes["v1/jobs"] = { job: { ...routes["v1/jobs"].job, state: "failed", error: "No room." } };
await run("refresh()"); await settle();
check(!busy("add-go") && element("job").attrs["aria-busy"] === "false", "the job's end frees them");
check(!/Adding/.test(element("live-text").textContent), "and the header drops it in the same poll, not one later");
// A Qwen child that fails while it loads answers failed with its model: that ends the reload too, Try again works, and
// another model can be started, as when the front answers failed.
routes["v1/jobs"] = { job: null }; await run("refresh()"); await settle();
run('$("quality").value = "1"; dialMoved()'); await run("apply()"); await settle();
routes["v1/status"] = loading; await run("refresh()"); await settle();
routes["v1/status"] = { status: "failed", error: "out of memory", version: "2.3.1-dev", model: { name: "Qwen", fingerprint: "q" }, requests: { recent: [] } };
await run("refresh()"); await settle();
check(!run("reloading") && !element("model").disabled && !element("notice-retry").disabled && element("chip-text").textContent === "Failed"
  && "data-failed" in element("apply").attrs && run("noModel") && /out of memory/.test(element("notice-text").textContent),
  "a model that fails while loading ends the reload, frees Try again and offers another model");
routes["v1/status"] = glm;
unreachable = true; await run("refresh()"); await settle();
check(element("page").disabled && element("chip-text").textContent === "Not reachable", "with no answer from Origami the page is disabled");
unreachable = false; await run("refresh()"); await settle();
check(!element("page").disabled, "and enabled again once it answers");

// Storage: each drive a bar with its items, names escaped; Remove asks first; a move sends the item and the folder chosen,
// after saying the running model stops while it moves, and turns its own button.
routes["v1/storage"] = { running: "glm-5.3-flash", drives: [{ name: "B", path: "/Volumes/B", total_bytes: 1e12, free_bytes: 4e11, roles: { model: 1e11, staging: 1e9 },
  items: [{ path: "/Volumes/B/m<x>", name: "m<x>", role: "model", model: "m<x>", bytes: 1e11, in_use: true, unused: false, actions: ["move"] },
    { path: "/Volumes/B/.origami-moving-q", name: ".origami-moving-q", role: "staging", bytes: 1e9, in_use: false, unused: true, actions: ["remove"], note: "an interrupted move" },
    { path: "/Volumes/B/s/q-4bit", name: "q-4bit", role: "pack", model: "qwen", bytes: 0, in_use: false, unused: false, actions: ["move"] },
    { path: "/Volumes/B/s/0123456789abcdef", name: "0123456789abcdef", role: "support", model: "glm-5.3-flash", bytes: 0, in_use: true, unused: false, actions: [] }] }] };
await run("loadStorage()"); await settle();
const map = element("storage").innerHTML;
check(/m&lt;x&gt;/.test(map) && !/m<x>/.test(map) && /data-role="model" style="width:10.00%"/.test(map) && /in use/.test(map) && !/Use as store|Hot copy here|Rebuild elsewhere/.test(map)
  && /aria-label="B: models 100.0 GB, other Origami files 1.0 GB, other files 499.0 GB, 400.0 GB free"/.test(map), "storage shows each drive's bar and items, names escaped");
check(/<h3>qwen<\/h3>/.test(map) && /All experts, 4-bit/.test(map) && /<h3>glm-5.3-flash<\/h3>/.test(map), "files group under their owning model");
const jobPosts = () => sent.filter((r) => r.path === "v1/jobs" && r.method === "POST").length;
const before = jobPosts();
routes["v1/storage/plan"] = { kind: "remove", path: "/Volumes/B/.origami-moving-q", name: "unfinished copy", bytes: 1e9, parts: [], revision: null };
const removal = run('storageAction("remove", "/Volumes/B/.origami-moving-q", $("storage-refresh"))'); await settle();
check(element("storage-confirm").open && /permanently/.test(element("storage-confirm-body").innerHTML), "deletion has a concrete preview");
run('finishStorageConfirm(false)'); await removal;
check(jobPosts() === before, "cancelling the preview deletes nothing");
routes["v1/storage/plan"] = { kind: "remove", path: "/Volumes/B/.origami-moving-q", name: "Qwen", role: "download", bytes: 1.2e11, parts: [], revision: null };
const downloadRemoval = run('storageAction("remove", "/Volumes/B/.origami-moving-q", $("storage-refresh"))'); await settle();
check(/Delete 120\.0 GB permanently\. This is a whole downloaded model: to use it again, all 120\.0 GB must be downloaded again\./.test(element("storage-confirm-body").innerHTML),
  "deleting an unlisted download names its size and that it would have to be downloaded again");
run('finishStorageConfirm(false)'); await downloadRemoval;
// A model whose experts are read from a file outside its folder: Delete names each file that goes, and the one another
// model reads too, which stays.
const modelDelete = run('confirmStorage(' + JSON.stringify({ kind: "delete", path: "/Volumes/B/m", name: "m", role: "model", bytes: 1e9, allocated_bytes: 6e9,
  parts: ["/Volumes/B/m"], deletes: ["/Volumes/B/m", "/Volumes/C/mine.gguf"], keeps: [{ path: "/Volumes/C/theirs.gguf", why: "reader reads it too" }] }) + ')');
check(/Delete 6\.0 GB permanently\.<\/p><p>Deletes \/<wbr>Volumes\/<wbr>B\/<wbr>m, \/<wbr>Volumes\/<wbr>C\/<wbr>mine\.gguf<\/p><p>Keeps \/<wbr>Volumes\/<wbr>C\/<wbr>theirs\.gguf: reader reads it too<\/p>/
  .test(element("storage-confirm-body").innerHTML), "deleting a model whose experts are elsewhere names the files that go and the one that stays: " + element("storage-confirm-body").innerHTML);
run('finishStorageConfirm(false)'); await modelDelete;
// Saved prompts get rows per model: what the running build reads, and apart what nothing can read any more, with its size
// and age, in `origami cache`'s words; the storage total says so above 1 GB. Remove unusable… lists what goes; a folder
// with pinned or named conversations needs its own tick.
const storageBefore = routes["v1/storage"], day = Date.now() / 1000 - 14 * 86400;
routes["v1/storage"] = { ...storageBefore, drives: [{ ...storageBefore.drives[0], items: [...storageBefore.drives[0].items,
    { path: "/c/0123456789ab-32768", name: "0123456789ab-32768", role: "prompts", bytes: 1e9, in_use: false, unused: true, actions: ["remove"], note: "saved prompts no build can read" }] }],
  saved_prompts: [{ path: "/c/aaaaaaaaaaaa-32768/1111111111111111", label: "aaaaaaaaaaaa-32768/1111111111111111", model: "qwen", bytes: 2e9, conversations: 3, kept: ["chat-1"], unreadable: false, state: "in_use", used: day + 86400 * 13 },
    { path: "/c/aaaaaaaaaaaa-32768/2222222222222222", label: "aaaaaaaaaaaa-32768/2222222222222222", model: "qwen", bytes: 6e9, conversations: 2, kept: ["plan"], unreadable: false, state: "unusable", reason: "made by another Origami build (version 3.1.3)", used: day },
    { path: "/c/0123456789ab-32768/3333333333333333", label: "0123456789ab-32768/3333333333333333", model: "qwen", bytes: 1e9, conversations: 0, kept: [], unreadable: false, state: "unusable", reason: "a model file no model uses", used: day - 86400 }] };
await run("loadStorage()"); await settle();
const prompts = element("storage").innerHTML;
check(/<h3>Saved prompts<\/h3>/.test(prompts) && /qwen <span class="tag">in use<\/span><\/div><div class="file-location">2\.0 GB · 3 conversations · 1 pinned or named/.test(prompts)
  && /qwen <span class="tag">unusable: made by another Origami build \(version 3\.1\.3\); a model file no model uses<\/span><\/div><div class="file-location">7\.0 GB · 2 folders · last used 14 d ago/.test(prompts)
  && !/Saved prompts <span class="tag">not used/.test(prompts) && /data-sact="prune"/.test(prompts), "saved prompts: the running build's and the unusable ones per model, in the words origami cache uses");
// Rows of no registered model name the one they were made for (its last name, or its fingerprint), and an unused one what
// deleting it loses; saved prompts of a replaced model are listed under its last name.
const named = { ...routes["v1/storage"], drives: [{ ...routes["v1/storage"].drives[0], items: [
    { path: "/s/older-dddddddddddd/hot", name: "hot", role: "hot copy", owner: "older (replaced by newer 2026-11-26)", bytes: 6.8e10, unused: true, actions: ["remove"], state: "unused" },
    { path: "/s/gone-ffffffffffff/hot", name: "hot", role: "hot copy", owner: "from a model no longer registered · ffffffffffff", bytes: 1e9, unused: true, actions: ["remove"], state: "unused" }] }],
  saved_prompts: [{ path: "/c/dddddddddddd-32768/1111111111111111", label: "dddddddddddd-32768/1111111111111111", model: "dddddddddddd", owner: "older (replaced by newer 2026-11-26)",
    bytes: 6.5e9, conversations: 6, kept: [], unreadable: false, state: "usable" }] };
const unnamed = routes["v1/storage"]; routes["v1/storage"] = named; await run("loadStorage()"); await settle();
const owned = element("storage").innerHTML;
check(/Hot copy · older \(replaced by newer 2026-11-26\) <span class="tag">unused<\/span><\/div><div class="file-location">[^<]* · 68\.0 GB · deleting it loses nothing a model reads now/.test(owned)
  && /Hot copy · from a model no longer registered · ffffffffffff <span class="tag">unused/.test(owned)
  && /older \(replaced by newer 2026-11-26\) <span class="tag">usable by this build<\/span><\/div><div class="file-location">6\.5 GB · 6 conversations/.test(owned) && !/>dddddddddddd </.test(owned),
  "rows of a replaced model name it, marked replaced, an older one its fingerprint, and an unused row what deleting it loses");
routes["v1/storage"] = unnamed; await run("loadStorage()"); await settle();
check(/ on disk · 7\.0 GB of saved prompts no build can read$/.test(element("storage-total").textContent), "the storage total names unusable saved prompts above 1 GB: " + element("storage-total").textContent);
routes["v1/storage"] = { ...routes["v1/storage"], saved_prompts: routes["v1/storage"].saved_prompts.slice(0, 1) };
await run("loadStorage()"); await settle();
check(/ on disk$/.test(element("storage-total").textContent) && !/data-sact="prune"/.test(element("storage").innerHTML), "with none unusable, no total line and no Remove unusable…");
routes["v1/storage/plan"] = { kind: "prune", name: "unusable saved prompts", bytes: 7e9, revision: null, folders: [
  { path: "/c/aaaaaaaaaaaa-32768/2222222222222222", label: "aaaaaaaaaaaa-32768/2222222222222222", bytes: 6e9, kept: ["plan"], unreadable: false, reason: "made by another Origami build" },
  { path: "/c/0123456789ab-32768/3333333333333333", label: "0123456789ab-32768/3333333333333333", bytes: 1e9, kept: [], unreadable: false, reason: "a model file no model uses" }] };
const pruneJobs = () => sent.filter((r) => r.path === "v1/jobs" && r.method === "POST").map((r) => JSON.parse(r.body)).filter((b) => b.kind === "prune");
for (const tick of [false, true]) {
  element("storage-confirm-body").querySelectorAll = () => tick ? [{ dataset: { yes: "/c/aaaaaaaaaaaa-32768/2222222222222222" } }] : [];
  const pruning = run('storageAction("prune", "", $("storage-refresh"))'); await settle();
  const body = element("storage-confirm-body").innerHTML;
  check(element("storage-confirm").open && /Delete 7\.0 GB permanently/.test(body) && /0123456789ab-32768\/3333333333333333 · 1\.0 GB · a model file no model uses/.test(body)
    && /<input type="checkbox" data-yes="\/c\/aaaaaaaaaaaa-32768\/2222222222222222"> Also remove aaaaaaaaaaaa-32768\/2222222222222222 · 6\.0 GB: it holds plan/.test(body),
    "Remove unusable… lists each folder with its size, and a folder with named conversations has its own tick");
  run('finishStorageConfirm(true)'); await pruning; await settle();
  check(JSON.stringify(pruneJobs().pop().yes) === JSON.stringify(tick ? ["/c/aaaaaaaaaaaa-32768/2222222222222222"] : []), "the job carries the ticked folder only " + (tick ? "when ticked" : "(none ticked)"));
}
element("storage-confirm-body").querySelectorAll = () => [];
const jobsBefore = routes["v1/jobs"];
routes["v1/jobs"] = { job: { kind: "prune", title: "Removing unusable saved prompts", state: "done", stage: "done", seconds: 1, started: 7000, lines: [],
  message: "Removed 1 of 2 folders of unusable saved prompts (1.0 GB). Kept aaaaaaaaaaaa-32768/2222222222222222  6.0 GB: it holds plan; ticking it removes it too." } };
await run("refresh()"); await settle();
check(/^Removed 1 of 2 folders of unusable saved prompts \(1\.0 GB\)\. Kept .*it holds plan; ticking it removes it too\.$/.test(element("job-stage").textContent), "the outcome says what was removed and what was kept, and why: " + element("job-stage").textContent);
routes["v1/jobs"] = jobsBefore; routes["v1/storage"] = storageBefore; await run("refresh()"); await run("loadStorage()"); await settle();
run('browse = async () => ({ path: "/Volumes/C/Models", drive: "C", free_bytes: 2e12 })');
routes["v1/storage/plan"] = { kind: "move", path: "/Volumes/B/m<x>", to: "/Volumes/C/Models", name: "m<x>", in_use: true, copy_bytes: 1e11, required_bytes: 1.1e11, free_bytes: 2e12 };
routes["v1/jobs"] = { job: { kind: "move", title: "Moving m<x>", state: "running", stage: "Copying and checking", seconds: 3, started: 5000, percent: 0.1, lines: [] } };
const moving = run('storageAction("move", "/Volumes/B/m<x>", $("storage-refresh"))'); await settle();
check(/pauses during/.test(element("storage-confirm-body").innerHTML), "move preview explains the loaded model pause");
run('finishStorageConfirm(true)'); await moving; await settle();
const moved = JSON.parse(sent.filter((r) => r.path === "v1/jobs" && r.method === "POST").pop().body);
check(moved.kind === "move" && moved.path === "/Volumes/B/m<x>" && moved.to === "/Volumes/C/Models", "Move sends only the previewed item and destination");
// While that move runs, the folder and drive choices and each saved prompt's Open wait too; a folder being saved keeps turning.
const fake = (busy) => ({ disabled: false, dataset: {}, attrs: {}, setAttribute(k, v) { this.attrs[k] = String(v); }, toggleAttribute: noop, hasAttribute: () => false, getAttribute: (k) => k === "aria-busy" && busy ? "true" : null });
const folderButtons = [fake(false), fake(true)], openButton = fake(false);
element("folders").querySelectorAll = () => folderButtons; element("prompts").querySelectorAll = () => [openButton];
await run("refresh()"); await settle();
check(folderButtons[0].disabled && !folderButtons[1].disabled && openButton.disabled, "folder and saved-prompt buttons wait while a job runs");

// The Experts card, from stub statuses: each place the experts are read from, in the card's words; the read split; the
// hot copy's actions only where they apply; Change… sends a drive and a size to v1/hot-copy.
routes["v1/jobs"] = { job: null }; await run("refresh()"); await settle();
const gguf = { role: "all", drive: "ExternalSSD", path: "/Volumes/ExternalSSD/m/UD-IQ1_S", bytes: 233e9, state: "reading", read_share: 0, detail: "As released (IQ2_XXS + IQ1_S). No smaller width exists." };
const choices = [{ drive: "internal drive", path: "/Users/me/o", free_bytes: 129.7e9, status: "goes here", sizes: ["auto", 2e10, 4e10, 8e10, "off"] },
  { drive: "ExternalSSD", path: "/Volumes/ExternalSSD", free_bytes: 359e9, status: "holds all experts" }, { drive: "Fast", path: "/Volumes/Fast", free_bytes: 5e11, status: "has room", sizes: ["auto", 2e10, "off"] }];
const hotRow = (extra) => ({ role: "hot", drive: "internal drive", path: "/Users/me/o/family/k/hot", bytes: 0, budget_bytes: 4e10, records: 0, topup_bytes: 0, progress: null,
  has_counts: false, can_build: false, read_share: 0, custom_folder: null, choices, detail: null, ...extra });
const deepseek = (hot, split = { reads: 0, memory: 0, hot_copy: 0, model_drive: 0 }) => ({ status: "ready", version: "2.3.3-dev", model: { name: "DeepSeek-V4.1-Flash", fingerprint: "ds" },
  requests: { recent: [] }, settings: { context: { value: "128k" }, quality: { value: "exact" }, hot_copy: { value: "(automatic)" } },
  experts_from: { released: "IQ2_XXS + IQ1_S", split, sources: [{ ...gguf, read_share: split.model_drive }, hot] } });
const card = (s) => { run("noModel = false; status = " + JSON.stringify(s) + "; expertsCard(status); gate()"); return { table: element("x-sources").innerHTML, said: element("x-said").textContent, acts: element("x-acts").innerHTML, lede: element("x-lede").innerHTML }; };
let c = card(deepseek(hotRow({ state: "learning", detail: "Uses up to 40 GB of 129.7 GB free. Replies are unchanged." })));
check(!element("experts").hidden && element("x-model").textContent === "DeepSeek-V4.1-Flash · released as IQ2_XXS + IQ1_S" && /ExternalSSD.*all experts.*233 GB.*Reading.*As released \(IQ2_XXS \+ IQ1_S\)\. No smaller width exists\./.test(c.table)
  && /This Mac.*hot copy.*up to 40 GB.*Learning: built after the first reply.*Uses up to 40 GB of 129\.7 GB free\. Replies are unchanged\./.test(c.table)
  && c.said === "No replies yet." && /Change…/.test(c.acts) && /data-xact="build" data-why="Available after the first reply" data-blocked/.test(c.acts) && /Hot copy: the experts replies use most/.test(c.lede),
  "DeepSeek learning: both drives with their roles, Build now waits for a first reply");
c = card(deepseek(hotRow({ state: "building", has_counts: true, budget_bytes: 4e10, detail: "Pauses while a reply runs.", progress: { done_bytes: 12e9, total_bytes: 4e10, seconds_left: 40 } })));
check(/12 of 40 GB/.test(c.table) && /data-m="live"[^>]*><\/i>Building · about 40 s left/.test(c.table) && /aria-valuenow="30"/.test(c.table) && /Pauses while a reply runs\./.test(c.table) && !/data-xact="build"/.test(c.acts),
  "DeepSeek building: progress, time left and the pause rule");
c = card(deepseek(hotRow({ state: "paused", has_counts: true, detail: "Pauses while a reply runs.", progress: { done_bytes: 12e9, total_bytes: 4e10, seconds_left: null } })));
check(/Paused · the model is reading/.test(c.table), "a paused build says the model is reading (a reply or the warm-up)");
c = card(deepseek(hotRow({ state: "ready", bytes: 4e10, records: 5212, has_counts: true, can_build: true, read_share: 0.11 }), { reads: 12410, memory: 0.71, hot_copy: 0.11, model_drive: 0.18 }));
check(/40 GB.*data-m="ink"[^>]*><\/i>Ready · 5,212 experts.*11%/.test(c.table) && /18%/.test(c.table) && !/data-xact="build"/.test(c.acts)
  && c.said === "Of 12,410 expert reads so far: 71% in memory, 11% from the hot copy on This Mac, 18% from ExternalSSD." && element("rs-mem").style.width === "71%" && element("rs-hot").style.width === "11%",
  "DeepSeek ready: experts held, each source's share of reads and the split in words");
c = card(deepseek(hotRow({ state: "ready", bytes: 4e10, records: 5212, has_counts: true, can_build: true, topup_bytes: 2.1e9, detail: "2.1 GB more to add after this reply" })));
check(/2\.1 GB more to add after this reply/.test(c.table) && /data-xact="build">Build now/.test(c.acts), "a top-up waiting offers Build now");
c = card(deepseek(hotRow({ state: "not_possible", detail: "The model and the copy would share This Mac's drive: no gain.", choices: choices.map((x) => x.drive === "internal drive" ? { ...x, status: "holds all experts", sizes: undefined } : x) })));
check(/Not possible here/.test(c.table) && /data-m="none"/.test(c.table) && /data-xact="put:\/Volumes\/Fast">Put it on Fast/.test(c.acts) && !/Put it on ExternalSSD/.test(c.acts), "not possible on the model's drive offers each other drive with room");
c = card(deepseek(hotRow({ state: "not_needed", path: "/Users/me/o/family/k/hot", detail: "This Mac keeps every expert in memory once it is read, so a copy would only help the first minutes after a start." })));
check(/Not needed/.test(c.table) && /keeps every expert in memory/.test(c.table) && !/family\/k\/hot/.test(c.table), "a Mac that keeps every expert in memory: the copy is not needed, and no folder is shown for it");
c = card(deepseek(hotRow({ state: "off", detail: "Turned off in the settings." })));
check(/Off.*Turned off in the settings\./.test(c.table) && /data-xact="on">Turn on/.test(c.acts), "off offers Turn on");
c = card(deepseek(hotRow({ state: "not_allowed", detail: "macOS has not let Origami use Fast yet; reading from the model's drive only.", drive: "Fast" })));
check(/Not allowed.*href="#notice"/.test(c.table), "not allowed links the permission notice");
// Turn on, Build now and Change… post to v1/hot-copy; Use this sends only what changed and says where the copy goes.
routes["v1/hot-copy"] = { hot_copy: {} };
const hotPosts = () => sent.filter((r) => r.path === "v1/hot-copy").map((r) => JSON.parse(r.body));
const click = (act) => element("x-acts").listeners.click({ target: { closest: () => ({ dataset: { xact: act }, setAttribute: noop }) } });
card(deepseek(hotRow({ state: "off", detail: "Turned off in the settings." }))); await click("on"); await settle();
check(JSON.stringify(hotPosts().pop()) === '{"size":"auto"}' && /Saved\. The hot copy is on; it is built after the first reply\./.test(element("experts-note").textContent), "Turn on saves the automatic size");
card(deepseek(hotRow({ state: "learning", has_counts: true, can_build: true }))); await click("build"); await settle();
check(JSON.stringify(hotPosts().pop()) === '{"build":true}', "Build now asks for a build");
card(deepseek(hotRow({ state: "learning", has_counts: true, can_build: true }))); click("change");
check(!element("x-change").hidden && /holds all experts/.test(element("x-drives").innerHTML) && /value="\/Volumes\/ExternalSSD" disabled/.test(element("x-drives").innerHTML)
  && /<option value="auto" selected>Automatic \(40 GB\)<\/option><option value="40GB">40 GB<\/option><option value="off">Off/.test(element("x-size").innerHTML), "Change… lists the drives with their words, and Automatic, a size or Off");
await element("x-drives").listeners.change({ target: { value: "/Volumes/Fast" } });
check(/value="\/Volumes\/Fast" checked/.test(element("x-drives").innerHTML) && !/80 GB/.test(element("x-size").innerHTML), "choosing a drive offers only its sizes");
element("x-size").listeners.change({ target: { value: "20GB" } }); await element("x-use").listeners.click(); await settle();
check(JSON.stringify(hotPosts().pop()) === '{"drive":"/Volumes/Fast","size":"20GB"}' && element("experts-note").textContent === "Saved. Building the hot copy on Fast now." && run("xOpen") === false,
  "Use this sends the drive and the size, and says where it builds");
// Each size offered says the share of counted expert uses a copy of it would hold, and why the automatic size is what it is.
card(deepseek(hotRow({ state: "ready", bytes: 8e10, records: 4253, has_counts: true, can_build: true, budget_bytes: 8e10, size_limit: "free_space",
  coverage: [{ bytes: 2e10, share: 0.41 }, { bytes: 4e10, share: 0.58 }, { bytes: 8e10, share: 0.79 }], size_note: "Automatic size: limited by free space on This Mac (20 GB is kept free)." }))); click("change");
check(/Automatic \(80 GB · 79% of uses\)/.test(element("x-size").innerHTML) && /<option value="80GB">80 GB · 79% of uses/.test(element("x-size").innerHTML)
  && element("x-size-note").textContent === "A copy of 80 GB would cover 79% of expert uses. Automatic size: limited by free space on This Mac (20 GB is kept free).",
  "Change… shows each size's coverage and why the automatic size is what it is");
click("change");
// The size slider over the experts' popularity: ten bars of 5 GB in the copy's order, This Mac with room for 30 GB, the
// copy now 20 GB. The bars inside follow the slider, the two lines with them, and nothing is sent before Use this.
{
  const uses = [1000, 800, 600, 400, 200, 100, 50, 20, 10, 0], bars = uses.map((u, i) => [10, 5e9, u, Math.round(u / 10) + 5, Math.max(0, Math.round(u / 10) - 5)]);
  const pop = (extra) => ({ bars, counted_uses: 3180, experts: 100, bytes: 5e10, tokens: 5000, held_experts: 40, held_bytes: 2e10, ...extra });
  const mac = { ...choices[0], free_bytes: 3e10, room_bytes: 3e10 }, at = (extra) => hotRow({ state: "ready", bytes: 2e10, records: 40, has_counts: true, can_build: true, budget_bytes: 2e10,
    choices: [mac, choices[1]], popularity: pop(extra) });
  const inside = () => (element("x-plot").innerHTML.match(/class="in"/g) || []).length, posts = hotPosts().length;
  const slide = (value) => element("x-cap").listeners.input({ target: { value: String(value) } });
  card(deepseek(at())); click("change");
  check(!element("x-graph").hidden && (element("x-plot").innerHTML.match(/<rect/g) || []).length === 10 && inside() === 4 && /class="x-now" style="left:40\.00%"/.test(element("x-plot").innerHTML)
    && /class="x-noroom" style="left:60\.00%"><span>no room on This Mac/.test(element("x-plot").innerHTML) && element("x-cap").max === "50" && element("x-cap").value === "20"
    && element("x-yaxis").textContent === "uses per expert, log scale", "the graph shows the copy's bars solid, the copy now, and no room beyond the drive's maximum");
  slide(12);
  check(inside() === 2 && element("x-size").value === "12GB" && element("x-size-note").textContent === "A copy of 12 GB would cover " + Math.round(100 * (1800 + 600 * 0.4) / 3180) + "% of expert uses."
    && element("x-size-edge").textContent === "The last expert in is used once per 67 tokens; the next one left out, once per 77 tokens." && hotPosts().length === posts,
    "dragging the slider moves the bars inside and both lines, with no request");
  slide(45);
  check(inside() === 6 && element("x-cap").value === "30" && element("x-size").value === "30GB" && /No room for more on This Mac: it keeps 20 GB free\.$/.test(element("x-size-edge").textContent),
    "the slider stops at the drive's room and says why");
  element("x-unit").listeners.click({ target: { closest: () => ({ dataset: { unit: "pct" } }) } });
  check(element("x-cap").max === "100" && element("x-cap").value === "60" && element("x-unit-pct").attrs["aria-pressed"] === "true" && /^A copy of 30 GB \(60\.0% of the experts\) would cover/.test(element("x-size-note").textContent),
    "the unit switches to a share of the experts' bytes");
  slide(37);
  check(element("x-size").value === "18.5GB" && run("capValue(capBytes(37, 'pct', 5e10), 'pct', 5e10)") === 37 && run("capBytes(capValue(18e9, 'gb', 5e10), 'gb', 5e10)") === 18e9
    && run("copySize(capBytes(capValue(27.3e9, 'pct', 68e9), 'pct', 68e9))") === "27.3GB", "percent and GB convert both ways");
  await element("x-use").listeners.click(); await settle();
  check(JSON.stringify(hotPosts().pop()) === '{"size":"18.5GB"}', "Use this saves the size chosen in percent as bytes (GB), as hot_copy takes it");
  run("xUnit = 'gb'");
  card(deepseek(at({ tokens: 300 }))); click("change");
  check(element("x-size-edge").textContent === "Still learning: 300 tokens counted." && !element("x-graph").hidden, "too few tokens counted: still learning");
  click("change");
  card(deepseek(at({ counted_uses: 0, bars: [] }))); click("change");
  check(element("x-graph").hidden && element("x-unit").hidden && element("x-size-edge").textContent === "Still learning: no replies counted yet." && element("x-cap").max === "30",
    "with no uses counted there is no graph; the slider still sets a size in GB");
  run("xUnit = 'pct'"); slide(12);
  check(element("x-size").value === "12GB" && element("x-cap").value === "12", "with no uses counted the slider reads GB even after a switch to percent");
  run("xUnit = 'gb'; xSize = null");
  click("change");
}
// More drives: a copy place on Fast after This Mac's (its ranks start where This Mac's 20 GB end), T7 offered unused,
// another drive offered with its reason. Ticking T7 adds a place with the automatic size, its graph after both; saving
// posts every place and offers the reload; unticking a place asks whether to delete its files (default keep).
{
  const uses = [1000, 800, 600, 400, 200, 100, 50, 20, 10, 0], bars = uses.map((u, i) => [10, 5e9, u, Math.round(u / 10) + 5, Math.max(0, Math.round(u / 10) - 5)]);
  const pop = (extra) => ({ bars, counted_uses: 3180, experts: 100, bytes: 5e10, tokens: 5000, held_experts: 40, held_bytes: 2e10, ...extra });
  const mac = { ...choices[0], free_bytes: 3e10, room_bytes: 3e10 };
  const first = hotRow({ state: "ready", bytes: 2e10, records: 40, has_counts: true, can_build: true, budget_bytes: 2e10, choices: [mac, choices[1]], popularity: pop(), place: 0, rank: 0 });
  const fast = hotRow({ drive: "Fast", path: "/Volumes/Fast/Origami/hot-copy/family-k/hot", place: 1, rank: 1, place_folder: "/Volumes/Fast/Origami/hot-copy", state: "ready", bytes: 1e10,
    records: 20, cap: 1e10, budget_bytes: 1e10, free_bytes: 6e10, reserve_bytes: 2e10, speed_bytes_per_second: 2.9e9, duplicates: 3,
    choices: [{ drive: "Fast", path: "/Volumes/Fast", free_bytes: 6e10, room_bytes: 5e10, status: "goes here" }], popularity: pop({ before_bytes: 2e10, held_bytes: 1e10, held_experts: 20 }) });
  const unused = [{ drive: "T7", path: "/Volumes/T7", folder: "/Volumes/T7/Origami/hot-copy", free_bytes: 2e11, speed_bytes_per_second: null, why_not: null },
    { drive: "ExternalSSD", path: "/Volumes/ExternalSSD", folder: "/Volumes/ExternalSSD/Origami/hot-copy", free_bytes: 359e9, speed_bytes_per_second: 2.1e9, why_not: "the model's own drive" }];
  const places = (rows) => { const s = deepseek(first); s.experts_from.sources = s.experts_from.sources.concat(rows); s.experts_from.unused_drives = unused; return s; };
  const box = () => element("x-places").innerHTML, posts = () => hotPosts();
  card(places([fast])); click("change");
  check(!element("x-places-row").hidden && /data-place="0" checked aria-label="Keep a copy on Fast"/.test(box()) && /data-place="1" aria-label="Keep a copy on T7"/.test(box())
    && /data-place="2" disabled aria-describedby="x-pl-2-why"/.test(box()) && /ExternalSSD: the model&#39;s own drive/.test(box())
    && element("x-pl-0-facts").textContent === "2.9 GB/s · 60.0 GB free · Ready · 20 experts" && element("x-pl-1-facts").textContent === "speed not measured · 200.0 GB free · not used",
    "two places render: each drive with its speed, free space and state; a drive that cannot hold one says why");
  check(/class="x-on" style="left:0\.00%;width:40\.00%"><span>on This Mac/.test(element("x-pl-0-plot").innerHTML) && (element("x-pl-0-plot").innerHTML.match(/class="in"/g) || []).length === 2
    && /class="x-now" style="left:60\.00%"/.test(element("x-pl-0-plot").innerHTML) && element("x-pl-0-cap").value === "30" && element("x-pl-0-cap").max === "50"
    && /^10 GB on Fast adds \d+% of expert uses after the faster drives\. 3 experts here are also on a faster drive \(spare room\)\.$/.test(element("x-pl-0-note").textContent),
    "a place's graph starts after the faster place's cap, with its duplicates said");
  check(element("x-places-save").disabled && element("x-places-why").textContent === "Tick a drive or move a size to save.", "nothing to save says so");
  element("x-places").listeners.change({ target: { dataset: { place: "1" }, checked: true } });
  check(/class="x-on" style="left:0\.00%;width:40\.00%"/.test(element("x-pl-1-plot").innerHTML) && /class="x-on" style="left:40\.00%;width:20\.00%"><span>on Fast/.test(element("x-pl-1-plot").innerHTML)
    && /^Automatic: 20 GB on T7 adds/.test(element("x-pl-1-note").textContent) && !element("x-places-save").disabled,
    "ticking an unused drive adds a place with the automatic size, its graph after both faster drives (three places)");
  // The slider spans the ranks: at 35 GB it gives Fast the 15 GB after This Mac's 20.
  element("x-places").listeners.input({ target: { dataset: { pcap: "0", start: "20000000000" }, value: "35" } });
  routes["v1/hot-copy"] = { hot_copy: {}, hot_copies: [], restart: true };
  // The status after the save: Fast at 15 GB, T7 a place of its own.
  const t7 = hotRow({ drive: "T7", path: "/Volumes/T7/Origami/hot-copy/family-k/hot", place: 2, rank: 2, place_folder: "/Volumes/T7/Origami/hot-copy", state: "learning", cap: "auto",
    free_bytes: 2e11, reserve_bytes: 2e10, choices: [{ drive: "T7", path: "/Volumes/T7", free_bytes: 2e11, room_bytes: 1.8e11, status: "goes here" }] });
  const after = places([{ ...fast, cap: 1.5e10 }, t7]), statusBefore = routes["v1/status"];
  after.experts_from.unused_drives = unused.slice(1); routes["v1/status"] = after;
  await element("x-places-save").listeners.click(); await settle();
  routes["v1/status"] = statusBefore;
  check(JSON.stringify(posts().pop()) === '{"places":[{"folder":"/Volumes/Fast/Origami/hot-copy","size":"15GB"},{"drive":"/Volumes/T7","size":"auto"}]}'
    && element("experts-note").textContent === "Saved. Reload to apply: a new or removed drive is used once the model reloads.", "saving posts every place with its size and offers the reload");
  check(/data-place="0" checked aria-label="Keep a copy on Fast"/.test(box()) && /data-place="1" checked aria-label="Keep a copy on T7"/.test(box()) && element("x-pl-0-cap").value === "35"
    && element("x-places-save").disabled && element("x-places-why").textContent === "Tick a drive or move a size to save.",
    "after a save the drives are the refreshed status's: the saved size and the new drive stay, with nothing left to save");
  card(places([fast])); if (run("xOpen")) click("change"); click("change");
  const asked = confirmed.length;
  confirmAnswer = false;
  element("x-places").listeners.change({ target: { dataset: { place: "0" }, checked: false } });
  await element("x-places-save").listeners.click(); await settle();
  check(confirmed.length === asked + 1 && /^Also delete the hot copy's files on Fast\? Cancel keeps them\.$/.test(confirmed.at(-1)) && JSON.stringify(posts().pop()) === '{"places":[],"delete_removed":false}',
    "removing a drive asks about its files, keeping them by default");
  click("change");
  check(run("copyEdge({ tokens: 5000, bars: [[1, 1, 1, 50, 50], [1, 1, 1, 50, 50]] }, 1)") === "The last expert in is used once per 100 tokens; the next one left out, about as often.",
    "two edge figures that round alike are said once");
}
// Qwen MLX: its hot copy row has the family engines' shape (built while serving); Build now asks the running model. A
// further copy on another drive is a setup job; Qwen GGUF: no hot copy row, the lede says so.
const qwen = (hot, copyTo = []) => ({ status: "ready", model: { name: "qwen3.8-flash-next-4bit" }, settings: {}, experts_from: { released: "4-bit",
  split: { reads: 100, memory: 0.6, hot_copy: 0, model_drive: 0.4 }, copy_to: copyTo, sources: [
  { role: "all", drive: "internal drive", path: "/Users/me/q", bytes: 2e11, state: "reading", read_share: 0.4, detail: null }, hot] } });
c = card(qwen(hotRow({ path: "/Users/me/o/store/q-0123456789ab/hot", state: "learning", has_counts: true, can_build: true, detail: "Uses up to 40 GB of 129.7 GB free. Replies are unchanged." })));
check(/hot copy.*up to 40 GB.*Uses up to 40 GB of 129\.7 GB free\. Replies are unchanged\./.test(c.table) && /data-xact="build">Build now/.test(c.acts),
  "Qwen's hot copy row is the family engines' row");
routes["v1/hot-copy"] = { hot_copy: {} }; await click("build"); await settle();
check(JSON.stringify(hotPosts().pop()) === '{"build":true}' && element("experts-note").textContent === "Building the hot copy now.", "Qwen's Build now asks the running model");
c = card(qwen(hotRow({ state: "ready", bytes: 18e9, records: 4400, has_counts: true, can_build: true }), [{ drive: "Fast", path: "/Volumes/Fast", free_bytes: 5e11, status: "has room", bytes: 2e11 }]));
routes["v1/hot-copy"] = [{ saved: ["stores"], job: "hot-copy" }, 202]; confirmAnswer = true; await click("copy:/Volumes/Fast"); await settle(); confirmAnswer = false;
check(JSON.stringify(hotPosts().pop()) === '{"copy":"/Volumes/Fast"}'
  && element("experts-note").textContent === "Saved. Copying the experts; if that changes the copies, the model reloads after the current reply.",
  "a further Qwen copy runs setup and says up front when the model reloads");
// Its end: no "Apply and reload" (nothing to apply; the model reloads by itself), the front's outcome line, and the note announcing it goes.
const hotJob = (state) => ({ job: { kind: "hot-copy", title: "Copying the experts", state, stage: "copy", seconds: 40, started: 7000, percent: state === "done" ? 1 : 0.5, lines: [],
  ...(state === "done" ? { ended: 7040, message: "18.0 GB copied; the model reloads after the current reply to use it." } : {}) } });
routes["v1/jobs"] = hotJob("running"); await run("refresh()"); await settle();
check(/Saved\. Copying the experts/.test(element("experts-note").textContent), "the note stays while the copy runs");
routes["v1/jobs"] = hotJob("done"); await run("refresh()"); await settle();
check(element("job-stage").textContent === "18.0 GB copied; the model reloads after the current reply to use it." && element("experts-note").textContent === "",
  "a finished copy job says what it copied and that the model reloads, not Apply, and the card's note goes");
routes["v1/jobs"] = { job: null }; await run("refresh()"); await settle();
c = card({ status: "ready", model: { name: "q-gguf" }, settings: {}, experts_from: { released: "Q4_K", no_hot_copy: "This model reads all its experts from one file; it has no hot copy yet.",
  split: { reads: 0 }, sources: [{ role: "all", drive: "internal drive", path: "/Users/me/g", bytes: 2e10, state: "reading", read_share: null, detail: null }] } });
check(!/hot copy</.test(c.table) && /no hot copy yet/.test(c.lede) && c.acts === "", "Qwen GGUF has no hot copy row and says why");
// One row per drive: the one in use shows the folder the copy is in and is marked; choosing it again sends nothing.
const roomChoices = [{ drive: "internal drive", path: "/Users/me/o", free_bytes: 82e9, status: "goes here", folder: "/Users/me/o/store/family-k/hot", sizes: ["auto", 2e10, 4e10, "off"] },
  { drive: "ExternalSSD", path: "/Volumes/ExternalSSD", free_bytes: 359e9, status: "holds all experts" }];
run("xOpen = false"); routes["v1/hot-copy"] = { hot_copy: {} };
card(deepseek(hotRow({ state: "learning", choices: roomChoices }))); click("change");
const drivesHtml = element("x-drives").innerHTML;
check((drivesHtml.match(/<label/g) || []).length === 2 && !/Automatic/.test(drivesHtml) && /This Mac<span class="sub">\/<wbr>Users\/<wbr>me\/<wbr>o\/<wbr>store\/<wbr>family-k\/<wbr>hot/.test(drivesHtml)
  && /in use<\/span><\/label>/.test(drivesHtml), "Change… has one row per drive; the one in use shows the copy's own folder and says so");
let postsBefore = hotPosts().length; await element("x-use").listeners.click(); await settle();
check(hotPosts().length === postsBefore && /^Nothing changed: the hot copy stays where it is\.$/.test(element("experts-note").textContent), "choosing the drive the copy is on changes nothing");
// The header's chip says the hot copy's state and links to the card.
card(deepseek(hotRow({ state: "building", has_counts: true, progress: { done_bytes: 16e9, total_bytes: 4e10, seconds_left: 20 } })));
check(!element("x-chip").hidden && element("x-chip-text").innerHTML === '<span class="x-hot">building </span>40%' && element("x-chip").dataset.state === "busy" && "data-hot" in element("top").attrs, "the header's chip shows the build's progress");
card({ ...deepseek(hotRow({})), experts_from: { ...deepseek(hotRow({})).experts_from, sources: [gguf] } });
check(element("x-chip").hidden && !("data-hot" in element("top").attrs), "no hot copy row, no chip");
// Build now before the first reply says why beside it, not in a tooltip.
c = card(deepseek(hotRow({ state: "learning" })));
check(/data-xact="build" data-why="Available after the first reply" data-blocked aria-describedby="x-why">Build now<\/button><span class="x-why" id="x-why">after the first reply<\/span>/.test(c.acts) && !/title=/.test(c.acts), "Build now's reason shows beside it");
// Is it faster: the median of replies with and without the hot copy, only with three or more on each side.
const replies = (list) => list.map(([tps, hot], i) => ({ time: "2026-10-31T10:0" + i + ":00Z", decode_tps: tps, completion_tokens: 200, hot_copy: hot }));
const withReplies = (list) => ({ ...deepseek(hotRow({ state: "ready", bytes: 4e10, records: 5212 })), requests: { recent: replies(list) } });
card(withReplies([[3.0, false], [3.2, false], [3.1, false], [3.5, true], [3.9, true], [3.6, true], [3.7, true]]));
check(element("x-speed").textContent === "Writing with the hot copy: 3.7 tokens a second (median of 4 replies), 3.1 without it (3 replies).", "the writing speed with and without the hot copy, as medians");
card(withReplies([[3.0, false], [3.2, false], [3.5, true], [3.9, true], [3.6, true]]));
check(element("x-speed").textContent === "", "too few replies on one side: nothing is said");
// Leftover settings: named, with one Clear that removes them from the settings file.
routes["v1/config"] = { saved: ["expert_bits", "stores"], restart: true };
const unused = [{ key: "expert_bits", label: "expert_bits 3 (settings file)", why: "no longer an option" }, { key: "stores", label: "Expert folders /Volumes/G/o", why: "this model writes nothing there" }];
card({ ...deepseek(hotRow({ state: "learning" })), experts_from: { ...deepseek(hotRow({ state: "learning" })).experts_from, unused_settings: unused } });
check(/Not used by DeepSeek-V4\.1-Flash: expert_bits 3 \(settings file\) .*Expert folders \/Volumes\/G\/o.*data-xact="clear">Clear/.test(element("x-unused").innerHTML), "settings that do nothing for this model are named with a Clear");
await element("x-unused").listeners.click({ target: { closest: () => ({ dataset: { xact: "clear" }, toggleAttribute: noop }) } }); await settle();
const cleared = JSON.parse(sent.filter((r) => r.path === "v1/config").pop().body);
check(cleared.expert_bits === "" && cleared.stores === "" && element("x-unused").innerHTML === "" && /^Cleared/.test(element("experts-note").textContent), "Clear removes them, and the line goes");
// Qwen: the hot copy shows where it will go and can be put on any drive; a pack moves to a drive (a job); a further copy goes on a drive with room.
const qwenRows = { status: "ready", model: { name: "qwen3.8-flash-next-4bit" }, settings: {}, requests: { recent: [] }, experts_from: { released: "4-bit",
  split: { reads: 100, memory: 0.6, hot_copy: 0.1, model_drive: 0.3 }, copy_to: [{ drive: "Fast", path: "/Volumes/Fast", free_bytes: 5e11, status: "has room", bytes: 1e11 }], sources: [
  { role: "all", drive: "internal drive", path: "/Users/me/q", bytes: 2e11, state: "not_read", read_share: 0, detail: "The 4-bit copy on G is read instead." },
  { role: "pack", bits: 4, drive: "G", path: "/Volumes/G/Origami/store/q-1", bytes: 1e11, state: "reading", read_share: 0.3,
    moves: [{ drive: "G", path: "/Volumes/G", free_bytes: 3e11, status: "goes here", folder: "/Volumes/G/Origami/store/q-1" }, { drive: "Fast", path: "/Volumes/Fast", free_bytes: 5e11, status: "has room" }] },
  { role: "hot", drive: "internal drive", path: "/Users/me/o/store/q-1", bytes: 18e9, state: "not_built", can_build: true, read_share: 0.1, meaning: "Hot copy: one read each.",
    choices: [{ drive: "internal drive", path: "/Users/me/o", free_bytes: 8e10, status: "goes here", folder: "/Users/me/o/store/q-1", sizes: ["auto", "off"] }, { drive: "G", path: "/Volumes/G", free_bytes: 3e11, status: "has room", sizes: ["auto", "off"] }] }] } };
c = card(qwenRows);
check(/This Mac<span class="sub wide">\/<wbr>Users\/<wbr>me\/<wbr>o\/<wbr>store\/<wbr>q-1<\/span>.*hot copy/.test(c.table) && /all experts, as released/.test(c.table) && /all experts, 4-bit.*Reading.*30%/.test(c.table)
  && /Change…/.test(c.acts) && !/data-xact="move:/.test(c.acts) && /data-xact="copy:\/Volumes\/Fast">Copy to Fast/.test(c.acts)
  && /Hot copy: one read each\./.test(c.lede) && /data-l="reads"/.test(c.table), "Qwen: the copy's place before it is built, Change…, file moves in the library, and a further copy");
routes["v1/hot-copy"] = [{ saved: ["stores"], job: "hot-copy" }, 202]; confirmAnswer = true; confirmed = []; await click("copy:/Volumes/Fast"); await settle();
check(JSON.stringify(hotPosts().pop()) === '{"copy":"/Volumes/Fast"}' && /^Saved\. Copying the experts/.test(element("experts-note").textContent)
  && /another copy of the 4-bit experts to Fast \(up to 100 GB, about 17 min\), so replies read from both drives, then reloads/.test(confirmed[0]), "Copy to says what it writes and how long, then sends the drive");
// Without a pack: the button names the 4-bit copy, and a declined confirm sends nothing.
c = card({ ...qwenRows, experts_from: { ...qwenRows.experts_from, sources: qwenRows.experts_from.sources.filter((r) => r.role !== "pack") } });
confirmAnswer = false; confirmed = []; postsBefore = hotPosts().length; await click("copy:/Volumes/Fast"); await settle();
check(/>Copy all experts, 4-bit, to Fast</.test(c.acts) && /all experts, 4-bit, to Fast \(100 GB, about 17 min\), then reloads/.test(confirmed[0]) && hotPosts().length === postsBefore,
  "Copy all experts names the 4-bit copy; declined, nothing starts");
// A family's saved memory limit shows as saved.
routes["v1/plan"] = { settings: { context: 131072, quality: "exact", memory_limit: 25.77e9 }, quality: { allowed: true, levels: [{ name: "exact" }, { name: "dynamic" }, { name: "fast" }] }, context: { limit: 131072 },
  experts: { cache_experts: 900, total: 10000 }, memory: { cap_bytes: 25.77e9, most_bytes: 27e9, ram_bytes: 34359738368, split: [] }, drives: [] };
await run("loadSetup(status)"); await settle();
check(/^25\.8 GB of 32 GB$/.test(element("o-memory").textContent) && !("memory_limit" in run("changes()")), "saved memory reads in the decimal GB the setting takes and is not resent");
// A server that needs its key: the page asks for it (no command to run on an iPad), keeps it in this browser and sends it.
const ready = routes["v1/status"];
routes["v1/status"] = [{ error: { message: "This server needs its API key" } }, 401]; await run("refresh()"); await settle();
check(!element("keyask").hidden && element("page").hidden && element("chip-text").textContent === "Needs its key" && element("keyask-note").textContent === "", "a 401 asks for the key, in a field, with nothing behind it");
element("keyask-input").value = " 0123456789abcdef0123 "; routes["v1/status"] = ready;
await element("keyask-use").listeners.click(); await settle();
check(run('localStorage.getItem("origami-key")') === "0123456789abcdef0123" && sent.filter((r) => r.path === "v1/status").pop().headers.Authorization === "Bearer 0123456789abcdef0123"
  && element("keyask").hidden && !element("page").hidden, "a pasted key is kept in this browser and sent; the field goes and the page shows once it works");
routes["v1/status"] = [{ error: { message: "This server needs its API key" } }, 401]; await run("refresh()"); await settle();
check(/not accepted/.test(element("keyask-note").textContent) && element("keyask-input").attrs["aria-invalid"] === "true", "a key that stops working says so");
routes["v1/status"] = ready; await run("refresh()"); await settle();
// Access: other devices on the network after a restart, which a launchd service gets from here; then the address
// with the key, and its QR code.
const lanState = (o) => ({ listen: "local", listening: "local", port: 11439, urls: ["http://MINI.local:11439/dashboard", "http://192.168.1.24:11439/dashboard"],
  key_required: true, key_setting: "lan_key", restart: true, ...o });
routes["v1/lan"] = lanState(); await run("loadAccess()"); await settle();
check(element("acc-open-box").hidden && element("acc-key-row").hidden && element("acc-save").disabled, "this Mac only: no address, no key choice, nothing to save");
element("acc-lan").checked = true; element("acc-listen").listeners.change(); await settle();
check(!element("acc-key-row").hidden && element("acc-key").checked && run("JSON.stringify(accessChanges())") === '{"listen":"lan"}' && !element("acc-save").disabled, "choosing the network offers the key, on, and Save");
element("acc-key").checked = false;
check(run("JSON.stringify(accessChanges())") === '{"listen":"lan","lan_key":"off"}', "turning the key off is saved with it");
element("acc-key").checked = true;
routes["v1/config"] = { saved: ["listen"], restart: true }; routes["v1/lan"] = lanState({ listen: "lan" });
await element("acc-save").listeners.click(); await settle();
check(JSON.parse(sent.filter((r) => r.path === "v1/config").pop().body).listen === "lan" && !element("acc-restart").hidden && /once it restarts: Restart Origami now/.test(element("acc-note").textContent),
  "saved, it says it applies at a restart and offers one");
routes["v1/restart"] = [{ restarting: true }, 202]; routes["v1/lan"] = lanState({ listen: "lan", listening: "lan", key: "feedface0123456789", qr: "data:image/png;base64,AAAA" });
await element("acc-restart").listeners.click(); await settle();
check(sent.some((r) => r.path === "v1/restart" && r.method === "POST") && run("restarting") && /Restarting/.test(element("acc-note").textContent), "Restart asks Origami to restart and waits for it");
await run("refresh()"); await settle();
check(!run("restarting") && !element("acc-open-box").hidden && element("acc-url").textContent === "http://MINI.local:11439/dashboard#key=feedface0123456789"
  && element("acc-qr").src === "data:image/png;base64,AAAA" && !element("acc-qr").hidden && element("acc-keytext").textContent === "feedface0123456789"
  && /Origami is on this network now/.test(element("acc-note").textContent), "once back on the network: the address with the key, its QR code and the key");
await settle();
check(element("c-url").textContent === "http://MINI.local:11439/v1" && JSON.parse(element("c-pi").textContent).providers.origami.baseUrl === "http://MINI.local:11439/v1",
  "on the network, Connect gives apps the network address, not 127.0.0.1");
routes["v1/lan"] = lanState({ listen: "lan", listening: "lan" }); await run("loadAccess()"); await settle();
check(element("acc-qr").hidden && element("acc-key-show").hidden && /shows on the Mac itself/.test(element("acc-open-hint").textContent), "another device is not shown the key or the code");
// On that device (it got in with its key): the key it holds, to copy for its home-screen app; Tailscale's address named.
run('location.hostname = "192.168.1.24"'); routes["v1/lan"] = lanState({ listen: "lan", listening: "lan", urls: [...lanState().urls, "http://100.64.0.10:11439/dashboard"] });
await run("loadAccess()"); await settle();
check(!element("acc-key-show").hidden && element("acc-keytext").textContent === "0123456789abcdef0123" && element("acc-open-head").textContent === "This device"
  && /Add to Home Screen/.test(element("acc-open-hint").textContent) && /loses its connection/.test(element("acc-local-hint").textContent) && element("acc-qr").hidden
  && /100\.64\.0\.10:11439\/dashboard \(Tailscale or another VPN\)$/.test(element("acc-more").textContent) && !/192\.168\.1\.124:11439\/dashboard \(/.test(element("acc-more").textContent),
  "another device sees its own key to copy, its own title and Tailscale's address named");
run('location.hostname = "127.0.0.1"');
// A request macOS kept interrupting on the GPU says so in the recent requests.
run('recentTable([{ time: "2026-10-31T10:00:00Z", prompt_tokens: 23273, finish: "gpu_interrupted", mode: "exact" }])');
check(/<span class="tag">interrupted by macOS \(GPU busy\)<\/span>/.test(element("recent").innerHTML)
  && /<span class="sub ending"><span class="tag">interrupted by macOS \(GPU busy\)<\/span><\/span>/.test(element("recent").innerHTML), "a GPU interruption shows as interrupted by macOS in the recent requests, also under its prompt on narrow screens");
run('recentTable([{ time: "2026-10-31T10:00:00Z", prompt_tokens: 23273, finish: "stop", mode: "exact", gpu_retries: 1 }])');
check(/retried after macOS needed the GPU/.test(element("recent").innerHTML), "a request run again after an interruption says so");
// Sampling: the model's recommendation per mode, Custom dials for every mode, the override; Apply saves and reloads.
const mode = (t, p, pp) => ({ temperature: t, top_p: p, top_k: 20, min_p: 0, presence_penalty: pp });
const rec = { thinking: mode(0.6, 0.95, 0), non_thinking: mode(0.7, 0.8, 1.5), tools: mode(0.2, 0.9, 0), source: "generation_config.json" };
run(`showSampling(${JSON.stringify({ defaults: { sampling: rec, sampling_recommended: rec, sampling_override: false }, settings: { sampling: { value: "recommended", source: "default" } } })})`);
check(/<td>Creativity<\/td><td class="r">0\.6<\/td><td class="r">0\.7<\/td><td class="r">0\.2<\/td>/.test(element("sm-table").innerHTML)
  && /<td>Repetition brake<\/td><td class="r">Off<\/td><td class="r">1\.5<\/td><td class="r">Off<\/td>/.test(element("sm-table").innerHTML) && /^What its generation_config\.json names/.test(element("sm-source").textContent)
  && element("sm-recommended").checked && element("sm-dials").disabled && element("sm-apply").disabled && element("sm-reset").disabled, "the recommendation shows per mode, its dials still, nothing to apply");
check(element("o-sm-temperature").textContent === "0.7" && element("o-sm-presence_penalty").textContent === "1.5" && element("o-sm-min_p").textContent === "None", "the dials show its values in plain words");
element("sm-custom").checked = true; element("sm-choice").listeners.change();
element("sm-temperature").value = "1.2"; element("sm-temperature").listeners.input();
check(!element("sm-dials").disabled && !element("sm-apply").disabled && !element("sm-reset").disabled
  && run("JSON.stringify(samplingChanges())") === '{"sampling":"custom","temperature":"1.2","top_p":"0.8","top_k":"20","min_p":"0","presence_penalty":"1.5"}', "Custom saves every value, for every mode");
check(!element("m-sm-temperature").hidden && /the model's 0\.7/.test(element("m-sm-temperature").textContent) && /the model's 1\.5/.test(element("m-sm-presence_penalty").textContent), "Custom marks the model's own value under each slider");
element("sm-override").checked = true; element("sm-override").listeners.change();
check(run("samplingChanges().sampling_override") === "true", "the switch is saved with them");
routes["v1/config"] = { saved: ["sampling"] }; routes["v1/reload"] = {};
await element("sm-apply").listeners.click(); await settle();
const smBody = JSON.parse(sent.filter((r) => r.path === "v1/config").pop().body);
check(smBody.temperature === "1.2" && smBody.sampling_override === "true" && sent[sent.length - 1].path !== "v1/config" && sent.some((r) => r.path === "v1/reload")
  && /Reloading/.test(element("sm-note").textContent) && run("reloadBy") === "sm-apply", "Apply saves, reloads and says so; its button turns while the model reloads");
run("reloading = false; liveState = 'ready'");
// The saved custom set comes back: the dials show it, Reset goes back to the recommendation and clears the values.
const custom = { thinking: mode(1.2, 0.8, 1.5), non_thinking: mode(1.2, 0.8, 1.5), tools: mode(1.2, 0.8, 1.5), source: "custom" };
run(`showSampling(${JSON.stringify({ defaults: { sampling: custom, sampling_recommended: rec, sampling_override: true }, settings: { sampling: { value: "custom", source: "model" } } })})`);
check(element("sm-custom").checked && element("o-sm-temperature").textContent === "1.2" && element("sm-override").checked && run("JSON.stringify(samplingChanges())") === "{}", "the saved custom set shows, nothing to apply");
element("sm-reset").listeners.click();
check(element("sm-recommended").checked && element("o-sm-temperature").textContent === "0.7"
  && run("JSON.stringify(samplingChanges())") === '{"sampling":"recommended","temperature":"","top_p":"","top_k":"","min_p":"","presence_penalty":""}', "Reset goes back to the recommendation and clears the custom values");
// Given at launch (a flag or ORIGAMI_*): the group stays put and says why.
run(`showSampling(${JSON.stringify({ defaults: { sampling: rec, sampling_recommended: rec, sampling_override: false }, settings: { temperature: { value: "0.3", source: "flag" } } })})`);
check(element("sm-group").disabled && element("sm-apply").disabled && /command that started Origami/.test(element("sm-note").textContent), "a sampling flag holds the group");
// Stop and unload: the front says stopping, then stopped (503, with what came back); the setup greys out, the strip shows
// the freed memory, and Start (the header's or the notice's) loads the model again. A busy model asks first.
const stoppedBody = (status, freed) => [{ status, version: "2.3.3-dev", model: "glm-5.3-flash", ram_bytes: 34359738368, freed_bytes: freed, note: null, wait_seconds: 30,
  error: { code: "stopped", message: "Origami is stopped; press Start on its dashboard (this server's address in a browser) or run origami start on its Mac", type: "server_unavailable" } }, 503];
routes["v1/jobs"] = { job: null };
routes["v1/status"] = { status: "ready", version: "2.3.3-dev", model: { name: "GLM-5.3-Flash", fingerprint: "glm2" }, requests: { recent: [], current: { phase: "decode", output_pieces: 40 } },
  settings: { context: { value: "32k", source: "config" }, quality: { value: "exact", source: "config" } }, quality_levels: ["exact", "dynamic", "fast"], mac: { gpu: "Apple M2 Pro", memory_bytes: 34359738368 } };
await run("refresh()"); await settle();
check(!element("power").hidden && !element("power").disabled && element("power").attrs["aria-label"] === "Stop and unload" && /^Stop<span class="power-more"> and unload/.test(element("power").innerHTML),
  "the front's page offers Stop and unload in the header");
confirmed = []; confirmAnswer = false; await element("power").listeners.click(); await settle();
check(confirmed.length === 1 && /gets up to 30 s to finish/.test(confirmed[0]) && !sent.some((r) => r.path === "v1/stop"), "a reply being written asks first; declined, nothing stops");
routes["v1/stop"] = [stoppedBody("stopping", null)[0], 202];
const readyBusy = routes["v1/status"]; routes["v1/status"] = stoppedBody("stopping", null);
run(`live(${JSON.stringify(readyBusy)})`);
confirmAnswer = true; await element("power").listeners.click(); await settle();
check(sent.some((r) => r.path === "v1/stop" && r.method === "POST") && element("chip-text").textContent === "Stopping" && element("power").attrs["aria-busy"] === "true"
  && element("power").disabled && element("dials").disabled && element("model").disabled, "stopping: the button turns, the chip says Unloading, the setup waits");
routes["v1/status"] = stoppedBody("stopped", 23500000000); await run("refresh()"); await settle();
check(element("chip-text").textContent === "Stopped" && /about 23\.5 GB back to macOS/.test(element("live-text").textContent) && element("power").innerHTML === "Start"
  && element("power").attrs["aria-label"] === "Start" && element("power").attrs["aria-busy"] === "false" && !element("power").disabled, "stopped: the chip, what came back, and Start");
check(element("notice").hidden && element("apply").disabled && element("measure").disabled, "stopped is indicated in the header without a redundant notice; inference actions wait");
check(!element("mem").hidden && /Origami holds none/.test(element("mem-label").textContent) && /Freed by the stop <b>23\.5 GB/.test(element("keys").innerHTML), "the memory strip shows the freed memory");
routes["v1/start"] = [{ status: "loading", model: "glm-5.3-flash" }, 202];
await element("notice-retry").listeners.click(); await settle();
check(sent.some((r) => r.path === "v1/start" && r.method === "POST") && run("reloading && reloadBy") === "power" && element("power").attrs["aria-busy"] === "true", "the notice's Start starts it; the header's button turns");
routes["v1/status"] = [{ status: "loading", version: "2.3.3-dev", model: "glm-5.3-flash" }, 503]; await run("refresh()"); await settle();
check(element("chip-text").textContent === "Loading" && element("power").innerHTML === "Start" && element("power").disabled, "loading again: Loading, the Start button turning");
// A one-model server (no jobs: not Origami's front) has no Stop.
routes["v1/jobs"] = [{ error: { message: "not here" } }, 404]; routes["v1/status"] = { status: "ready", version: "2.3.3-dev", model: { name: "GLM-5.3-Flash", fingerprint: "glm3" }, requests: { recent: [] } };
run("reloading = false"); await run("refresh()"); await settle();
check(element("power").hidden, "a one-model server's page shows no Stop");
// The self-tuner's line follows the running model's `tuning`; its buttons start the tune job; the learned memory line.
const knobs = { gpu_keepalive: { value: true, source: "default", default: true } };
const tuneLine = (tuning) => { run(`showSelfTune(${JSON.stringify({ tuning })})`); return element("st-line").textContent; };
check(tuneLine(undefined) === "" && element("selftune").hidden, "an older server (no tuning) shows no tuner");
check(tuneLine({ state: "none", entry: null, knobs }) === "" && !element("selftune").hidden && element("st-go").textContent === "Run tuning"
  && element("st-check").hidden && element("st-undo").hidden, "not tuned: Tune for this Mac only");
const entry = { values: { gpu_keepalive: false }, outcome: "faster", gain: 0.12, label: "GPU keep-alive off", date: "2026-11-01T12:00:00Z" };
check(/^Applied · \+12%$/.test(tuneLine({ state: "current", entry, knobs })) && element("st-go").textContent === "Run tuning"
  && !element("st-check").hidden && !element("st-undo").hidden, "tuned faster: the date, the gain and the setting; Re-check and Undo");
check(tuneLine({ state: "current", entry: { ...entry, values: {}, outcome: "already_fastest", gain: 0 }, knobs }) === "Tuned: defaults are fastest (2026-11-01)" && element("st-check").hidden,
  "already fastest: a result with its date, nothing to re-check");
check(tuneLine({ state: "current", entry: { ...entry, outcome: "already_fastest", gain: 0 }, knobs }) === "Tuned: GPU keep-alive off stays fastest (2026-11-01)",
  "a tuned setting nothing beat stays, with its date");
check(tuneLine({ state: "recheck", reason: "macOS was updated", entry, knobs }) === "Re-check · macOS was updated" && !element("st-check").hidden, "an update asks for a re-check");
check(/Setup changed/.test(tuneLine({ state: "stale", reason: "another Mac", entry, knobs })) && element("st-go").textContent === "Run tuning" && element("st-check").hidden,
  "a result for another setup is not applied: tune again");
check(/could not be read/.test(tuneLine({ state: "unreadable", entry: null, knobs })), "a broken file says the defaults are used");
routes["v1/jobs"] = { job: { kind: "tune", title: "Tuning for this Mac", state: "running", stage: "Round 1 of 3", lines: [], seconds: 5, started: 5000, percent: 0.1 } };
run("initial = current(); modelChoice = modelInitial; smSaved = null; reloading = false");
tuneLine({ state: "none", entry: null, knobs }); await fire("st-go"); await settle();
check(JSON.parse(sent.filter((r) => r.path === "v1/jobs" && r.method === "POST").pop().body).mode === "tune" && run("job.kind") === "tune"
  && element("st-go").attrs["aria-busy"] === "true" && element("st-go").disabled && element("st-check").disabled, "Tune starts the job; its button turns, the others wait");
run(`showJob(${JSON.stringify({ kind: "tune", title: "Tuning for this Mac", state: "done", stage: "done", lines: ["Tuned: writing +12% with GPU keep-alive off."], seconds: 480, started: 5000 })})`);
check(element("job-stage").textContent === "Tuned: writing +12% with GPU keep-alive off." && !element("st-go").disabled, "done: the tuner's own last line, and the buttons are back");
confirmed = []; confirmAnswer = false; await fire("st-undo"); await settle();
check(confirmed.length === 1 && !sent.some((r) => r.body && r.body.includes('"undo"')), "Undo asks first");
// A successful process exit is not a successful tuning outcome. Terminal reasons must not trigger completion work.
run("let tuningCompletions = 0; const savedJobDone = jobDone; jobDone = () => { tuningCompletions += 1; }");
for (const [state, outcome, reason] of [["interrupted", "aborted", "The loaded model changed. Settings unchanged."],
  ["incomplete", "incomplete", "The time limit ended before every mode was measured."],
  ["incomplete", "too_noisy", "Measurements varied too much to confirm an improvement."]]) {
  const started = 5100 + sent.length + reason.length;
  run(`showJob(${JSON.stringify({ kind: "tune", title: "Comparing quality modes", state: "running", stage: "Measuring", lines: [], seconds: 20, started, percent: 0.7 })})`);
  const sentBeforeOutcome = sent.length;
  run(`showJob(${JSON.stringify({ kind: "tune", title: "Comparing quality modes", state, outcome, stage: reason, message: reason, lines: [], seconds: 30, started, percent: 1 })})`);
  await settle();
  check(element("job-title").textContent.endsWith(": " + state) && element("job-stage").textContent === reason
    && element("job-track").hidden && element("job-track").getAttribute("aria-valuenow") === null
    && element("job-fill").style.width !== "100%" && element("job-button").textContent === "Dismiss"
    && run("tuningCompletions") === 0 && sent.length === sentBeforeOutcome,
    outcome + ": clear terminal reason, no success progress or completion/apply/reload work");
}
run(`showJob(${JSON.stringify({ kind: "tune", title: "Comparing quality modes", state: "running", stage: "Measuring", lines: [], seconds: 2, started: 5200, percent: 0.5 })})`);
run(`showJob(${JSON.stringify({ kind: "tune", title: "Comparing quality modes", state: "done", stage: "Comparison ready", lines: ["Comparison ready. Settings unchanged."], seconds: 4, started: 5200, percent: 1 })})`);
check(run("tuningCompletions") === 1 && !element("job-track").hidden && element("job-track").getAttribute("aria-valuenow") === "100",
  "a completed comparison still finishes normally after an incomplete job");
run("jobDone = savedJobDone; const savedTuneDashboard = dashboard, savedTuneRevision = revision; revision = 'tune-fixture'; job = null;");
const comparisonRow = { quality: "dynamic", decode_tps: 1.2, first_token_seconds: 3, samples: 2, spread: 0.45 };
const showComparison = (result) => {
  run(`dashboard = ${JSON.stringify({ tuning_proposal: { kind: "tradeoffs", key: "tune-key", revision: "tune-fixture", results: [result] } })}`);
  tuneLine({ key: "tune-key", state: "none", entry: null, knobs });
  return element("st-result").innerHTML;
};
let comparisonHTML = showComparison(comparisonRow);
check(/<th>Samples<\/th>/.test(comparisonHTML) && /<td>2<\/td>/.test(comparisonHTML) && /45%/.test(comparisonHTML)
  && /answer quality is not scored/.test(comparisonHTML) && /Large variation/.test(comparisonHTML)
  && /data-tune-apply="dynamic">Apply/.test(comparisonHTML) && !/data-blocked|disabled/.test(comparisonHTML),
  "a noisy comparison shows sample count, variation and caution while allowing manual Apply");
for (const invalid of [{ ...comparisonRow, samples: 1 }, { ...comparisonRow, decode_tps: null }, { ...comparisonRow, spread: -1 }]) {
  comparisonHTML = showComparison(invalid);
  check(/data-blocked disabled/.test(comparisonHTML), "incomplete or invalid measurements cannot be applied");
}
// A tuned prompt group says before Apply that it sets the saved Qwen prompts aside.
run(`dashboard = ${JSON.stringify({ tuning_proposal: { kind: "performance", key: "tune-key", revision: "tune-fixture", entry: { gain: 0.12, values: { qwen_prefill_scope: 768 } },
  sets_aside: "Applying it sets aside this model's 3 saved prompts (4.2 GB): the new prompt group starts their cache afresh, so the next prompts are read in full." } })}`);
tuneLine({ key: "tune-key", state: "none", entry: null, knobs });
check(/sets aside this model.{1,6}s 3 saved prompts \(4\.2 GB\)[\s\S]*data-tune-apply/.test(element("st-result").innerHTML), "a tuned prompt group names the saved prompts it sets aside, before Apply");
run("dashboard = { last_tuning: { key: 'tune-key', outcome: 'incomplete', reason: 'Two measurements per mode are needed.' } }");
check(tuneLine({ key: "tune-key", state: "none", entry: null, knobs }) === "Incomplete · settings unchanged"
  && element("st-note").textContent === "Two measurements per mode are needed.", "the tuning panel keeps the incomplete outcome and reason");
run("dashboard = savedTuneDashboard; revision = savedTuneRevision;");
const learned = (m) => { run(`showLearned(${JSON.stringify({ memory_learned: m })})`); return element("ml-text").textContent; };
check(learned({ bytes: null, days: 0, trims: 0, applied: false }) === "" || element("ml").hidden, "nothing recorded yet: no line");
learned({ bytes: null, days: 0.125, trims: 2, applied: false });
check(element("ml").hidden, "unused learned memory does not crowd the dashboard");
learned({ bytes: 21474836480, days: 2.5, trims: 205, applied: false });
check(element("ml").hidden, "an explicit memory limit hides the unused learned budget");
// Setup's Memory limit line says who decided the limit as Health > Memory does, and the learned start as a start.
const memStates = JSON.parse(readFileSync(new URL("Fixtures/status/memory-states.json", import.meta.url), "utf8")), keptDashboard = run("JSON.stringify(dashboard)");
const setupLine = (name) => {
  const st = memStates.find((x) => x.state.startsWith(name));
  run("dashboard = { ...(dashboard || {}), saved_settings: " + JSON.stringify(st.saved_settings) + " }; showLearned(" + JSON.stringify(st.status) + ")");
  return element("ml").hidden ? null : element("ml-text").textContent;
};
check(setupLine("automatic, started at the learned start") === "Memory limit: Origami chose this limit from this Mac's memory size, because its measurement from 16 Nov no longer counts (macOS took memory back on 3 days since then). Origami starts at 14.7 GB, what this Mac usually had free over the last 12 days, and grows while memory stays free."
  && !element("ml-reset").hidden && element("ml-reset").textContent !== "Reset", "Setup: automatic, a stale measurement, and the learned start as a start, with the button that forgets it");
check(setupLine("the measured limit") === "Memory limit: Origami measured this limit on 22 Nov by itself while the Mac was idle. It is the most memory this Mac kept for Origami without macOS taking any back."
  && element("ml-reset").hidden, "Setup: the measured limit says what measured means");
check(setupLine("your limit, which this Mac does not keep") === "Memory limit: You set this limit yourself.", "Setup: your own limit");
run("dashboard = " + keptDashboard);
routes["v1/memory/learned"] = { reset: true }; await fire("ml-reset"); await settle();
check(sent.some((r) => r.path === "v1/memory/learned" && r.method === "POST") && /^Forgotten/.test(element("ml-text").textContent), "Forget the learned start forgets it");
// The page itself: Storage and folders is one closed section; the drive buttons the card replaced are gone.
const html = readFileSync(process.argv[2], "utf8");
// A tuning run needs about five minutes on the slowest model to compare anything: no shorter limit is offered, and the
// default one grows to what the model's measurements need.
check(/<select id="st-budget"><option value="" selected>About 8 min \(more if needed\)<\/option><option value="8">8 min<\/option>/.test(html), "the tuning time limit starts at 8 min, growing if needed");
check(/<div id="view-library" hidden>/.test(html) && !/Use as store|Hot copy here|Rebuild elsewhere|<h2>Drives<\/h2>/.test(html), "Storage has its own view; obsolete drive action buttons are absent");

// Shared snapshot: draft preservation, revision conflicts and action eligibility.
run('initial = current(); modelChoice = modelInitial; smSaved = null; conflict = false; reloading = false; applying = null; job = null; front = true; liveState = "ready"');
const snapshot = { revision: "r1", generation: "g1", configured_model: "glm-5.3-flash", active_model: "glm-5.3-flash", host: "Test host", can_tune: true, can_measure: true, reload_required: false,
  saved_settings: { context: { value: "32k" }, quality: { value: "exact" }, reasoning_effort: { value: "none" } }, job: null };
run(`acceptSnapshot(${JSON.stringify(snapshot)}); syncSaved(); gate()`);
run('$("quality").value = "1"; dialMoved()');
run(`acceptSnapshot(${JSON.stringify(snapshot)}); syncSaved()`);
check(element("quality").value === "1" && !run("conflict"), "ordinary refresh preserves an unsaved draft");
run(`acceptSnapshot(${JSON.stringify({ ...snapshot, revision: "r2" })}); gate()`);
check(element("quality").value === "1" && !run("conflict") && /based on older settings/.test(element("save-state").textContent), "a newer revision keeps the draft, marked as based on older settings (the server checks its keys)");
run(`acceptSnapshot(${JSON.stringify({ ...snapshot, revision: "r2", generation: "g2" })}); gate()`);
check(run("conflict") && element("save").disabled && /Changed elsewhere/.test(element("save-state").textContent), "a restarted server, which no longer knows the draft's base, asks for a review first");
await element("discard").listeners.click(); await settle();
check(!run("conflict"), "discard resolves an external-edit conflict");
run('initial = current(); modelChoice = modelInitial; smSaved = null; reloading = false; liveState = "ready"; ready = true');
run(`acceptSnapshot(${JSON.stringify({ ...snapshot, revision: "r2", can_tune: false })}); gate()`);
check(element("st-go").disabled && element("measure").disabled, "busy server snapshot disables tuning and measurements");
run(`acceptSnapshot(${JSON.stringify({ ...snapshot, revision: "r2", reload_required: true })}); gate()`);
check(!element("reload-saved").hidden && /Reload needed/.test(element("save-state").textContent) && element("st-go").disabled, "saved settings clearly require reload before benchmarking");
await run('post("v1/config", { quality: "dynamic" })');
check(sent.filter(r => r.path === "v1/config").pop().headers["If-Match"] === "r2", "saves carry the revision that the user reviewed");
run('offline = true; gate()');
check(element("page").disabled && element("save").disabled && /out of date/.test(element("save-state").textContent), "disconnection makes stale values explicit and blocks mutations");
run('offline = false; fsState = { models: false, listing: { path: "/old", writable: true }, resolve: () => {} }');
check(!run('fsChoosable({ path: "/read-only", writable: false })'), "folder destinations must be writable");
routes["v1/fs"] = [{ error: { message: "Disk disconnected" } }, 404];
await run('fsOpen("/gone")');
check(element("fs-choose").disabled && run("fsState.listing") === null, "failed folder navigation cannot choose the previously displayed destination");
// A late response from a closed dialog must not replace a newly opened dialog.
let releaseFolder;
routes["v1/fs"] = () => new Promise(resolve => { releaseFolder = resolve; });
const oldNavigation = run('fsOpen("/older")'); await settle();
run('fsDone(null); fsState = { models: false, listing: { path: "/newer" }, resolve: () => {} }');
releaseFolder(answer({ path: "/older", folders: [], models: [], crumbs: [] })); await oldNavigation;
check(run("fsState.listing.path") === "/newer", "late folder responses cannot cross dialog sessions");
run('fsDone(null)');

// Unified wired RAM: host-provided bounds, truthful live/saved values, atomic edits, and the reboot repair.
// Memory amounts in the page's decimal GB, as settings read them; only this Mac's own size is binary.
const GB = 1e9;
let wiredFixture = { physical_memory_bytes: 32 * 1073741824, os_headroom_bytes: 4.5 * GB, safe_max_budget_bytes: 27.5 * GB,
  system_limit_mib: 24576, system_limit_bytes: 24 * GB, effective_limit_bytes: 24 * GB, system_default: false,
  metal_recommended_bytes: 24 * GB, configured_budget_bytes: 26 * GB, active_budget_bytes: 24 * GB,
  usable_budget_bytes: 22 * GB, physical_footprint_bytes: 20 * GB, budget_mode: "explicit", temporarily_reduced: true,
  requires_raise: true, supported: true, persistent: false };
let wiredSnapshot = { ...snapshot, revision: "wired1", generation: "wired-generation", reload_required: false, wired_memory: wiredFixture,
  saved_settings: { ...snapshot.saved_settings, memory_limit: { value: "26GB" } } };
const wiredStatus = () => ({ status: "ready", model: { name: "GLM", fingerprint: "wired" }, settings: wiredSnapshot.saved_settings,
  requests: { recent: [] }, dashboard: wiredSnapshot });
routes["v1/dashboard"] = () => answer(wiredStatus());
routes["v1/jobs"] = { job: null };
run('dashboard = null; initial = null; smSaved = null; conflict = false; applying = null; reloading = false; job = null; offline = false; noModel = false; generation = null; revision = null; liveState = "ready"');
run(`acceptSnapshot(${JSON.stringify(wiredSnapshot)}); familySetup({settings: ${JSON.stringify(wiredSnapshot.saved_settings)}, quality_levels: ["exact", "dynamic", "fast"]}); gate()`);
check(run("memoryStops.at(-1)") === 27.5 && !element("memory").disabled && element("wired-saved").textContent === "26.0 GB"
  && element("wired-active").textContent === "24.0 GB" && element("wired-system").textContent === "24.0 GB",
  "wired RAM uses the host's fractional maximum and separates saved, active and system budgets");
check(/administrator.*this Mac/.test(element("wired-help").textContent) && /until reboot/.test(element("wired-help").textContent)
  && !element("wired-restore").hidden, "wired RAM explains local authorization and reboot lifetime, with repair when needed");
check(element("wired-usable").textContent === "22.0 GB" && element("wired-footprint").textContent === "20.0 GB",
  "wired RAM distinguishes the current allowance and process footprint from the loaded limit");
check(/Restore the ceiling/.test(element("wired-state").textContent),
  "the actionable system-ceiling mismatch takes priority over temporary pressure relief");
run('dashboard.wired_memory = { ...dashboard.wired_memory, requires_raise: false }; showWiredMemory()');
check(/temporarily reduced/.test(element("wired-state").textContent) && /recovers/.test(element("wired-state").textContent),
  "explicit pressure relief explains automatic recovery without a reload action");
run('dashboard.wired_memory = { ...dashboard.wired_memory, temporarily_reduced: false }; showWiredMemory()');
check(element("wired-state").textContent === "", "the pressure notice clears after recovery");
run(`dashboard.wired_memory = ${JSON.stringify(wiredFixture)}; showWiredMemory()`);
run('$("memory").value = $("memory").max; autoMemory = false; $("quality").value = "1"; dialMoved()');
const parseMemoryGB = value => /GB$/.test(value) ? Math.trunc(parseFloat(value) * 1e9) : NaN;
check(parseMemoryGB(run("changes().memory_limit")) === 27.5 * GB && /administrator approval/.test(element("wired-state").textContent),
  "the maximum saves explicit GB to byte precision without exceeding the machine's bound");
for (const ram of [32, 64, 128, 192]) {
  for (const extraMiB of [0, 1, 17, 123]) {
    const maximum = (ram - Math.max(3, ram / 8)) * 1073741824 - extraMiB * 1048576;
    const encoded = run(`memorySetting(${maximum} / GB)`), decoded = parseMemoryGB(encoded);
    check(Number.isFinite(decoded) && decoded <= maximum && maximum - decoded <= 1,
      `${ram} GB host, ${extraMiB} MiB adjustment: GB serialization roundtrips within one byte below its exact ceiling`);
  }
}
const wiredConfigBefore = sent.filter(r => r.path === "v1/config").length;
const wiredReloadBefore = sent.filter(r => r.path === "v1/reload").length;
routes["v1/memory/wired"] = [{ error: { message: "Administrator approval cancelled; nothing saved." } }, 403];
await run("apply(false)"); await settle();
const wiredDenied = sent.filter(r => r.path === "v1/memory/wired").at(-1);
check(wiredDenied.headers["If-Match"] === "wired1" && wiredDenied.signal.milliseconds === 135000 && JSON.parse(wiredDenied.body).quality === "dynamic"
  && sent.filter(r => r.path === "v1/config").length === wiredConfigBefore && run("dirty()")
  && /nothing saved/.test(element("apply-note").textContent), "wired save includes every edited setting and preserves the draft after authorization fails");
routes["v1/memory/wired"] = (_path, options) => {
  const values = JSON.parse(options.body);
  const parsed = values.memory_limit === "auto" ? null : parseMemoryGB(values.memory_limit);
  if (parsed != null && (!Number.isFinite(parsed) || parsed > wiredFixture.safe_max_budget_bytes)) return answer({error:{message:"Invalid memory size"}}, 400);
  wiredFixture = { ...wiredFixture, configured_budget_bytes: parsed,
    system_limit_bytes: 27.5 * GB, effective_limit_bytes: 27.5 * GB, requires_raise: false };
  wiredSnapshot = { ...wiredSnapshot, revision: "wired2", wired_memory: wiredFixture, reload_required: true,
    saved_settings: { ...wiredSnapshot.saved_settings, ...Object.fromEntries(Object.entries(values).map(([key, value]) => [key, {value}])) } };
  return answer({ saved: true, revision: "wired2", reload_required: true, wired_memory: wiredFixture });
};
await run("apply(false)"); await settle();
check(sent.filter(r => r.path === "v1/reload").length === wiredReloadBefore && !run("dirty()")
  && element("wired-saved").textContent === "27.5 GB" && element("wired-active").textContent === "24.0 GB",
  "Save applies both ceilings without reloading or presenting the saved budget as active");
// A reboot reports the automatic ceiling explicitly; a repair can leave an already-compatible model running.
wiredFixture = { ...wiredFixture, system_limit_mib: 0, system_limit_bytes: 0, system_default: true,
  effective_limit_bytes: 24 * GB, requires_raise: true };
wiredSnapshot = { ...wiredSnapshot, wired_memory: wiredFixture, reload_required: false };
run(`acceptSnapshot(${JSON.stringify(wiredSnapshot)}); gate()`);
check(element("wired-system").textContent === "Automatic · 24.0 GB" && !element("wired-restore").disabled,
  "the automatic macOS ceiling is not mislabeled as zero or unavailable");
routes["v1/memory/wired"] = (_path, options) => {
  wiredFixture = { ...wiredFixture, system_default: false, system_limit_bytes: 27.5 * GB, effective_limit_bytes: 27.5 * GB, requires_raise: false };
  wiredSnapshot = { ...wiredSnapshot, revision: "wired3", wired_memory: wiredFixture, reload_required: false };
  return answer({ saved: true, revision: "wired3", wired_memory: wiredFixture, reload_required: false });
};
await fire("wired-restore"); await settle();
check(sent.filter(r => r.path === "v1/reload").length === wiredReloadBefore && /restored until reboot/.test(element("wired-note").textContent)
  && element("wired-restore").hidden
  && parseMemoryGB(JSON.parse(sent.filter(r => r.path === "v1/memory/wired").at(-1).body).memory_limit) === 27.5 * GB,
  "restoring the saved ceiling sends explicit GB and avoids a reload when the server says none is needed");
run('autoMemory = true; dialMoved()');
routes["v1/memory/wired"] = () => answer({ saved: true, revision: "wired4", reload_required: true });
await run("apply(true)"); await settle();
check(JSON.parse(sent.filter(r => r.path === "v1/memory/wired").at(-1).body).memory_limit === "auto"
  && sent.filter(r => r.path === "v1/reload").length === wiredReloadBefore + 1,
  "Save and reload sends Automatic through the unified endpoint before reloading");
run('memoryRange(224 * GB, 200 * GB)');
check(run("memoryStops.at(-1)") === 224, "wired bounds follow a different Mac rather than a hardcoded 32 GB machine");
check(/grid-template-columns: 300px/.test(html) && /grid-template-columns: 104px/.test(html), "the ASCII crane has its own column on desktop and on narrow screens");

// An upstream error normalized by the front must not fall through to Ready or stale generation progress.
run('reloading = false; applying = null; job = null; offline = false; render({ status: "unavailable", upstream_http_status: 503, failure: "Reconnecting…", requests: { current: { phase: "decode", completion_tokens: 10 } } })');
check(element("chip-text").textContent === "Not responding" && element("chip").dataset.state === "unavailable"
  && run("liveState") === "unavailable" && element("progress").hidden && element("basis").textContent === "Model not responding",
  "unavailable upstream status preserves the unavailable state instead of Ready or stale Working");
run('live({ status: "ready", requests: { recent: [] } })');
check(element("chip-text").textContent === "Ready" && run("liveState") === "ready", "a later healthy status recovers the unavailable live strip");

// The four views. Every element the script names by id exists in the page (this stub page would make any up).
const pageIds = new Set([...html.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1]));
const missingIds = [...new Set([...script.matchAll(/\$\("([a-z0-9_-]+)"\)/g)].map((m) => m[1]))].filter((id) => !pageIds.has(id));
check(missingIds.length === 0, "every element the script names exists in the page" + (missingIds.length ? ": missing " + missingIds.join(", ") : ""));
check(["run", "setup", "library", "settings"].every((v) => new RegExp('data-view="' + v + '"').test(html) && new RegExp('<div id="view-' + v + '"').test(html)), "Now, Setup, Models and Connect each have a tab and a view");

// Now: where a request's time went, the health lines and what they offer.
run('reloading = false; applying = null; conflict = false; job = null; offline = false');
const timed = { time: new Date().toISOString(), seconds: 40, prompt_tokens: 1000, cached_tokens: 250, completion_tokens: 200, decode_tps: 10, ttft_s: 12, finish: "stop",
  stage_seconds: { prompt_encoding: 1, restore_and_setup: 1, prefill_and_capture: 10, generation: 20, finalization: 2 } };
check(run("timeParts(" + JSON.stringify(timed) + ").map((p) => p[2]).join()") === "6,2,10,20,2" && run("timeParts({ seconds: 3 })") === null,
  "a request's time splits into waiting, setting up, reading the prompt, writing and saving; one without stage times has no bar");
const nowStatus = { status: "ready", model: { name: "q2", experts: "low-bit experts from x.gguf (Q2_0): replies differ from the exact model" }, context: 131072, mac: { memory_bytes: 32 * 2 ** 30 },
  settings: { quality: { value: "exact" } }, requests: { recent: [timed] },
  memory: { cap_bytes: 30e9, budget_bytes: 24e9, footprint_bytes: 16e9, budget_mode: "explicit", temporarily_reduced: true },
  memory_learned: { limit_advice: { suggested_bytes: 25.5e9, text: "Fastest that this Mac kept: 25.5GB — 10.1 tok/s; your current limit 30.0GB — cut 2 times; set it with `origami config set memory_limit 25.5GB`" } },
  experts_from: { split: { reads: 100, memory: 0.5, hot_copy: 0.3, model_drive: 0.2 }, sources: [
    { role: "all", state: "reading", drive: "SSD", detail: "SSD read 38% of the time far below its own speed. Stop Origami, then run `origami refresh`.", slow_reads: { high: true, share: 0.38, hour_share: 0.42, busy_seconds: 900 } },
    { role: "hot", state: "building", drive: "internal drive", bytes: 1e10, budget_bytes: 4e10, progress: { done_bytes: 1e10, total_bytes: 4e10, seconds_left: 300 } }] } };
run("access = null; showNow(" + JSON.stringify(nowStatus) + ")");
const health = element("health").innerHTML;
// Health > Memory in plain words: each state in Tests/Fixtures/status/memory-states.json (built from a 3.1.5 server's
// own fields and the shapes the code emits) shows its title, its sentences and its buttons exactly; a `last_cut.at` of
// "-22m" is that long before now.
{
  const states = JSON.parse(readFileSync(new URL("Fixtures/status/memory-states.json", import.meta.url), "utf8"));
  const kept = run("JSON.stringify(dashboard)");
  const strip = (x) => x.replace(/<[^>]+>/g, "").replace(/&#39;/g, "'").replace(/&amp;/g, "&");
  for (const st of states) {
    const status = { ...JSON.parse(JSON.stringify(nowStatus)), memory: JSON.parse(JSON.stringify(st.status.memory)), memory_learned: st.status.memory_learned };
    const cut = status.memory.last_cut;
    if (cut && /^-\d+m$/.test(cut.at)) cut.at = new Date(Date.now() - parseInt(cut.at.slice(1), 10) * 60e3 - 20e3).toISOString();
    run("dashboard = { ...(dashboard || {}), saved_settings: " + JSON.stringify(st.saved_settings) + " }; job = " + JSON.stringify(st.job) + "; showNow(" + JSON.stringify(status) + ")");
    const h = element("health").innerHTML, row = /data-m="(\w+)"[^>]*><\/i><span class="label">Memory<\/span><div>([\s\S]*?)<\/div><div class="hact">([\s\S]*?)<\/div><\/div>/.exec(h);
    const [title, sub] = row ? row[2].split('<span class="sub">') : ["", ""];
    const got = row ? { mark: row[1], title: strip(title), text: strip(sub || ""), buttons: [...row[3].matchAll(/>([^<>]+)<\/button>/g)].map((x) => x[1]) } : null;
    check(!!st.words && JSON.stringify(got) === JSON.stringify(st.words), "Health > Memory, " + st.state + (st.words && JSON.stringify(got) === JSON.stringify(st.words) ? "" : ": " + JSON.stringify(got)));
  }
  run("dashboard = " + kept + "; job = null");
  const legend = element("flow-mem").innerHTML;
  check(/Origami uses/.test(legend) && /free/.test(legend) && /other apps and macOS/.test(legend) && !/left to macOS/.test(legend) && !/room in its limit/.test(legend), "the memory bar's legend says what Origami uses, what is free and what other apps and macOS hold");
  // Row C7's first reading (production 3.1.8.1, 2026-11-26): the budget below the cap after a cut, and what the Mac had free.
  const c7 = { ...JSON.parse(JSON.stringify(nowStatus)), memory: { cap_bytes: 24.05e9, budget_bytes: 20.21e9, reserved_bytes: 19.14e9, footprint_bytes: 17.76e9, available_bytes: 5.77e9,
    budget_mode: "explicit", temporarily_reduced: true, last_cut: { reason: "memory is short (3.8 GB free)", at: new Date(Date.now() - 3 * 3600e3).toISOString() } } };
  run("showNow(" + JSON.stringify(c7) + ")");
  const mem = element("flow-mem").innerHTML.replace(/<[^>]+>/g, "");
  check(/Origami uses 17\.8 GB/.test(mem) && /free 5\.8 GB/.test(mem) && /other apps and macOS 10\.8 GB/.test(mem) && /May still grow by 2\.5 GB within its budget of 20\.2 GB/.test(mem)
    && /3\.8 GB of its 24\.1 GB limit held back: macOS took memory back 3 h ago because free memory had run low/.test(mem) && !/may still grow by 6/.test(mem),
    "the memory flow splits the old 'may still grow by' into growth within the budget, what is held back and why, and what the Mac has free" + (/held back/.test(mem) ? "" : ": " + mem));
  delete c7.memory.last_cut; c7.memory.learned_start_bytes = 20e9;
  run("showNow(" + JSON.stringify(c7) + ")");
  check(/held back: Origami started at what this Mac usually had free/.test(element("flow-mem").innerHTML), "held back after a learned start says so");
}
run("showNow(" + JSON.stringify(nowStatus) + ")");
run('live({ status: "loading", model: "first", loading: { seconds: 42, line: "reading the dense weights" } })');
check(/Loading the model · 42 s · reading the dense weights/.test(element("live-text").textContent), "loading shows its seconds and the model's last line in the header");
check(/<b>42%<\/b> of its reading far below its own speed/.test(health) && /data-copy-text="origami refresh"/.test(health) && /run `origami refresh`/.test(health), "a slow drive says so, with the cure");
check(/Hot copy<\/span><div><b>Building<\/b>/.test(health) && /width:25\.0%/.test(health) && /about 5 min left/.test(health), "a hot copy being built shows its progress");
// A healthy drive counted toward its next refresh: the amount read and the sentence; near the last slowdown's amount, a
// warning with the command. A hot copy's drive slowing down says to rebuild it.
const counted = JSON.parse(JSON.stringify(nowStatus)), drive = counted.experts_from.sources[0], copyRow = counted.experts_from.sources[1];
Object.assign(drive, { detail: null, slow_reads: { high: false, share: 0.01, busy_seconds: 900 }, since_rewrite: { bytes: 4.2e12, due: false, high: false, text: "4.2 TB read since 2026-11-15; slow 1% of the reading time." } });
Object.assign(copyRow, { state: "ready", progress: null, since_rewrite: { bytes: 9e12, due: false, high: true, text: "9.0 TB read since 2026-11-10; slow 30% of the reading time. The drive reads these files slowly now: stop the server and run `origami refresh --copy`." } });
run("showNow(" + JSON.stringify(counted) + ")");
let counting = element("health").innerHTML;
check(/data-m="ok"[^>]*><\/i><span class="label">Drive<\/span><div><b>[^<]*<\/b> · reads at its own speed · 4\.2 TB read/.test(counting) && /slow 1% of the reading time/.test(counting)
  && !/data-copy-text="origami refresh"/.test(counting), "a healthy drive shows what was read since its files were rewritten, quietly");
check(/data-m="warn"[^>]*><\/i><span class="label">Hot copy/.test(counting) && /origami refresh --copy/.test(counting), "a hot copy read slowly says to refresh it");
drive.since_rewrite = { bytes: 5.1e12, due: true, high: false, text: "5.1 TB read since 2026-11-15; slow 1% of the reading time. Last time the slowdown began after about 6.0 TB. A slowdown is likely soon: stop the server and run `origami refresh`." };
run("showNow(" + JSON.stringify(counted) + ")");
counting = element("health").innerHTML;
check(/data-m="warn"[^>]*><\/i><span class="label">Drive/.test(counting) && /likely soon/.test(counting) && /data-copy-text="origami refresh"/.test(counting), "near the last slowdown's amount, the Drive line warns ahead with the command");
run("showNow(" + JSON.stringify(nowStatus) + ")");
check(/data-b="differs"[^>]*><b>Low-bit experts<\/b> · replies differ from exact/.test(element("badges").innerHTML) && !/Exact/.test(element("badges").innerHTML),
  "a low-bit variant says its replies differ from exact, beside the model's name");
check(/First word<\/dt><dd>12\.0/.test(element("figures").innerHTML) && /Prompt reused<\/dt><dd>25/.test(element("figures").innerHTML) && /reading the prompt 10\.0 s/.test(element("flow-time").innerHTML)
  && /from the hot copy 30%/.test(element("flow-reads").innerHTML) && /Origami uses 16\.0 GB/.test(element("flow-mem").innerHTML), "Now shows the first word, the reuse, and where time, reads and memory went");
run('showNow({ status: "stopped" })');
check(element("health").innerHTML === "" && element("health-block").hidden && element("figures").innerHTML === "" && element("badges").innerHTML === "", "stopped, Now keeps none of the last model's numbers");
// The measured limit is saved as its advice names it, after a confirm, then the model reloads.
confirmed = []; confirmAnswer = false; await run('useLimit("25.5GB", $("health-note"))'); await settle();
const limitPosts = () => sent.filter((r) => r.path === "v1/memory/wired" && /25\.5GB/.test(r.body || "")).length;
check(confirmed.length === 1 && limitPosts() === 0, "using the measured limit asks first");
confirmAnswer = true; routes["v1/memory/wired"] = { revision: "r9" }; routes["v1/reload"] = {}; await run('useLimit("25.5GB", $("health-note"))'); await settle();
check(limitPosts() === 1 && sent.filter((r) => r.path === "v1/memory/wired").pop().headers["If-Match"] !== undefined, "then saves it against the revision this page read");
run('reloading = false; applying = null; conflict = false');
// "Use 25.5 GB" (a limit of 25.5GB): asked first; declined, nothing is sent; accepted, the limit is saved, then the model reloads; a save
// refused reloads nothing.
const pressUse = () => element("health").listeners.click({ target: { closest: () => ({ dataset: { hact: "use-limit", value: "25.5GB" }, disabled: false, setAttribute: noop, removeAttribute: noop, toggleAttribute: noop }) } });
const usePosts = () => sent.filter((r) => r.method === "POST" && ["v1/memory/wired", "v1/reload"].includes(r.path)).map((r) => r.path + " " + (r.body || ""));
let useBefore = usePosts().length; confirmed = []; confirmAnswer = false;
await pressUse(); await settle();
// The server's value goes out as it came ("25.5GB": decimal, as `origami config set memory_limit` reads it); the page
// shows the same number, as `origami status` and `origami config` print it.
check(confirmed.length === 1 && /limit to 25\.5 GB and/.test(confirmed[0]) && usePosts().length === useBefore, "Use … GB declined sends nothing");
routes["v1/memory/wired"] = [{ error: { message: "Wired memory cannot be raised now" } }, 400]; confirmAnswer = true;
await pressUse(); await settle();
check(usePosts().slice(useBefore).join("|") === 'v1/memory/wired {"memory_limit":"25.5GB"}' && /cannot be raised/.test(element("health-note").textContent),
  "Use … GB whose save is refused reloads nothing and says why");
useBefore = usePosts().length;
routes["v1/memory/wired"] = { revision: "r2" }; routes["v1/reload"] = [{ reloading: true }, 202];
await pressUse(); await settle(); confirmAnswer = false;
check(usePosts().slice(useBefore).join("|") === 'v1/memory/wired {"memory_limit":"25.5GB"}|v1/reload {}' && /Loading with a limit of 25\.5 GB…/.test(element("health-note").textContent),
  "Use … GB saves the limit, then reloads the model");
run('reloading = false; applying = null; conflict = false');

// Models: every model a row; a low-bit variant marked; a .gguf of experts added as a variant of an exact model.
run('models = ' + JSON.stringify([{ id: "q4", origami: { model: "q4", setting: "q4", running: true, family: "qwen3.8-flash-next", bytes: 1e11, availability: "available", drive: "SSD" } },
  { id: "q2", origami: { model: "q2", setting: "q2", variant_of: "q4", family: "qwen3.8-flash-next", bytes: 3e10, availability: "available", drive: "SSD" } }])
  + '; modelInitial = "q4"; noModel = false; liveState = "ready"; live.chip = STATE.ready; found = { models: [{ name: "glm", path: "/m/glm.gguf", bytes: 4e11, drive: "SSD", state: "available", verdict: "runs" }] }; modelList()');
const modelRows = element("model-list").innerHTML;
check(/q4<\/strong><span class="sub">Ready · /.test(modelRows) && /q2<\/strong><span class="badge" data-b="differs">low-bit experts · replies differ from exact/.test(modelRows) && /its experts on q4/.test(modelRows) && /data-mload="q2"/.test(modelRows)
  && !/data-mload="q4"/.test(modelRows) && /Found on this Mac[\s\S]*data-madd="\/m\/glm\.gguf"/.test(modelRows), "Models lists each model, names the loaded one's state as the header does, marks a low-bit variant, and offers Load and Add");
element("add-base").value = "q4"; const checksBefore = sent.filter((r) => r.path.startsWith("v1/models/check")).length;
await run('checkSource("/m/experts.gguf")'); await settle();
check(!element("add-base-row").hidden && !element("add-go").hidden && /replies differ from exact; q4 itself stays exact/.test(element("add-info").textContent)
  && sent.filter((r) => r.path.startsWith("v1/models/check")).length === checksBefore, "a .gguf with a base chosen says what the variant is, and that its replies differ");
routes["v1/jobs"] = { job: null }; fire("add-go"); await settle();
const variantJob = JSON.parse(sent.filter((r) => r.path === "v1/jobs" && r.method === "POST").pop().body);
check(variantJob.kind === "add" && variantJob.source === "/m/experts.gguf" && variantJob.base === "q4" && !("to" in variantJob), "and is added as that model's low-bit variant");
// An experts-only file the pairing check cannot fully verify: the page asks with the check's words, and only a yes adds it.
const addPosts = () => sent.filter((r) => r.path === "v1/jobs" && r.method === "POST").map((r) => JSON.parse(r.body));
routes["v1/jobs"] = (path, options) => options.method === "POST" && !JSON.parse(options.body).acknowledged
  ? answer({ confirm: "experts.gguf has no tokenizer: only its banks' shapes say it is q4 quantized another way." })
  : answer({ job: { kind: "add", title: "Adding experts", state: "running", stage: "Starting", lines: [], seconds: 0, started: 9000 } }, 202);
confirmed = []; confirmAnswer = false; let postsBefore2 = addPosts().length; fire("add-go"); await settle();
check(confirmed.length === 1 && /only its banks' shapes say/.test(confirmed[0]) && addPosts().length === postsBefore2 + 1 && element("add-info").textContent === "Not added.",
  "an unverifiable experts file asks first; declined, nothing is added");
confirmed = []; confirmAnswer = true; postsBefore2 = addPosts().length; fire("add-go"); await settle();
check(confirmed.length === 1 && addPosts().length === postsBefore2 + 2 && addPosts().pop().acknowledged === true && addPosts().pop().base === "q4",
  "a yes sends the add again, acknowledged");
routes["v1/jobs"] = [{ error: { message: "These experts do not belong with this GLM-5.3-Flash GGUF: layer 1 router differs" } }, 409];
confirmed = []; fire("add-go"); await settle();
check(confirmed.length === 0 && /router differs/.test(element("add-info").textContent), "a refused pairing shows its reason and asks nothing");
confirmAnswer = false; routes["v1/jobs"] = { job: null }; run("jobPost = null; jobLive = null"); await settle();
element("add-base").value = ""; await run('checkSource("owner/name")'); await settle();
check(element("add-base-row").hidden, "a download offers no base");
// The source edited while its check is pending: that check's answer brings no Add back, so A is never posted under B.
let releaseCheck; const keptCheck = routes["v1/models/check"];
routes["v1/models/check"] = () => new Promise((done) => { releaseCheck = () => done(answer({ verdict: "runs", family: "Qwen", bytes: 1e11, needs_bytes: 0, fits: true })); });
element("add-source").value = "/m/a"; const pendingCheck = run('checkSource("/m/a")'); await settle();
element("add-source").value = "/m/b"; element("add-source").listeners.input({ target: element("add-source") });
releaseCheck(); await pendingCheck; await settle();
check(element("add-go").hidden && run("addSource") === "", "a check answered after its source was edited brings no Add for the old source");
routes["v1/models/check"] = keptCheck;

// The save bar shows only while there is something to save, reload or review.
run('smSaved = null; initial = current(); modelChoice = modelInitial; conflict = false; reloading = false; dashboard = { ...(dashboard || {}), reload_required: false }; gate()');
check(element("savebar").hidden && element("setup-dot").textContent === "", "with nothing to save the save bar is out of the way");
run('modelChoice = "q2"; gate()');
check(element("savebar").hidden && !element("use-model").hidden, "a model chosen and nothing else has one button, Load model");
run('modelChoice = modelInitial; globalThis.keptContext = $("context").value; $("context").value = keptContext === "0" ? "1" : "0"; dialMoved("context")');
check(!element("savebar").hidden && element("setup-dot").textContent === "●" && /Unsaved changes/.test(element("save-state").textContent), "an unsaved dial brings it back and marks Setup");
run('$("context").value = keptContext; gate()');

// The model's drive went away: the notice says so; its button reloads (for a different file found at the path).
run('notice({ status: "degraded", degraded: "The model\'s drive is not answering: reconnect it; reading resumes by itself, with no reload" })');
check(run("noticeKind") === "degraded" && !element("notice").hidden && element("notice-title").textContent === "The model's drive is not answering"
  && element("notice-retry").textContent === "Reload" && !element("notice-retry").hidden, "a drive that is gone says so in the notice, with Reload");
// A first run: what Origami found, the download it is made for, or your own file, as the one next action.
run('models = []; found = { models: [{ name: "qwen", path: "/m/qwen", bytes: 1e11, drive: "SSD", verdict: "runs" }] }; notice({ status: "failed", failure: "No model found", download: "o/Qwen" })');
check(run("noticeKind") === "missing" && element("notice").hidden && !element("first").hidden && element("picker").hidden
  && /data-madd="\/m\/qwen"[^>]*>Add qwen · 100 GB/.test(element("first").innerHTML) && /Download Qwen…/.test(element("first").innerHTML) && /Choose a file or folder…/.test(element("first").innerHTML),
  "a first run offers what Origami found, the download and your own file");
run('notice({ status: "stopped", freed_bytes: 16e9 })');
check(/data-first="start"/.test(element("first").innerHTML) && /16\.0 GB of memory is back/.test(element("first").innerHTML), "stopped, the next action is Start");

// What a live server showed. Before any model is added the list holds the built-in default as "missing" with no setting:
// the first run still offers Add, not a Load that cannot work.
routes["v1/models"] = { data: [{ id: "qwen3.8-flash-next", origami: { model: "qwen3.8-flash-next", setting: "", availability: "missing", default: true } }] };
await run("loadModels()"); await settle();
check(run("models.length") === 0, "the not-found default is not listed as an added model");
run('found = { models: [] }; reloading = false; notice({ status: "failed", failure: "No model found", download: "o/Qwen" })');
check(/Download Qwen…/.test(element("first").innerHTML) && !/data-mload/.test(element("first").innerHTML) && element("picker").hidden, "a first run with nothing found offers the download and your own file");
// A model added but not loaded: Load, and no empty Model menu beside it.
run('models = [{ id: "q4", origami: { model: "q4", setting: "q4" } }]; notice({ status: "failed", failure: "No model found" }); gate()');
check(/data-mload="q4"[^>]*>Load q4/.test(element("first").innerHTML) && element("picker").hidden && element("use-model").hidden, "a model added but not loaded has one Load button");
// The model an add job added is chosen once a list holds it, also when the first read was dropped for its old revision.
routes["v1/models"] = { data: [{ id: "q4", origami: { model: "q4", setting: "q4", default: true } }, { id: "new", origami: { model: "new", setting: "new" } }] };
run('noticeKind = null; addedNow = "new"'); await run("loadModels()"); await settle();
check(run("modelChoice") === "new" && run("addedNow") === null && /Added new\./.test(element("model-note").textContent), "an added model is chosen when the list that holds it arrives");
run('modelChoice = modelInitial; modelByJob = false; $("model-note").textContent = ""');
// Measure waits for its whole reply: the 15 s of every other call would cancel it.
sent.length = 0; routes["v1/chat/completions"] = { choices: [] };
element("measure").listeners.click(); await settle();
const measureCall = sent.find((r) => r.path === "v1/chat/completions");
check(!!measureCall && measureCall.signal.milliseconds === 300000, "Measure gives its reply five minutes, not 15 s");
// The memory search runs the model at limits under test: no "reload required" bar, the limit named as under test, sizes
// in the decimal GB the setting takes.
const searching = JSON.parse(JSON.stringify(nowStatus));
searching.memory.budget_mode = "explicit";
run('dashboard = { ...(dashboard || {}), reload_required: true }; initial = current(); modelChoice = modelInitial');
run('showJob(' + JSON.stringify({ kind: "tune", title: "Finding the fastest memory limit", state: "running", stage: "Loading at 21.0GB", lines: [], seconds: 5, started: 77 }) + ')');
run("showNow(" + JSON.stringify(searching) + "); gate()");
check(element("savebar").hidden && /A memory search is running/.test(element("health").innerHTML) && !/data-hact="auto-measure"/.test(element("health").innerHTML)
  && /Loading at 21\.0 GB/.test(element("job-stage").textContent), "a memory search shows its limit as under test, in the page's GB, and no reload bar");
run('showJob(null); gate()');
check(!element("savebar").hidden && !element("reload-saved").hidden && element("save").hidden && element("apply").hidden, "a saved draft shows Reload alone");
run('dashboard = { ...dashboard, reload_required: false }; gate()');
// Settings Origami has no plan for cannot be saved.
run('globalThis.keptMemory = $("memory").value; autoMemory = false; planRefused = true; $("context").value = $("context").value === "0" ? "1" : "0"; gate(); showEffects()');
check(element("apply").disabled && element("save").disabled && element("effects").hidden, "settings that cannot run are not saved, and promise no speed");
run('planRefused = false; $("context").value = $("context").value === "0" ? "1" : "0"; gate()');

// A model that is no longer where it was added from: the notice names it and leads to Models, where Locate… is.
run('models = [{ id: "q4", origami: { model: "q4", setting: "q4", availability: "missing" } }]; modelInitial = modelChoice = "q4"; notice({ status: "failed", failure: "q4 did not start: No config.json in /x; is this a model folder?" })');
check(element("notice-title").textContent === "q4 is not where it was" && /Locate…/.test(element("notice-text").textContent) && element("notice-setup").textContent === "Open Models" && !element("notice-setup").hidden,
  "a model that moved away says so and leads to Models");
run('models = [{ id: "q4", origami: { model: "q4", setting: "q4", availability: "available" } }]; notice({ status: "failed", failure: "q4 did not start: No config.json in /x; is this a model folder?" })');
check(element("notice-title").textContent === "q4 is found again" && /new location is recorded\. Try again/.test(element("notice-text").textContent) && !/config\.json/.test(element("notice-text").textContent)
  && element("notice-retry").textContent === "Try again", "after Locate… the notice drops the old place's failure and says Try again loads it");
run('notice({ status: "failed", failure: "q4 did not start: out of memory" })');
check(element("notice-title").textContent === "The model did not start", "a new failure after that is shown as it is");

// The page writes only what the user edited. As it happened: a page showed a model with a 24 GB limit (S1); the limit was
// removed elsewhere (S2: automatic); nothing the user then pressed may bring the 24 GB back, and what is saved is S2 plus
// the one edited key.
{
  const G = 1e9, memoryWorld = { revision: "s1", pid: 1, saved: 24 * G, running: 24 * G, status: "ready" };
  const shown = (bytes) => ({ value: bytes ? (bytes / G).toFixed(1) + " GB" : "(fitted to this Mac)", source: bytes ? "file" : "default" });
  const saved = (bytes) => ({ context: { value: "128k" }, quality: { value: "exact" }, reasoning_effort: { value: "none" }, memory_limit: shown(bytes), expert_bits: { value: "4" } });
  const differs = () => memoryWorld.status === "ready" && memoryWorld.saved !== memoryWorld.running;
  routes["v1/dashboard"] = () => answer({ status: memoryWorld.status, version: "3.1", model: memoryWorld.status === "ready" ? { name: "qwen", fingerprint: "q", quantization: "affine" } : "qwen", requests: { recent: [] },
    settings: saved(memoryWorld.running), mac: { gpu: "Apple M2 Pro", memory_bytes: 34359738368 },
    dashboard: { revision: memoryWorld.revision, generation: "f:" + memoryWorld.pid, configured_model: "glm-5.3-flash", active_model: "glm-5.3-flash", host: "h", can_tune: true, can_measure: true, job: null,
      reload_required: differs(), saved_settings: saved(memoryWorld.saved),
      pending_changes: differs() ? [{ key: "memory_limit", label: "Memory", saved: "automatic", running: "24.0 GB" }] : [], pending_revert: differs() ? { memory_limit: "24.0 GB" } : {},
      wired_memory: { physical_memory_bytes: 34359738368, os_headroom_bytes: 4.3 * G, safe_max_budget_bytes: 30 * G, effective_limit_bytes: 30 * G, system_default: true, metal_recommended_bytes: 24 * G,
        configured_budget_bytes: memoryWorld.saved, active_budget_bytes: memoryWorld.running || 25.5 * G, supported: true, requires_raise: false } } });
  routes["v1/plan"] = () => answer({ settings: { context: 131072, quality: "exact", expert_bits: 4, memory_limit: memoryWorld.running || "auto" }, quality: { allowed: true, levels: [{ name: "exact" }, { name: "dynamic" }, { name: "fast" }] },
    context: { limit: 262144, in_memory_up_to: 262144 }, experts: { pack_ready: true, bits: 4 }, memory: { cap_bytes: memoryWorld.running || 25.5 * G, most_bytes: 30 * G, ram_bytes: 34359738368, split: [] }, drives: [], notes: [] });
  routes["v1/jobs"] = { job: null };
  const writes = [];
  for (const path of ["v1/config", "v1/memory/wired", "v1/reload", "v1/start", "v1/stop"]) routes[path] = (_path, options) => { writes.push([path, options.body]); return answer({ saved: true, revision: memoryWorld.revision, reloading: true }, path === "v1/reload" ? 202 : 200); };
  const poll = async () => { for (let i = 0; i < 3; i++) { await run("refresh()"); await settle(); } };
  run('dashboard = null; initial = null; conflict = false; applying = null; reloading = false; job = null; jobPost = null; offline = false; generation = null; revision = null; setupFor = null; smSaved = null; measuring = false; front = true; unanswered = 0; status = null; setNoModel(false)');
  await poll();
  check(run('!autoMemory && memoryStops[+$("memory").value]') === 24 && !run("dirty()"), "a model with a 24 GB limit shows it");
  memoryWorld.revision = "s2"; memoryWorld.saved = null; await poll();
  check(run("autoMemory") && !run("dirty()"), "a limit removed elsewhere shows as Automatic");
  // Saved versus running: one line that says what differs, with both ways out.
  check(!element("pending").hidden && element("pending-text").textContent === "Reload needed: saved settings differ from the running model. Memory: saved automatic, running 24.0 GB."
    && !element("pending-reload").hidden && !element("pending-reload").disabled && !element("pending-discard").hidden, "saved versus running is said in one line, with Reload to apply and Discard the saved change");
  // No silently disabled action: tuning says why it waits and offers the reload.
  check(element("st-go").disabled && element("st-why").textContent === "Reload to apply saved settings first." && !element("st-reload").hidden && !element("st-reload").disabled
    && element("st-go").title === "Reload to apply saved settings first", "tuning blocked by a pending reload says so next to it, with the Reload button");
  // Automatic is a state, not a toggle: pressing it while automatic cannot save the slider's resting place as a limit.
  fire("auto");
  check(run("autoMemory") && run("JSON.stringify(changes())") === "{}", "pressing Automatic while automatic changes nothing");
  // A dial left differing by anything but the user's own move is not an edit.
  run("autoMemory = false; gate()");
  check(run("JSON.stringify(changes())") === "{}" && !run("dirty()"), "a memory value the user did not move is never part of a save");
  run("autoMemory = true");
  await fire("pending-discard"); await settle();
  check(JSON.stringify(writes) === JSON.stringify([["v1/memory/wired", '{"memory_limit":"24.0 GB"}']]), "Discard the saved change saves the running value again");
  writes.length = 0;
  await fire("pending-reload"); await settle();
  check(JSON.stringify(writes) === '[["v1/reload","{}"]]', "Reload to apply sends no settings");
  memoryWorld.running = null; memoryWorld.pid = 2; await poll(); run("reloading = false; gate()"); await poll();
  check(element("pending").hidden && !element("st-go").disabled && element("st-why").textContent === "", "after the reload nothing differs and tuning is free");
  writes.length = 0;
  run('$("reasoning").value = "1"; dialMoved("reasoning")');
  await fire("save"); await settle();
  check(JSON.stringify(writes) === '[["v1/config","{\\"reasoning_effort\\":\\"low\\"}"]]', "saving an unrelated setting sends that key alone: the old limit does not come back");
  await poll(); writes.length = 0;
  await run("power('power')"); await settle(); await run("retryNotice()"); await settle();
  check(writes.every(([path, body]) => ["v1/stop", "v1/start", "v1/reload"].includes(path) && body === "{}") && writes.length === 2, "Stop, Start and Try again send no settings");
  writes.length = 0; run('reloading = false; applying = null; liveState = "ready"; $("reasoning").value = "2"; dialMoved("reasoning")');
  confirmAnswer = true; await run('loadChosen("qwen3.8-flash-next-4bit")'); await settle();
  check(writes.length > 0 && writes[0][1] === '{"model":"qwen3.8-flash-next-4bit"}', "Load sends the model alone");
}

// A stale page cannot overwrite newer settings: a save carries the revision its draft was based on, and a refusal that
// names what changed elsewhere asks Keep mine or Take theirs.
{
  const world = { revision: "r1" }, saves = [];
  routes["v1/dashboard"] = () => answer({ status: "ready", version: "3.1", model: { name: "qwen", fingerprint: "q", quantization: "affine" }, requests: { recent: [] },
    settings: { context: { value: "128k" }, quality: { value: "exact" }, reasoning_effort: { value: "none" } }, mac: { gpu: "Apple M2 Pro", memory_bytes: 34359738368 },
    dashboard: { revision: world.revision, generation: "f:9", configured_model: "glm-5.3-flash", active_model: "glm-5.3-flash", host: "h", can_tune: !world.block, can_measure: true, job: null,
      reload_required: false, pending_changes: [], pending_revert: {}, tune_block: world.block ?? "", saved_settings: { context: { value: "128k" }, quality: { value: "exact" }, reasoning_effort: { value: "none" } } } });
  routes["v1/config"] = (_path, options) => {
    saves.push(options.headers["If-Match"]);
    return options.headers["If-Match"] === world.revision ? answer({ saved: true, revision: world.revision })
      : answer({ error: { message: "Changed elsewhere", code: "settings_conflict" }, changes: [{ key: "reasoning_effort", base: "none", current: "high" }] }, 409);
  };
  const poll = async () => { for (let i = 0; i < 3; i++) { await run("refresh()"); await settle(); } };
  run('dashboard = null; initial = null; conflict = false; review = null; applying = null; reloading = false; job = null; jobPost = null; offline = false; generation = null; revision = null; setupFor = null; smSaved = null; measuring = false; front = true; unanswered = 0; status = null; modelChoice = modelInitial = null; setNoModel(false)');
  await poll();
  run('$("reasoning").value = "1"; dialMoved("reasoning")');
  world.revision = "r2"; await poll();
  check(run("dirty()") && /based on older settings/.test(element("save-state").textContent) && !element("save").disabled, "a draft based on older settings is marked so and can still be saved");
  await fire("save"); await settle();
  check(saves[0] === "r1", "the save carries the revision the draft was based on, not the latest one polled");
  check(!element("review").hidden && /reasoning effort none → high/.test(element("review-text").textContent) && !element("review-keep").disabled && !element("review-take").disabled,
    "the refusal shows what changed elsewhere, with Keep mine and Take theirs");
  await fire("review-keep"); await settle();
  check(saves[1] === "r2" && element("review").hidden && !run("conflict"), "Keep mine saves the draft over the current settings");
  // The server's reason for tuning is shown as it is.
  world.block = "The model is not ready"; run("reloading = false; applying = null"); await poll();
  check(element("st-go").disabled && element("st-why").textContent === "The model is not ready.", "tuning shows the server's reason, the one its job start refuses with");
}

// No silently disabled action: in each state below, every control the page holds back says why (its tooltip, and the
// note next to it where it has one).
{
  const silent = () => [...elements.values()].filter((e) => e.disabled && e.attrs["aria-busy"] !== "true" && !e.title
    && (e.listeners.click || "aria-busy" in e.attrs || "data-held" in e.attrs)).map((e) => e.id);
  const states = {
    "ready": 'offline = false; job = null; jobPost = null; applying = null; reloading = false; conflict = false; review = null; liveState = "ready"',
    "a job is running": 'job = { state: "running", kind: "move", title: "Moving a model" }',
    "loading": 'job = null; liveState = "loading"',
    "stopped": 'liveState = "stopped"',
    "failed": 'liveState = "failed"',
    "a reload is needed": 'liveState = "ready"; dashboard.reload_required = true; dashboard.tune_block = "Reload to apply saved settings first"',
    "settings changed elsewhere": 'conflict = true; review = { changes: [{ key: "context", base: "32k", current: "64k" }], reload: true }',
    "disconnected": 'conflict = false; review = null; offline = true',
  };
  for (const [name, setup] of Object.entries(states)) {
    run(setup + "; gate()");
    const quiet = silent();
    check(quiet.length === 0, "no silently disabled control when " + name + (quiet.length ? ": " + quiet.join(", ") : ""));
  }
  run('offline = false; liveState = "ready"; dashboard.reload_required = false; dashboard.tune_block = ""; gate()');
}

// One name per state: the header says each state in the words the views use.
{
  const said = (setup) => { run(setup + '; live(live.s)'); return element("chip-text").textContent; };
  check(said('job = null; reloading = false; dashboard.reload_required = true; live.s = { status: "ready", requests: { recent: [] } }') === "Reload needed"
    && said('job = { state: "running", kind: "move", title: "Moving a model" }') === "Job running"
    && said('job = null; dashboard.reload_required = false; reloading = true; reloadBy = "apply"; live.s = { status: "ready", requests: { recent: [] } }') === "Loading"
    && said('reloading = false; live.s = { status: "stopping" }') === "Stopping" && said('live.s = { status: "stopped" }') === "Stopped"
    && said('live.s = { status: "failed", failure: "out of memory" }') === "Failed" && said('live.s = { status: "ready", requests: { recent: [] } }') === "Ready"
    && said('live.s = { status: "degraded", degraded: "gone" }') === "Drive gone",
    "the header names each state as the views do: Ready, Loading, Stopping, Stopped, Failed, Reload needed, Job running, Drive gone");
}

// Why the last request was slow: each verdict rule on its synthetic request (the fixture `SlowReason` in the engine is
// checked against too), and the breakdown under the last request's bar.
{
  const cases = JSON.parse(readFileSync(new URL("Fixtures/status/slow-reasons.json", import.meta.url), "utf8"));
  for (const c of cases) {
    const got = run("slowReason(" + JSON.stringify(c.request) + ", " + JSON.stringify(c.status) + ")");
    check(got === c.verdict, "slow reason, " + c.rule + (got === c.verdict ? "" : ": " + got));
  }
  const html = run("whyHtml(" + JSON.stringify(cases[0].request) + ", {})");
  check(/First word 31\.0 s<\/b> · 6,100 prompt tokens read, 2,000 reused from a saved copy \(8,100 could have been reused\)/.test(html)
    && /Writing 10\.0 s/.test(html) && /data-slow="">Slower than usual: 6,100 tokens of the prompt had to be read again because there was too little free disk space/.test(html), "the breakdown names the prompt read, the restore's reason and the verdict");
  const calm = run("whyHtml(" + JSON.stringify(cases[cases.length - 1].request) + ", {})");
  check(/GPU 8\.0 s · waiting on the model&#39;s drive 1\.0 s/.test(calm) && /\(1,950 could have been reused, but its saved copy had been replaced by newer conversations\)/.test(calm) && /<p class="verdict">Nothing unusual<\/p>/.test(calm), "an ordinary request: GPU, waits, and nothing unusual" + (/Nothing unusual/.test(calm) ? "" : ": " + calm));
}

// Slowed by: the notice at the top of Now shows the status's list (each fixture case's `slowed_by`, the lists the engine's
// `SlowedBy` gives for `origami status`), one line per condition, what then the fix, also while a prompt is read; an
// empty list hides it.
{
  const cases = JSON.parse(readFileSync(new URL("Fixtures/status/slowed-by.json", import.meta.url), "utf8"));
  const order = cases.find((c) => /^the order/.test(c.rule)).slowed_by;
  const reading = { status: "ready", queue: { busy: true, queued: 0 }, requests: { recent: [], current: { phase: "prefill", prompt_tokens: 50392, completed_prompt_tokens: 12000, reused_tokens: 0 } } };
  run("live(" + JSON.stringify({ ...reading, slowed_by: order }) + ")");
  const html = element("slowed-list").innerHTML, items = html.split("<li>").slice(1);
  check(!element("slowed").hidden && /^Reading a 50,392-token prompt/.test(element("live-text").textContent) && items.length === order.length && order.length >= 3
    && items.every((line, i) => line.startsWith("<b>" + run("esc(" + JSON.stringify(order[i].text) + ")") + ".</b> ")),
    "while a 50,392-token prompt is read, the notice lists each condition in the status's order, what first" + (items.length ? "" : ": " + html));
  check(cases.every((c) => { run("live(" + JSON.stringify({ ...reading, slowed_by: c.slowed_by }) + ")"); return !element("slowed").hidden && (c.slowed_by.length ? element("slowed-list").innerHTML.split("<li>").length - 1 === c.slowed_by.length && element("slowed").dataset.calm === "" : /Nothing is slowing Origami down right now/.test(element("slowed-list").innerHTML) && element("slowed").dataset.calm === "1"); }),
    "every fixture case shows one line per condition, and the quiet line when its list is empty");
  const drive = cases.find((c) => /drive reads slowly/.test(c.rule)).slowed_by;
  run("live(" + JSON.stringify({ status: "ready", requests: { recent: [] }, slowed_by: drive }) + ")");
  check(/run <code>origami refresh<\/code>: it rewrites/.test(element("slowed-list").innerHTML) && !element("slowed").hidden, "a fix's command shows as code, also when idle");
  run('live({ status: "ready", requests: { recent: [] }, slowed_by: [] })');
  check(!element("slowed").hidden && element("slowed").dataset.calm === "1" && /Nothing is slowing Origami down right now/.test(element("slowed-list").innerHTML), "nothing slowing a loaded model: the quiet line, not an empty page");
  run('live({ status: "stopped", slowed_by: [] })');
  check(element("slowed").hidden, "no model loaded: no notice");
  // The quiet line does not contradict a last reply that waited on a drive: it says the delay was that reply's own.
  const waited = JSON.parse(readFileSync(new URL("Fixtures/status/slow-reasons.json", import.meta.url), "utf8")).find((c) => /^GLM or DeepSeek waiting on the model/.test(c.rule)).request;
  run("live(" + JSON.stringify({ status: "ready", requests: { recent: [waited] }, slowed_by: [] }) + ")");
  check(/Nothing is slowing Origami down right now\. The last reply was slower for a reason of its own/.test(element("slowed-list").innerHTML), "the quiet line owns up to a slow last reply");
  // The headline speed: the last replies' tokens over their writing time (two short slow replies do not halve it), the
  // reply being written while there is one, and warm-up named beside it.
  const reply = (tokens, tps) => ({ decode_tps: tps, completion_tokens: tokens, mode: "exact" });
  const speedOf = (extra) => { run("base = null; status = " + JSON.stringify({ status: "ready", settings: { quality: { value: "exact" } }, slowed_by: [], ...extra, requests: { recent: [reply(120, 5), reply(150, 6), reply(1500, 11)], ...(extra.requests || {}) } }) + "; showSpeed()"); return element("speed").textContent + " | " + element("basis").textContent; };
  const idle = speedOf({}), warm = speedOf({ slowed_by: [{ id: "warming_up", text: "", fix: "" }] }), busy = speedOf({ requests: { current: { phase: "decode", completion_tokens: 182, decode_tps: 10.3 } } });
  check(idle === "9.5 | Writing speed · average of the last 3 replies" && warm === "9.5 | Writing speed · average of the last 3 replies · warming up, so slower than usual at first"
    && busy === "10.3 | Writing speed right now · the last 3 replies averaged 9.5", "the headline speed weighs replies by their tokens, follows the reply being written and names warm-up: " + [idle, warm, busy].join(" / "));
  // Health > Memory says first whether the limit is what slows replies: not while warming up, yes when replies wait
  // on drives at full memory (the server's finding and its fix), else that it is enough.
  const calm = { ...JSON.parse(JSON.stringify(nowStatus)), memory: { cap_bytes: 22.6e9, budget_bytes: 22.6e9, footprint_bytes: 18.3e9, budget_mode: "explicit" }, memory_learned: {} };
  const memoryLine = (slowed) => { run("showNow(" + JSON.stringify({ ...calm, slowed_by: slowed }) + ")"); return element("health").innerHTML; };
  const full = cases.find((c) => /more of the model than fits/.test(c.rule)).slowed_by.find((c) => c.id === "memory_full");
  check(/warming up, so the limit is not what slows it now/.test(memoryLine([{ id: "warming_up", text: "", fix: "" }]))
    && /The limit is holding Origami back: the last replies spent about \d+% of their writing time[^<]*\. Nothing to change: /.test(memoryLine([full]))
    && /The limit is enough right now: replies hardly wait for the drive\./.test(memoryLine([])), "Health > Memory says whether the limit holds Origami back");
  run("showNow(" + JSON.stringify(nowStatus) + ")");
  // A page left open across an upgrade reloads once, and not over unsaved settings.
  run('versionSeen("9.9.8"); versionSeen("9.9.8")');
  const before = context.location.reloads;
  run('versionSeen("9.9.8"); versionSeen("9.9.9"); versionSeen("9.9.9")');
  check(context.location.reloads - before === 1, "the page reloads once when the server's version changes: " + (context.location.reloads - before));
}

// The copy control lists every model's copies on a shared drive, each with its own keep box, and the swap backlog and the
// disk budget are shown.
{
  const budget = [{ volume: "/Volumes/Fast", free_bytes: 300e9, reserve_bytes: 10e9, given_back_bytes: 2e9, give_backs: 1, saves_skipped: 2, holders: [
    { name: "hot copy", model: "glm-5.3-flash", loaded: true, bytes: 60e9, took: [{ from: "deepseek-v4.1-flash", bytes: 5e9, at: "2026-11-17T10:00:00Z" }] },
    { name: "hot copy", model: "deepseek-v4.1-flash", loaded: false, bytes: 40e9, cap: "auto", floor_bytes: 22e9, keep_when_idle: false,
      gave: [{ to: "glm-5.3-flash", bytes: 5e9, records: 300, at: "2026-11-17T10:00:00Z" }] },
    { name: "hot copy", model: "qwen3.8-flash-next-4bit", loaded: false, bytes: 30e9, cap: 30e9, floor_bytes: 15e9, keep_when_idle: true, gave: [] }] },
    { volume: "/", free_bytes: 50e9, reserve_bytes: 10e9, holders: [{ name: "prompt cache", bytes: 3e9 }] }];
  const shared = run("sharedCopies(" + JSON.stringify({ disk_budget: budget, settings: { hot_copy_keep: { value: "false" } } }) + ")");
  const boxes = [...shared.matchAll(/data-keep-model="([^"]+)"( checked)?/g)].map((m) => m[1] + (m[2] ? "+" : "-"));
  check(boxes.join() === "deepseek-v4.1-flash-,qwen3.8-flash-next-4bit+" && /glm-5\.3-flash<span class="sub">loaded/.test(shared), "a keep box per idle model on the shared drive, each as saved (the loaded one's is the box above): " + boxes.join());
  check(/Copies on Fast/.test(shared) && /keeps 22 GB when another model grows · gave 5\.0 GB/.test(shared) && /took 5\.0 GB/.test(shared) && /keeps 30 GB when another model grows/.test(shared)
    && !/prompt cache/.test(shared), "each row says its size, what an idle copy keeps, and what it gave or took; a drive without copies is left out");
  sent.length = 0; routes["v1/hot-copy"] = { hot_copy: null, hot_copies: [], restart: false };
  element("x-shared").listeners.change({ target: { dataset: { keepModel: "deepseek-v4.1-flash" }, checked: true } });
  await settle();
  const post = sent.find((r) => r.path === "v1/hot-copy");
  check(post && JSON.stringify(JSON.parse(post.body)) === '{"keep_when_idle":true,"model":"deepseek-v4.1-flash"}', "ticking an idle model's box saves keep_when_idle for that model");
  const row = run("sourceRow(" + JSON.stringify({ role: "hot", state: "ready", drive: "Fast", records: 900, bytes: 40e9, backlog_records: 120, backlog_bytes: 1.9e9, backlog_uses: 3400 }) + ", {})");
  check(/120 swaps wait for the hourly allowance \(1\.9 GB, 3,400 uses\)/.test(row), "the hot copy row shows its swap backlog");
  const kept = run("diskBudget(" + JSON.stringify({ disk_budget: budget }) + ")");
  check(/Fast: keeps 10 GB free, .* free now · copies gave back 2\.0 GB for saves · 2 saves skipped for space/.test(kept), "the disk budget per drive where storage is shown: " + kept);
}

// Presets: a row each with its settings in words; Use posts it with the page's revision; saving one waits for a draft
// changed elsewhere to be reviewed.
{
  const presets = [{ name: "Long chats", values: { model: "glm-5.3-flash", context: "128k", quality: "dynamic", reasoning_effort: "high", memory_limit: "25.500000000GB", sampling: "recommended" } }];
  run(`dashboard = { ...dashboard, presets: ${JSON.stringify(presets)} }; revision = "r-presets"; offline = false; applying = null; reloading = false; job = null; jobPost = null; conflict = false; liveState = "ready"; showPresets()`);
  const html = element("presets").innerHTML;
  check(/<strong>Long chats<\/strong><span class="sub">glm-5\.3-flash · 128k context · dynamic · high thinking · 25\.5 GB memory<\/span>/.test(html)
    && /data-preset="use" data-name="Long chats"/.test(html) && /data-preset="delete"/.test(html) && element("presets-count").textContent === "1", "a preset row names its model and settings in words: " + html);
  sent.length = 0; routes["v1/presets/Long%20chats"] = { saved: ["context"], presets };
  await element("presets").listeners.click({ target: { closest: () => ({ dataset: { preset: "use", name: "Long chats" }, disabled: false }) } }); await settle();
  const use = sent.find((r) => r.path === "v1/presets/Long%20chats");
  check(use && use.method === "POST" && use.headers["If-Match"] === "r-presets" && /^Using Long chats · reload/.test(element("preset-note").textContent),
    "Use posts the preset with the page's revision and says Reload runs it: " + element("preset-note").textContent);
  run("conflict = true; gate()");
  check(element("preset-save").disabled && /Settings changed elsewhere/.test(element("preset-save").title), "saving a preset waits while settings changed elsewhere");
  run("conflict = false; gate()");
}

process.exit(failures ? 1 : 0);
