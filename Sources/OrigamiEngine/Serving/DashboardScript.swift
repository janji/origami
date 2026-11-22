enum DashboardScript {
    static let js = #"""
        "use strict";
        const $ = (id) => document.getElementById(id);
        const fmt = new Intl.NumberFormat("en-US"), one = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1, minimumFractionDigits: 1 });
        // Memory amounts in decimal GB, as settings, the CLI and the logs read and print them ("25.5GB" reads 25.5 GB here
        // too); only this Mac's own size is named as macOS does (binary: a "32 GB" Mac). Drives as Finder does (decimal GB).
        const GIB = 1073741824, GB = 1e9;
        const gb = (bytes) => one.format(bytes / GB) + " GB", disk = (bytes) => one.format(bytes / 1e9) + " GB";
        const memoryText = (text) => String(text || "").replace(/(\d+(?:\.\d+)?) ?GB\b/g, (all, n) => gb(n * 1e9));
        const esc = (text) => String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
        const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
        // iPadOS before 15.4 has no <dialog>.showModal: the folder browser and the saved-prompt viewer open as a fixed overlay
        // there, close with Escape and say so with a close event, as a modal dialog does.
        document.querySelectorAll("dialog").forEach((dialog) => {
          if (typeof dialog.showModal === "function") return;
          if (!("open" in dialog)) Object.defineProperty(dialog, "open", { get: () => dialog.hasAttribute("open") });
          dialog.showModal = () => {
            dialog.setAttribute("open", ""); dialog.setAttribute("data-fallback", "");
            const first = dialog.querySelector("button, input"); if (first) first.focus();
          };
          dialog.close = () => { if (!dialog.hasAttribute("open")) return; dialog.removeAttribute("open"); dialog.dispatchEvent(new Event("close")); };
          dialog.addEventListener("keydown", (event) => { if (event.key === "Escape") dialog.close(); });
        });
        // The server's key when it has one: `origami dashboard` (and the QR code under Access) opens this page with it after
        // #key=, which never reaches a server or a log, or it is pasted when the page asks; this browser keeps it and sends it
        // with every request.
        const keyStore = (key) => { try { if (key !== undefined) localStorage.setItem("origami-key", key); return localStorage.getItem("origami-key") || ""; } catch (error) { return key || ""; } };
        let apiKey = (() => {
          const given = /(?:^#|&)key=([^&]*)/.exec(location.hash);
          if (given) history.replaceState(null, "", location.pathname);
          return keyStore(given ? decodeURIComponent(given[1]) : undefined);
        })();
        // `draftBase`: the revision the dials' draft was based on, sent with its save so the server refuses it when a key it
        // writes changed since; `review`: what changed elsewhere, from that refusal, until Keep mine or Take theirs.
        let workspace = "run", revision = null, generation = null, dashboard = null, conflict = false, syncedRevision = null, draftBase = null, review = null;
        let observed = 0, unanswered = 0, pollBusy = false, pollSeq = 0, auxAt = 0, storageAt = 0, setupSeq = 0;
        const api = (path, options = {}) => fetch(path, { cache: "no-store", ...(typeof AbortSignal !== "undefined" && AbortSignal.timeout ? { signal: AbortSignal.timeout(15000) } : {}), ...options,
          headers: { ...(options.headers || {}), ...(apiKey ? { Authorization: "Bearer " + apiKey } : {}) } });
        const post = (path, body, base = revision) => api(path, { method: "POST", headers: { "Content-Type": "application/json", ...(base && ["v1/config", "v1/hot-copy", "v1/memory/wired"].includes(path) ? { "If-Match": base } : {}) },
          // The administrator dialog and a whole reply (Measure) take longer than the 15 s every other call gets.
          ...(SLOW_POSTS[path] && typeof AbortSignal !== "undefined" && AbortSignal.timeout ? { signal: AbortSignal.timeout(SLOW_POSTS[path]) } : {}), body: JSON.stringify(body) });
        const SLOW_POSTS = { "v1/memory/wired": 135000, "v1/chat/completions": 300000 };
        const reduced = matchMedia("(prefers-reduced-motion: reduce)");
        function ago(when) {
          const s = Math.max(0, (Date.now() - new Date(when).getTime()) / 1000);
          return s < 60 ? "just now" : s < 3600 ? Math.round(s / 60) + " min ago" : s < 86400 ? Math.round(s / 3600) + " h ago" : Math.round(s / 86400) + " d ago";
        }
        // A few-token answer measures start-up and saving, not writing speed.
        const measured = (r) => r.decode_tps > 0 && r.completion_tokens >= 16;
        const endings = { length: "hit the length limit", tool_calls: "called a tool", error: "failed", cancelled: "cancelled", gpu_interrupted: "interrupted by macOS (GPU busy)" };
        // What the narrow table keeps under the prompt: the endings that say a request did not go as asked.
        const alarming = ["error", "cancelled", "gpu_interrupted"];
        const clock = (when) => new Date(when).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

        // The header art: a paper crane folded from flat facets, drawn through a character-density ramp.
        const RAMP = " .:-=+*#%@";
        const split = (a, b, tip, light, dark) => { const m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]; return [[[a, m, tip], light], [[m, b, tip], dark]]; };
        const CRANE = [
          ...split([47, 36], [61, 37], [76, 5], 0.70, 0.55),              // far wing
          ...split([56, 37], [67, 41], [96, 11], 0.92, 0.72),             // tail
          [[[33, 41], [50, 33], [50, 41]], 1.00], [[[50, 33], [67, 41], [50, 41]], 0.80],
          [[[33, 41], [50, 41], [50, 49]], 0.68], [[[50, 41], [67, 41], [50, 49]], 0.52],   // body
          ...split([34, 41], [44, 37], [10, 9], 0.98, 0.76),              // neck
          [[[10, 9], [3, 15], [13.5, 13]], 1.00],                         // head, folded down
          ...split([40, 37], [58, 37], [32, 0], 1.00, 0.84),              // near wing
        ];
        function inside(x, y, poly) {
          let hit = false;
          for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
            const [x1, y1] = poly[i], [x2, y2] = poly[j];
            if ((y1 > y) !== (y2 > y) && x < ((x2 - x1) * (y - y1)) / (y2 - y1) + x1) hit = !hit;
          }
          return hit;
        }
        function edge(x, y, poly) {
          let best = 1e9;
          for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
            const [x1, y1] = poly[j], [x2, y2] = poly[i], dx = x2 - x1, dy = y2 - y1;
            const t = Math.max(0, Math.min(1, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)));
            best = Math.min(best, Math.hypot(x - x1 - t * dx, y - y1 - t * dy));
          }
          return best;
        }
        const noise = (c, r) => { const v = Math.sin(c * 127.1 + r * 311.7) * 43758.5453; return v - Math.floor(v); };
        let art = null, sweeping = false;
        function layoutArt() {
          const canvas = $("art"), ctx = canvas.getContext("2d"), dpr = window.devicePixelRatio || 1;
          const width = canvas.clientWidth, height = canvas.clientHeight, size = width < 420 ? 5 : 6.5;
          canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          ctx.font = size + "px " + css("--mono"); ctx.textBaseline = "top";
          const cw = ctx.measureText("M").width, lh = size * 1.18, cols = Math.floor(width / cw), rows = Math.floor(height / lh);
          // The crane fills the height; a character cell is about twice as tall as it is wide.
          const scale = Math.min(cols * 0.9 / 100, (rows * 2) / 54), left = Math.round(cols / 2 - 50 * scale), top = (rows - 52 * scale / 2) / 2;
          const cells = [];
          for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
            const x = (c + 0.5 - left) / scale, y = ((r + 0.5 - top) * 2) / scale;
            let face = null;
            for (const f of CRANE) if (inside(x, y, f[0])) face = f;
            if (face) {
              // Light falls from the upper left across each face; folds read as darker creases.
              const crease = edge(x, y, face[0]) < 0.9 / scale, light = 0.1 * (0.5 - (x + y) / 150);
              cells.push([c, r, Math.max(0.14, face[1] + light - (crease ? 0.28 : 0) + (noise(c, r) - 0.5) * 0.1), x, y]);
            } else if (c % 2 === 0 && r % 2 === 0) {
              // A halftone field that fades away from the crane and before the canvas edges.
              const fade = Math.min(1, Math.min(c, cols - 1 - c) / (cols * 0.12), Math.min(r, rows - 1 - r) / (rows * 0.2));
              const glow = Math.max(0, 1 - Math.hypot((x - 50) / 90, (y - 26) / 60)) * fade;
              if (glow > 0.04) cells.push([c, r, -glow, x, y]);
            }
          }
          art = { ctx, width, height, cw, lh, cells };
        }
        function drawArt(phase) {
          if (!art) return;
          const parse = (v) => v.split(",").map(Number), { ctx, width, height, cw, lh, cells } = art;
          const [h0, h1, h2] = parse(css("--art-hi")), [l0, l1, l2] = parse(css("--art-lo")), [d0, d1, d2] = parse(css("--art-dot")), bg = parse(css("--art-bg"));
          ctx.clearRect(0, 0, width, height);
          for (const [c, r, v0, x, y] of cells) {
            if (v0 < 0) {
              const g = -v0;
              ctx.fillStyle = "rgb(" + Math.round(bg[0] + (d0 - bg[0]) * g) + "," + Math.round(bg[1] + (d1 - bg[1]) * g) + "," + Math.round(bg[2] + (d2 - bg[2]) * g) + ")";
              ctx.fillText(".", c * cw, r * lh); continue;
            }
            // While a request runs or the model loads, a band of light crosses the paper.
            const v = phase == null ? v0 : Math.min(1, v0 + 0.35 * Math.exp(-(((x + y * 0.6) - phase * 150 + 20) ** 2) / 120));
            const k = Math.min(1, Math.max(0, v));
            ctx.fillStyle = "rgb(" + Math.round(l0 + (h0 - l0) * k) + "," + Math.round(l1 + (h1 - l1) * k) + "," + Math.round(l2 + (h2 - l2) * k) + ")";
            ctx.fillText(RAMP[Math.max(1, Math.min(RAMP.length - 1, Math.round(k * (RAMP.length - 1))))], c * cw, r * lh);
          }
        }
        function sweep() {
          if (!sweeping || reduced.matches) { drawArt(null); return; }
          drawArt((performance.now() % 3200) / 3200);
          setTimeout(sweep, 90);
        }
        function setSweeping(on) {
          if (on === sweeping) return;
          sweeping = on;
          if (on) sweep(); else drawArt(null);
        }

        // This Mac's memory as an ASCII strip: Origami's fixed parts, the experts held, the context, and what macOS and other
        // apps keep. When a dial moves, the boundaries re-fold column by column toward their new place.
        // Stopped, the columns Origami held are the freed part.
        const PARTS = { freed: "Freed by the stop", fixed: "Origami", experts: "Experts held", context: "Context", others: "macOS and apps" }, ORDER = ["freed", "fixed", "experts", "context"];
        const macGB = (bytes) => Math.round(bytes / GIB) + " GB";
        let strip = { from: null, to: null, start: 0 };
        function stripColumns(split, cols, ram) {
          const ends = [];
          let at = 0, end = 0;
          for (const part of ORDER) {
            const p = split.find((x) => x.part === part), bytes = p ? p.bytes : 0;
            at += bytes;
            // A part that holds anything shows at least one column.
            end = Math.min(cols, Math.max(end + (bytes > 0 ? 1 : 0), Math.round(cols * Math.min(1, at / ram))));
            ends.push(end);
          }
          return ends;
        }
        function drawStrip(now) {
          const canvas = $("memstrip"), ctx = canvas.getContext("2d"), dpr = window.devicePixelRatio || 1, width = canvas.clientWidth, height = canvas.clientHeight;
          if (!strip.to || !width) return;
          canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          ctx.font = "12px " + css("--mono"); ctx.textBaseline = "top";
          const cw = ctx.measureText("M").width, cols = Math.max(10, Math.floor(width / cw)), to = stripColumns(strip.to.split, cols, strip.to.ram);
          const from = strip.from ? stripColumns(strip.from.split, cols, strip.from.ram) : to;
          const t = reduced.matches || !strip.from ? 1 : Math.min(1, (now - strip.start) / 420);
          // Columns flip one after another from the left, as folded paper turns over.
          const sweepCol = Math.floor(t * cols);
          const ink = css("--ink"), accent = "rgb(" + css("--art-accent") + ")", dim = "rgb(" + css("--art-lo") + ")";
          for (let c = 0; c < cols; c++) {
            const ends = c < sweepCol ? to : from;
            const part = ORDER[ends.findIndex((end) => c < end)] || "others";
            for (let r = 0; r < 3; r++) {
              const y = r * 14;
              // The legend's characters, one per part.
              if (part === "freed") { ctx.fillStyle = css("--ink-2"); ctx.fillText("-", c * cw, y); }
              else if (part === "fixed") { ctx.fillStyle = ink; ctx.fillText("#", c * cw, y); }
              else if (part === "experts") { ctx.fillStyle = accent; ctx.fillText("@", c * cw, y); }
              else if (part === "context") { ctx.fillStyle = accent; ctx.fillText("/", c * cw, y); }
              else if (c % 2 === 0) { ctx.fillStyle = dim; ctx.fillText(".", c * cw, y); }
            }
          }
          if (t < 1) requestAnimationFrame(drawStrip);
        }
        function showMemory(m) {
          const next = { split: m.split, ram: m.ram_bytes };
          strip = { from: strip.to, to: next, start: performance.now() };
          requestAnimationFrame(drawStrip);
          $("mem-label").textContent = "This Mac's memory · " + macGB(m.ram_bytes);
          $("memstrip").setAttribute("aria-label", "This Mac's " + macGB(m.ram_bytes) + ": " + m.split.map((p) => (PARTS[p.part] || p.label) + " " + gb(p.bytes)).join(", "));
          // Each key is the strip's own character for its part.
          const glyph = { freed: "-", fixed: "#", experts: "@", context: "/", others: "." };
          $("keys").innerHTML = m.split.filter((p) => p.bytes > 0).map((p) => '<span><i class="k-' + esc(p.part) + '" aria-hidden="true">' + (glyph[p.part] || "#") + "</i>" + esc(PARTS[p.part] || p.label) + " <b>" + gb(p.bytes) + "</b></span>").join("");
        }

        // Stopped: Origami holds no memory; what the stopped model held shows as freed, where Origami's parts were.
        function stoppedMemory(s) {
          const ram = s.ram_bytes || (strip.to && strip.to.ram);
          if (!ram) return;
          const freed = Math.min(ram, s.freed_bytes || 0);
          $("mem").hidden = false;
          showMemory({ ram_bytes: ram, split: [{ part: "freed", bytes: freed }, { part: "others", bytes: ram - freed }] });
          $("mem-label").textContent = "This Mac's memory · " + macGB(ram) + " · Origami holds none";
        }

        // The dials. Each holds a list of stops; the plan endpoint answers for the stop each one points at.
        // Quality: the levels the running model offers (its plan's or status's list), each with its readout.
        const LEVEL_LABELS = { exact: "Exact", dynamic: "Dynamic", faster: "Faster", fastest: "Fastest", "dynamic-fastest": "Dynamic fastest" };
        let LEVELS = [["exact", "Exact"]];
        function setLevels(names) {
          LEVELS = ["exact"].concat((names || []).filter((n) => n !== "exact" && LEVEL_LABELS[n])).map((n) => [n, LEVEL_LABELS[n]]);
          $("quality").max = String(LEVELS.length - 1);
          const floor = $("st-floor").value;
          $("st-floor").innerHTML = LEVELS.slice(1).map((l) => '<option value="' + l[0] + '">' + l[1] + "</option>").join("");
          if (LEVELS.some((l) => l[0] === floor)) $("st-floor").value = floor;
        }
        const chosenLevel = () => LEVELS[+$("quality").value] || LEVELS[LEVELS.length - 1];
        // A level the model does not offer runs as its mildest fast one, as the server runs it; anything else shows as exact.
        const levelAt = (name) => { const at = LEVELS.findIndex((l) => l[0] === name); return at >= 0 ? at : LEVEL_LABELS[name] && name !== "exact" && LEVELS.length > 1 ? 1 : 0; };
        const CONTEXTS = [[8192, "8K"], [32768, "32K"], [65536, "64K"], [131072, "128K"], [262144, "256K"]];
        let base = null, memoryStops = [], autoMemory = true, ready = false, kind = "qwen", modelChoice = null, modelInitial = null, initial = null, status = null, launched = [];
        // The dashboard's job (preparing smaller experts, adding a model), models found on this Mac, and the model being added.
        // `jobLive`: the job this page saw running or started (kind and start time), whose end alone chooses or previews anything;
        // `modelByJob`: the model menu's choice came from an add job, not from the menu itself.
        let job = null, jobLive = null, modelByJob = false, addedNow = null, found = null, models = [], addSource = null;
        const jobId = (j) => j && j.kind + "@" + j.started;
        const running = () => !!job && job.state === "running";
        // No model runs (none yet, or one that did not start): only the model menu and what fixes it show.
        let noModel = false;
        function setNoModel(on) {
          noModel = on;
          $("tune").hidden = on; $("setup-empty").hidden = !on;
          $("lede").textContent = dashboard ? dashboard.host : "";
          document.querySelectorAll(".running-only").forEach((el) => { el.hidden = on; });
          if (on) { $("experts").hidden = true; hotChip(null); }
          showStart(); showFolders();
        }
        function showStart() { $("use-model").hidden = !noModel || !modelChoice || modelChoice === modelInitial || noticeKind === "missing"; gate(); }
        const driveLabel = (name) => name === "internal drive" ? "This Mac" : name;
        function ticks(id, labels, on, far = []) {
          const n = labels.length;
          $(id).innerHTML = labels.map((text, i) => '<span class="' + (i === on ? "on" : far.includes(i) ? "far" : "") + '" style="left:calc(7px + (100% - 14px) * ' + (n > 1 ? i / (n - 1) : 0) + ')">' + esc(text) + "</span>").join("");
        }
        function setFill(input) { const max = +input.max || 1; input.style.setProperty("--fill", (100 * (+input.value) / max) + "%"); }
        // The thinking a reply does when its request asks for none (`reasoning_effort`); a request's own level still wins.
        const EFFORT_INFO = [["none", "None", "Answers at once, without thinking first."], ["low", "Low", "Thinks briefly before answering."],
          ["medium", "Medium", "Thinks before answering: better on harder questions, slower to start."],
          ["high", "High", "Thinks at length: best on hard problems, takes much longer."], ["max", "Max", "Thinks as long as it needs."]];
        // The stops are the levels the running model tells apart (`reasoning_efforts` in `v1/models`), each once.
        const EFFORTS = EFFORT_INFO.slice();
        const effortRank = (name) => ["none", "minimal", "low", "medium", "high", "xhigh", "max"].indexOf(name);
        // A level's stop: its own, else the highest listed level below it (`minimal` is `none`; `xhigh` is Qwen's `high`).
        const effortIndex = (name) => { const rank = effortRank(name); let at = 0;
          EFFORTS.forEach((e, i) => { if (e[0] === name || effortRank(e[0]) <= rank) at = i; }); return at; };
        function setEfforts(names) {
          const list = EFFORT_INFO.filter((e) => names.includes(e[0]));
          if (!list.length || list.map((e) => e[0]).join() === EFFORTS.map((e) => e[0]).join()) return;
          const name = EFFORTS[+$("reasoning").value][0];
          EFFORTS.splice(0, EFFORTS.length, ...list);
          $("reasoning").max = String(EFFORTS.length - 1);
          $("reasoning").value = String(effortIndex(name));
          setFill($("reasoning"));
          const effort = EFFORTS[+$("reasoning").value];
          $("o-reasoning").textContent = effort[1];
          ticks("t-reasoning", EFFORTS.map((x) => x[1]), +$("reasoning").value);
          $("h-reasoning").textContent = effort[2] + " Apps that ask for a level keep theirs.";
        }
        function setReasoning(s) {
          const at = s && s.settings && s.settings.reasoning_effort;
          $("reasoning").value = String(effortIndex(at ? at.value : "none"));
          $("reasoning").disabled = launched.includes("reasoning"); $("d-reasoning").toggleAttribute("data-off", $("reasoning").disabled);
        }
        function current() {
          return { context: CONTEXTS[+$("context").value][0], quality: chosenLevel()[0],
                   memory: autoMemory ? "auto" : memoryStops[+$("memory").value], reasoning: EFFORTS[+$("reasoning").value][0] };
        }
        function sizeText(bytes) { return Math.round(bytes / GIB) + " GB"; }
        const wiredMemory = () => dashboard && dashboard.wired_memory;
        // Explicit decimal GB to byte precision: unitless values mean GB to Settings.parseBytes, and rounding a final
        // slider stop to hundredths of GB could exceed this Mac's safe maximum.
        const memorySetting = (value) => value === "auto" ? "auto" : (Math.floor(value * GB) / 1e9).toFixed(9) + "GB";
        function memoryRange(maximum, selected) {
          const most = maximum / GB;
          memoryStops = [];
          for (let g = Math.min(8, Math.floor(most)); g <= most; g += 1) memoryStops.push(g);
          if (!memoryStops.length || memoryStops[memoryStops.length - 1] < most) memoryStops.push(most);
          $("memory").max = String(Math.max(0, memoryStops.length - 1));
          const at = memoryStops.findIndex(g => g * GB >= selected);
          $("memory").value = String(at < 0 ? memoryStops.length - 1 : at);
        }
        function showWiredMemory() {
          const w = wiredMemory(); $("wired-memory").hidden = !w;
          if (!w) return;
          const amount = (value, absent = "Unavailable") => typeof value === "number" && value > 0 ? gb(value) : absent;
          $("wired-saved").textContent = amount(w.configured_budget_bytes, "Automatic");
          $("wired-active").textContent = amount(w.active_budget_bytes, "Not loaded");
          $("wired-usable").textContent = amount(w.usable_budget_bytes, "Not loaded");
          $("wired-footprint").textContent = amount(w.physical_footprint_bytes, "Not loaded");
          $("wired-system").textContent = w.system_default ? "Automatic · " + amount(w.effective_limit_bytes) : amount(w.system_limit_bytes);
          $("wired-metal").textContent = amount(w.metal_recommended_bytes);
          $("wired-help").textContent = w.supported ? "Save sets Origami’s target and raises the macOS ceiling if needed. Raising it asks for an administrator on this Mac. The macOS change lasts until reboot; the saved limit is kept. Footprint includes all model-process memory."
            : "The macOS memory limit for the GPU is unavailable on this Mac. Origami’s saved budget remains separate from Metal’s recommendation.";
          const changedMemory = initial && current().memory !== initial.memory;
          const target = current().memory === "auto" ? null : current().memory * GB;
          const effective = w.effective_limit_bytes ?? w.system_limit_bytes;
          $("wired-state").textContent = changedMemory ? (w.supported && target > effective ? "Saving this budget requires administrator approval on this Mac. " : "") + "Save & reload to use it."
            : w.requires_raise ? "The saved budget exceeds the current macOS ceiling. Restore the ceiling after a reboot."
            : dashboard.reload_required ? pendingText()
            : w.temporarily_reduced ? "The allowance is temporarily reduced to relieve memory pressure or stay within the loaded limit. It recovers when macOS is stable." : "";
          $("wired-restore").hidden = !w.supported || !w.requires_raise;
        }
        function drawDials(plan) {
          const v = current(), c = plan ? plan.context : null, inMemory = c ? c.in_memory_up_to : 262144;
          $("o-context").textContent = fmt.format(v.context / 1024) + "K tokens";
          // Only the lengths the model takes (GLM and DeepSeek stop at their own).
          const stops = CONTEXTS.slice(0, +$("context").max + 1);
          ticks("t-context", stops.map((x) => x[1]), +$("context").value, stops.map((x, i) => x[0] > inMemory ? i : -1).filter((i) => i >= 0));
          $("h-context").textContent = c && c.paged ? "Above " + fmt.format(inMemory / 1024) + "K this Mac keeps the context in a file on disk: long replies slow down."
            : v.context <= 8192 ? "Questions and short chats." : v.context <= 32768 ? "Code, agents and long system prompts." : "Whole documents; prompts take minutes to read.";
          $("o-quality").textContent = chosenLevel()[1];
          ticks("t-quality", LEVELS.map((x) => x[1]), +$("quality").value);
          const effort = EFFORTS[+$("reasoning").value];
          $("o-reasoning").textContent = effort[1];
          ticks("t-reasoning", EFFORTS.map((x) => x[1]), +$("reasoning").value);
          $("h-reasoning").textContent = effort[2] + " Apps that ask for a level keep theirs.";
          const level = plan && (plan.quality.levels || []).find((l) => l.name === v.quality);
          $("h-quality").textContent = v.quality === "exact" ? "The model's own replies." : level && level.quality ? level.quality.split(";")[0].replace(/^./, (c) => c.toUpperCase()) + "." : "Replies can differ slightly from exact.";
          const w = wiredMemory(), m = plan ? plan.memory : w ? { cap_bytes: w.active_budget_bytes || w.configured_budget_bytes || w.safe_max_budget_bytes, ram_bytes: w.physical_memory_bytes } : null;
          if (m) {
            // A saved limit between two stops shows as saved until the dial moves (Apply sends it only then).
            const kept = !autoMemory && storedMemory && initial && v.memory === initial.memory;
            $("o-memory").textContent = autoMemory ? "Automatic" + (plan && plan.memory.cap_bytes ? " · " + gb(plan.memory.cap_bytes) : "") : gb(kept ? storedMemory : v.memory * GB) + " of " + macGB(m.ram_bytes);
            const first = memoryStops[0], last = memoryStops[memoryStops.length - 1];
            ticks("t-memory", [gb(first * GB), gb((first + last) / 2 * GB), gb(last * GB)], -1);
            $("h-memory").textContent = (autoMemory ? "Adapts to other apps and this Mac’s memory history." : "The most Origami uses. macOS can cut it for a while when memory runs short.")
              + (w ? " Up to " + gb(w.safe_max_budget_bytes) + " here, leaving " + gb(w.os_headroom_bytes) + " for macOS and other apps." : "");
          }
          $("auto").setAttribute("aria-pressed", String(autoMemory));
          const lowBit = /low-bit/.test(status && status.model && status.model.experts || "") && modelChoice === modelInitial;
          if (lowBit && v.quality === "exact") $("h-quality").textContent = "This model's low-bit experts make its replies differ from the exact model at every level.";
          $("d-quality").toggleAttribute("data-differs", v.quality !== "exact" || lowBit);
          launched.forEach((id) => { $("h-" + id).textContent = "Set by the command that started Origami."; });
          ["context", "quality", "reasoning", "memory"].forEach((id) => setFill($(id)));
          showWiredMemory();
        }
        // What moved since the setup loaded (a setting the dials cannot show exactly, such as a 100K context, moved only when its dial does).
        // Only a dial the user moved in this page is ever saved: a value a refresh, a reload or another model left on a
        // dial is not an edit. `rebase` takes what the dials show as the saved state and forgets the moves.
        const touched = new Set();
        function rebase() { initial = current(); touched.clear(); draftBase = revision; }
        function changes() {
          const out = {}, v = current();
          if (initial) {
            const moved = (k) => touched.has(k) && v[k] !== initial[k];
            if (moved("context")) out.context = v.context / 1024 + "k";
            if (moved("quality")) out.quality = v.quality;
            if (moved("reasoning")) out.reasoning_effort = v.reasoning;
            if (moved("memory")) out.memory_limit = memorySetting(v.memory);
          }
          if (modelChoice !== null && modelChoice !== modelInitial) out.model = modelChoice;
          return out;
        }
        const changed = () => Object.keys(changes()).length > 0;
        const dirty = () => changed() || Object.keys(samplingChanges()).length > 0;
        let planTimer = null, planSeq = 0, storedMemory = null;
        // A dial names itself when the user moves it; a call without a name takes every dial that differs as moved.
        function dialMoved(id) {
          if (typeof id === "string") touched.add(id);
          else if (initial) { const v = current(); for (const k of Object.keys(v)) if (v[k] !== initial[k]) touched.add(k); }
          redrawDials();
        }
        function redrawDials() {
          if (!applying && !reloading) $("apply-note").textContent = "";
          drawDials(lastPlan);
          showStart();
          clearTimeout(planTimer); planTimer = setTimeout(previewPlan, 140);
        }
        let lastPlan = null, planRefused = false;
        async function previewPlan() {
          if (!base) { showEffects(); return; }
          const v = current(), seq = ++planSeq, started = generation;
          try {
            const query = { context: v.context, quality: v.quality, memory_limit: memorySetting(v.memory) };
            const response = await api("v1/plan?" + new URLSearchParams(query)), plan = await response.json();
            if (seq !== planSeq || started !== generation) return;
            planRefused = response.status === 400 || response.status === 422;
            if (!response.ok) { $("fits").dataset.state = "warn"; $("fits").textContent = ((plan.error && plan.error.message) || "No plan for these settings.").replace(/^No plan for these settings: /, "These settings cannot run: "); showEffects(); gate(); return; }
            lastPlan = plan; readout(plan); drawDials(plan); gate();
          } catch (error) { if (seq === planSeq) { $("fits").dataset.state = "warn"; $("fits").textContent = "Could not reach Origami."; } }
        }
        // This Mac's own speed: the middle of its last replies at the current setup. Nothing here is predicted.
        function nowSpeed() {
          // The level replies run at: the plan's, else the server's setting (a level name; a custom mix counts every reply).
          const setting = status && status.settings && status.settings.quality ? status.settings.quality.value : null;
          const quality = base ? base.settings.quality : LEVELS.some((l) => l[0] === setting) ? setting : null;
          const values = ((status && status.requests && status.requests.recent) || []).filter((r) => measured(r) && (!quality || (r.mode || "exact") === quality)).slice(-5).map((r) => r.decode_tps).sort((a, b) => a - b);
          const n = values.length, middle = n % 2 ? values[(n - 1) / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
          return n ? { tps: middle, count: n } : null;
        }
        function showSpeed() {
          const now = status && status.status === "ready" && !reloading ? nowSpeed() : null;
          $("speed").textContent = now ? one.format(now.tps) : status && status.status === "stopped" ? "Stopped" : status && status.status === "ready" && !reloading ? "No replies yet" : "—";
          $("speed-line").classList.toggle("none", !now);
          if (!measuring) $("basis").textContent = now ? "Median · " + now.count + (now.count === 1 ? " reply" : " replies") : status && status.status === "ready" && !reloading ? "A test reply takes a few seconds:" : "No current measurement";
          $("measure").textContent = "Measure";
        }
        // The self-tuner (`origami tune`, run as a job): what the running model's `tuning` says, and Tune, Re-check and Undo.
        function showSelfTune(s) {
          const t = s.tuning, e = t && t.entry;
          $("selftune").hidden = !t || !Object.keys(t.knobs || {}).length;
          if ($("selftune").hidden) return;
          const run = dashboard && dashboard.last_tuning, relevant = run && run.key === t.key;
          // Nothing faster than what runs is a result: "Tuned: defaults are fastest (date)", from the run or the saved entry.
          const fastest = (x) => "Tuned: " + (e && e.values && Object.keys(e.values).length ? e.label + " stays fastest" : "defaults are fastest") + (x && x.date ? " (" + String(x.date).slice(0, 10) + ")" : "");
          const labels = { too_noisy: "Inconclusive · settings unchanged", incomplete: "Incomplete · settings unchanged", aborted: "Interrupted · settings unchanged", already_fastest: fastest(run), faster: "Improvement measured", comparison: "Comparison ready" };
          $("st-line").textContent = t.state === "unreadable" ? "Tuning file could not be read" : t.state === "stale" ? "Setup changed · tune again" : t.state === "recheck" ? "Re-check · " + (t.reason || "setup updated") : relevant ? labels[run.outcome] || "" : e ? e.outcome === "faster" ? "Applied · +" + Math.round((e.gain || 0) * 100) + "%" : fastest(e) : "";
          $("st-go").textContent = "Run tuning";
          $("st-check").hidden = !e || t.state === "stale" || !e.values || !Object.keys(e.values).length;
          $("st-undo").hidden = !e;
          if (relevant && ["too_noisy", "incomplete", "aborted"].includes(run.outcome) && !running()) $("st-note").textContent = run.reason || (run.measured && (run.measured.notes || [])[0]) || "No reliable improvement measured.";
          const p = dashboard && dashboard.tuning_proposal;
          if (!p || p.key !== t.key || p.revision !== revision) { $("st-result").innerHTML = ""; return; }
          if (p.kind === "performance") {
            const entry = p.entry || {}, values = Object.entries(entry.values || {}).map(([k,v]) => esc(k.replace(/_/g," ")) + ": " + esc(v)).join(" · ");
            $("st-result").innerHTML = '<p>+' + Math.round((entry.gain || 0) * 100) + '% ' + (entry.measured && entry.measured.objective === 'prompt' ? 'prompt speed' : 'writing speed') + ' measured</p><p class="note">' + values + '</p>' + (p.sets_aside ? '<p class="note">' + esc(p.sets_aside) + '</p>' : '') + '<button type="button" data-tune-apply="">Apply &amp; reload</button>';
          } else {
            $("st-result").innerHTML = '<p class="note">Speed measurements only; answer quality is not scored. Large variation makes close speed differences unreliable.</p><table><thead><tr><th>Mode</th><th>tok/s</th><th>First token</th><th>Samples</th><th>Variation</th><th></th></tr></thead><tbody>' + (p.results || []).map(r => {
              const valid = Number.isFinite(r.decode_tps) && r.decode_tps > 0 && Number.isFinite(r.first_token_seconds) && r.first_token_seconds >= 0 && Number.isFinite(r.spread) && r.spread >= 0 && Number.isInteger(r.samples) && r.samples >= 2;
              return '<tr><td>' + esc(r.quality) + '</td><td>' + (valid ? one.format(r.decode_tps) : '—') + '</td><td>' + (valid ? one.format(r.first_token_seconds) + ' s' : '—') + '</td><td>' + (Number.isInteger(r.samples) && r.samples >= 0 ? fmt.format(r.samples) : '—') + '</td><td>' + (valid ? Math.round(r.spread * 100) + '%' : 'Not measured') + '</td><td><button type="button" data-tune-apply="' + esc(r.quality) + '"' + (valid ? '' : ' data-why="At least two complete speed measurements are required" data-blocked disabled title="At least two complete speed measurements are required"') + '>Apply</button></td></tr>';
            }).join("") + '</tbody></table>';
          }
        }
        // The memory budget learned from this Mac's other apps (`memory_learned`), and its Reset.
        function showLearned(s) {
          const m = s.memory_learned, advice = m && m.limit_advice;
          // `in_force`: which memory rule is in force (the measured limit, the user's own, or automatic), first.
          $("ml").hidden = !(m && m.in_force) && !advice && (!m || !m.days || !m.applied);
          if ($("ml").hidden) return;
          const said = advice ? memoryText(advice.text).replace(/[;,]? set it with `[^`]*`\.?$/, ".") : !m.days || !m.applied ? "" : m.bytes == null ? "Learning the available memory…" : "Automatic budget · " + gb(m.bytes);
          const rule = memoryText(m.in_force).replace(/^memory: /, "").replace(/ \(not measured: run `origami tune --memory`\)/, " · not measured yet").replace(/^./, (c) => c.toUpperCase());
          $("ml-text").textContent = [rule && "Memory limit: " + rule, said].filter(Boolean).join(" · ");
          $("ml-reset").hidden = !advice && !(m.days && m.applied);
        }
        let tuneBy = null;
        // What a moved dial does to speed here, as a step from much slower (-3) to much faster (+3), judged from this Mac's
        // facts: how often its experts are missed (only misses are skipped by the faster levels), how many experts its memory
        // holds at each setting, and which context tier the setting lands in.
        const WORDS = ["much slower", "slower", "a little slower", "about the same", "a little faster", "faster", "much faster"];
        const clampStep = (x) => Math.max(-3, Math.min(3, Math.round(x)));
        function missShare() {
          const c = status && status.expert_cache;
          if (c && c.hits + c.misses > 5000) return c.misses / (c.hits + c.misses);
          const e = base && base.experts;
          return e && e.total ? Math.max(0, 1 - e.cache_experts / e.total) : 0.5;
        }
        const QUALITY_GAIN = { exact: 0, dynamic: 2, "dynamic-fastest": 2, faster: 3, fastest: 4 };
        const contextPace = (c) => c.paged ? 0.35 : c.tier <= 8192 ? 1 : c.tier <= 32768 ? 0.89 : c.tier <= 65536 ? 0.73 : c.tier <= 131072 ? 0.63 : 0.6;
        // Much faster or slower means about twice or half the speed.
        function paceStep(ratio) { return ratio >= 1.8 ? 3 : ratio >= 1.25 ? 2 : ratio >= 1.05 ? 1 : ratio > 0.95 ? 0 : ratio > 0.8 ? -1 : ratio > 0.55 ? -2 : -3; }
        function effects() {
          const rows = [], v = current();
          // Each row: what moved, its step, an optional word, and the dial it belongs to.
          if (modelChoice !== null && modelChoice !== modelInitial) rows.push([modelSwitch(), null, "shows once it runs", "model"]);
          if (!initial) return rows;
          const misses = missShare(), weight = misses < 0.05 ? 0 : misses < 0.25 ? 0.5 : 1;
          if (v.quality !== initial.quality) rows.push(["Quality " + LEVELS.find((l) => l[0] === v.quality)[1].toLowerCase(), clampStep((QUALITY_GAIN[v.quality] - QUALITY_GAIN[initial.quality]) * weight), null, "quality"]);
          // Thinking changes how long a reply takes to start, not the speed it writes at.
          if (v.reasoning !== initial.reasoning) {
            const more = effortIndex(v.reasoning) > effortIndex(initial.reasoning);
            rows.push(["Thinking " + EFFORTS[effortIndex(v.reasoning)][1].toLowerCase(), null, more ? "starts later" : "starts sooner", "reasoning"]);
          }
          // Without a plan (GLM, DeepSeek) a context's cost here is known once it runs.
          if (!base || !lastPlan) { if (v.context !== initial.context) rows.push(["Context " + fmt.format(v.context / 1024) + "K", null, "shows once it runs", "context"]); return rows; }
          if (v.context !== initial.context) rows.push(["Context " + fmt.format(v.context / 1024) + "K", paceStep(contextPace(lastPlan.context) / contextPace(base.context)), null, "context"]);
          const heldNow = base.experts.cache_experts, heldNext = lastPlan.experts.cache_experts;
          if (v.memory !== initial.memory && heldNow > 0) {
            const r = heldNext / heldNow, step = weight === 0 ? 0 : r >= 2 ? 2 : r >= 1.2 ? 1 : r > 0.83 ? 0 : r > 0.5 ? -1 : -2;
            rows.push([fmt.format(heldNext) + " experts in memory", step, null, "memory"]);
          }
          return rows;
        }
        function showEffects() {
          const rows = planRefused ? [] : effects();
          for (const dial of ["context", "quality", "memory"]) {
            const row = rows.find((r) => r[3] === dial);
            $("e-" + dial).textContent = row && row[1] != null ? "Here: " + (row[1] === 0 ? "about the same as now." : WORDS[row[1] + 3] + " than now.") : "";
          }
          $("effects").hidden = !rows.length;
          if (!rows.length) return;
          const steps = rows.filter((r) => r[1] != null).map((r) => r[1]), total = clampStep(steps.reduce((a, b) => a + b, 0));
          const mixed = steps.some((x) => x > 0) && steps.some((x) => x < 0);
          const thinkingOnly = rows.every((r) => r[3] === "reasoning");
          $("overall").textContent = thinkingOnly ? "The same writing speed; the answer " + (rows[0][2] === "starts later" ? "starts later." : "starts sooner.")
            : !steps.length ? "Its speed shows once it runs." : mixed && total === 0 ? "Faster in one way, slower in another." : total === 0 ? "About the same speed as now." : WORDS[total + 3].replace(/^./, (c) => c.toUpperCase()) + " than now.";
          $("effect-rows").innerHTML = rows.map(([what, step, word, dial]) => '<div class="effect"><span class="what' + (dial === "model" ? " whole" : "") + '">' + esc(what) + "</span>" +
            // Slower at the left, the same in the middle, faster at the right: `---|-*-` (a mark the memory strip does not use).
            (step == null ? "<span></span>" : '<span class="scale" aria-hidden="true">' + [-3, -2, -1, 0, 1, 2, 3].map((x) => x === step ? "<b>*</b>" : x === 0 ? "|" : "-").join("") + "</span>") +
            '<span class="word">' + esc(word || WORDS[step + 3]) + "</span></div>").join("");
        }
        function readout(plan) {
          showSpeed(); showEffects();
          showMemory(plan.memory);
          const warn = [];
          if (plan.context.paged) warn.push("The context is kept in a file on disk, which needs about 37 GB free on its drive.");
          (plan.notes || []).slice(0, 1).forEach((n) => warn.push(n.charAt(0).toUpperCase() + n.slice(1) + (/[.!?]$/.test(n) ? "" : ".")));
          $("fits").dataset.state = warn.length ? "warn" : "ok";
          $("fits").textContent = warn.length ? warn.join(" ") : "Fits in this Mac's memory, with room for your other apps.";
        }
        // What the page lets you do, from Origami's state: one place (`gate`) enables and disables the actions, and marks the
        // one at work. `liveState` is the chip's state; `offline` a status poll that failed; `applying` the control whose
        // request is on its way (a save, a reload, a look again); `reloadBy` the one that started a reload, which turns until
        // the new model runs or fails; `jobPost` a job being started; `measuring` the speed test; `checking` the source being checked.
        let reloading = false, reloadBy = null, reloadAt = 0, reloadSeen = false, liveState = "down", offline = true;
        // Whether Origami's front answers (its jobs do), which stops and starts the model; a one-model server has no Stop.
        let front = false;
        let applying = null, jobPost = null, measuring = false, checking = false, addFits = true;
        // Sampling as saved (`smSaved`: the choice, the dials' values, the override, a setting given at launch) and as the dials show it.
        let smSaved = null, smSeen = null, smValues = {};
        // Access as last read (`v1/lan`), and a restart this page asked for, until Origami listens as saved.
        let access = null, restarting = false, restartSince = 0;
        // The storage map as last read, the item whose move or removal this page started, and a read under way.
        let storageMap = null, storagePath = null, storageLoading = false;
        // A new reload clears the marks an earlier failed one left.
        function reloadStarted(by) {
          reloading = true; reloadBy = by; reloadAt = Date.now(); reloadSeen = false; setupFor = null;
          ["apply", "use-model", "notice-retry", "folders-reload", "sm-apply", "power"].forEach((id) => failed($(id), false));
        }
        // A control that cannot act now is disabled; `working` gives it its spinner, and a failed try's mark goes. The keyboard
        // focus a disabled control drops comes back to it once it can act again.
        // No silently disabled action: a control held back (not one turning while it works) says why, as its tooltip and in
        // the note next to it (`<id>-why`) when it has one. The reason: the one given, the control's own `data-why`, or
        // what holds the whole page back.
        function hold(el, disabled, working = false, why = "") {
          if (disabled && document.activeElement === el) el.dataset.refocus = "1";
          if (working) el.toggleAttribute("data-failed", false);
          el.disabled = disabled; el.setAttribute("aria-busy", String(working));
          if (!disabled && el.dataset.refocus) { delete el.dataset.refocus; if (!el.hidden && document.activeElement === document.body) el.focus(); }
          const reason = disabled && !working ? why || el.getAttribute("data-why") || waitReason() : "";
          if (reason || el.hasAttribute("data-held")) { el.title = reason; el.toggleAttribute("data-held", !!reason); }
          // "Nothing changed" stays a tooltip: only a reason that asks for a step stands under a control in view.
          const note = el.id && document.getElementById(el.id + "-why");
          if (note) note.textContent = reason && !el.hidden && !QUIET.includes(reason) ? reason + "." : "";
        }
        const QUIET = ["Nothing changed", "Nothing to discard"];
        // What holds every action back now ("" when nothing does).
        function waitReason() {
          return offline ? "Not connected to Origami" : running() || jobPost ? "Another job is running" + (job && job.title ? ": " + job.title : "")
            : applying ? "Wait for the current change to finish" : liveState === "stopping" ? "Origami is stopping"
            : reloading || ["loading", "reloading", "unavailable"].includes(liveState) ? "The model is loading" : "";
        }
        // A list's buttons share one note: the first reason one of them is held for.
        function listWhy(note, buttons) {
          const held = [...buttons].find((b) => b.disabled && b.getAttribute("aria-busy") !== "true" && b.title && !QUIET.includes(b.title));
          $(note).textContent = held ? held.title + "." : "";
        }
        const failed = (el, on) => el.toggleAttribute("data-failed", on);
        // Every disabled control says so to assistive technology as well.
        new MutationObserver((list) => list.forEach((m) => m.target.setAttribute("aria-disabled", String(!!m.target.disabled))))
          .observe(document.body, { attributes: true, attributeFilter: ["disabled"], subtree: true });
        function coreGate() {
          // A model loading or reloading takes no dial, switch or test until it runs; with no answer at all nothing acts.
          // Stopped (or stopping), nothing that needs the model acts until Start.
          const halted = liveState === "stopped" || liveState === "stopping";
          const starting = reloading || liveState === "loading" || liveState === "stopping", serving = !starting && (liveState === "ready" || liveState === "busy");
          const by = (id) => applying === id || (reloading && reloadBy === id), jobFor = (kind) => jobPost === kind || (running() && job.kind === kind);
          // The control a moment without an answer takes the keyboard from gets it back when Origami answers again.
          const page = $("page"), at = document.activeElement;
          if (offline && !page.disabled && at && at !== document.body && page.contains(at)) page.refocus = at;
          page.disabled = offline;
          if (!offline && page.refocus) { if (document.activeElement === document.body && !page.refocus.disabled) page.refocus.focus(); page.refocus = null; }
          // Busy on the dials alone: on the whole setup it would hold back the notes its live regions announce meanwhile.
          $("dials").disabled = starting || !!applying; $("dials").setAttribute("aria-busy", String(starting || !!applying));
          hold($("model"), starting || !!applying);
          hold($("use-model"), starting || !!applying, by("use-model"));
          hold($("apply"), !ready || !changed() || starting || !!applying, by("apply"));
          hold($("reset"), !changed() || starting || !!applying, false, !changed() ? "Nothing changed" : "");
          hold($("measure"), measuring || !serving, measuring, liveState === "stopped" ? "The model is stopped: Start it first" : "The model is not ready");
          ["st-go", "st-check", "st-undo"].forEach((id) => hold($(id), running() || !!jobPost || !serving || !front, jobFor("tune") && tuneBy === id));
          hold($("ml-reset"), !!applying, applying === "ml-reset");
          hold($("notice-retry"), !!applying || reloading || liveState === "stopping", by("notice-retry"));
          // Stop and unload, Start once stopped, Try again after a failed start: it turns while the model unloads or loads
          // again (by this page or another).
          const powering = liveState === "stopping" || by("power") || by("notice-retry") && noticeKind === "stopped", starts = halted && liveState !== "stopping" || powering && reloading;
          const powerLabel = starts ? "Start" : liveState === "failed" ? "Try again" : "Stop";
          $("power").hidden = offline || !front && !halted; $("top").toggleAttribute("data-power", !$("power").hidden);
          hold($("power"), offline || powering || !halted && (reloading || !!applying), powering);
          if ($("power").dataset.power !== powerLabel) {
            $("power").dataset.power = powerLabel; $("power").dataset.starts = String(powerLabel !== "Stop");
            $("power").innerHTML = powerLabel === "Stop" ? 'Stop<span class="power-more"> and unload</span>' : powerLabel;
            $("power").setAttribute("aria-label", powerLabel === "Stop" ? "Stop and unload" : powerLabel);
          }
          hold($("folders-reload"), starting || !!applying, by("folders-reload"));
          // Sampling: the dials move only for Custom; a setting the command line gave stays put.
          const smWaits = starting || !!applying || !smSaved || smSaved.launched;
          $("sm-group").disabled = smWaits; $("sm-dials").disabled = smWaits || !$("sm-custom").checked;
          const smWhy = smSaved && smSaved.launched ? "Set by the command that started Origami" : !smSaved ? "The model's sampling is not loaded yet" : "";
          hold($("sm-apply"), smWaits || !Object.keys(samplingChanges()).length, by("sm-apply"), smWhy || (!starting && !applying ? "Nothing changed" : ""));
          hold($("sm-reset"), smWaits || !$("sm-custom").checked, false, smWhy || (!starting && !applying ? "Only Custom sampling is reset" : ""));
          hold($("acc-save"), !Object.keys(accessChanges()).length || !!applying || restarting, by("acc-save"), restarting ? "Origami is restarting" : !applying ? "Nothing changed" : "");
          hold($("acc-restart"), !!applying || restarting, by("acc-restart") || restarting);
          // One job at a time: its own button turns while it runs.
          const busyJob = running() || !!jobPost;
          hold($("add-go"), !addFits || busyJob, jobFor("add"), !busyJob ? "Not enough room on that drive: Write it elsewhere…" : "");
          hold($("add-to"), busyJob);
          hold($("add-check"), !!checking, !!checking);
          $("add-info").setAttribute("aria-busy", String(!!checking));
          $("job").setAttribute("aria-busy", String(running()));
          // A move or removal turns the button that started it; while any job runs the others wait.
          const storing = jobPost === "move" || jobPost === "remove" || running() && (job.kind === "move" || job.kind === "remove");
          $("storage").querySelectorAll('button[data-sact="move"], button[data-sact="remove"]').forEach((b) => hold(b, busyJob, storing && b.dataset.path === storagePath));
          hold($("storage-refresh"), storageLoading, storageLoading);
          // Folder and drive choices and each saved prompt's Open wait for a job, a start or an Apply too; one saving keeps turning.
          const waits = busyJob || starting || !!applying;
          for (const b of [...$("folders").querySelectorAll("button[data-act]"), ...$("prompts").querySelectorAll("button[data-open]"),
          ]) if (b.getAttribute("aria-busy") !== "true") hold(b, waits);
          // The Experts card: a family model changes its hot copy live; Qwen's is a job, so it waits for one running.
          // A pack's Move… is a job: it waits for one running.
          const xWaits = starting || !!applying || (kind !== "family" && busyJob), panelWaits = xWaits || !!xPost || (!!xOpen && xOpen !== "hot" && busyJob);
          hold($("x-change"), panelWaits, false, xPost ? "Wait for the current change to finish" : "");
          hold($("x-use"), panelWaits, xPost === "use", xPost ? "Wait for the current change to finish" : "");
          $("x-acts").querySelectorAll("button[data-xact]").forEach((b) => b.dataset.xact === "change" || b.dataset.xact.startsWith("move:") ? hold(b, xWaits)
            : hold(b, xWaits || !!xPost || b.hasAttribute("data-blocked"), xPost === b.dataset.xact));
          $("x-unused").querySelectorAll("button[data-xact]").forEach((b) => hold(b, starting || !!applying || !!xPost, xPost === "clear", xPost ? "Wait for the current change to finish" : ""));
          listWhy("x-acts-why", $("x-acts").querySelectorAll("button"));
          // What waits for you: the model menu with no model chosen yet, else the way to the permission macOS asks for.
          $("model").toggleAttribute("data-ask", false);
          const asks = (noticeKind === "permission" || noticeKind === "found") && !reloading;
          $("notice-settings").toggleAttribute("data-ask", asks && !$("notice-settings").hidden);
          $("notice-retry").toggleAttribute("data-ask", asks && $("notice-settings").hidden);
        }
        async function apply(reload = true, restoreWired = false, modelOnly = false) {
          const w = wiredMemory();
          if (restoreWired && (!w || !w.supported || !w.requires_raise || dirty())) return;
          const body = restoreWired ? { memory_limit: w.configured_budget_bytes == null ? "auto" : memorySetting(w.configured_budget_bytes / GB) } : modelOnly ? { model: modelChoice } : { ...changes(), ...samplingChanges() };
          const by = restoreWired ? "wired-restore" : noModel ? "use-model" : reload ? "apply" : "save", note = $(restoreWired ? "wired-note" : "apply-note");
          if (conflict || !Object.keys(body).length) return;
          if (body.model && modelByJob && Object.keys(body).length > 1 && !confirm("This also switches the model. " + modelSwitch() + ". Apply both?")) return;
          const wired = "memory_limit" in body;
          applying = by; gate(); note.textContent = wired ? "Saving the memory limit… If macOS asks, approve the administrator dialog on this Mac." : "Saving…";
          let ok = false;
          try {
            const saved = await post(wired ? "v1/memory/wired" : "v1/config", body, restoreWired ? revision : draftBase || revision), result = await saved.json();
            if (!saved.ok) {
              if (saved.status === 409) { conflict = true; if (result.changes && !restoreWired) review = { changes: result.changes, reload }; }
              throw Error(review ? "" : result.error && result.error.message || "Not saved");
            }
            review = null; revision = result.revision || revision; syncedRevision = null;
            if (dashboard && result.wired_memory) { dashboard.wired_memory = result.wired_memory; showWiredMemory(); }
            rebase(); modelInitial = modelChoice;
            if (smSaved) smSaved = { ...smSaved, custom: $("sm-custom").checked, override: $("sm-override").checked, values: { ...smValues } };
            const shouldReload = reload && (!restoreWired || result.reload_required && liveState !== "stopped");
            if (shouldReload) {
              const response = await post(liveState === "stopped" ? "v1/start" : "v1/reload", {});
              if (!response.ok) throw Error("Saved. Reload when the current operation finishes.");
              reloadStarted(by); base = lastPlan = null; ready = false; smSeen = null;
            }
            ok = true; note.textContent = shouldReload ? "Loading saved settings…" : restoreWired ? "macOS ceiling restored until reboot." : "";
            auxAt = storageAt = 0; await refresh();
          } catch (error) { note.textContent = error.message || "Could not save"; }
          finally { applying = null; failed($(by), !ok); gate(); }
        }
        // A dial given at launch (a flag such as --context 8k, or ORIGAMI_*) would win again after the reload: it stays put.
        function launchedDials(status) {
          const given = (key) => { const at = status && status.settings && status.settings[key]; return !!at && (at.source === "flag" || at.source === "env"); };
          return [["context", "context"], ["quality", "quality"], ["reasoning", "reasoning_effort"], ["memory", "memory_limit"]]
            .filter(([, key]) => given(key)).map(([id]) => id);
        }
        // A model whose server gives no plan (`v1/plan` refused): context and quality are set from the server's settings, and
        // the expert memory is sized to what this Mac has free when it starts. GLM and DeepSeek have plans: `loadSetup`.
        function familySetup(s) {
          const value = (key) => s.settings[key] ? s.settings[key].value : "";
          const text = value("context"), tokens = parseFloat(text) * (/k$/i.test(text) ? 1024 : 1);
          ready = true; base = null; lastPlan = null; kind = "family"; autoMemory = true;
          $("context").max = String(CONTEXTS.length - 1);
          $("context").value = String(Math.max(0, CONTEXTS.findIndex((x) => x[0] >= tokens)));
          setLevels(s.quality_levels); $("quality").value = String(levelAt(value("quality")));
          launched = launchedDials(s); setReasoning(s);
          $("context").disabled = launched.includes("context");
          $("quality").disabled = launched.includes("quality") || value("allow_fast") === "false";
          $("memory").disabled = $("auto").disabled = true;
          const wired = wiredMemory();
          if (wired && wired.safe_max_budget_bytes > 0) {
            autoMemory = wired.configured_budget_bytes == null; storedMemory = wired.configured_budget_bytes;
            memoryRange(wired.safe_max_budget_bytes, storedMemory || wired.active_budget_bytes || wired.safe_max_budget_bytes);
            $("memory").disabled = $("auto").disabled = launched.includes("memory") || !wired.supported;
          }
          ["context", "quality"].forEach((id) => $("d-" + id).toggleAttribute("data-off", $(id).disabled));
          $("d-memory").toggleAttribute("data-off", $("memory").disabled);
          $("mem").hidden = true; $("fits").textContent = "";
          reloadedNotes();
          if (reloading) $("model-note").textContent = "";
          reloading = false;
          rebase();
          drawDials(null);
          if (!wired) {
            $("o-memory").textContent = "Automatic";
            $("h-memory").textContent = "This model sizes its expert memory to what this Mac has free when it starts.";
          }
          showSpeed(); showEffects(); syncSaved(); showFolders(); gate();
        }
        async function loadSetup(status) {
          const seq = ++setupSeq, started = generation;
          try {
            const response = await api("v1/plan");
            if (seq !== setupSeq || started !== generation) return;
            if (!response.ok) { if (status && status.settings) familySetup(status); else { reloading = false; setupUnavailable(); } return; }
            const plan = await response.json(), s = plan.settings, m = plan.memory;
            if (seq !== setupSeq || started !== generation) return;
            base = plan; lastPlan = plan; ready = true;
            // GLM and DeepSeek; Qwen's affine 4-bit builds; NVFP4 and GGUF experts ("fixed").
            kind = status && status.model && status.model.family && status.model.family !== "qwen" ? "family"
              : status && status.model && !/^affine/.test(status.model.quantization || "affine") ? "fixed" : "qwen";
            const longest = (plan.context && plan.context.limit) || 262144;
            $("context").max = String(Math.max(0, CONTEXTS.filter((x) => x[0] <= longest).length - 1));
            $("context").value = String(Math.min(+$("context").max, Math.max(0, CONTEXTS.findIndex((x) => x[0] >= s.context))));
            setLevels((plan.quality.levels || []).map((l) => l.name)); $("quality").value = String(levelAt(s.quality));
            launched = launchedDials(status); setReasoning(status);
            $("mem").hidden = false;
            $("context").disabled = launched.includes("context");
            $("memory").disabled = $("auto").disabled = launched.includes("memory");
            $("quality").disabled = !plan.quality.allowed || launched.includes("quality");
            ["context", "memory"].forEach((id) => $("d-" + id).toggleAttribute("data-off", $(id).disabled));
            $("d-quality").toggleAttribute("data-off", launched.includes("quality"));
            // The host policy supplies the upper bound, even when it exceeds the ceiling in the current model's plan.
            autoMemory = s.memory_limit === "auto"; storedMemory = autoMemory ? null : s.memory_limit;
            memoryRange(wiredMemory() && wiredMemory().safe_max_budget_bytes || m.most_bytes, autoMemory ? m.cap_bytes : s.memory_limit);
            reloadedNotes();
            if (reloading) $("model-note").textContent = "";
            reloading = false;
            rebase();
            drawDials(plan); readout(plan); syncSaved(); showFolders(); gate();
          } catch (error) { if (seq === setupSeq && started === generation) { reloading = false; setupUnavailable(); } }
        }
        // Once a reload is back: the note under what started it says so, and Sampling's "Reloading" goes (or says it, for its own).
        function reloadedNotes() {
          const sm = reloading && reloadBy === "sm-apply";
          $("apply-note").textContent = reloading && !sm ? "Reloaded with the new setup." : "";
          if (reloading) $("health-note").textContent = "";
          if (reloading && $("sm-note").textContent !== SM_HELD) $("sm-note").textContent = sm ? "Reloaded with the new sampling." : "";
        }
        function setupUnavailable(why) {
          ready = true; base = null; initial = null; lastPlan = null;
          ["context", "quality", "reasoning", "memory"].forEach((id) => { $(id).disabled = true; $("d-" + id).toggleAttribute("data-off", true); });
          $("h-context").textContent = why || "This model's dials are set from Terminal (origami config set); its model can be switched here.";
          $("mem").hidden = true; showSpeed(); $("fits").textContent = ""; gate();
        }

        // The model menu: the models Origami has, those found on this Mac that it could add, and "Add a model…". The front keeps
        // both lists and each model's state; a failed ask keeps the last lists, and a check or search still running is asked
        // for again in a few seconds (up to 20 times), so entries settle in place instead of coming and going.
        let modelPolls = 0, foundPolls = 0;
        const pending = (state) => state === "checking" || state === "slow";
        async function loadModels(again) {
          if (!again) modelPolls = 0;
          const expected = revision, request = loadModels.request = (loadModels.request || 0) + 1;
          try {
            const response = await api("v1/models?all"), result = response.ok ? await response.json() : null;
            if (expected !== revision || request !== loadModels.request) return;
            // Before any model is added the list holds the built-in default as "missing", with no setting: not a model to load.
            if (result) models = (result.data || []).filter((m) => !m.origami || m.id === m.origami.model && !(m.origami.availability === "missing" && !m.origami.setting));
          } catch (error) {}
          const def = models.find((m) => m.origami && m.origami.default) || models[0];
          const wasInitial = modelInitial;
          modelInitial = dashboard ? dashboard.configured_model : def && def.origami ? def.origami.setting : "";
          if (modelChoice === wasInitial && !conflict) modelChoice = modelInitial;
          if (modelChoice === null || !models.some((m) => (m.origami ? m.origami.setting : m.id) === modelChoice)) modelChoice = modelInitial;
          // The model an add job just added is chosen once a list holds it: the add changes the revision, so the first read can be dropped.
          if (addedNow && models.some((m) => m.origami && m.origami.setting === addedNow)) {
            modelChoice = addedNow; modelByJob = true; addedNow = null;
            $("model-note").textContent = "Added " + modelChoice + "." + (noModel ? "" : " Load model switches to it.");
            redrawDials();
          }
          fillPicker();
          if (status) notice(status);
          if (!found) loadFound(false);
          clearTimeout(loadModels.t);
          if (models.some((m) => m.origami && pending(m.origami.availability)) && modelPolls++ < 20) loadModels.t = setTimeout(() => loadModels(true), 3000);
        }
        async function loadFound(fresh, again) {
          if (!again) foundPolls = 0;
          try {
            const response = await api("v1/models/found" + (fresh ? "?fresh" : ""));
            if (response.ok) { found = await response.json(); fillPicker(); if (status) notice(status); }
          } catch (error) {}
          clearTimeout(loadFound.t);
          if (found && found.searching && foundPolls++ < 20) loadFound.t = setTimeout(() => loadFound(false, true), 3000);
        }
        // A model set by its folder shows the folder's name.
        const named = (m) => m.origami && /^[\/~]/.test(m.origami.setting) ? m.origami.setting.replace(/\/+$/, "").split("/").pop() : m.id;
        const settingName = (setting) => { const m = models.find((x) => x.origami && x.origami.setting === setting); return m ? named(m) : setting || "none"; };
        // What Apply does to the model, named in full: "Model: GLM-5.3-Flash → qwen3.8-flash-next-4bit".
        const modelSwitch = () => "Model: " + settingName(modelInitial) + " → " + settingName(modelChoice);
        // A model's state, as the front last checked it, in a few words: a drive that sleeps, is unplugged or is not allowed yet
        // is named as such, never taken for a missing model.
        function stateText(state, drive) {
          const d = drive && drive !== "internal drive" ? drive : "drive";
          return { checking: "checking…", slow: d + " slow to answer", not_connected: d + " not connected", no_permission: "needs permission to read " + (d === "drive" ? "it" : d), missing: "not found" }[state] || "";
        }
        function fillPicker() {
          const label = (m) => [named(m), m.origami && stateText(m.origami.availability, m.origami.drive), m.origami && m.origami.bytes ? Math.round(m.origami.bytes / 1e9) + " GB" : "",
            m.origami && m.origami.variant_of ? "low-bit experts, replies differ" : m.origami && m.origami.quantization && m.origami.quantization !== "unknown" ? m.origami.quantization : ""].filter(Boolean).join(" · ");
          const extra = (found && found.models) || [];
          // Rewritten only when it changes: a poll that brings nothing new leaves a menu being chosen from (and a screen reader) alone.
          const html = '<optgroup label="Added">' + models.map((m) => '<option value="' + esc(m.origami ? m.origami.setting : m.id) + '">' + esc(label(m)) + "</option>").join("") + "</optgroup>" +
            (extra.length ? '<optgroup label="Found on this Mac">' + extra.map((m) => '<option value="' + esc("found:" + m.path) + '">' + esc([m.name, stateText(m.state, m.drive), Math.round(m.bytes / 1e9) + " GB", m.verdict === "converts" ? "converts once" : ""].filter(Boolean).join(" · ")) + "</option>").join("") + "</optgroup>" : "") +
            '<option value="add">Add a model…</option>';
          if (html !== fillPicker.html) $("model").innerHTML = fillPicker.html = html;
          modelList();
          $("model").value = addSource && $("model").querySelector('option[value="' + CSS.escape("found:" + addSource) + '"]') ? "found:" + addSource : addSource !== null && $("add-row").hidden === false ? "add" : modelChoice || "";
        }
        function closeAdd() { $("add").hidden = true; $("add-open").setAttribute("aria-expanded", "false"); $("add-base-row").hidden = true; $("add-base").value = ""; addSource = null; addTo = null; $("add-info").textContent = ""; $("add-go").hidden = true; $("add-to").hidden = true; }
        // Where a download or conversion is written: the folder chosen with the browser, else the data folder's models folder.
        let addTo = null, addDestination = null;
        // What adding a model would do, from its headers: run as it is, convert once, or download; and whether there is room.
        async function checkSource(source) {
          addSource = source; $("add-go").hidden = true; $("add-to").hidden = true; $("add-source").setAttribute("aria-invalid", "false");
          if (!source) { $("add-info").textContent = ""; $("add-base-row").hidden = true; return; }
          if (baseChoices(source) && $("add-base").value) return baseChosen();
          $("add-info").textContent = "Checking…"; checking = source; gate();
          try {
            const response = await api("v1/models/check?" + new URLSearchParams(addTo ? { source, to: addTo } : { source })), c = await response.json();
            if (source !== addSource) return;
            if (!response.ok || c.verdict === "refused") { $("add-info").textContent = ((c.error && c.error.message) || c.reason || "Origami cannot add this.").replace(/([^.])$/, "$1.") +
              ($("add-base-row").hidden ? "" : " If it holds a model's low-bit experts, choose that model under Add as.");
              $("add-source").setAttribute("aria-invalid", "true"); return; }
            const hub = !/^[\/~]/.test(source), converts = c.verdict === "converts";
            const what = hub ? "downloads " + disk(c.bytes) + (converts ? ", then converts once" : "") : converts ? "converts once" + (c.needs_bytes > 1e10 ? ", writing about " + disk(c.needs_bytes - 1e10) : "") : "runs as it is";
            const room = c.needs_bytes > 1e10 ? " to " + (addTo ? addTo + " on " : "") + driveLabel(c.drive) + ", which has " + disk(c.free_bytes) + " free" : "";
            addDestination = c.destination || null; $("add-to").hidden = !(c.needs_bytes > 1e10);
            $("add-info").textContent = c.family + ": " + what + room + "." + (c.fits === false ? " Not enough room: it needs " + disk(c.needs_bytes) + ", 10 GB to spare included." : "");
            $("add-go").hidden = false; addFits = c.fits !== false;
            $("add-go").textContent = hub ? "Download and add" : converts ? "Convert and add" : "Add model";
          } catch (error) { if (source === addSource) $("add-info").textContent = "Could not reach Origami."; }
          finally { if (checking === source) { checking = false; gate(); } }
        }

        // What keeps Origami from working, and how to fix it: a drive macOS has not let it read, or a model that did not start.
        let noticeKind = null, noticeGone = false, goneFailure = null;
        function notice(s) {
          // Drives macOS has not let Origami read: the model search's, and those of added models refused.
          const storeDrive = ((s.stores || []).find((d) => d.permission) || {}).drive, unreadable = ((found && found.unreadable) || [])
            .concat(models.filter((m) => m.origami && m.origami.availability === "no_permission" && m.origami.drive !== "internal drive").map((m) => m.origami.drive));
          let title = "", text = "";
          noticeKind = null;
          if (s.status === "stopped" || s.status === "stopping") {
            noticeKind = "stopped"; title = s.status === "stopped" ? "Origami is stopped" : "Origami is stopping";
            text = "The model is " + (s.status === "stopped" ? "unloaded" + (s.freed_bytes ? " and about " + gb(s.freed_bytes) + " of memory is back with macOS" : "") : "unloading")
              + ". Apps get “Origami is stopped” until you start it; it stays stopped when the Mac restarts.";
          } else if (s.permission && s.permission.drive) {
            noticeKind = "permission"; title = "Origami can't read " + s.permission.drive + " yet";
            text = "The model is on " + s.permission.drive + ", and macOS asks before an app reads an external drive. Origami starts once you allow it.";
          } else if (storeDrive) {
            noticeKind = "permission"; title = "Origami can't read " + storeDrive + " yet";
            text = "The experts it keeps on " + storeDrive + " are skipped until you allow it, so replies read from the model's drive alone and can be slower.";
          } else if (s.status === "degraded") {
            // The model's own drive went away: its files and copies reopen by themselves once it is back (Reload loads what
            // is there when the model process refuses a different file at the path).
            noticeKind = "degraded"; title = "The model's drive is not answering";
            text = s.degraded || "";
          } else if (s.status === "failed" && /No model (found|at)/.test(s.failure || "")) {
            noticeKind = "missing"; title = "No model yet";
            text = "Add one below: Origami lists what it found on this Mac, and can download one from Hugging Face.";
          } else if (s.status === "failed") {
            const chosen = models.find((m) => m.origami && m.origami.setting === modelInitial), gone = chosen && chosen.origami.availability === "missing" ? chosen : null;
            // Once Locate… has recorded the new place, the failure from the old one no longer says what to do.
            if (gone) goneFailure = s.failure; const located = !gone && chosen && goneFailure && s.failure === goneFailure;
            noticeKind = "failed"; noticeGone = !!gone; title = gone ? named(gone) + " is not where it was" : located ? named(chosen) + " is found again" : "The model did not start";
            text = located ? "Its new location is recorded. Try again loads it from there."
              : (s.failure || s.error || "It stopped while loading.").replace(/^[^ ]+ did not start: /, "") + (gone ? " If it moved, Locate… under Models records where it is now; another model can be loaded there too." : "");
            const limit = dashboard && dashboard.saved_settings && dashboard.saved_settings.memory_limit;
            if (/too little memory/.test(text) && limit && /^\d/.test(limit.value || "")) text = memoryText(text).replace(/\.?$/, ".") + " Your memory limit is " + memoryText(limit.value) + ": raise it, or set it to Automatic, in Setup.";
          } else if (unreadable.length) {
            noticeKind = "found"; title = "Origami can't read " + unreadable[0] + " yet";
            text = "macOS asks before an app reads an external drive. Models there show in the Model menu once you allow it.";
          }
          showFirst(s);
          $("notice").hidden = !noticeKind || noticeKind === "missing"; $("notice").dataset.kind = noticeKind || "";
          if (!noticeKind) return;
          $("notice-title").textContent = title; $("notice-text").textContent = text;
          const asks = noticeKind === "permission" || noticeKind === "found";
          $("notice-steps").hidden = !asks;
          // System Settings opens only from a browser on this Mac.
          $("notice-settings").hidden = !asks || !["127.0.0.1", "localhost", "[::1]"].includes(location.hostname);
          $("notice-retry").hidden = noticeKind === "missing"; $("notice-setup").hidden = noticeKind !== "failed" || !noticeGone && $("tune").hidden;
          $("notice-setup").textContent = noticeGone ? "Open Models" : "Change the setup";
          $("notice-retry").textContent = noticeKind === "stopped" ? "Start" : noticeKind === "degraded" ? "Reload" : noticeKind === "failed" ? "Try again" : noticeKind === "found" ? "Look again" : "Check again";
        }

        // The job band: progress while running, then its actual outcome until dismissed.
        const duration = (s) => s < 60 ? "under a minute" : s < 3600 ? Math.round(s / 60) + " min" : Math.floor(s / 3600) + " h " + Math.round((s % 3600) / 60) + " min";
        function showJob(j) {
          job = j;
          $("job").hidden = !j;
          if (!j) return live.s && !offline ? live(live.s) : gate();
          const going = j.state === "running", unfinished = ["interrupted", "incomplete"].includes(j.state), pct = j.percent != null ? Math.round(j.percent * 100) : null;
          const stage = /pack$|^converting$/i.test(j.stage) ? "Converting" : j.stage === "Downloading" ? "Downloading" : (j.kind === "tune" ? memoryText(j.stage) : j.stage).replace(/\s+/g, " ").slice(0, 120);
          // A tuning run ends at its time limit, whatever its early percent would predict.
          const left = going && j.kind !== "tune" && j.percent > 0.02 && j.seconds > 20 ? j.seconds * (1 - j.percent) / j.percent : null;
          $("job").dataset.state = j.state;
          $("job-title").textContent = j.title + ({ done: ": done", failed: ": stopped", cancelled: ": cancelled", interrupted: ": interrupted", incomplete: ": incomplete" }[j.state] || "");
          $("job-figure").textContent = going && pct != null ? pct + "%" : "";
          $("job-stage").textContent = going ? stage + " · " + duration(j.seconds) + " so far" + (left ? " · about " + duration(left) + " left" : "")
            : unfinished ? j.message || j.error || stage || "Tuning did not finish. Settings were not changed."
            : j.state === "failed" ? j.error || "It stopped." : j.state === "cancelled" ? (j.kind === "tune" ? (/memory/i.test(j.title || "") ? "Stopped. Nothing was changed." : "Stopped. No setting changed; measurements finished so far are kept, and the next run continues from them.") : j.kind === "move" ? "Stopped. The original is where it was." : "Stopped. Anything half written is removed the next time it runs.")
            : j.message ? j.message : j.kind === "tune" ? memoryText(j.lines[j.lines.length - 1] || "Done.").replace(/[;,]? set it with `[^`]*`/, "")
            : j.kind === "hot-copy" ? "Done." : "Added" + (j.added ? " as " + j.added + (noticeKind === "missing" ? "" : "; it is in the Model menu") : "") + ".";
          $("job-track").hidden = unfinished;
          $("job-fill").style.width = (going ? pct || 0 : unfinished ? 0 : 100) + "%";
          if (unfinished) $("job-track").removeAttribute("aria-valuenow");
          else $("job-track").setAttribute("aria-valuenow", going ? pct || 0 : 100);
          $("job-button").textContent = going ? "Cancel" : "Dismiss";
          // `v1/jobs` keeps the last job until it is dismissed: one that ended before this page loaded (or on another page) only
          // shows its result line, and never chooses a model here.
          if (going) jobLive = jobId(j);
          else if (jobLive === jobId(j)) {
            jobLive = null;
            if (j.kind === "hot-copy") $("experts-note").textContent = "";
            if (j.state === "done") jobDone(j); else if (j.kind === "move" || j.kind === "remove") loadStorage();
          }
          // The header's job text follows this poll's job, not the last one's (a page without its key keeps its own words).
          if (live.s && !offline) live(live.s); else gate();
        }
        async function jobDone(j) {
          if (j.kind === "add") {
            found = null; closeAdd(); addedNow = j.added || null; auxAt = 0; await loadModels();
            redrawDials();
          } else { loadModels(); loadFolders(); }
          loadStorage();
        }

        // One name per state, in the header and wherever a view says it (docs/dashboard.md lists them).
        const STATE = { ready: "Ready", loading: "Loading", stopping: "Stopping", stopped: "Stopped", failed: "Failed", reload: "Reload needed", job: "Job running", drive: "Drive gone" };
        // The live strip: state, speed and the prompt being read.
        function live(s) {
          live.s = s;
          const current = s.requests && s.requests.current, queue = s.queue || {};
          let state = s.status === "ready" ? "ready" : s.status === "loading" ? "loading" : "failed", chip, text = "", keyed = null, reading = null;
          const last = ((s.requests && s.requests.recent) || []).filter(measured).slice(-1)[0];
          // A reload is seen through once Origami has answered as not ready: until then a ready answer is still the old model.
          if (reloading && s.status !== "ready" && s.status !== "failed") reloadSeen = true;
          if (s.status === "stopping") { state = "stopping"; chip = STATE.stopping; text = "Unloading the model; a running reply gets up to " + (s.wait_seconds || 30) + " s to finish."; }
          else if (s.status === "stopped") { state = "stopped"; chip = STATE.stopped; text = "Model unloaded" + (s.freed_bytes ? " · about " + gb(s.freed_bytes) + " back to macOS" : "") + (s.note ? " · " + s.note : ""); }
          else if (reloading && reloadBy === "power") { state = "reloading"; chip = STATE.loading; text = "Loading the model again…"; }
          else if (reloading && !s.pending && reloadBy === "use-model") { state = "reloading"; chip = STATE.loading; text = "Loading the model…"; }
          else if (reloading && !s.pending) { state = "reloading"; chip = STATE.loading; text = "Starting with the new setup…"; }
          else if (s.status === "degraded") { chip = STATE.drive; text = s.degraded || ""; }
          else if (s.status === "loading") { chip = STATE.loading; text = "Loading the model" + (s.loading ? " · " + s.loading.seconds + " s" + (s.loading.line ? " · " + s.loading.line : "") : "") + "; requests wait for it."; }
          else if (s.status === "failed" && /No model (found|at)/.test(s.failure || "")) { state = "down"; chip = "No model"; }
          else if (s.status === "failed") { chip = STATE.failed; text = "Not running"; }
          else if (s.status === "unavailable") { state = "unavailable"; chip = "Not responding"; text = "Waiting for the model to answer…"; }
          else if (current && current.phase === "prefill") { state = "busy"; chip = "Working"; reading = current; }
          else if (current && (current.completion_tokens > 0 || current.output_pieces > 0)) { state = "busy"; chip = "Working"; const known = Number.isInteger(current.completion_tokens); const count = known ? current.completion_tokens : current.output_pieces; text = "Writing · " + fmt.format(count) + (known ? " tokens" : " output pieces") + (known && current.decode_tps && count >= 16 ? " · " + one.format(current.decode_tps) + " tok/s" : ""); }
          else if (current) { state = "busy"; chip = "Working"; text = "Starting a request…"; }
          else { chip = running() ? STATE.job : dashboard && dashboard.reload_required ? STATE.reload : STATE.ready; text = last ? "Last reply " + one.format(last.decode_tps) + " tok/s · " + ago(last.time) : s.requests && s.requests.recent && s.requests.recent.length ? "No timed reply yet" : "No requests yet"; }
          if (queue.queued > 0) text += " · " + fmt.format(queue.queued) + " waiting";
          if (running() && !current) text = job.title + (job.percent != null ? " · " + Math.round(job.percent * 100) + "%" : "");
          $("progress").hidden = !reading;
          if (reading) {
            const total = Math.max(1, reading.prompt_tokens), done = reading.completed_prompt_tokens, reused = Math.min(done, reading.reused_tokens || 0);
            $("reused").style.width = (100 * reused / total) + "%";
            $("read").style.left = (100 * reused / total) + "%"; $("read").style.width = (100 * (done - reused) / total) + "%";
            const percent = Math.round(100 * done / total), share = Math.round(100 * reused / total);
            $("progress").setAttribute("aria-valuenow", percent);
            $("progress").setAttribute("aria-valuetext", percent + "% read" + (reused ? "; " + share + "% of the prompt reused from a saved one" : ""));
            text = "Reading a " + fmt.format(reading.prompt_tokens) + "-token prompt · " + percent + "%";
            // The reused share, keyed to the bar's grey part; a fresh prompt shows none.
            if (reused) { keyed = esc(text) + ' · <i class="key" aria-hidden="true"></i>' + share + "% reused"; text += " · " + share + "% reused"; }
          }
          $("chip").dataset.state = state; $("chip-text").textContent = chip; $("page").dataset.state = state;
          if (live.chip !== chip) { live.chip = chip; modelList(); }
          // While a model loads the page names it, in place of the numbers of the one before.
          if (s.status === "loading" && typeof s.model === "string" && s.model) $("title").textContent = modelInitial ? settingName(modelInitial) : s.model;
          if ($("live-text").textContent !== text) { if (keyed) $("live-text").innerHTML = keyed; else $("live-text").textContent = text; }
          setSweeping(state === "busy" || state === "loading" || state === "reloading" || state === "stopping" || running());
          liveState = state; gate();
        }

        // The API's base URL as apps need it: this page's own, also behind a proxy path.
        // On this Mac the page is at 127.0.0.1; once Origami listens on the network, the address it reports reaches it from anywhere.
        const lanPage = () => local() && access && access.listening === "lan" && (access.urls || [])[0];
        const v1Base = () => (lanPage() ? access.urls[0].replace(/#.*$/, "") : location.origin + location.pathname).replace(/\/?(dashboard)?$/, "/v1");
        function recentTable(recent) {
          if (!recent.length) { $("recent").innerHTML = '<p class="empty">No requests yet. Press Measure for a test reply, or point any OpenAI-compatible app at <code>' + esc(v1Base()) + "</code> with the model <code>origami</code>.</p>"; return; }
          const longest = Math.max(0, ...recent.slice(-10).map((r) => r.seconds || 0));
          $("recent").innerHTML = '<table><thead><tr><th>When</th><th>Prompt</th><th class="r">First word</th><th class="r">Tok/s</th><th class="wide">Time</th><th class="wide"></th></tr></thead><tbody>' +
            recent.slice(-10).reverse().map((r) => {
              const retried = r.gpu_retries ? "retried after macOS needed the GPU" : "", alarm = alarming.includes(r.finish) ? endings[r.finish] : retried;
              return '<tr><td><time datetime="' + esc(r.time) + '" title="' + esc(clock(r.time)) + '">' + ago(r.time) + "</time></td><td>" + (r.prompt_tokens != null ? fmt.format(r.prompt_tokens) + " tokens" : "–") +
              (r.cached_tokens ? '<span class="sub">' + fmt.format(r.cached_tokens) + " reused · " + Math.round(100 * r.cached_tokens / Math.max(1, r.prompt_tokens)) + "%</span>" : "") +
              (alarm ? '<span class="sub ending"><span class="tag">' + esc(alarm) + "</span></span>" : "") + '</td><td class="r">' + (r.ttft_s != null ? one.format(r.ttft_s) + " s" : "–") +
              '</td><td class="r">' + (measured(r) ? one.format(r.decode_tps) : "–") + '</td><td class="wide t-split">' + timeBar(r, longest) + '</td><td class="wide">' + [endings[r.finish], retried, r.mode && r.mode !== "exact" ? r.mode + (r.exact_from != null ? ", exact in the tool call" : "") : ""].filter(Boolean).map((t) => '<span class="tag">' + esc(t) + "</span>").join(" · ") + "</td></tr>";
            }).join("") + "</tbody></table>";
        }
        // Every saved entry, system prompts first; each opens in the dialog below.
        function prompts(s) {
          const entries = ((s.prompt_cache && s.prompt_cache.entries) || []).slice().sort((a, b) => (b.kind === "system") - (a.kind === "system"));
          $("prompts-count").textContent = entries.length ? fmt.format(entries.length) + " · " + fmt.format(entries.reduce((n, e) => n + e.tokens, 0)) + " tokens" : "";
          $("prompts").innerHTML = !entries.length ? '<p class="empty">No prompts saved yet. A chat app\'s system prompt is kept here after its first reply, so the next chat starts at once; recent prompts and conversations are kept for follow-ups.</p>' :
            '<table><thead><tr><th>Name</th><th class="r">Tokens</th><th class="wide">Last used</th><th></th></tr></thead><tbody>' +
            entries.map((e) => "<tr><td>" + (e.name ? esc(e.name) : '<span class="muted">unnamed</span>') + (e.pinned ? ' <span class="tag kept">kept</span>' : "") +
              '<span class="sub">' + (e.kind === "system" ? "system prompt" : "prompt or conversation") + '</span></td><td class="r">' + fmt.format(e.tokens) +
              '</td><td class="wide">' + ago(e.last_used) + '</td><td class="r">' + (e.id ? '<button type="button" data-open="' + esc(e.id) + '" aria-label="' +
              esc("Open " + (e.name || "unnamed") + ", " + fmt.format(e.tokens) + " tokens") + '">Open</button>' : "") + "</td></tr>").join("") + "</tbody></table>";
          gate();
        }
        // The Experts card: each place the running model's experts are read from (`experts_from`), with its role, size and
        // state in the card's words, the share of expert reads it served, and the actions on the hot copy and the packs. Redrawn
        // only when it changes.
        const ROLES = { all: "all experts, as released", pack: "all experts, N-bit", hot: "hot copy", more: "next most-used" };
        const DEFINE = { hot: "Hot copy: the experts replies use most, copied to a second drive so both drives read at once. Replies are unchanged.",
          pack: "All experts, N-bit: every expert packed once into one record, one read instead of nine. Replies are unchanged.", more: "Next most-used: more of the most-used experts, on a further drive." };
        // Each state's mark (as the status chip's: ink ready, accent live work, graphite waiting, none when not possible) and words.
        const STATES = { reading: ["ink", "Reading"], not_read: ["dim", "Not read"], ready: ["ink", "Ready"], building: ["live", "Building"],
          paused: ["dim", "Paused · the model is reading"], learning: ["dim", "Learning: built after the first reply"], off: ["dim", "Off"],
          not_needed: ["dim", "Not needed"], not_possible: ["none", "Not possible here"], not_connected: ["none", "Not connected"], not_allowed: ["none", "Not allowed"],
          failed: ["fail", "Could not build"], not_built: ["dim", "Not built yet"] };
        const place = (drive) => !drive || drive === "internal drive" ? "This Mac" : drive;
        const gbs = (bytes) => (bytes >= 1e10 ? fmt.format(Math.round(bytes / 1e9)) : one.format(bytes / 1e9)) + " GB";
        const tbs = (bytes) => (bytes >= 1e12 ? one.format(bytes / 1e12) + " TB" : gbs(bytes || 0));
        const soon = (s) => s < 60 ? Math.max(5, Math.round(s / 5) * 5) + " s" : duration(s);
        const pct = (x) => Math.round(100 * (x || 0)) + "%";
        // A path, escaped, that wraps after its slashes rather than inside a folder's name.
        const wpath = (p) => esc(p).replace(/\//g, "/<wbr>");
        // The hot copy row and the page's own requests to `v1/hot-copy` (`xPost`: the action on its way), and the Change… or
        // Move… panel (`xOpen`: "hot", or the path of the pack it moves; false while closed).
        let xFrom = null, xHot = null, xPost = null, xOpen = false, xDrive = null, xSize = null, xCleared = [];
        // The size slider's bytes and unit ("gb", or "pct": a share of the experts' bytes).
        let xCap = null, xUnit = "gb";
        const roleText = (r) => (ROLES[r.role] || r.role).replace("N-bit", (r.bits || "") + "-bit");
        const defineText = (r) => (r.meaning || DEFINE[r.role] || "").replace(/N-bit/g, (r.bits || "") + "-bit").replace("N bits", (r.bits || "") + " bits");
        // A copy's popularity, when it has counted uses to draw (else the slider is in GB with no graph).
        const xPop = (row) => row && row.popularity && row.popularity.counted_uses > 0 && row.popularity.bytes > 0 ? row.popularity : null;
        const xPanel = () => !xFrom || !xOpen ? null : xOpen === "hot" ? xHot : (xFrom.sources || []).find((r) => r.path === xOpen && (r.moves || []).length) || null;
        function sourceRow(r, from) {
          const [mark, word] = STATES[r.state] || ["dim", r.state], p = r.progress, live = (r.state === "building" || r.state === "paused") && p && p.total_bytes;
          let state = word;
          if (r.state === "ready" && r.records) state += " · " + fmt.format(r.records) + " experts";
          if (r.state === "building" && p && p.seconds_left != null) state += " · about " + soon(p.seconds_left) + " left";
          const size = r.state === "learning" && r.budget_bytes ? "up to " + gbs(r.budget_bytes) : live ? gbs(p.done_bytes).replace(" GB", "") + " of " + gbs(p.total_bytes) : r.bytes ? gbs(r.bytes) : "–";
          const detail = [r.detail || "", r.backlog_records > 0 ? fmt.format(r.backlog_records) + " swaps wait for the hourly allowance (" + gbs(r.backlog_bytes || 0) + ", " +
            fmt.format(r.backlog_uses || 0) + " uses)" : ""].filter(Boolean).join(" · ");
          const done = live ? Math.round(100 * p.done_bytes / p.total_bytes) : 0;
          const track = live ? '<span class="job-track x-track" role="progressbar" aria-label="Hot copy written" aria-valuemin="0" aria-valuemax="100" aria-valuenow="' + done + '"><span style="width:' + done + '%"></span></span>' : "";
          const reads = r.read_share == null || !(from.split && from.split.reads) || r.state === "not_read" || (r.role === "hot" && !r.read_share) ? "–" : pct(r.read_share);
          // A copy that is off or cannot be built has no folder to show.
          const path = r.path && !(r.role === "hot" && (r.state === "off" || r.state === "not_needed" || r.state === "not_possible")) ? r.path : "";
          const define = defineText(r) ? ' title="' + esc(defineText(r)) + '"' : "";
          return "<tr><td>" + esc(place(r.drive)) + (path ? '<span class="sub wide">' + wpath(path) + "</span>" : "") + '</td><td class="x-role"' + define + ">" + esc(roleText(r)) +
            '</td><td class="r x-sz">' + size + '</td><td class="x-state"><i class="mark" data-m="' + mark + '" aria-hidden="true"></i>' + esc(state) + track +
            (detail ? '<span class="sub">' + esc(detail) + (r.state === "not_allowed" ? ' <a href="#notice">How to allow it</a>' : "") + "</span>" : "") +
            '</td><td class="r x-reads"' + (reads === "–" ? "" : ' data-l="reads"') + ">" + reads + "</td></tr>";
        }
        // "Is it faster?": the median writing speed of recent replies the hot copy served reads to against those it did not, only
        // with three replies or more on each side.
        const median = (a) => { const b = a.slice().sort((x, y) => x - y), m = b.length >> 1; return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2; };
        function hotSpeed(s) {
          const done = ((s.requests && s.requests.recent) || []).filter((r) => measured(r) && typeof r.hot_copy === "boolean");
          const on = done.filter((r) => r.hot_copy).map((r) => r.decode_tps), off = done.filter((r) => !r.hot_copy).map((r) => r.decode_tps);
          return on.length < 3 || off.length < 3 ? "" : "Writing with the hot copy: " + one.format(median(on)) + " tokens a second (median of " + on.length + " replies), "
            + one.format(median(off)) + " without it (" + off.length + " replies).";
        }
        // The header's summary of the hot copy, linking to the card: its state in a word or two.
        function hotChip(hot) {
          const p = hot && hot.progress, words = !hot ? "" : hot.state === "building" ? "building" + (p && p.total_bytes ? " " + Math.round(100 * p.done_bytes / p.total_bytes) + "%" : "")
            : { learning: "learning", paused: "paused", ready: "ready", failed: "could not build", not_built: "not built" }[hot.state] || "";
          $("x-chip").hidden = !words; $("top").toggleAttribute("data-hot", !!words);
          if (!words) return;
          $("x-chip").dataset.state = hot.state === "building" ? "busy" : hot.state === "ready" ? "ready" : "down";
          // A phone shows a build as its percentage alone, beside the turning mark.
          const html = esc(words).replace(/^building (\d)/, '<span class="x-hot">building </span>$1');
          if ($("x-chip-text").innerHTML !== html) $("x-chip-text").innerHTML = html;
          $("x-chip").setAttribute("aria-label", "Hot copy: " + words + ". Go to Experts");
        }
        function expertsCard(s) {
          const from = s.experts_from;
          $("experts").hidden = noModel || !from || !(from.sources || []).length;
          if ($("experts").hidden) { xFrom = xHot = null; hotChip(null); return; }
          xFrom = from; const sources = from.sources, hot = xHot = sources.find((r) => r.role === "hot") || null;
          hotChip(hot);
          $("x-model").textContent = [s.model && s.model.name, s.model && s.model.experts || from.released && "released as " + from.released].filter(Boolean).join(" · ");
          const lede = ["Each reply reads a few experts per layer. The most-used stay in memory; the rest are read from these drives."].concat(from.no_hot_copy ? [from.no_hot_copy] : [],
            ["hot", "pack", "more"].map((role) => sources.find((r) => r.role === role)).filter(Boolean).map(defineText));
          const ledeHtml = lede.map((line) => "<span>" + esc(line) + "</span>").join("");
          if ($("x-lede").innerHTML !== ledeHtml) $("x-lede").innerHTML = ledeHtml;
          const table = '<table><thead><tr><th>Drive</th><th>Role</th><th class="r">Size</th><th>State</th><th class="r">Reads</th></tr></thead><tbody>' + sources.map((r) => sourceRow(r, from)).join("") + "</tbody></table>";
          if (expertsCard.table !== table) $("x-sources").innerHTML = expertsCard.table = table;
          // The read split: memory in the accent (what memory holds), the hot copy in ink, the model's drive in graphite.
          const sp = from.split || {}, reads = sp.reads || 0, hotShown = sp.hot_copy != null && hot && (sp.hot_copy > 0 || hot.state === "ready");
          $("rs-mem").style.width = reads ? pct(sp.memory) : "0%"; $("rs-hot").style.width = reads && hotShown ? pct(sp.hot_copy) : "0%"; $("rs-drive").style.width = reads ? pct(sp.model_drive) : "0%";
          const drives = [...new Set(sources.filter((r) => r.role !== "hot" && r.state === "reading").map((r) => place(r.drive)))];
          $("x-said").textContent = !reads ? "No replies yet." : "Of " + fmt.format(reads) + " expert reads so far: " + [pct(sp.memory) + " in memory",
            hotShown ? pct(sp.hot_copy) + " from the hot copy on " + place(hot.drive) : "", pct(sp.model_drive) + " from " + (drives.length === 1 ? drives[0] : "the model's drives")].filter(Boolean).join(", ") + ".";
          $("x-speed").textContent = hot ? hotSpeed(s) : "";
          // Settings saved that do nothing for this model, each named, with one Clear.
          const unused = (from.unused_settings || []).filter((u) => !xCleared.includes(u.key));
          const unusedHtml = unused.length ? "Not used by " + esc(s.model && s.model.name || "this model") + ": " + unused.map((u) => esc(u.label) + ' <span class="muted">(' + esc(u.why) + ")</span>").join("; ")
            + ' <button type="button" data-xact="clear">Clear</button>' : "";
          if ($("x-unused").innerHTML !== unusedHtml) { $("x-unused").innerHTML = unusedHtml; expertsCard.acts = null; }
          // The actions, only where they apply: the hot copy's Change… and Build now (which waits for a first reply, and says
          // so), Move… for a pack, and for Qwen a further copy on each drive with room that holds none yet.
          const acts = [];
          if (hot && (hot.choices || []).length) acts.push(["change", "Change…", ' aria-expanded="' + (xOpen === "hot") + '" aria-controls="x-change"']);
          if (hot && (["learning", "failed", "not_built"].includes(hot.state) || hot.state === "ready" && hot.topup_bytes > 0))
            acts.push(["build", "Build now", hot.can_build ? "" : ' data-why="Available after the first reply" data-blocked aria-describedby="x-why"', hot.can_build ? "" : "after the first reply"]);
          if (hot && hot.state === "off") acts.push(["on", "Turn on", ""]);
          if (hot && hot.state === "not_possible" && /share/.test(hot.detail || ""))
            (hot.choices || []).filter((c) => c.status === "has room").forEach((c) => acts.push(["put:" + c.path, "Put it on " + place(c.drive), ""]));
          const packed = sources.some((r) => r.role === "pack");
          (from.copy_to || []).forEach((c) => acts.push(["copy:" + c.path, (packed ? "Copy to " : "Copy all experts, 4-bit, to ") + place(c.drive), ""]));
          const actsHtml = acts.map(([act, label, extra, why]) => { const b = '<button type="button" data-xact="' + esc(act) + '"' + extra + ">" + esc(label) + "</button>";
            return why ? '<span class="x-pair">' + b + '<span class="x-why" id="x-why">' + esc(why) + "</span></span>" : b; }).join("");
          if (expertsCard.acts !== actsHtml) { $("x-acts").innerHTML = expertsCard.acts = actsHtml; gate(); }
          showChange();
        }
        // The hot copy's size as saved: "auto", "off" or a size such as "40GB".
        function savedSize() {
          const v = status && status.settings && status.settings.hot_copy ? String(status.settings.hot_copy.value) : "auto";
          return v === "off" ? "off" : /^\(?auto/.test(v) || !parseFloat(v) ? "auto" : copySize(parseFloat(v) * (/TB/i.test(v) ? 1e12 : 1e9));
        }
        // Change… (the hot copy: a drive and a size) or Move… (a pack: a drive): one row per drive, the data folder's first,
        // the one in use marked and showing the folder itself; choosing it again changes nothing.
        function showChange() {
          const row = xPanel(), hot = !!row && row.role === "hot", choices = !row ? [] : (hot ? row.choices : row.moves) || [];
          $("x-change").hidden = !choices.length;
          if ($("x-change").hidden) return;
          $("x-change-head").textContent = hot ? "Where the hot copy goes" : "Where the " + roleText(row).replace(/^all experts, /, "") + " experts go";
          $("x-change-text").textContent = hot ? "Choose a drive; Origami makes the folder on it." : "Origami copies them to the drive, checks the copy, then deletes them here; the model stops meanwhile.";
          const can = (c) => c.status !== "holds all experts" && c.status !== "too small", current = choices.find((c) => c.status === "goes here");
          const chosen = choices.find((c) => c.path === xDrive && can(c)) || current || choices.find(can);
          xDrive = chosen ? chosen.path : null;
          const html = choices.map((c) => '<label class="' + (can(c) ? "" : "no") + '"><input type="radio" name="x-drive" value="' + esc(c.path) + '"' + (c === chosen ? " checked" : "") +
            (can(c) ? "" : ' disabled aria-disabled="true"') + "><span>" + esc(place(c.drive)) + '<span class="sub">' + wpath(c.folder || c.path) + "</span></span><span>" +
            (c.free_bytes != null ? disk(c.free_bytes) + " free" : "") + "</span><span>" + (c === current ? "in use" : c === chosen ? "goes here" : c.status === "goes here" ? "has room" : esc(c.status)) + "</span></label>").join("");
          if ($("x-drives").innerHTML !== html) $("x-drives").innerHTML = html;
          $("x-size-row").hidden = !hot;
          if (!hot) { $("x-places-row").hidden = true; return; }
          const saved = savedSize(), pop = xPop(row);
          // The most the chosen drive takes (its free space, with the copy already on it, less what it keeps free); the slider
          // spans the whole model.
          const room = chosen && chosen.room_bytes != null ? chosen.room_bytes : Infinity, whole = pop ? pop.bytes : Math.min(room, 2e11);
          const keep = chosen && chosen.room_bytes != null ? chosen.free_bytes + (chosen === current ? row.bytes || 0 : 0) - chosen.room_bytes : 0;
          if (xSize == null) { xSize = saved; xCap = null; }
          const auto = chosen === current && row.budget_bytes ? row.budget_bytes : Math.min(room, whole);
          if (xCap == null) xCap = capClamp(xSize === "auto" ? auto : xSize === "off" ? 0 : parseFloat(xSize) * 1e9, room, whole);
          // The share of counted expert uses a copy of each size would hold: from the bars, else the server's `coverage` points.
          const share = (bytes) => pop ? copyCover(pop, bytes) : ((row.coverage || []).find((x) => Math.abs(x.bytes - bytes) < 5e8) || {}).share;
          const covers = (bytes) => share(bytes) != null ? " · " + pct(share(bytes)) + " of uses" : "";
          const sized = xSize === "auto" || xSize === "off" ? copySize(xCap || auto) : xSize;
          const label = (v) => v === "auto" ? "Automatic" + (chosen === current && row.budget_bytes ? " (" + gbs(row.budget_bytes) + covers(row.budget_bytes) + ")" : "")
            : v === "off" ? "Off" : gbs(parseFloat(v) * 1e9) + covers(parseFloat(v) * 1e9);
          const select = ["auto", sized, "off"].map((v) => '<option value="' + esc(v) + '"' + (v === xSize ? " selected" : "") + ">" + esc(label(v)) + "</option>").join("");
          if ($("x-size").innerHTML !== select) { $("x-size").innerHTML = select; $("x-size").value = xSize; }
          // The size in words with the share of uses it covers, then what sits at its edge (`popularity`: the ranking the copy is
          // built from, so the graph and the copy agree).
          const cap = xSize === "off" ? 0 : xCap, of = pop && xUnit === "pct" ? " (" + one.format(100 * cap / whole) + "% of the experts)" : "";
          const note = (cap === 0 ? "Off: every expert is read from the model's drive." : share(cap) != null ? "A copy of " + gbs(cap) + of + " would cover " + pct(share(cap)) + " of expert uses." : "")
            + (saved === "auto" && xSize === "auto" && row.size_note ? " " + row.size_note : "");
          if ($("x-size-note").textContent !== note.trim()) $("x-size-note").textContent = note.trim();
          const full = cap > 0 && cap >= room - 5e8 && room < whole;
          const edge = !pop ? "Still learning: no replies counted yet." : cap === 0 ? "" : copyEdge(pop, cap) + (full ? " No room for more on " + place(chosen.drive) + (keep > 0 ? ": it keeps " + gbs(keep) + " free." : ".") : "");
          if ($("x-size-edge").textContent !== edge) $("x-size-edge").textContent = edge;
          $("x-graph").hidden = $("x-unit").hidden = !pop;
          if (pop) {
            // The copy places, in the order they take the ranks: one per model today.
            swap("x-plot", copyGraph(pop, [{ start: 0, cap, held: pop.held_bytes || 0, room, drive: place(chosen ? chosen.drive : row.drive) }]));
            $("x-yaxis").textContent = "uses per expert" + (copyLog(pop) ? ", log scale" : "");
          }
          const gbUnit = xUnit === "gb" || !pop, slider = $("x-cap");
          ["gb", "pct"].forEach((u) => $("x-unit-" + u).setAttribute("aria-pressed", String(gbUnit === (u === "gb"))));
          slider.min = "0"; slider.step = "1"; slider.max = String(gbUnit ? whole / 1e9 : 100);
          const at = capValue(cap, gbUnit ? "gb" : "pct", whole);
          if (document.activeElement !== slider || Math.abs(parseFloat(slider.value) - at) >= 1) slider.value = String(at);
          slider.style.setProperty("--fill", (100 * at / (parseFloat(slider.max) || 1)).toFixed(2) + "%");
          slider.setAttribute("aria-valuetext", cap === 0 ? "Off" : gbs(cap) + (pop ? ", " + one.format(100 * cap / whole) + "% of the experts" : "") + covers(cap).replace(" · ", ", "));
          showPlaces(pop, cap, place(chosen ? chosen.drive : row.drive));
          // `hot_copy_keep`: only with a size set does an idle copy keep it.
          const kept = status && status.settings && status.settings.hot_copy_keep;
          $("x-keep-row").hidden = !kept || saved === "auto" || saved === "off";
          $("x-keep").checked = !!kept && kept.value === "true";
          swap("x-shared", sharedCopies(status || {}));
        }
        // Every model's copies on each drive (`disk_budget` holders): size, limit, what an idle one keeps (its floor) and what
        // it gave or took, with a box per idle model to keep its size when idle (`hot_copy_keep`; the loaded one's is the box above);
        // drives with an idle model's copy only.
        const volumeName = (v) => v === "/" ? "This Mac" : v.split("/").pop() || v;
        function sharedCopies(s) {
          const kept = ((s.settings || {}).hot_copy_keep || {}).value === "true", loadedName = s.model && s.model.name;
          return (s.disk_budget || []).map((v) => [v, (v.holders || []).filter((h) => h.model)]).filter(([, hs]) => hs.length > 1 || hs.some((h) => !h.loaded)).map(([v, hs]) =>
            '<p class="label">Copies on ' + esc(volumeName(v.volume)) + "</p><table><tbody>" + hs.map((h) => {
              const gave = (h.gave || []).reduce((n, g) => n + (g.bytes || 0), 0), took = (h.took || []).reduce((n, g) => n + (g.bytes || 0), 0);
              const keep = h.loaded ? kept : !!h.keep_when_idle, cap = h.cap == null || h.cap === "auto" ? "automatic" : gbs(+h.cap);
              return "<tr><td>" + esc(h.model) + '<span class="sub">' + (h.loaded ? "loaded" : "idle") + " · limit " + cap + "</span></td><td class=\"r\">" + gbs(h.bytes || 0) + "</td><td>" +
                [h.loaded ? "" : "keeps " + gbs(keep ? h.bytes || 0 : h.floor_bytes || 0) + " when another model grows", gave ? "gave " + gbs(gave) : "", took ? "took " + gbs(took) : ""].filter(Boolean).join(" · ") +
                '</td><td>' + (h.loaded ? "" : '<label class="x-keep"><input type="checkbox" data-keep-model="' + esc(h.model) + '"' + (keep ? " checked" : "") + "> Keep when idle</label>") + "</td></tr>";
            }).join("") + "</tbody></table>").join("");
        }
        // The disk budget where storage is shown: per drive, what it keeps free, what copies gave back for saves and the saves
        // skipped for space.
        function diskBudget(s) {
          return ((s && s.disk_budget) || []).filter((v) => (v.holders || []).length || v.give_backs || v.saves_skipped).map((v) =>
            '<p class="sub">' + esc(volumeName(v.volume)) + ": keeps " + gbs(v.reserve_bytes || 0) + " free" + (v.free_bytes != null ? ", " + disk(v.free_bytes) + " free now" : "") +
            (v.given_back_bytes ? " · copies gave back " + gbs(v.given_back_bytes) + " for saves" : "") + (v.saves_skipped ? " · " + fmt.format(v.saves_skipped) + " save" + (v.saves_skipped === 1 ? "" : "s") + " skipped for space" : "") + "</p>").join("");
        }
        // The size the slider stands for: at least 1 GB, at most the drive's room and the whole model.
        const capClamp = (bytes, room, whole) => Math.max(Math.min(1e9, room, whole), Math.min(bytes, room, whole));
        // A size as saved (`hot_copy`): decimal GB to a tenth, as `origami config` takes it.
        const copySize = (bytes) => Math.round(bytes / 1e8) / 10 + "GB";
        // The slider's value in its unit to bytes and back: whole GB, or percent of the experts' bytes (`whole`) to a tenth of a GB.
        const capBytes = (value, unit, whole) => unit === "pct" ? Math.round(value / 100 * whole / 1e8) * 1e8 : Math.round(value) * 1e9;
        const capValue = (bytes, unit, whole) => unit === "pct" ? 100 * bytes / whole : bytes / 1e9;
        // The share of counted uses a copy of `bytes` holds, from the bars (a bar the edge cuts counts for its part).
        const copyCover = (pop, bytes) => { let at = 0, held = 0; for (const b of pop.bars) { held += b[2] * Math.min(1, Math.max(0, (bytes - at) / b[1])); at += b[1]; } return held / pop.counted_uses; };
        // The copy place a bar is in (its middle is), or undefined.
        const copyIn = (places, at, bytes) => places.find((p) => at + bytes / 2 >= p.start && at + bytes / 2 <= p.start + p.cap);
        // Uses per expert span orders of magnitude on GLM and DeepSeek (log scale), about one on Qwen (linear).
        const copyLog = (pop) => { const v = pop.bars.map((b) => b[2] / b[0]).filter((x) => x > 0); return Math.max(...v) > 20 * Math.min(...v); };
        // The bars: x the experts by rank (most used first), each bar as wide as its bytes; y the uses per expert in it. Inside
        // a place's size solid, outside faint; each place's copy now (a dashed mark), its chosen size (a line), no room hatched.
        function copyGraph(pop, places) {
          const v = pop.bars.map((b) => b[2] / b[0]), top = Math.max(...v), low = Math.min(...v.filter((x) => x > 0)) / 2, log = copyLog(pop);
          const h = (x) => x <= 0 ? 0 : Math.max(0.015, log ? Math.log(x / low) / Math.log(top / low) : x / top);
          let at = 0, bars = "";
          pop.bars.forEach((b, i) => {
            bars += '<rect x="' + (600 * at / pop.bytes).toFixed(2) + '" width="' + (600 * b[1] / pop.bytes + 0.6).toFixed(2) + '" y="' + (120 * (1 - h(v[i]))).toFixed(1)
              + '" height="' + (120 * h(v[i])).toFixed(1) + '"' + ((p) => !p ? "" : p.on ? ' class="on"' : ' class="in"')(copyIn(places, at, b[1])) + "/>";
            at += b[1];
          });
          const x = (bytes) => (100 * Math.min(1, bytes / pop.bytes)).toFixed(2) + "%";
          return '<svg viewBox="0 0 600 120" preserveAspectRatio="none" aria-hidden="true">' + bars + "</svg>" + places.map((p) => p.on
            ? (p.cap > 0 ? '<span class="x-on" style="left:' + x(p.start) + ";width:" + x(p.cap) + '"><span>on ' + esc(p.drive) + "</span></span>" : "") :
            (p.room < pop.bytes - p.start ? '<span class="x-noroom" style="left:' + x(p.start + p.room) + '"><span>no room on ' + esc(p.drive) + "</span></span>" : "") +
            (p.held > 0 ? '<i class="x-now" style="left:' + x(p.start + p.held) + '"></i>' : "") +
            (p.cap > 0 ? '<i class="x-edge" style="left:' + x(p.start + p.cap) + '"></i>' : "")).join("");
        }
        // The answer at the edge: how often the last expert in and the first one left out are used, per token counted.
        function copyEdge(pop, cap) {
          if (!(pop.tokens >= 1000)) return "Still learning: " + fmt.format(pop.tokens || 0) + " tokens counted.";
          const once = (u) => !(u > 0) ? "not used yet" : pop.tokens / u < 1.5 ? "every token" : "once per " + fmt.format(Math.round(pop.tokens / u)) + " tokens";
          let at = 0, last = -1;
          pop.bars.forEach((b, i) => { if (at + b[1] / 2 <= cap) last = i; at += b[1]; });
          if (last < 0) return "The first expert in is used " + once(pop.bars[0][3]) + ".";
          if (last === pop.bars.length - 1) return "Every counted expert is in" + (pop.bars[last][4] > 0 ? "; the last one is used " + once(pop.bars[last][4]) : "") + ".";
          const inside = once(pop.bars[last][4]), outside = once(pop.bars[last + 1][3]);
          return "The last expert in is used " + inside + "; the next one left out, " + (inside === outside ? "about as often" : outside) + ".";
        }
        // More drives (`hot_copy_more`): one row per further copy place (`experts_from.sources`, role hot, `place` > 0) and per
        // drive that could hold one (`unused_drives`). A ticked drive holds a copy with its own size over the same graph, its
        // ranks after the faster drives' (drawn "on <drive>"); saving posts the whole list (`v1/hot-copy {"places": [...]}`).
        // `xPlaces`: the edited list (null: as the server has it). `xSavedAt`: after a save, the polls asked for before it
        // (their status is the old list): the saved list stays, and cannot be saved again, until a later one arrives.
        let xPlaces = null, xSavedAt = null;
        const speedText = (bps) => bps > 0 ? one.format(bps / 1e9) + " GB/s" : "speed not measured";
        function placeList() {
          const hots = ((xFrom && xFrom.sources) || []).filter((r) => r.role === "hot"), more = hots.filter((r) => (r.place || 0) > 0);
          // The server's rows stay current (state, size held); what the owner ticked or sized stays until saved or closed.
          for (const p of xPlaces || []) if (p.was) p.row = more.find((r) => (r.place_folder || r.path) === p.key) || p.row;
          if (xPlaces == null) {
            xPlaces = more.map((r) => ({ key: r.place_folder || r.path, folder: r.place_folder, drive: r.drive, row: r, on: true, was: true,
              cap: r.cap === "auto" ? null : r.cap === "off" ? 0 : r.cap })).concat(((xFrom && xFrom.unused_drives) || []).map((d) => ({ key: d.folder || d.path,
              path: d.path, folder: d.folder, drive: d.drive, unused: d, on: false, was: false, cap: null })));
          }
          return xPlaces;
        }
        // The bytes a place's copy would take: its size, else (automatic) what the keeper chose or the drive's room.
        // Past the ranks no faster drive holds, a place's room holds the most-used again (duplicates), up to the whole
        // model; automatic, it starts at the ranks left, at least 20 GB, as the keeper's automatic size does.
        function placeCap(p, pop, start = 0) {
          const room = placeRoom(p), whole = pop ? pop.bytes : room, left = pop ? Math.max(0, pop.bytes - start) : room;
          if (p.cap != null) return Math.min(p.cap, room, whole);
          return Math.min(p.row && p.row.budget_bytes ? p.row.budget_bytes : Math.max(left, 2e10), room, whole);
        }
        function placeRoom(p) {
          if (p.unused) return Math.max(0, (p.unused.free_bytes || 0) - 2e10);
          const c = ((p.row && p.row.choices) || []).find((x) => x.status === "goes here");
          return c && c.room_bytes != null ? c.room_bytes : Math.max(0, (p.row.free_bytes || 0) + (p.row.bytes || 0) - (p.row.reserve_bytes || 0));
        }
        function showPlaces(pop, firstCap, firstDrive) {
          const list = placeList(), shown = list.length > 0;
          $("x-places-row").hidden = !shown;
          if (!shown) return;
          // The rows, redrawn only when which drives are ticked changes (a slider being dragged is never replaced); the graphs
          // and lines are filled in each time. Faster first: the server's start for a place it has, a new one after them.
          const facts = (p) => {
            const r = p.row, u = p.unused, why = u && u.why_not;
            const state = r ? (STATES[r.state] || ["", r.state])[1] + (r.records ? " · " + fmt.format(r.records) + " experts" : "") : why ? "" : "not used";
            return [speedText((r || u).speed_bytes_per_second), (r || u).free_bytes != null ? disk((r || u).free_bytes) + " free" : "", state].filter(Boolean).join(" · ");
          };
          const html = list.map((p, i) => {
            const u = p.unused, why = u && u.why_not ? u.why_not : "", name = place(p.drive), id = "x-pl-" + i;
            return '<div class="x-place"><label class="' + (why ? "no" : "") + '"><input type="checkbox" data-key="' + id + '" data-place="' + i + '"' + (p.on ? " checked" : "")
              + (why ? ' disabled aria-describedby="' + id + '-why"' : "") + ' aria-label="Keep a copy on ' + esc(name) + '"><span>' + esc(name)
              + (p.folder ? '<span class="sub">' + wpath(p.folder) + "</span>" : "") + '</span><span id="' + id + '-facts">' + esc(facts(p)) + "</span></label>"
              + (why ? '<span class="sub" id="' + id + '-why">' + esc(name) + ": " + esc(why) + "</span>" : "")
              + (p.on && pop ? '<div class="x-copy"><div class="x-graph"><div class="x-plot" id="' + id + '-plot"></div></div>'
                + '<input type="range" data-key="' + id + '-cap" data-pcap="' + i + '" id="' + id + '-cap" step="1" aria-label="Copy size on ' + esc(name) + '" aria-describedby="' + id + '-note ' + id + '-edge">'
                + '<span class="sub" id="' + id + '-note"></span><span class="sub" id="' + id + '-edge"></span></div>' : "") + "</div>";
          }).join("");
          swap("x-places", html);
          let start = firstCap;
          const placed = [{ start: 0, cap: firstCap, held: 0, room: Infinity, drive: firstDrive, on: true }];
          list.forEach((p, i) => {
            const id = "x-pl-" + i, r = p.row;
            if ($(id + "-facts")) $(id + "-facts").textContent = facts(p);
            if (!(p.on && pop) || !$(id + "-cap")) return;
            const name = place(p.drive), room = placeRoom(p);
            const at = r && r.popularity && r.popularity.before_bytes != null ? r.popularity.before_bytes : start, cap = placeCap(p, pop, at);
            const adds = pct(copyCover(pop, at + cap) - copyCover(pop, at)), again = Math.max(0, at + cap - pop.bytes);
            const dup = r && r.duplicates ? " " + fmt.format(r.duplicates) + " experts here are also on a faster drive (spare room)." : "";
            swap(id + "-plot", copyGraph(pop, placed.concat([{ start: at, cap, held: r ? r.bytes || 0 : 0, room, drive: name }])));
            // The slider spans the graph's ranks, so its thumb sits on the edge it sets: from the start of this drive's
            // share to the end of the ranking or its room; a size past the ranks left (duplicates) pins it at the end.
            const slider = $(id + "-cap"), max = Math.max(1, Math.round(Math.min(pop.bytes, at + room) / 1e9));
            slider.min = "0"; slider.max = String(max); slider.dataset.start = String(at);
            if (document.activeElement !== slider) slider.value = String(Math.min(max, Math.round((at + cap) / 1e9)));
            slider.style.setProperty("--fill", (100 * parseFloat(slider.value) / max).toFixed(2) + "%");
            slider.setAttribute("aria-valuetext", gbs(cap) + ", adds " + adds + " of uses");
            $(id + "-note").textContent = (p.cap == null ? "Automatic: " : "") + gbs(cap) + " on " + name + " adds " + adds + " of expert uses after the faster drives"
              + (again > 5e8 ? "; " + gbs(again) + " of it holds the most-used again, read from whichever drive is sooner." : ".") + dup;
            $(id + "-edge").textContent = copyEdge(pop, at + cap);
            placed.push({ start: at, cap, held: 0, room: Infinity, drive: name, on: true });
            start = at + cap;
          });
          const sizeOf = (r) => r.cap === "auto" ? null : r.cap === "off" ? 0 : r.cap;
          const changed = list.some((p) => p.on !== p.was || p.on && p.was && p.cap !== sizeOf(p.row));
          $("x-places-save").disabled = !changed || !!xPost || xSavedAt != null;
          const why = xPost || xSavedAt != null ? "Wait for the current change to finish." : changed ? "" : "Tick a drive or move a size to save.";
          $("x-places-why").textContent = $("x-places-save").title = why;
        }
        $("x-places").addEventListener("change", (event) => {
          const i = event.target.dataset && event.target.dataset.place;
          if (i == null) return;
          xPlaces[+i].on = event.target.checked;
          showChange();
        });
        $("x-places").addEventListener("input", (event) => {
          const i = event.target.dataset && event.target.dataset.pcap;
          if (i == null) return;
          const start = parseFloat(event.target.dataset.start) || 0;
          xPlaces[+i].cap = Math.max(1e9, parseFloat(event.target.value) * 1e9 - start);
          showChange();
        });
        $("x-places-save").addEventListener("click", async () => {
          const list = placeList(), removed = list.filter((p) => p.was && !p.on);
          const body = { places: list.filter((p) => p.on).map((p) => Object.assign(p.folder && !p.unused ? { folder: p.folder } : { drive: p.path }, { size: p.cap == null ? "auto" : p.cap === 0 ? "off" : copySize(p.cap) })) };
          if (removed.length) body.delete_removed = confirm("Also delete the hot copy's files on " + removed.map((p) => place(p.drive)).join(", ") + "? Cancel keeps them.");
          const note = $("experts-note");
          xPost = "places"; gate();
          let ok = false;
          try {
            const response = await post("v1/hot-copy", body), result = await response.json();
            if (!response.ok) { note.textContent = result.error && result.error.message ? result.error.message.replace(/^./, (c) => c.toUpperCase()) : "Not saved."; return; }
            ok = true; xSavedAt = pollSeq;
            note.textContent = result.restart ? "Saved. Reload to apply: a new or removed drive is used once the model reloads." : result.restart === false && !result.hot_copy ? "Saved. Origami uses it once a model runs." : "Saved.";
            refresh();
          } catch (error) { note.textContent = "Could not reach Origami."; }
          finally { xPost = null; failed($("x-places-save"), !ok); gate(); showChange(); }
        });
        $("x-keep").addEventListener("change", (event) => hotCopy({ keep_when_idle: event.target.checked }, "keep", "Saved."));
        $("x-shared").addEventListener("change", (event) => { const model = event.target.dataset && event.target.dataset.keepModel; if (model) hotCopy({ keep_when_idle: event.target.checked, model }, "keep", "Saved."); });
        async function hotCopy(body, act, said) {
          const note = $("experts-note");
          xPost = act; gate();
          let ok = false;
          try {
            const response = await post("v1/hot-copy", body), result = await response.json();
            if (!response.ok) { note.textContent = (result.error && result.error.message ? result.error.message.replace(/^./, (c) => c.toUpperCase()).replace(/([^.])$/, "$1.") : "Not saved."); return; }
            ok = true;
            // A further Qwen copy is built by a job, after which the model reloads if the copies changed; with no model running the change
            // waits for one.
            note.textContent = result.job ? "Saved. Copying the experts; if that changes the copies, the model reloads after the current reply."
              : result.restart === false ? "Saved. Origami uses it once a model runs." : said;
            if (act === "use") { xOpen = false; xDrive = null; xSize = null; xCap = null; }
            refresh();
          } catch (error) { note.textContent = "Could not reach Origami."; }
          finally { xPost = null; const b = $("x-acts").querySelector('button[data-xact="' + CSS.escape(act) + '"]') || (act === "use" ? $("x-use") : null); if (b) failed(b, !ok); gate(); }
        }
        // Clear: the leftover settings go from the settings file; they did nothing, so nothing reloads.
        async function clearUnused(button) {
          const keys = ((xFrom && xFrom.unused_settings) || []).map((u) => u.key).filter((k) => !xCleared.includes(k)), note = $("experts-note");
          if (!keys.length) return;
          xPost = "clear"; gate();
          let ok = false;
          try {
            const response = await post("v1/config", Object.fromEntries(keys.map((k) => [k, ""]))), result = await response.json();
            if (!response.ok) { note.textContent = (result.error && result.error.message) || "Not cleared."; return; }
            ok = true; xCleared = xCleared.concat(keys); note.textContent = "Cleared; they did nothing for this model.";
            expertsCard(status); loadFolders();
          } catch (error) { note.textContent = "Could not reach Origami."; }
          finally { xPost = null; if (button) failed(button, !ok); gate(); }
        }
        const toggle = (key) => { xOpen = xOpen === key ? false : key; xDrive = null; xCap = null; xPlaces = null; expertsCard.acts = null; expertsCard(status); if (xOpen) { const r = $("x-drives").querySelector("input:checked"); if (r) r.focus(); } };
        $("x-acts").addEventListener("click", (event) => {
          const button = event.target.closest && event.target.closest("button[data-xact]"), act = button && button.dataset.xact, hot = xHot;
          if (!act) return;
          if (act.startsWith("move:")) return toggle(act.slice(5));
          if (act.startsWith("copy:")) {
            const c = ((xFrom && xFrom.copy_to) || []).find((x) => x.path === act.slice(5)), packed = (xFrom.sources || []).some((r) => r.role === "pack");
            // About 100 MB/s, as the hot copy's own estimate.
            const size = c && c.bytes ? (packed ? "up to " : "") + gbs(c.bytes) + ", about " + duration(c.bytes / 1e8) : "";
            if (!confirm("Origami writes " + (packed ? "another copy of the 4-bit experts" : "all experts, 4-bit,") + " to " + place(c && c.drive) + (size ? " (" + size + ")" : "")
              + (packed ? ", so replies read from both drives" : "") + ", then reloads the model; requests wait while it reloads. Replies are unchanged. Copy them?")) return;
            return hotCopy({ copy: act.slice(5) }, act, "Saved. Origami copies the experts to " + place(c && c.drive) + " at its next setup.");
          }
          if (!hot) return;
          if (act === "change") return toggle("hot");
          if (act === "build") return hotCopy({ build: true }, act, "Building the hot copy now.");
          if (act === "on") return hotCopy({ size: "auto" }, act, hot.has_counts ? "Saved. Building the hot copy now." : "Saved. The hot copy is on; it is built after the first reply.");
          if (act.startsWith("put:")) { const c = (hot.choices || []).find((x) => x.path === act.slice(4)); return hotCopy({ drive: act.slice(4) }, act, building(c && c.drive)); }
        });
        $("x-unused").addEventListener("click", (event) => { const b = event.target.closest && event.target.closest("button[data-xact]"); if (b) clearUnused(b); });
        $("x-drives").addEventListener("change", (event) => { xDrive = event.target.value; xCap = null; showChange(); });
        $("x-size").addEventListener("change", (event) => { xSize = event.target.value; xCap = null; showChange(); });
        // Moving the slider chooses that size: the graph and the lines follow at once; nothing is sent before Use this.
        $("x-cap").addEventListener("input", (event) => {
          const row = xPanel(), pop = xPop(row), whole = pop ? pop.bytes : 0, c = row && (row.choices || []).find((x) => x.path === xDrive);
          const room = c && c.room_bytes != null ? c.room_bytes : Infinity, unit = whole ? xUnit : "gb";
          xCap = capClamp(capBytes(parseFloat(event.target.value), unit, whole), room, whole || room);
          xSize = copySize(xCap); showChange();
        });
        $("x-unit").addEventListener("click", (event) => { const u = event.target.closest && event.target.closest("button[data-unit]"); if (u) { xUnit = u.dataset.unit; showChange(); } });
        $("x-use").addEventListener("click", () => {
          const row = xPanel();
          if (!row) return;
          const hot = row.role === "hot", choices = (hot ? row.choices : row.moves) || [], current = choices.find((c) => c.status === "goes here"), body = {};
          if (xDrive && (!current || xDrive !== current.path)) body.drive = xDrive;
          if (hot && xSize && xSize !== savedSize()) body.size = xSize;
          if (!Object.keys(body).length) { xOpen = false; expertsCard.acts = null; $("experts-note").textContent = "Nothing changed: " + (hot ? "the hot copy stays where it is." : "they stay where they are."); expertsCard(status); return; }
          const c = choices.find((x) => x.path === xDrive);
          if (!hot) {
            if (!confirm((status.model && status.model.name || "The model") + " stops while its " + roleText(row).replace(/^all experts, /, "") + " experts move to " + place(c && c.drive) + " and starts again; requests wait meanwhile. Move them?")) return;
            xOpen = false; expertsCard.acts = null; storagePath = row.path;
            return startJob({ kind: "move", path: row.path, to: xDrive, drive: true }, $("x-use"), $("experts-note"));
          }
          hotCopy(body, "use", body.size === "off" ? "Saved. Hot copy off; existing files retained." : building(c ? c.drive : current && current.drive));
        });
        // What a saved drive or size does next: a build now, or after the first reply when there are no counts yet.
        const building = (drive) => xHot && xHot.has_counts ? "Saved. Building the hot copy on " + place(drive) + " now." : "Saved. The hot copy goes on " + place(drive) + "; it is built after the first reply.";
        // Redraws a list only when it changed, and gives the keyboard its place back (each control names itself in data-key).
        function swap(id, html) {
          const el = $(id);
          if (el.innerHTML === html) return false;
          const at = document.activeElement, key = at && el.contains && el.contains(at) && at.getAttribute("data-key");
          el.innerHTML = html;
          if (key) { const again = el.querySelector('[data-key="' + CSS.escape(key) + '"]'); if (again) again.focus(); }
          return true;
        }
        // A bar drawn in characters, as the memory strip is: one character per part ([character, colour, amount, name, value]),
        // each part a clipped run of its character, with the key underneath.
        function abar(parts, label, keys = true, width = 1) {
          const shown = parts.filter((p) => p[2] > 0), total = shown.reduce((n, p) => n + p[2], 0) || 1, said = label + ": " + shown.map((p) => p[3] + " " + p[4]).join(", ");
          return '<div class="abar" role="img" style="width:' + Math.max(4, Math.round(100 * width)) + '%" title="' + esc(said) + '" aria-label="' + esc(said) + '">' +
            shown.map((p) => '<span data-k="' + p[1] + '" style="width:' + (100 * p[2] / total).toFixed(1) + '%" aria-hidden="true">' + p[0].repeat(200) + "</span>").join("") + "</div>" +
            (keys ? '<div class="akeys" aria-hidden="true">' + shown.map((p) => '<span><i data-k="' + p[1] + '">' + p[0] + "</i>" + esc(p[3]) + " <b>" + esc(p[4]) + "</b></span>").join("") + "</div>" : "");
        }
        // Where a finished request's time went, from its `stage_seconds`: waiting its turn, setting up (and restoring a saved
        // prompt), reading the prompt, writing, saving. Null for a request without them.
        const secs = (s) => s >= 100 ? fmt.format(Math.round(s)) + " s" : one.format(s) + " s";
        function timeParts(r) {
          const st = r.stage_seconds;
          if (!st || !(r.seconds > 0)) return null;
          const v = (k) => +st[k] || 0, setup = v("prompt_encoding") + v("state_prepare") + v("restore_and_setup") + v("prior_save_wait");
          const read = st.prefill_and_capture != null ? v("prefill_and_capture") : v("prefill_and_save"), write = v("generation"), save = v("finalization");
          const rest = r.seconds - setup - read - write - save, wait = rest > 0.02 * r.seconds ? rest : 0;
          return [[".", "d", wait, "waiting", secs(wait)], ["-", "c", setup, "setting up", secs(setup)], ["/", "b", read, "reading the prompt", secs(read)],
            ["#", "a", write, "writing", secs(write)], ["=", "c", save, "saving", secs(save)]];
        }
        const timeBar = (r, longest) => { const parts = timeParts(r); return parts ? abar(parts, secs(r.seconds), false, longest ? r.seconds / longest : 1) : ""; };
        // Why the last request was slow: its first word split into the prompt read (new, restored, and why fewer were restored
        // than wanted), its writing into the GPU and each drive's wait, and one verdict by the rules `SlowReason` (the engine)
        // applies for `origami status`; both are checked against one fixture.
        const RESTORE_WHY = { save_skipped_disk: "the saved copy was skipped for disk space", save_failed: "saving it had failed", save_not_finished: "its save had not finished",
          entry_evicted: "its saved copy had been replaced by newer ones", entry_not_used: "the saved copy was not used", no_entry: "it had not been saved" };
        const restoreWhy = (why) => /^prefix_differs_at=/.test(why) ? "the prompt differed from the saved one at token " + why.slice(18) : RESTORE_WHY[why] || why;
        function slowParts(r) {
          const pc = r.phase_counters || {}, st = r.stage_seconds || {}, v = (k) => +pc[k] || 0, waits = [];
          for (const k in pc) { const m = /^decode\.drive_activity\.(model_drive|copy_place_(\d+))\.busy_seconds\.reads$/.exec(k); if (m) waits.push([m[2] == null ? "the model's drive" : "copy " + (+m[2] + 1), +pc[k] || 0]); }
          const late = v("generation.decode_late_wait_seconds"), times = [["the model's drive", v("generation.reads_internal_seconds")], ["the pack's drive", v("generation.reads_external_seconds")], ["the hot copy", v("generation.reads_hot_seconds")]];
          const all = times.reduce((n, t) => n + t[1], 0);
          if (late > 0 && all > 0) times.filter((t) => t[1] > 0).forEach((t) => waits.push([t[0], late * t[1] / all]));
          waits.sort((a, b) => b[1] - a[1]);
          const pr = r.prompt_restore && typeof r.prompt_restore === "object" ? r.prompt_restore : null, restored = r.cached_tokens || 0;
          return { read: +(st.prefill_and_capture != null ? st.prefill_and_capture : st.prefill_and_save) || 0, write: +st.generation || 0, waits, restored,
            gpu: "generation.decode_gpu_seconds" in pc ? v("generation.decode_gpu_seconds") : v("decode.drive_activity.wall_seconds.gpu"),
            fresh: Math.max(0, (r.prompt_tokens || 0) - restored), lost: pr ? Math.max(0, (pr.wanted || 0) - (pr.restored || 0)) : 0, wanted: pr ? pr.wanted || 0 : 0, why: pr && pr.reason };
        }
        function slowReason(r, s) {
          const p = slowParts(r), t = +r.seconds || 0;
          if (!(t > 0)) return "nothing unusual";
          const short = (+r.memory_budget_deficit_gb_seconds || 0) / t;
          if (short >= 1) return "slow because: memory was cut by about " + short.toFixed(1) + " GB for this request's " + Math.round(t) + " s";
          const reread = p.lost >= 1000 && p.fresh > 0 ? p.read * Math.min(1, p.lost / p.fresh) : 0, wait = p.waits[0], enough = (cost) => cost >= 2 && cost >= 0.2 * t;
          if (enough(reread) && reread >= (wait ? wait[1] : 0)) return "slow because: " + fmt.format(p.lost) + " prompt tokens were read again" + (p.why ? " — " + restoreWhy(p.why) : "");
          if (wait && enough(wait[1])) {
            const backlog = (((s || {}).experts_from || {}).sources || []).filter((x) => x.role === "hot").reduce((n, x) => n + (x.backlog_records || 0), 0);
            return "slow because: " + wait[0] + " kept the reply waiting for " + Math.round(wait[1]) + " s" + (wait[0] === "the model's drive" && backlog > 0 ? " — the hot copy has " + fmt.format(backlog) + " swaps waiting" : "");
          }
          return "nothing unusual";
        }
        // The breakdown under the last request's bar: two lines of figures and the verdict.
        function whyHtml(r, s) {
          const p = slowParts(r), verdict = slowReason(r, s), slow = verdict !== "nothing unusual";
          const first = r.ttft_s != null ? "<b>First word " + secs(r.ttft_s) + "</b> · " + fmt.format(p.fresh) + " prompt tokens read, " + fmt.format(p.restored) + " restored" +
            (p.lost > 0 && p.why ? " (" + fmt.format(p.wanted) + " wanted" + (verdict.includes(restoreWhy(p.why)) ? "" : ": " + esc(restoreWhy(p.why))) + ")" : "") : "";
          const waits = p.waits.filter((w) => w[1] >= 0.05).map((w) => esc(w[0]) + " " + secs(w[1]));
          const write = p.write > 0 ? "<b>Writing " + secs(p.write) + "</b>" + (p.gpu > 0 ? " · GPU " + secs(p.gpu) : "") + (waits.length ? " · waiting on " + waits.join(", ") : "") : "";
          return '<div class="why">' + [first, write].filter(Boolean).map((l) => "<p>" + l + "</p>").join("") +
            '<p class="verdict"' + (slow ? ' data-slow=""' : "") + ">" + esc(verdict.replace(/^./, (c) => c.toUpperCase())) + "</p></div>";
        }
        function flow(id, row) {
          const html = row ? '<div class="flow-head"><span class="label">' + row[0] + "</span><span>" + esc(row[1]) + "</span></div>" + row[2] : "";
          if ($(id).innerHTML !== html) $(id).innerHTML = html;
        }
        const share = (part, whole) => (100 * Math.min(1, Math.max(0, part / whole))).toFixed(1) + "%";
        const MEMORY_ASK = "Find the fastest memory limit? Origami reloads the model at a few limits and times each: about 10 to 20 minutes, and requests wait meanwhile. Your setting is not changed.";
        // Now: this Mac's own numbers from its last replies; where the last request's time, the expert reads and the memory
        // went; what the model runs as (badges); and Health, a line per thing that can need you.
        const SPARK = "▁▂▃▄▅▆▇█";
        function showNow(s) {
          const ready = !!s && s.status === "ready", halted = !!s && (s.status === "stopped" || s.status === "stopping");
          // Stopped, the speed, the requests and the saved prompts give way to Start.
          if (halted || ready && !noModel) document.querySelectorAll("#view-run .running-only").forEach((el) => { el.hidden = halted; });
          $("health-block").hidden = !ready;
          const recent = (ready && s.requests && s.requests.recent) || [], speeds = recent.filter(measured).slice(-32).map((r) => r.decode_tps), top = Math.max(0.001, ...speeds), low = Math.min(top * 0.8, ...speeds) * 0.9;
          $("spark").textContent = speeds.length > 1 ? speeds.map((v) => SPARK[Math.min(7, Math.floor(7.99 * (v - low) / (top - low)))]).join("") : "";
          $("spark").title = speeds.length > 1 ? "The last " + speeds.length + " replies, " + one.format(Math.min(...speeds)) + " to " + one.format(top) + " tok/s" : "";
          const few = recent.filter((r) => r.ttft_s != null).slice(-5), reading = recent.filter((r) => r.prefill_tps > 0).slice(-5).map((r) => r.prefill_tps);
          const asked = few.reduce((n, r) => n + (r.prompt_tokens || 0), 0), reused = few.reduce((n, r) => n + (r.cached_tokens || 0), 0);
          const figure = (name, value, unit, title) => '<div title="' + esc(title) + '"><dt>' + name + "</dt><dd>" + value + "<small>" + unit + "</small></dd></div>";
          const figures = !few.length ? "" : figure("First word", one.format(median(few.map((r) => r.ttft_s))), "s", "How long a reply takes to start: the middle of the last " + few.length) +
            (reading.length ? figure("Prompt reading", fmt.format(Math.round(median(reading))), "tok/s", "How fast a new prompt is read: the middle of the last " + reading.length) : "") +
            (asked ? figure("Prompt reused", Math.round(100 * reused / asked), "%", "The share of the last prompts that came from a saved one instead of being read again") : "");
          if ($("figures").innerHTML !== figures) $("figures").innerHTML = figures;
          const last = recent.filter(timeParts).slice(-1)[0];
          flow("flow-time", last ? ["Last request", secs(last.seconds) + " · " + ago(last.time), abar(timeParts(last), "Last request") + whyHtml(last, s)] : null);
          const from = (ready && s.experts_from) || {}, sp = from.split;
          flow("flow-reads", sp && sp.reads ? ["Expert reads", fmt.format(sp.reads), abar([["@", "b", sp.memory || 0, "from memory", pct(sp.memory)],
            ["#", "a", sp.hot_copy || 0, "from the hot copy", pct(sp.hot_copy)], [":", "c", sp.model_drive || 0, "from the model's drive", pct(sp.model_drive)]], "Expert reads")] : null);
          const m = ready && s.memory, ram = ready && s.mac && s.mac.memory_bytes, room = m ? Math.max(0, m.cap_bytes - m.footprint_bytes) : 0;
          flow("flow-mem", m && ram && m.footprint_bytes ? ["Memory", "this Mac has " + macGB(ram), abar([["#", "a", m.footprint_bytes, "Origami", gb(m.footprint_bytes)], ["-", "c", room, "room in its limit", gb(room)],
            [".", "d", Math.max(0, ram - m.footprint_bytes - room), "left to macOS and apps", gb(Math.max(0, ram - m.footprint_bytes - room))]], "Memory")] : null);
          // What the model runs as: its context, and whether its replies are the model's own.
          const about = ready && s.model && typeof s.model === "object" ? s.model : {}, set = (ready && s.settings) || {}, q = set.quality && LEVEL_LABELS[set.quality.value] ? set.quality.value : null;
          const variant = /low-bit/.test(about.experts || ""), context = ready && (s.context || about.context);
          const badge = (text, differs, title) => '<span class="badge"' + (differs ? ' data-b="differs"' : "") + (title ? ' title="' + esc(title) + '"' : "") + ">" + text + "</span>";
          const badges = !ready ? "" : (context ? badge("<b>" + fmt.format(Math.round(context / 1024)) + "K</b> context") : "") +
            (variant ? badge("<b>Low-bit experts</b> · replies differ from exact", true, about.experts) : "") +
            (q && q !== "exact" ? badge("<b>" + LEVEL_LABELS[q] + "</b> · replies differ slightly", true) : q && !variant ? badge("<b>Exact</b> replies") : "");
          if ($("badges").innerHTML !== badges) $("badges").innerHTML = badges;
          if (!ready) { swap("health", ""); return; }
          const rows = [], row = (mark, name, body, sub, acts) => '<div class="hrow"><i class="mark" data-m="' + mark + '" aria-hidden="true"></i><span class="label">' + name + "</span><div>" + body +
            (sub ? '<span class="sub">' + sub + "</span>" : "") + '</div><div class="hact">' + (acts || "") + "</div></div>";
          const go = (view, label, at) => '<button type="button" class="small" data-key="go:' + (at || view) + '" data-go="' + view + '"' + (at ? ' data-at="' + at + '"' : "") + ">" + label + "</button>";
          const act = (name, label, extra = "") => '<button type="button" class="small" data-key="' + name + '" data-hact="' + name + '"' + extra + ">" + label + "</button>";
          if (m && m.cap_bytes) {
            // The limit in force: automatic, yours, or the one the memory search measured; the search's result, with the
            // value its own advice names, one press away.
            const advice = s.memory_learned && s.memory_learned.limit_advice, command = advice && /`origami config set memory_limit ([^` ]+)`/.exec(advice.text || "");
            const held = advice && advice.suggested_bytes && Math.abs(m.cap_bytes - advice.suggested_bytes) < 3e8;
            const testing = running() && job.kind === "tune" && /memory limit/.test(job.title || "");
            const which = testing ? "a limit under test" : held ? "the measured limit" : m.budget_mode === "explicit" ? "your limit" : "automatic";
            // `memory_auto_measure`: what the search while idle last did, and its switch while no limit of yours is set.
            const auto = s.memory_learned && s.memory_learned.auto_measure, saved = dashboard && dashboard.saved_settings || {}, autoOn = (saved.memory_auto_measure || {}).value === "true";
            const yours = saved.memory_limit ? /^\d/.test(saved.memory_limit.value || "") : which === "your limit";
            // A cut, in either mode: the budget below what it climbs back to, with the last cut's reason and when.
            const cut = m.last_cut ? " (" + m.last_cut.reason + ", " + ago(m.last_cut.at) + ")" : "";
            const sub = [m.temporarily_reduced ? "Origami is held to " + gb(m.budget_bytes) + " for now" + cut + "; it climbs back when memory is free." : "", advice ? memoryText(advice.text).replace(/[;,]? set it with `[^`]*`\.?$/, ".") : "",
              auto ? auto.replace(/^memory: /, "").replace(/^./, (c) => c.toUpperCase()) + "." : ""].filter(Boolean).map(esc).join(" ");
            rows.push(row(m.temporarily_reduced ? "warn" : "ok", "Memory", "<b>" + gb(m.footprint_bytes || 0) + "</b> of " + gb(m.cap_bytes) + " · " + which, sub,
              (command && !held ? act("use-limit", "Use " + esc(memoryText(command[1])), ' data-value="' + esc(command[1]) + '"') : "") + (advice ? "" : act("tune-memory", "Find the fastest", ' title="Times a few memory limits on this Mac and names the fastest one it keeps: about 4 minutes for Qwen, 6 for GLM, 10 for DeepSeek"')) +
              (yours || testing ? "" : act("auto-measure", autoOn ? "Stop measuring when idle" : "Measure when idle", ' data-value="' + !autoOn + '" title="Find the fastest limit by itself after half an hour without requests, on power, at most once a day; a request stops it"')) + go("setup", "Change…")));
          }
          // The drives the experts are read from: one that reads far below its own speed says so, with the cure.
          const sources = from.sources || [], speeds2 = s.drive_speeds || [], drives = new Set();
          for (const r of sources) {
            if (r.role === "hot" || r.role === "more" || r.state === "not_read" || drives.has(r.drive)) continue;
            drives.add(r.drive);
            const slow = r.slow_reads, bad = ["not_connected", "not_allowed"].includes(r.state), speed = speeds2.find((d) => d.drive === r.drive), rate = speed && speed.read_gbps && (speed.read_gbps.qd8 || speed.read_gbps.qd1);
            // `since_rewrite`: what was read from these files since they were last written, and from the drive's own
            // history whether a refresh is due before reading slows down.
            const since = !bad && r.since_rewrite, due = !!since && (since.due || since.high), high = !!slow && slow.high || (!!since && since.high);
            const body = bad ? "<b>" + esc((STATES[r.state] || ["", r.state])[1]) + "</b>" : slow && slow.high ? "<b>" + pct(Math.max(slow.share || 0, slow.hour_share || 0)) + "</b> of its reading far below its own speed"
              : since && since.high ? "<b>" + pct(since.share || 0) + "</b> of its reading time slow" + (rate ? " · " + one.format(rate) + " GB/s when new" : "")
              // What was read before this start is in the tally: "nothing read yet" is for a drive with none.
              : [slow && slow.busy_seconds > 0 ? "reads at its own speed" : since && since.bytes ? "" : "nothing read yet", rate ? one.format(rate) + " GB/s measured" : ""].filter(Boolean).join(" · ");
            rows.push(row(bad ? "fail" : high || due ? "warn" : "ok", "Drive", ["<b>" + esc(place(r.drive)) + "</b>", body, since ? tbs(since.bytes) + " read" : ""].filter(Boolean).join(" · "),
              bad || (slow && slow.high) ? esc(r.detail || "") : since ? esc((since.text || "").replace(/^[\d.,]+ [TG]B read s/, "S")) : "",
              high || due ? '<button type="button" class="small" data-key="copy-refresh" data-copy-text="origami refresh">Copy the command</button>' : ""));
          }
          const hot = sources.find((r) => r.role === "hot");
          // The copy's drive can slow down on heavily read blocks too: a rebuilt copy is written on fresh blocks.
          const worn = hot && hot.since_rewrite && (hot.since_rewrite.high || hot.since_rewrite.due) ? hot.since_rewrite : null;
          if (hot) {
            const state = STATES[hot.state] || ["dim", hot.state], p = hot.progress, building = (hot.state === "building" || hot.state === "paused") && p && p.total_bytes;
            const cover = (hot.coverage || []).find((c) => Math.abs(c.bytes - (hot.budget_bytes || 0)) < 5e8);
            rows.push(row(worn ? "warn" : state[0], "Hot copy", "<b>" + esc(state[1]) + "</b>" + (hot.bytes ? " · " + gbs(hot.bytes) + " on " + esc(place(hot.drive)) : "") + (hot.state === "ready" && cover ? " · holds " + pct(cover.share) + " of expert uses" : "") +
              (building ? '<span class="hbar" data-m="warn" aria-hidden="true"><span style="width:' + share(p.done_bytes, p.total_bytes) + '"></span></span>' : ""),
              building ? gbs(p.done_bytes).replace(" GB", "") + " of " + gbs(p.total_bytes) + (p.seconds_left != null ? " · about " + soon(p.seconds_left) + " left" : "") : worn ? esc(worn.text || "") : hot.state === "ready" ? "" : esc(hot.detail || ""), go("library", "Change…", "experts")));
          } else if (from.no_hot_copy) rows.push(row("none", "Hot copy", "None", esc(from.no_hot_copy), ""));
          const t = s.tuning;
          const proposal = dashboard && dashboard.tuning_proposal, waits = !!t && !!proposal && proposal.key === t.key && proposal.revision === revision;
          const lastRun = dashboard && dashboard.last_tuning, tuned = !!t && (!!t.entry || !!lastRun && lastRun.key === t.key && lastRun.outcome === "already_fastest");
          if (t && Object.keys(t.knobs || {}).length) rows.push(row(waits || ["recheck", "stale", "unreadable"].includes(t.state) ? "warn" : tuned ? "ok" : "none", "Tuning",
            waits ? proposal.kind === "performance" ? "<b>+" + Math.round(((proposal.entry && proposal.entry.gain) || 0) * 100) + "%</b> measured · not applied yet" : "<b>Comparison ready</b> · nothing applied"
              : esc($("st-line").textContent || "Not tuned for this Mac yet"), "", go("setup", waits ? "Review…" : tuned ? "Open…" : "Tune…", "selftune")));
          if (access) rows.push(row(access.listening === "lan" && !access.key_required ? "warn" : "ok", "Access", access.listening === "lan" ? "<b>This network</b> · " + (access.key_required ? "other devices need the key" : "no key: any device on it can use Origami and change its settings")
            : "<b>This Mac only</b>", "", go("settings", "Change…")));
          swap("health", rows.join(""));
        }
        async function useLimit(value, button) {
          if (!confirm("Set the memory limit to " + memoryText(value) + " and reload the model? Requests wait while it reloads.")) return;
          const note = $("health-note");
          applying = "use-limit"; gate(); note.textContent = "Saving… If macOS asks, approve the administrator dialog on this Mac.";
          let ok = false;
          try {
            const saved = await post("v1/memory/wired", { memory_limit: value }), result = await saved.json();
            if (!saved.ok) { if (saved.status === 409) conflict = true; throw Error(result.error && result.error.message || "Not saved"); }
            revision = result.revision || revision; syncedRevision = null;
            if (!(await post("v1/reload", {})).ok) throw Error("Saved. Reload when the current operation finishes.");
            reloadStarted("apply"); base = lastPlan = null; ready = false; ok = true; note.textContent = "Saved. Loading with a limit of " + memoryText(value) + "…";
            auxAt = 0; await refresh();
          } catch (error) { note.textContent = error.message || "Could not save"; }
          finally { applying = null; failed(button, !ok); gate(); }
        }
        $("health").addEventListener("click", async (event) => {
          const b = event.target.closest && event.target.closest("button");
          if (!b || b.disabled) return;
          if (b.dataset.go) {
            chooseWorkspace(b.dataset.go);
            const at = b.dataset.at && $(b.dataset.at);
            if (at && !at.hidden) { if (at.tagName === "DETAILS") at.open = true; at.scrollIntoView({ block: "start" }); }
            return;
          }
          if (b.dataset.copyText) {
            try { await navigator.clipboard.writeText(b.dataset.copyText); $("health-note").textContent = "Copied. Stop Origami, then run it in Terminal on this Mac."; }
            catch (error) { $("health-note").textContent = "In Terminal on the Mac: " + b.dataset.copyText; }
            return;
          }
          $("health-note").textContent = "";
          if (b.dataset.hact === "use-limit") return useLimit(b.dataset.value, b);
          if (b.dataset.hact === "auto-measure") {
            const saved = await post("v1/config", { memory_auto_measure: b.dataset.value }), result = await saved.json();
            if (!saved.ok) { if (saved.status === 409) conflict = true; $("health-note").textContent = result.error && result.error.message || "Not saved"; return; }
            revision = result.revision || revision; syncedRevision = null; auxAt = 0; await refresh(); return;
          }
          if (b.dataset.hact === "tune-memory" && confirm(MEMORY_ASK)) { tuneBy = "st-go"; startJob({ kind: "tune", mode: "memory" }, b, $("health-note")); }
        });
        // The Models view: every added model as a row (the running one marked with the header's state, a low-bit variant saying its replies differ),
        // Load and Remove… on each, then what Origami found on this Mac, one Add away.
        const FAMILIES = { "qwen3.8-flash-next": "Qwen3.8-Flash-Next", "glm-5.3-flash": "GLM-5.3-Flash", "deepseek-v4.1-flash": "DeepSeek-V4.1-Flash" };
        function modelList() {
          const files = storageMap ? storageMap.drives.flatMap((d) => d.items || []) : [];
          const rows = models.map((m) => {
            const o = m.origami || {}, setting = o.setting || m.id, running = !!o.running && !noModel && liveState !== "stopped", state = stateText(o.availability, o.drive), away = !!state && !pending(o.availability);
            const file = files.find((i) => i.role === "model" && i.model === m.id && (i.actions || []).includes("forget"));
            return '<div class="mrow"><i class="mark" data-m="' + (running ? "ok" : away ? "none" : "dim") + '" aria-hidden="true"></i><div><strong>' + esc(named(m)) + "</strong>" +
              (o.variant_of ? '<span class="badge" data-b="differs">low-bit experts · replies differ from exact</span>' : "") + '<span class="sub">' +
              [running ? live.chip || STATE.ready : state, FAMILIES[o.family] || "", o.bytes ? Math.round(o.bytes / 1e9) + " GB" : "", o.drive ? esc(place(o.drive)) : "",
                o.variant_of ? "its experts on " + esc(o.variant_of) : o.quantization && o.quantization !== "unknown" ? esc(o.quantization) : ""].filter(Boolean).join(" · ") + '</span></div><div class="hact">' +
              (setting && setting !== modelInitial ? '<button type="button" class="small" data-key="load:' + esc(setting) + '" data-mload="' + esc(setting) + '"' + (away ? ' data-blocked data-why="' + esc("It is " + state) + '"' : "") + ">Load</button>" : "") +
              (file ? '<button type="button" class="small" data-key="rm:' + esc(file.path) + '" data-mremove="' + esc(file.path) + '" aria-label="' + esc("Remove " + named(m)) + '">Remove…</button>' : "") + "</div></div>";
          });
          const extra = ((found && found.models) || []).map((f) => '<div class="mrow"><i class="mark" data-m="none" aria-hidden="true"></i><div><strong>' + esc(f.name) + '</strong><span class="sub">' +
            [stateText(f.state, f.drive), Math.round(f.bytes / 1e9) + " GB", esc(place(f.drive)), f.verdict === "converts" ? "converts once" : ""].filter(Boolean).join(" · ") +
            '</span></div><div class="hact"><button type="button" class="small" data-key="add:' + esc(f.path) + '" data-madd="' + esc(f.path) + '">Add</button></div></div>');
          swap("model-list", (rows.join("") || '<p class="empty">Nothing added yet: a folder, a .gguf file, or a Hugging Face name to download.</p>') +
            (extra.length ? '<p class="mhead">Found on this Mac</p>' + extra.join("") : found && found.searching ? '<p class="mhead">Looking for models on this Mac…</p>' : ""));
        }
        // A found model that runs as it is is added at once; one that converts first shows what that writes.
        function addFound(path, button) {
          const f = ((found && found.models) || []).find((x) => x.path === path);
          if (f && f.verdict === "converts") return openAdd(path);
          startJob({ kind: "add", source: path }, button);
        }
        const loadChosen = (setting) => { if (chooseModel(setting)) { $("model").value = modelChoice; apply(true, false, true); } };
        $("model-list").addEventListener("click", (event) => {
          const b = event.target.closest && event.target.closest("button");
          if (!b || b.disabled) return;
          if (b.dataset.mremove) return storageAction("forget", b.dataset.mremove, b);
          if (b.dataset.madd) return addFound(b.dataset.madd, b);
          if (b.dataset.mload) loadChosen(b.dataset.mload);
        });
        // The one next action where no model runs: Start when stopped; on a first run what Origami found, the download it is
        // made for, or your own file; a model added but not loaded yet.
        function showFirst(s) {
          const none = noticeKind === "missing", list = (found && found.models) || [];
          $("picker").hidden = none;
          const button = (attrs, label, primary) => '<button type="button"' + (primary ? ' class="primary"' : "") + " " + attrs + ">" + label + "</button>";
          let html = "";
          const there = models.filter((m) => !(m.origami && m.origami.availability === "missing"));
          if (s.status === "stopped") html = "<p>Stopped: the model is unloaded" + (s.freed_bytes ? " and about " + gb(s.freed_bytes) + " of memory is back with macOS" : "") +
            ". Apps get an error until you start it.</p>" + '<div class="actions">' + button('data-key="start" data-first="start"', "Start", true) + "</div>";
          else if (none && reloading) html = "";
          else if (none && there.length) html = "<p>No model is loaded.</p>" + '<div class="actions">' + there.slice(0, 3).map((m, i) => button('data-key="load:' + esc(m.origami ? m.origami.setting : m.id) +
            '" data-mload="' + esc(m.origami ? m.origami.setting : m.id) + '"', "Load " + esc(named(m)), !i)).join("") + "</div>";
          else if (none) html = "<p>" + (list.length ? "No model added yet. Origami found " + (list.length === 1 ? "one" : list.length) + " on this Mac." : found && found.searching ? "No model yet. Looking on this Mac…"
            : "No model yet. Download the one Origami is made for, or choose your own.") + '</p><div class="actions">' +
            list.slice(0, 3).map((f, i) => button('data-key="add:' + esc(f.path) + '" data-madd="' + esc(f.path) + '"', "Add " + esc(f.name) + " · " + Math.round(f.bytes / 1e9) + " GB", !i)).join("") +
            (s.download ? button('data-key="download" data-first="download"', "Download " + esc(s.download.split("/").pop()) + "…", !list.length) : "") + button('data-key="own" data-first="own"', "Choose a file or folder…") + "</div>";
          $("first").hidden = !html; swap("first", html);
        }
        $("first").addEventListener("click", (event) => {
          const b = event.target.closest && event.target.closest("button");
          if (!b || b.disabled) return;
          if (b.dataset.first === "start") return power("power");
          if (b.dataset.first === "download") return openAdd((status && status.download) || "");
          if (b.dataset.first === "own") return openAdd("");
          if (b.dataset.madd) return addFound(b.dataset.madd, b);
          if (b.dataset.mload) loadChosen(b.dataset.mload);
        });
        // A .gguf can be a model of its own or, read beside an exact model already added, that model's low-bit experts
        // (`origami model add <experts.gguf> --base <model>`): the choice shows for a .gguf once there is a model to base it on.
        function baseChoices(source) {
          const exact = models.filter((m) => m.origami && m.origami.setting && !m.origami.variant_of), offer = /\.gguf$/i.test(source) && exact.length > 0;
          $("add-base-row").hidden = !offer;
          if (!offer) { $("add-base").value = ""; return false; }
          const html = '<option value="">A model of its own</option>' + exact.map((m) => '<option value="' + esc(m.origami.setting) + '">Low-bit experts for ' + esc(named(m)) + "</option>").join("");
          if (baseChoices.html !== html) { const kept = $("add-base").value; $("add-base").innerHTML = baseChoices.html = html; $("add-base").value = kept; }
          return true;
        }
        function baseChosen() {
          const name = settingName($("add-base").value);
          $("add-source").setAttribute("aria-invalid", "false"); $("add-to").hidden = true; addFits = true;
          $("add-info").textContent = "Reads its experts from this file and everything else from " + name + ", after checking both are the same model. Nothing is copied or converted. Its replies differ from exact; " + name + " itself stays exact.";
          $("add-go").hidden = false; $("add-go").textContent = "Add the low-bit variant";
        }
        function events(s) {
          const list = (s.events || []).slice().reverse();
          $("events").innerHTML = list.length ? '<div class="events">' + list.map((e) => '<div><time datetime="' + esc(e.time) + '">' + clock(e.time) + "</time><span>" + esc(e.text) + "</span></div>").join("") + "</div>" : '<p class="empty">Nothing reported yet.</p>';
        }

        // What the model that serves says is coming: a reload after the current reply (the page stays on that model, not a
        // loading screen, until it stops), and saved prompts paused for lack of disk.
        function banner(s) {
          const paused = s.prompt_cache && s.prompt_cache.saves_paused, lines = [];
          if (s.pending) lines.push("The model will " + s.pending + ". It keeps serving until then.");
          if (paused) lines.push(paused.charAt(0).toUpperCase() + paused.slice(1) + ".");
          $("banner").hidden = !lines.length; $("banner-text").textContent = lines.join(" ");
        }
        let setupFor = null;
        function render(s) {
          banner(s);
          // A stopped answer from before this page's Start is not the news (the next one says loading).
          if ((s.status === "stopped" || s.status === "stopping") && reloading && reloadBy === "power" && Date.now() - reloadAt < 5000) return;
          notice(s);
          if (s.status === "unavailable") {
            status = s; live(s); showNow(s); $("speed").textContent = "—"; $("basis").textContent = "Model not responding"; $("experts").hidden = true; gate(); return;
          }
          // Stopped or stopping: the front answers; the setup greys out (`gate`), and once stopped the memory strip shows what came back.
          if (s.status === "stopped" || s.status === "stopping") {
            if (reloading) { reloading = false; $("model-note").textContent = ""; }
            $("apply-note").textContent = ""; if ($("sm-note").textContent !== SM_HELD) $("sm-note").textContent = "";
            $("experts").hidden = true; xFrom = xHot = null;
            status = s; live(s); hotChip(null); showNow(s);
            $("version").textContent = s.version ? "v" + s.version : "";
            if (s.status === "stopped" && setupFor !== "stopped") {
              // Dials this page never drew (it opened on a stopped Origami) stay out of sight until Start.
              const drawn = !!initial;
              setupFor = "stopped"; setupUnavailable("Stopped: Start loads the model again."); stoppedMemory(s); loadModels();
              if (dashboard && dashboard.saved_settings) {
                familySetup({ settings: dashboard.saved_settings }); syncedRevision = null; syncSaved();
                $("tune").hidden = false;
              } else if (!drawn) $("tune").hidden = true;
            }
            return;
          }
          // No model runs (none yet, or one that did not start, answered by the front or by a Qwen child that failed while it
          // loaded): the menu and Apply still work, through the front.
          if (s.status === "failed") {
            // The reload ends here when it failed (a failed answer from before it, within a few seconds, does not count).
            // A Start that failed says so in the notice; the header's button is then Stop again, without a mark.
            if (reloading && (reloadSeen || Date.now() - reloadAt > 5000)) { reloading = false; if (reloadBy !== "power") failed($(reloadBy), true); $("model-note").textContent = ""; $("apply-note").textContent = ""; $("health-note").textContent = ""; }
            status = s; live(s); showNow(s);
            if (setupFor !== "failed") {
              setupFor = "failed"; setupUnavailable("The dials show once a model runs."); setNoModel(true); loadModels();
              if (dashboard && dashboard.saved_settings && dashboard.configured_model) {
                familySetup({ settings: dashboard.saved_settings }); syncedRevision = null; syncSaved();
                $("tune").hidden = false; $("setup-empty").hidden = true;
              }
            }
            $("version").textContent = s.version ? "v" + s.version : "";
            return;
          }
          // Between two model processes the front answers alone: only the live strip changes.
          if (s.status !== "ready" && (!s.model || typeof s.model !== "object")) { live(s); return; }
          $("version").textContent = s.version ? "v" + s.version : "";
          const mac = s.mac;
          $("title").textContent = dashboard && dashboard.active_model || s.model && s.model.name || "Origami";
          $("lede").textContent = dashboard ? dashboard.host : "";
          status = s;
          live(s); recentTable((s.requests && s.requests.recent) || []); prompts(s); expertsCard(s); events(s); showSampling(s); showSelfTune(s); showLearned(s); showNow(s);
          if (ready) { showSpeed(); showEffects(); }
          // The setup loads once per model start (a reload starts a new one).
          const settled = !reloading || reloadSeen || !s.pending && Date.now() - reloadAt > 20000;
          const started = s.status === "ready" && settled ? (generation || (s.model ? s.model.fingerprint || s.model.name : "ready")) : null;
          // The chip follows at once when the setup ends a reload.
          if (started && started !== setupFor) { setupFor = started; setNoModel(false); loadModels().then(() => { if (!conflict) return loadSetup(s); }).then(() => live(status)); }
          $("footer").textContent = [s.model && s.model.name ? s.model.name : "", s.context ? fmt.format(s.context) + "-token context" : "", mac ? mac.gpu + ", " + Math.round(mac.memory_bytes / 1073741824) + " GB" : ""].filter(Boolean).join("  ·  ");
        }

        // "Connect an app": the base URL, the model apps send, and Pi's models.json block, from the listing apps read
        // (`v1/models`: its first entry is the name requests reach, with the levels it tells apart).
        async function loadConnect() {
          let m = null;
          try { const response = await api("v1/models"); if (response.ok) m = ((await response.json()).data || [])[0] || null; } catch (error) {}
          const can = (m && m.capabilities) || ["thinking"], id = m ? m.id : "origami", context = (m && m.context_length) || 8192, thinks = can.includes("thinking");
          const efforts = (m && m.reasoning_efforts) || ["none", "low", "medium", "high", "max"];
          setEfforts(efforts);
          // Pi offers the model's distinct levels, each once (Origami still takes every name).
          const levels = Object.fromEntries([["off", "none"], ...efforts.filter((n) => n !== "none").map((n) => [n, n])]);
          const model = { id, name: "Origami", reasoning: thinks, ...(thinks ? { thinkingLevelMap: levels } : {}),
            input: can.includes("vision") ? ["text", "image"] : ["text"], contextWindow: context, maxTokens: Math.floor(context / 2) };
          // A server with a key: Pi reads it from the environment, so the page never prints it.
          const pi = { providers: { origami: { baseUrl: v1Base(), api: "openai-completions", apiKey: apiKey ? "$ORIGAMI_API_KEY" : "origami",
            compat: { supportsReasoningEffort: true, thinkingFormat: "openai", maxTokensField: "max_tokens" }, models: [model] } } };
          $("c-url").textContent = v1Base(); $("c-model").textContent = id; $("c-pi").textContent = JSON.stringify(pi, null, 2);
          $("c-key").textContent = apiKey ? "this server's API key" : "any API key";
        }
        $("connect").addEventListener("click", async (event) => {
          const button = event.target.closest && event.target.closest("[data-copy]");
          if (!button) return;
          copy($(button.dataset.copy), $("connect-note"));
        });
        // The clipboard needs a secure page (this Mac's own address); elsewhere the text is selected to copy by hand.
        async function copy(node, note) {
          try { await navigator.clipboard.writeText(node.textContent); note.textContent = "Copied."; }
          catch (error) { getSelection().selectAllChildren(node); note.textContent = local() ? "Selected: press ⌘C to copy." : "Selected: copy it from the menu."; }
        }

        // The key asked for: kept in this browser, tried at once.
        let keyUsing = false;
        async function useKey() {
          const key = $("keyask-input").value.trim();
          if (!key) { $("keyask-note").textContent = "Paste the key first."; $("keyask-input").focus(); return; }
          apiKey = keyStore(key); keyUsing = true; hold($("keyask-use"), true, true);
          await refresh();
          keyUsing = false; hold($("keyask-use"), false);
          if (!$("keyask").hidden) refresh();
        }
        $("keyask-use").addEventListener("click", useKey);
        $("keyask-input").addEventListener("keydown", (event) => { if (event.key === "Enter") useKey(); });

        // Access: who reaches Origami (`listen`, applied when Origami restarts, which a launchd service does from here) and,
        // on this network, the address other devices open; this Mac alone is sent the key and the QR code with it.
        async function loadAccess() {
          const expected = revision;
          try {
            const response = await api("v1/lan"), fresh = await response.json();
            if (!response.ok || expected !== revision) return;
            const editing = Object.keys(accessChanges()).length > 0, previous = access;
            access = editing ? { ...fresh, listen: previous.listen, key_required: previous.key_required } : fresh;
            showAccess(editing);
            if ($("c-url").textContent && $("c-url").textContent !== v1Base()) loadConnect();
          } catch (error) {}
        }
        const listenChoice = () => $("acc-lan").checked ? "lan" : "local";
        const keyChoice = () => !access || access.key_setting === "api_key" ? null : $("acc-key").checked;
        function accessChanges() {
          if (!access) return {};
          const body = {}, listen = listenChoice(), key = keyChoice();
          if (listen !== access.listen) body.listen = listen;
          if (listen === "lan" && key !== null && key !== access.key_required) body.lan_key = key ? "on" : "off";
          return body;
        }
        function showAccess(keepChoice) {
          const a = access;
          if (!a) return;
          if (!keepChoice) { $("acc-" + a.listen).checked = true; $("acc-key").checked = a.key_required; }
          const lan = listenChoice() === "lan", apiKeySet = a.key_setting === "api_key";
          $("acc-key-row").hidden = !lan;
          $("acc-key").disabled = apiKeySet;
          $("acc-key-hint").textContent = apiKeySet ? "This server has an API key (api_key), which every app and device sends." : "Without it, anything on this network can use Origami and change its settings.";
          // Saved but not applied: the socket is opened once, when Origami starts.
          const pending = a.listen !== a.listening;
          if (restarting && (!pending || Date.now() - restartSince > 90000)) { restarting = false; $("acc-note").textContent = pending ? "Origami has not restarted yet; restart it to apply the change." : a.listening === "lan" ? "Origami is on this network now." : "Only this Mac reaches Origami now."; }
          $("acc-restart").hidden = !pending || !a.restart;
          if (pending && !restarting) $("acc-note").textContent = "Saved. " + (a.listen === "lan" ? "Other devices reach Origami" : "Only this Mac reaches Origami") + " once it restarts" +
            (a.restart ? ": Restart Origami now, or it applies at its next start." : ". Restart origami serve, or the app that runs it, to apply it.") + (a.listen === "local" && !local() ? " This device then loses its connection." : "");
          const open = a.listening === "lan" && (a.urls || []).length > 0, remote = !local();
          $("acc-local-hint").textContent = remote ? "Apps and browsers on this Mac. This device then loses its connection." : "Apps and browsers on this Mac. The default.";
          $("acc-open-box").hidden = !open;
          if (open) {
            const url = a.urls[0] + (a.key ? "#key=" + encodeURIComponent(a.key) : "");
            // Another device that got in holds the key already: it may copy it for its home-screen app, which has its own storage.
            const key = a.key || (remote && a.key_required ? apiKey : "");
            $("acc-open-head").textContent = remote ? "This device" : "Open it on another device";
            $("acc-url").textContent = url;
            // 100.64.0.0/10 is the shared range Tailscale gives its devices.
            $("acc-more").textContent = a.urls.slice(1).map((u) => /^http:\/\/100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\./.test(u) ? u + " (Tailscale or another VPN)" : u).join("\n");
            $("acc-more-row").hidden = a.urls.length < 2;
            $("acc-key-show").hidden = !key; $("acc-keytext").textContent = key;
            $("acc-qr").hidden = !a.qr; if (a.qr) { $("acc-qr").src = a.qr; $("acc-qr").alt = "QR code of " + url; }
            $("acc-open-hint").textContent = remote ? (key ? "Copy the key before Add to Home Screen: the home-screen app keeps its own storage and asks for the key once." : "Other devices open the same address.")
              : a.qr ? "Scan the code with the camera of an iPad or a phone on the same network, or type the address into its browser" +
              (a.key ? " and paste the key when it asks." : ".") : a.key_required ? "Open the address there and paste the key, which shows on the Mac itself." : "Open the address there.";
          }
          gate();
        }
        // Sampling: the defaults a request without its own values gets, per mode (`defaults.sampling` in v1/status), the model's
        // recommendation, and a custom set for every mode (`sampling custom` and its five values) with `sampling_override`.
        const SM_KEYS = ["temperature", "top_p", "top_k", "min_p", "presence_penalty"];
        const SM_MODES = ["thinking", "non_thinking", "tools"], SM_NAMES = ["Creativity", "Word pool", "Shortlist", "Minimum chance", "Repetition brake"];
        const smNumber = (v) => String(+(+v).toFixed(2));
        const smText = { temperature: smNumber, top_p: (v) => +v >= 1 ? "1, every word" : smNumber(v), top_k: (v) => +v ? fmt.format(v) + " words" : "No limit",
          min_p: (v) => +v ? smNumber(v) : "None", presence_penalty: (v) => +v ? smNumber(v) : "Off" };
        const SM_SOURCES = { "generation_config.json": "What its generation_config.json names, over Origami's general defaults for the rest.",
          "GGUF metadata": "What its GGUF file names, over Origami's general defaults for the rest." };
        const SM_HELD = "Set by the command that started Origami.";
        function showSampling(s) {
          const d = s && s.defaults, set = (s && s.settings) || {};
          if (!d || !d.sampling || !d.sampling_recommended) return;
          // The values are the running model's: another one picked in the menu gets its own once it runs.
          const name = s.model && typeof s.model === "object" && s.model.name;
          $("sm-hint").textContent = "";
          const custom = d.sampling.source === "custom";
          const launched = ["sampling", "sampling_override"].concat(SM_KEYS).some((k) => set[k] && (set[k].source === "flag" || set[k].source === "env"));
          const seen = JSON.stringify([d.sampling, d.sampling_recommended, d.sampling_override, launched]);
          if (seen === smSeen || conflict || dashboard && smSaved && Object.keys(samplingChanges()).length) return;
          smSeen = seen;
          const rec = d.sampling_recommended;
          smSaved = { custom, override: !!d.sampling_override, launched, rec, values: Object.assign({}, (custom ? d.sampling : rec).non_thinking) };
          $("sm-source").textContent = SM_SOURCES[rec.source] || "Its files name none; these are Origami's general defaults.";
          // In the dials' words; the word pool as a bare number, which fits a phone's column.
          $("sm-table").innerHTML = SM_KEYS.map((k, i) => "<tr><td>" + SM_NAMES[i] + "</td>" + SM_MODES.map((mode) => '<td class="r">' + esc((k === "top_p" ? smNumber : smText[k])(rec[mode][k])) + "</td>").join("") + "</tr>").join("");
          $("sm-custom").checked = custom; $("sm-recommended").checked = !custom; $("sm-override").checked = smSaved.override;
          smShow(smSaved.values);
          if (launched) $("sm-note").textContent = SM_HELD; else if ($("sm-note").textContent === SM_HELD) $("sm-note").textContent = "";
        }
        function smShow(values) {
          smValues = Object.assign({}, values);
          for (const k of SM_KEYS) { const input = $("sm-" + k); input.value = String(smValues[k]); smDraw(k); }
          gate();
        }
        function smDraw(k) {
          const input = $("sm-" + k), min = +input.min, max = +input.max;
          $("o-sm-" + k).textContent = smText[k](smValues[k]);
          const at = (v) => (100 * (Math.min(max, Math.max(min, +v)) - min) / (max - min));
          input.style.setProperty("--fill", at(smValues[k]) + "%");
          // Custom: where the model's own value (the Not thinking column) sits on the slider.
          const mark = $("m-sm-" + k), rec = smSaved && $("sm-custom").checked ? smSaved.rec.non_thinking[k] : null;
          mark.hidden = rec == null;
          if (rec == null) return;
          const left = at(rec);
          // The arrow sits on the value: the words run right of it on the slider's left half, left of it on the right half.
          mark.textContent = left < 50 ? "↑ the model's " + smText[k](rec) : "the model's " + smText[k](rec) + " ↑";
          mark.style.marginLeft = left + "%"; mark.style.transform = left < 50 ? "none" : "translateX(-100%)";
        }
        // What Apply saves: the choice, all five values with Custom (one set for every mode), and the override.
        function samplingChanges() {
          if (!smSaved) return {};
          const out = {}, custom = $("sm-custom").checked, override = $("sm-override").checked;
          if (custom !== smSaved.custom) out.sampling = custom ? "custom" : "recommended";
          if (custom) for (const k of SM_KEYS) if (!smSaved.custom || +smValues[k] !== +smSaved.values[k]) out[k] = smNumber(smValues[k]);
          // Back to the recommendation: the custom values go too.
          if (!custom && smSaved.custom) for (const k of SM_KEYS) out[k] = "";
          if (override !== smSaved.override) out.sampling_override = String(override);
          return out;
        }
        SM_KEYS.forEach((k) => $("sm-" + k).addEventListener("input", () => { smValues[k] = +$("sm-" + k).value; smDraw(k); gate(); }));
        $("sm-choice").addEventListener("change", () => {
          // Custom starts from what the dials show; the recommendation shows its values again.
          if (smSaved && !$("sm-custom").checked) smShow(smSaved.rec.non_thinking); else { SM_KEYS.forEach(smDraw); gate(); }
        });
        $("sm-override").addEventListener("change", () => gate());
        $("sm-reset").addEventListener("click", () => { $("sm-recommended").checked = true; $("sm-custom").checked = false; if (smSaved) smShow(smSaved.rec.non_thinking); $("sm-note").textContent = ""; });
        $("sm-apply").addEventListener("click", async () => {
          const body = samplingChanges();
          if (!Object.keys(body).length) return;
          applying = "sm-apply"; gate();
          $("sm-note").textContent = "Saving…";
          let ok = false;
          try {
            const saved = await post("v1/config", body), result = await saved.json();
            if (!saved.ok) { $("sm-note").textContent = (result.error && result.error.message) || "Not saved."; return; }
            const reload = await post("v1/reload", {});
            if (!reload.ok) { $("sm-note").textContent = "Saved; restart Origami to use it."; return; }
            ok = true; reloadStarted("sm-apply"); smSeen = null;
            $("sm-note").textContent = "Saved. Reloading; requests wait until it is back.";
            refresh();
          } catch (error) { $("sm-note").textContent = "Could not reach Origami."; }
          finally { applying = null; failed($("sm-apply"), !ok); gate(); }
        });
        $("acc-listen").addEventListener("change", () => showAccess(true));
        $("acc-key").addEventListener("change", () => gate());
        $("acc-save").addEventListener("click", async () => {
          const body = accessChanges();
          if (!Object.keys(body).length) return;
          applying = "acc-save"; gate();
          let ok = false;
          try {
            const response = await post("v1/config", body), result = await response.json();
            ok = response.ok;
            if (ok) {
              access = { ...access, listen: body.listen || access.listen, key_required: body.lan_key ? body.lan_key !== "off" : access.key_required };
              if (result.revision) revision = result.revision;
            }
            $("acc-note").textContent = ok ? "Saved." : (result.error && result.error.message) || "Not saved.";
          } catch (error) { $("acc-note").textContent = "Could not reach Origami."; }
          applying = null; failed($("acc-save"), !ok); gate();
          if (ok) await loadAccess();
        });
        $("acc-restart").addEventListener("click", async () => {
          applying = "acc-restart"; gate();
          let ok = false;
          try {
            const response = await post("v1/restart", {});
            ok = response.ok;
            if (ok) { restarting = true; restartSince = Date.now(); $("acc-note").textContent = "Restarting; this page reconnects in a moment" + (access.listen === "lan" || local() ? "." : ", from this Mac only."); }
            else { const result = await response.json(); $("acc-note").textContent = (result.error && result.error.message) || "Origami did not restart."; }
          } catch (error) { $("acc-note").textContent = "Could not reach Origami."; }
          applying = null; failed($("acc-restart"), !ok); gate();
        });
        $("acc-open-box").addEventListener("click", (event) => {
          const button = event.target.closest && event.target.closest("[data-copy]");
          if (button) copy($(button.dataset.copy), $("acc-copy-note"));
        });

        let failures = 0, timer = null, last = null;
        async function refresh() {
          if (pollBusy) return;
          pollBusy = true;
          const seq = ++pollSeq;
          try {
            let response = await api("v1/dashboard");
            if (response.status === 404) response = await api("v1/status");
            if (response.status === 409) return;
            offline = response.status === 401;
            if (response.status === 401) {
              $("chip").dataset.state = "down"; $("chip-text").textContent = "Needs its key";
              $("live-text").textContent = "Paste the key below.";
              $("page").hidden = true;
              if ($("keyask").hidden) { $("keyask").hidden = false; if (!apiKey) $("keyask-input").focus(); }
              const tried = !!apiKey && !keyUsing;
              $("keyask-input").setAttribute("aria-invalid", String(tried));
              $("keyask-note").textContent = tried ? "That key was not accepted. Paste the one the Mac shows now." : "";
              $("progress").hidden = true; setSweeping(false); gate();
            // A 503 means every connection is in use (a busy 256K server) or the model is starting: keep the last view and ask again.
            } else if (response.status === 503) {
              const body = await response.json().catch(() => ({}));
              if (["failed", "stopped", "stopping"].includes(body.status)) render(body);
              else if (body.status) { setupFor = null; live(body); }
            } else {
              if (!response.ok) throw Error("No current state");
              last = await response.json(); failures = 0; observed = Date.now();
              if (last.dashboard) acceptSnapshot(last.dashboard);
              if (["loading", "stopping", "unavailable"].includes(last.status)) setupFor = null;
              // A model busy reading a long prompt can miss one status read: the page says so only when the next one fails too.
              unanswered = last.status === "unavailable" ? unanswered + 1 : 0;
              if (unanswered === 1 && status && status.status === "ready") return;
              // The first status asked for after the drives were saved: their list is rebuilt from it.
              if (xSavedAt != null && seq > xSavedAt) { xSavedAt = null; xPlaces = null; }
              render(last);
              if (last.dashboard) { front = true; showJob(last.dashboard.job); }
              if (Date.now() - auxAt > 15000) { auxAt = Date.now(); loadModels(); loadFolders(); loadAccess(); loadConnect(); }
              if (Date.now() - storageAt > (workspace === "library" ? 6000 : 20000)) { storageAt = Date.now(); loadStorage(); }
              syncSaved(); gate();
            }
            // A key that was missing or wrong now works: what the first loads could not read comes in.
            if (response.status !== 401 && !$("keyask").hidden) { $("keyask").hidden = true; $("page").hidden = false; $("keyask-note").textContent = ""; loadFolders(); loadStorage(); }
            if (restarting) loadAccess();
            if (!dashboard) {
              const jobs = await api("v1/jobs"); front = jobs.ok;
              if (jobs.ok) showJob((await jobs.json()).job);
            }
          } catch (error) {
            failures += 1; offline = true;
            $("chip").dataset.state = "down"; $("chip-text").textContent = "Not reachable";
            $("live-text").textContent = "Disconnected · reconnecting";
            $("speed").textContent = "—"; $("basis").textContent = "No current measurement";
            $("progress").hidden = true; setSweeping(false); gate();
          }
          finally {
            pollBusy = false; clearTimeout(timer);
            if (!document.hidden) timer = setTimeout(refresh, failures ? Math.min(10000, 2000 * failures) : reloading ? 1000 : 2000);
          }
        }
        document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
        const redraw = () => { layoutArt(); drawArt(null); requestAnimationFrame(drawStrip); };
        window.addEventListener("resize", () => { clearTimeout(redraw.t); redraw.t = setTimeout(redraw, 120); });
        matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);
        ["context", "quality", "reasoning"].forEach((id) => $(id).addEventListener("input", () => dialMoved(id)));
        $("memory").addEventListener("input", () => { autoMemory = false; dialMoved("memory"); });
        // Automatic is where the control returns to in one press; only moving the slider sets a limit.
        $("auto").addEventListener("click", () => { if (autoMemory) return; autoMemory = true; if (base) { const at = Math.round(base.memory.cap_bytes / GB); $("memory").value = String(Math.max(0, memoryStops.findIndex((g) => g >= at))); } dialMoved("memory"); });
        // Add a model…: the panel in the Models view, with a source to check or empty to type one.
        function openAdd(source) {
          chooseWorkspace("library");
          $("add").hidden = false; $("add-row").hidden = false; $("add-open").setAttribute("aria-expanded", "true"); addSource = ""; $("add-info").textContent = ""; $("add-go").hidden = true;
          // Opened empty, it is empty: not the source an earlier press left in it.
          $("add-source").value = source || "";
          if (source) checkSource(source);
          $("add-source").focus();
        }
        // Whether the choice was taken (a found model or Add a model… opens the add panel instead).
        function chooseModel(value) {
          $("model-note").textContent = "";
          if (value === "add") { openAdd(""); return false; }
          if (value.startsWith("found:")) { chooseWorkspace("library"); $("add").hidden = false; $("add-row").hidden = true; checkSource(value.slice(6)); return false; }
          if (dirty() && value !== modelChoice && !confirm("Discard unsaved settings before choosing another model?")) { $("model").value = modelChoice; return false; }
          if (dirty()) { rebase(); syncedRevision = null; syncSaved(); }
          closeAdd(); modelChoice = value; modelByJob = false; redrawDials();
          return true;
        }
        $("model").addEventListener("change", (event) => { chooseModel(event.target.value); });
        $("add-check").addEventListener("click", () => checkSource($("add-source").value.trim()));
        $("add-source").addEventListener("keydown", (event) => { if (event.key === "Enter" && !checking) checkSource($("add-source").value.trim()); });
        // An edited source drops the check under way for the one before, so its answer cannot bring Add back for it.
        $("add-source").addEventListener("input", () => { addSource = ""; $("add-go").hidden = true; $("add-info").textContent = ""; });
        async function startJob(body, button, note = $("model-note")) {
          body = { ...body, ...(revision ? { revision } : {}) };
          jobPost = body.kind; gate();
          let ok = false, ask = null;
          try {
            const response = await post("v1/jobs", body), result = await response.json();
            if (!response.ok) { note.textContent = (result.error && result.error.message) || "It did not start."; return; }
            // A low-bit experts file that cannot show it belongs with its base: nothing starts until the person says so.
            ok = true; if (result.confirm) ask = result.confirm; else { jobLive = jobId(result.job); showJob(result.job); refresh(); }
          } catch (error) { note.textContent = "Could not reach Origami."; }
          finally { jobPost = null; failed(button, !ok); gate(); }
          if (ask) { if (confirm(ask + " Add it?")) await startJob({ ...body, acknowledged: true }, button, note); else note.textContent = "Not added."; }
        }
        $("add-go").addEventListener("click", () => { const base = !$("add-base-row").hidden && $("add-base").value;
          startJob({ kind: "add", source: addSource, ...(base ? { base } : addTo ? { to: addTo } : {}) }, $("add-go"), $("add-info")); });
        $("add-base").addEventListener("change", () => checkSource(addSource));
        $("add-open").addEventListener("click", () => { if ($("add").hidden || $("add-row").hidden) openAdd(""); else { closeAdd(); fillPicker(); } });
        [["st-go", "tune"], ["st-check", "recheck"], ["st-undo", "undo"]].forEach(([id, mode]) => $(id).addEventListener("click", () => {
          if (mode === "undo" && !confirm("Remove the tuned settings? Origami reloads with its defaults.")) return;
          if (mode === "tune" && $("st-mode").value === "memory" && !confirm(MEMORY_ASK)) return;
          tuneBy = id; $("st-note").textContent = ""; startJob({ kind: "tune", mode: mode === "tune" ? $("st-mode").value || "tune" : mode, minutes: +$("st-budget").value || undefined, quality_floor: $("st-floor").value || undefined }, $(id), $("st-note"));
        }));
        $("ml-reset").addEventListener("click", async () => {
          applying = "ml-reset"; gate();
          let ok = false;
          try { const r = await post("v1/memory/learned", {}); ok = r.ok; $("ml-text").textContent = ok ? "Forgotten: Origami learns this Mac's memory use again." : "Not reset."; }
          catch (error) { $("ml-text").textContent = "Could not reach Origami."; }
          finally { applying = null; failed($("ml-reset"), !ok); gate(); }
        });
        $("add-browse").addEventListener("click", async () => {
          const chosen = await browse("Choose a model: a .gguf file or a model folder", { models: true });
          if (chosen) { $("add-source").value = chosen.path; checkSource(chosen.path); }
        });
        $("add-to").addEventListener("click", async () => {
          const chosen = await browse("Where the model is written", { start: addTo || addDestination });
          if (chosen) { addTo = chosen.path; checkSource(addSource); }
        });
        $("job-button").addEventListener("click", async () => {
          hold($("job-button"), true, true);
          let ok = false;
          try { const response = await api("v1/jobs", { method: "DELETE", headers: { "Content-Type": "application/json" }, body: "{}" }); if (response.ok) { ok = true; showJob((await response.json()).job); } } catch (error) {}
          hold($("job-button"), false); failed($("job-button"), !ok);
        });
        // Stop and unload: a running reply gets its time, then is cancelled (asked first); Start loads the model again.
        async function power(by) {
          const start = liveState === "stopped";
          if (!start && liveState === "busy" && !confirm("A reply is being written. It gets up to 30 s to finish, then it is cancelled. Stop and unload anyway?")) return;
          applying = by; gate();
          let ok = false;
          try {
            const response = await post(start ? "v1/start" : "v1/stop", {}), result = await response.json().catch(() => ({}));
            ok = response.ok;
            if (!ok) $("model-note").textContent = (result.error && result.error.message) || (start ? "Origami did not start." : "Origami did not stop.");
            else if (start) { reloadStarted("power"); setupFor = null; }
            else render(result);
          } catch (error) { $("model-note").textContent = "Could not reach Origami."; }
          applying = null; failed($(by), !ok); gate(); if (ok) refresh();
        }
        $("power").addEventListener("click", () => liveState === "failed" ? retryNotice() : power("power"));
        $("notice-setup").addEventListener("click", () => chooseWorkspace(noticeGone ? "library" : "setup"));
        async function retryNotice() {
          if (noticeKind === "stopped") return power("notice-retry");
          applying = "notice-retry"; gate();
          let ok = true;
          try {
            if (noticeKind === "found") await Promise.all([loadFound(true), loadModels()]);
            else { ok = (await post("v1/reload", {})).ok; if (ok) reloadStarted("notice-retry"); }
          } catch (error) { ok = false; }
          applying = null; failed($("notice-retry"), !ok); gate(); refresh();
        }
        $("notice-retry").addEventListener("click", retryNotice);
        $("apply").addEventListener("click", () => apply(true));
        $("use-model").addEventListener("click", () => apply(true));
        $("measure").addEventListener("click", async () => {
          measuring = true; gate(); $("basis").textContent = "Measuring: a short reply at the current setup…";
          let ok = false;
          try {
            ok = (await post("v1/chat/completions", { model: "origami", stream: false, max_tokens: 96, temperature: 0, reasoning_effort: "none",
              messages: [{ role: "user", content: "In four short steps, explain how a paper crane is folded." }] })).ok;
          } catch (error) { $("basis").textContent = "Could not reach Origami."; }
          measuring = false; gate(); failed($("measure"), !ok); refresh();
        });
        $("reset").addEventListener("click", () => { closeAdd(); modelChoice = modelInitial; modelByJob = false; $("model").value = modelInitial || ""; $("model-note").textContent = ""; if (!base) { redrawDials(); return; } setupFor = null; refresh(); });
        // A saved prompt opened from the list: its text read-only, rename, keep, a system prompt's text edited into a new one
        // (read as `origami save` reads it, the old one removed once it is saved), and delete after a confirmation here.
        let pvEntry = null, pvOpened = null, pvEditing = false;
        $("prompts").addEventListener("click", (event) => {
          const id = event.target.dataset && event.target.dataset.open;
          if (id) pvOpen(id);
        });
        async function pvOpen(id) {
          pvOpened = id; pvEntry = null; pvEdit(false);
          $("pv-confirm").hidden = true; $("pv-actions").hidden = false; $("pv-edit-start").hidden = true;
          $("pv-title").textContent = "Saved prompt"; $("pv-meta").textContent = ""; $("pv-text").textContent = "Reading…"; $("pv-note").textContent = ""; $("pv-name").value = "";
          if (!$("pv").open) $("pv").showModal();
          $("pv-text").setAttribute("aria-busy", "true");
          const e = await pvCall("GET", "v1/prompts/" + encodeURIComponent(id));
          if (pvOpened !== id) return;
          $("pv-text").setAttribute("aria-busy", "false");
          if (e) pvShow(e); else $("pv-text").textContent = "";
          pvLock(!e);
        }
        // While a saved prompt is read or changed, its other actions wait.
        const pvLock = (on) => ["pv-rename", "pv-keep", "pv-edit-start", "pv-delete", "pv-delete-yes", "pv-delete-no"].forEach((id) => { $(id).disabled = on; });
        function pvShow(e) {
          pvEntry = Object.assign(pvEntry || {}, e);
          $("pv-title").textContent = e.name || "Unnamed";
          $("pv-meta").textContent = [e.kind === "system" ? "System prompt" : "Prompt or conversation", fmt.format(e.tokens) + " tokens", disk(e.bytes) + " on disk",
            "used " + ago(e.last_used), e.pinned ? "kept" : "", pvEntry.system_whole === false ? "the start of a longer system prompt" : ""].filter(Boolean).join(" · ");
          if (pvEntry.text != null && !pvEditing) $("pv-text").textContent = pvEntry.text;
          $("pv-name").value = e.name || "";
          $("pv-keep").setAttribute("aria-pressed", String(!!e.pinned));
          $("pv-edit-start").hidden = pvEntry.system_text == null;
        }
        function pvEdit(on) {
          pvEditing = on; $("pv-edit").hidden = !on; $("pv-text").hidden = on;
          $("pv-edit-start").textContent = on ? "Save as new" : "Edit text";
          if (!on) return;
          $("pv-edit").value = pvEntry.system_text;
          // A save that stops before the prompt ends (Qwen saves at 256-token steps) shows only its start: say so, or its tail is lost unseen.
          $("pv-note").textContent = (pvEntry.system_whole === false ? "This save holds only the start of the system prompt: the text ends early. " : "")
            + "Saving reads the edited text as a new system prompt, under the name above, and removes this one.";
          $("pv-edit").focus();
        }
        // One request, from `button` (which turns meanwhile, and is marked if it fails); its refusal shows in the dialog. The
        // answer, or null.
        async function pvCall(method, path, body, button) {
          pvLock(true); if (button) button.setAttribute("aria-busy", "true");
          let reply = null;
          try {
            const response = await api(path, body ? { method, headers: { "Content-Type": "application/json", ...(revision && ["v1/config", "v1/hot-copy"].includes(path) ? { "If-Match": revision } : {}) }, body: JSON.stringify(body) } : { method });
            const answer = await response.json();
            if (response.ok) reply = answer; else $("pv-note").textContent = (answer.error && answer.error.message) || "Not done.";
          } catch (error) { $("pv-note").textContent = "Could not reach Origami."; }
          pvLock(false);
          if (button) { button.setAttribute("aria-busy", "false"); failed(button, !reply); }
          return reply;
        }
        const pvPath = () => "v1/prompts/" + encodeURIComponent(pvEntry.id);
        $("pv-rename").addEventListener("click", async () => {
          const name = $("pv-name").value.trim();
          if (!pvEntry || !name || name === pvEntry.name) return;
          const e = await pvCall("PATCH", pvPath(), { name }, $("pv-rename"));
          if (e) { pvShow(e); $("pv-note").textContent = "Renamed."; refresh(); }
        });
        $("pv-name").addEventListener("keydown", (event) => { if (event.key === "Enter" && !pvEditing) $("pv-rename").click(); });
        $("pv-keep").addEventListener("click", async () => {
          if (!pvEntry) return;
          const e = await pvCall("PATCH", pvPath(), { keep: !pvEntry.pinned }, $("pv-keep"));
          if (e) { pvShow(e); $("pv-note").textContent = e.pinned ? "Kept: it is not removed to make room." : "Released: it may be removed to make room."; refresh(); }
        });
        $("pv-edit-start").addEventListener("click", async () => {
          if (!pvEntry) return;
          if (!pvEditing) return pvEdit(true);
          const text = $("pv-edit").value, name = $("pv-name").value.trim();
          if (text === pvEntry.system_text) { $("pv-note").textContent = "The text is unchanged."; return; }
          if (!name) { $("pv-note").textContent = "Give it a name first."; $("pv-name").focus(); return; }
          $("pv-note").textContent = "Reading the new system prompt…";
          const reply = await pvCall("POST", "v1/prompts", { text, name, replace: pvEntry.id }, $("pv-edit-start"));
          if (!reply) return;
          refresh(); await pvOpen(reply.id); $("pv-note").textContent = reply.message;
        });
        $("pv-delete").addEventListener("click", () => { $("pv-actions").hidden = true; $("pv-confirm").hidden = false; $("pv-delete-no").focus(); });
        $("pv-delete-no").addEventListener("click", () => { $("pv-confirm").hidden = true; $("pv-actions").hidden = false; $("pv-delete").focus(); });
        $("pv-delete-yes").addEventListener("click", async () => {
          if (!pvEntry) return;
          const name = pvEntry.name;
          if (!(await pvCall("DELETE", pvPath(), null, $("pv-delete-yes")))) return;
          $("pv").close(); $("prompt-note").textContent = "Deleted " + (name || "the saved prompt") + "."; refresh();
        });
        $("pv-close").addEventListener("click", () => $("pv").close());
        // Escape closes it too; focus goes back to the entry's Open button when the list still has it.
        $("pv").addEventListener("close", () => {
          const back = pvOpened && $("prompts").querySelector('button[data-open="' + CSS.escape(pvOpened) + '"]');
          pvEntry = null; pvOpened = null;
          if (back) back.focus();
        });
        // The folder browser: one dialog for every path the page takes. It resolves with the chosen folder's listing (its path,
        // drive and free space), with `models` a model's {path} (a .gguf file or a checkpoint folder), or null when closed.
        let fsState = null, fsHome = null;
        const local = () => ["127.0.0.1", "localhost", "[::1]"].includes(location.hostname);
        function browse(title, opts = {}) {
          return new Promise((resolve) => {
            if (fsState) fsDone(null);
            fsState = { resolve, models: !!opts.models, listing: null, from: document.activeElement };
            $("fs-search").value = ""; $("fs-path").value = opts.start || "";
            $("fs-host").textContent = "Folders on " + (dashboard && dashboard.host || "the Origami server");
            fsHome = null;
            $("fs-title").textContent = title; $("fs-new").hidden = true; $("fs-note").textContent = ""; $("fs-list").innerHTML = ""; $("fs-crumbs").innerHTML = "";
            hold($("fs-choose"), true, false, "The folder is still loading"); $("fs-choose").textContent = fsState.models ? "Choose this model" : "Choose this folder"; $("fs-mkdir").hidden = fsState.models;
            $("fs").showModal(); $("fs").setAttribute("aria-busy", "true");
            fsStart(opts.start);
          });
        }
        async function fsStart(start) {
          const opened = fsState;
          try {
            const response = await api("v1/fs"), places = await response.json();
            if (opened !== fsState) return;
            if (response.ok) {
              const roots = places.roots || [], home = roots.find((r) => r.name === "Home");
              fsHome = home ? home.path : roots.length ? roots[0].path : null;
              $("fs-roots").innerHTML = roots.map((r) => '<li><button type="button" data-path="' + esc(r.path) + '">' + esc(r.name) + '<span class="sub">' +
                (r.free_bytes != null ? disk(r.free_bytes) + " free" : "not readable yet") + "</span></button></li>").join("");
            }
          } catch (error) {}
          // The folder asked for, else the nearest one above it that exists (a store not made yet), else the home folder.
          if (opened !== fsState) return;
          for (let at = start || ""; at && at !== "/" && opened === fsState; at = at.replace(/\/[^\/]*\/?$/, "")) if (await fsOpen(at, true)) return;
          if (opened !== fsState) return;
          if (fsHome) await fsOpen(fsHome);
          else { $("fs-note").textContent = "Could not reach Origami."; $("fs").setAttribute("aria-busy", "false"); }
        }
        // While a folder opens, the list turns and Choose waits for it (the folder shown until then is not the one asked for).
        async function fsOpen(path, quiet, offset = 0) {
          if (!fsState) return false;
          const opened = fsState, seq = fsOpen.seq = (fsOpen.seq || 0) + 1;
          $("fs").setAttribute("aria-busy", "true"); hold($("fs-choose"), true, false, "The folder is still loading");
          let shown = false;
          try {
            const response = await api("v1/fs?" + new URLSearchParams({ path, search: $("fs-search").value || "", offset, ...(fsState.models ? { models: "" } : {}) })), listing = await response.json();
            if (seq !== fsOpen.seq || opened !== fsState) return false;
            if (!response.ok) { if (!quiet) $("fs-note").textContent = (listing.error && listing.error.message) || "Origami cannot open it."; return false; }
            if (fsState) { fsShow(listing); shown = true; }
            return true;
          } catch (error) { if (!quiet && seq === fsOpen.seq) $("fs-note").textContent = "Could not reach Origami."; return false; }
          finally { if (seq === fsOpen.seq) { $("fs").setAttribute("aria-busy", "false"); if (!shown && fsState) { fsState.listing = null; hold($("fs-choose"), true, false, "The folder could not be read"); hold($("fs-mkdir"), true, false, "The folder could not be read"); } } }
        }
        const fsChoosable = (l) => !!l && !!fsState && !l.permission && (fsState.models ? !!l.model : l.writable !== false);
        function fsShow(l) {
          fsState.listing = l; $("fs-path").value = l.path; $("fs-more").hidden = l.next_offset == null;
          $("fs-crumbs").innerHTML = l.crumbs.map((c, i) => '<li><button type="button" data-path="' + esc(c.path) + '"' + (i === l.crumbs.length - 1 ? ' aria-current="location"' : "") + ">" + esc(c.name) + "</button></li>").join("");
          $("fs-here").textContent = l.drive ? driveLabel(l.drive) + (l.free_bytes != null ? " · " + disk(l.free_bytes) + " free" : "") : "Folders";
          const up = l.parent ? [[l.parent, "folder", "..", "the folder above"]] : [];
          const items = up.concat(l.folders.map((f) => [f.path, f.model ? "model" : "folder", f.name, f.model ? "model folder" : ""]),
            l.models.map((m) => [m.path, "model", m.name, "GGUF · " + disk(m.bytes)]));
          $("fs-list").innerHTML = items.map(([path, kind, name, sub]) => '<li><button type="button" data-path="' + esc(path) + '" data-kind="' + kind + '">' +
            (kind === "folder" ? '<i aria-hidden="true">/</i>' : "") + esc(name) + (sub ? '<span class="sub">' + esc(sub) + "</span>" : "") + "</button></li>").join("") +
            (l.folders.length || l.models.length || l.permission ? "" : '<li class="empty">' + (fsState.models ? "No folders or models here." : "No folders here.") + "</li>");
          $("fs-note").textContent = l.permission ? l.permission + "." : l.next_offset != null ? "More entries available. Filter or open the next page." : l.writable === false && !fsState.models ? "Read-only folder" : "";
          $("fs-settings").hidden = !l.permission || !local();
          hold($("fs-choose"), !fsChoosable(l), false, l.permission ? "macOS has not let Origami read this folder" : fsState.models ? "Choose a model in the list" : "Origami cannot write in this folder");
          hold($("fs-mkdir"), !!l.permission || l.writable === false, false, l.permission ? "macOS has not let Origami read this folder" : "Origami cannot write in this folder");
          const first = $("fs-list").querySelector("button");
          if (first) first.focus();
        }
        function fsDone(result) {
          const state = fsState;
          if (!state) return;
          fsState = null; fsOpen.seq = (fsOpen.seq || 0) + 1; $("fs").close();
          if (state.from && state.from.focus) state.from.focus();
          state.resolve(result);
        }
        async function fsMake() {
          const name = $("fs-name").value.trim();
          if (!name || !fsState || !fsState.listing || $("fs-make").disabled) return;
          hold($("fs-make"), true, true);
          let ok = false; const opened = fsState;
          try {
            const response = await post("v1/fs/mkdir", { path: fsState.listing.path.replace(/\/$/, "") + "/" + name }), listing = await response.json();
            if (!response.ok) { $("fs-note").textContent = (listing.error && listing.error.message) || "Not made."; return; }
            if (opened !== fsState) return;
            ok = true; $("fs-new").hidden = true; if (fsState) fsShow(listing);
          } catch (error) { $("fs-note").textContent = "Could not reach Origami."; }
          finally { hold($("fs-make"), false); failed($("fs-make"), !ok); }
        }
        $("fs").addEventListener("click", (event) => {
          const button = event.target.closest && event.target.closest("button[data-path]");
          if (!button) return;
          if (button.dataset.kind === "model") fsDone({ path: button.dataset.path }); else fsOpen(button.dataset.path);
        });
        // Escape, or the browser closing the dialog, chooses nothing.
        $("fs").addEventListener("close", () => fsDone(null));
        $("fs-close").addEventListener("click", () => fsDone(null));
        $("fs-choose").addEventListener("click", () => { if (fsState && fsState.listing) fsDone(fsState.listing); });
        $("fs-mkdir").addEventListener("click", () => { $("fs-new").hidden = false; $("fs-name").value = ""; $("fs-name").focus(); });
        $("fs-make").addEventListener("click", fsMake);
        $("fs-name").addEventListener("keydown", (event) => { if (event.key === "Enter") fsMake(); });

        // Where Origami keeps things, each set with the browser: the saved prompts' and long contexts' folder, more folders the
        // model search looks in and, for a model with a smaller width, the expert folders (stores). The hot copy is the Experts card's.
        let folderSettings = null;
        async function loadFolders() {
          const expected = revision, request = loadFolders.request = (loadFolders.request || 0) + 1;
          try {
            const response = await api("v1/fs"), result = response.ok ? await response.json() : null;
            if (expected === revision && request === loadFolders.request && result) { folderSettings = result.settings; showFolders(); }
          } catch (error) {}
        }
        function showFolders() {
          const f = folderSettings;
          if (!f) return;
          const place = (p) => wpath(p.path) + '<span class="sub">' + esc(p.free_bytes != null ? driveLabel(p.drive) + " · " + disk(p.free_bytes) + " free" : p.permission || "") + "</span>";
          const row = (p, act, label) => '<div class="frow"><div>' + place(p) + "</div>" + (act ? '<button type="button" data-act="' + act + '" data-path="' + esc(p.path) + '" aria-label="' + esc(label + " " + p.path) + '">' + label + "</button>" : "") + "</div>";
          const group = (title, hint, rows, add) => '<div class="fgroup"><h3>' + title + '</h3><p class="fhint">' + hint + "</p>" + rows + add + "</div>";
          $("folders").innerHTML =
            group("Saved prompts and long contexts", "The fastest drive with room is best.", row(f.context_dir), '<button type="button" data-act="context-choose">Choose…</button>') +
            group("Model search", "The Model menu also lists the models found in these folders.", f.models_dirs.map((x) => row(x, "search-remove", "Remove")).join(""),
              '<button type="button" data-act="search-add">Add a folder…</button>') +
            (storesMatter() ? group("Expert folders", "Where Origami may write copies of the experts. It always also uses its data folder.",
              ownStores().map((x) => row(x, "store-remove", "Remove")).join(""), '<button type="button" data-act="store-add">Add a folder…</button>') : "");
          gate();
        }
        // The expert folders (stores) matter only to a running model that writes there: Qwen's 4-bit build, whose pack and
        // further copies go there (not a GGUF, whose experts stay in their file).
        const storesMatter = () => !noModel && ready && (kind === "qwen" || kind === "fixed" && !!xFrom && !xFrom.no_hot_copy);
        // The stores but the data folder's own, which Origami always uses.
        const dataStore = () => folderSettings && folderSettings.data_dir ? folderSettings.data_dir.replace(/\/+$/, "") + "/store" : null;
        const ownStores = () => folderSettings.stores.filter((x) => x.path.replace(/\/+$/, "") !== dataStore());
        async function folderAction(act, path, button, start, note = $("folders-note")) {
          const f = folderSettings, paths = (list) => list.map((x) => x.path);
          if (!f) return;
          if (act === "store-remove") return saveFolder({ stores: JSON.stringify(paths(ownStores()).filter((p) => p !== path)) }, button, "Removed; what it holds stays on the drive, listed under Storage, where it can be removed.");
          if (act === "search-remove") return saveFolder({ models_dirs: JSON.stringify(paths(f.models_dirs).filter((p) => p !== path)) }, button, "Removed.");
          const titles = { "store-add": "Add a folder for expert copies", "context-choose": "The folder for saved prompts and long contexts", "search-add": "Add a folder to the model search" };
          const chosen = await browse(titles[act], { start: start || (act === "context-choose" ? f.context_dir.path : null) });
          if (!chosen) return;
          // Lists are saved comma-separated.

          const at = chosen.path.replace(/\/+$/, "");
          if (act === "store-add" && (at === (f.data_dir || "").replace(/\/+$/, "") || at === dataStore())) { note.textContent = "Origami already uses its data folder."; return; }
          saveFolder(act === "store-add" ? { stores: JSON.stringify(paths(ownStores()).concat(chosen.path)) } : act === "search-add" ? { models_dirs: JSON.stringify(paths(f.models_dirs).concat(chosen.path)) }
            : { context_dir: chosen.path }, button, null, note);
        }
        async function saveFolder(body, button, said, note = $("folders-note")) {
          if (button) hold(button, true, true);
          note.textContent = "Saving…";
          try {
            const response = await post("v1/config", body), result = await response.json();
            if (!response.ok) { note.textContent = (result.error && result.error.message) || "Not saved."; if (button) failed(button, true); return; }
            const kept = result.overridden && result.overridden.length ? " " + result.overridden.join(", ").replace(/_/g, " ") + " stay as this server's command line sets them." : "";
            // The model search reads the folders at once; the others are read when the model starts.
            const search = "models_dirs" in body;
            note.textContent = (said || "Saved.") + (search ? " The Model menu looks there now." : " Origami uses it once it reloads" + (note === $("folders-note") ? "." : " (Reload now, under Folders).")) + kept;
            $("folders-reload").hidden = search;
            if (search) loadFound(true);
            await loadFolders(); loadStorage();
          } catch (error) { note.textContent = "Could not reach Origami."; if (button) failed(button, true); }
          finally { if (button) hold(button, false); }
        }
        $("folders").addEventListener("click", (event) => {
          const button = event.target.closest && event.target.closest("button[data-act]");
          if (button) folderAction(button.dataset.act, button.dataset.path, button);
        });
        $("folders-reload").addEventListener("click", async () => {
          applying = "folders-reload"; gate();
          let ok = false;
          try { const response = await post("v1/reload", {}); if (response.ok) { ok = true; reloadStarted("folders-reload"); $("folders-reload").hidden = true; $("folders-note").textContent = "Reloading; requests wait until it is back."; } }
          catch (error) { $("folders-note").textContent = "Could not reach Origami."; }
          applying = null; failed($("folders-reload"), !ok); gate(); refresh();
        });
        // Storage: the drives Origami's files are on, the running model's items first, each with its role and size, and what can
        // be done with it: a model, a converted set or the saved prompts move (a job), and an item nothing uses is removed after
        // a confirm. Where the hot copy goes is the Experts card's. The read has its own status line: notes others wrote stay.
        const roleNames = { model: "Model", pack: "All experts, converted", "hot copy": "Hot copy", prompts: "Saved prompts", support: "Support files", download: "Download",
          conversion: "Conversion", staging: "Left over" };
        // A converted set names its width where its folder or note does ("q4_k experts", "…-3bit").
        const roleName = (i) => { const bits = i.role === "pack" && /(?:q|\b)(\d)(?:_|-?bit)/i.exec(i.name + " " + (i.note || "")); return bits ? "All experts, " + bits[1] + "-bit" : roleNames[i.role] || i.role; };
        const itemName = (i) => !i.model ? i.name : i.role === "support" ? i.model + " support files" : i.role === "hot copy" && i.name === "hot" ? i.model + " hot copy" : i.name;
        async function loadStorage() {
          if (storageLoading) return;
          storageLoading = true; gate();
          const expected = revision;
          $("storage-read").setAttribute("aria-busy", "true");
          try {
            const response = await api("v1/storage"), map = await response.json();
            if (!response.ok) throw Error(map.error && map.error.message || "Storage unavailable");
            if (expected !== revision || map.revision && revision && map.revision !== revision) { storageAt = 0; return; }
            storageMap = map; $("storage-read").textContent = ""; showStorage();
          } catch (error) {
            storageMap = null; $("storage").innerHTML = ""; $("storage-total").textContent = "";
            $("storage-read").textContent = error.message || "Storage unavailable";
          } finally { storageLoading = false; $("storage-read").setAttribute("aria-busy", "false"); gate(); }
        }
        function storageDrive(d) {
          const total = d.total_bytes || 0, free = d.free_bytes || 0, roles = d.roles || {}, items = d.items || [];
          const groups = ["model", "pack", "hot copy", "prompts"].map((r) => [r, roles[r] || 0]);
          groups.push(["origami", Object.keys(roles).filter((r) => !groups.some((g) => g[0] === r)).reduce((sum, r) => sum + roles[r], 0)]);
          const ours = groups.reduce((sum, g) => sum + g[1], 0), other = Math.max(0, total - free - ours), share = (b) => total ? (100 * b / total).toFixed(2) : 0;
          const names = { model: "models", pack: "converted experts", "hot copy": "hot copies", prompts: "saved prompts", origami: "other Origami files", other: "other files" };
          const shown = groups.filter((g) => g[1] > 0).concat(total ? [["other", other]] : []);
          const name = d.name === "internal drive" ? "This Mac" : d.name;
          const head = '<div class="sdrive-head"><h3>' + esc(name) + "</h3><span>" + (total ? disk(free) + " free of " + disk(total) : d.not_connected ? "not connected" : esc(d.permission || "not readable yet")) + "</span></div>";
          const bar = total ? '<div class="sbar" role="img" aria-label="' + esc(name + ": " + shown.map((g) => names[g[0]] + " " + disk(g[1])).join(", ") + ", " + disk(free) + " free") + '">' +
            shown.map((g) => '<span data-role="' + g[0] + '" style="width:' + share(g[1]) + '%"></span>').join("") + "</div>" +
            '<div class="skeys" aria-hidden="true">' + shown.map((g) => '<span><i data-role="' + g[0] + '"></i>' + names[g[0]] + " " + disk(g[1]) + "</span>").join("") + "</div>" : "";
          const button = (act, item, label) => '<button type="button" data-sact="' + act + '" data-path="' + esc(item.path) + '" aria-label="' + esc(label + " " + item.name) + '">' + label + "</button>";
          const ul = (list) => '<ul class="sitems">' + list.map((i) => '<li class="frow"><div>' + esc(itemName(i)) + ' <span class="tag">' + esc(roleName(i)) +
            (i.model && i.role !== "model" && i.role !== "support" && itemName(i) === i.name ? " · " + esc(i.model) : "") + "</span>" + (i.in_use ? ' <span class="tag kept">in use</span>' : "") + (i.unused ? ' <span class="tag">not used</span>' : "") +
            '<span class="sub">' + wpath(i.path) + " · " + disk(i.bytes) + (i.note ? " · " + esc(i.note) : "") + '</span></div><div class="sact">' +
            (i.actions.includes("move") ? button("move", i, "Move…") : "") + (i.actions.includes("remove") ? button("remove", i, "Remove") : "") + "</div></li>").join("") + "</ul>";
          // Another model's items go below, under their own heading, so they never read as the running model's.
          const who = storageMap && storageMap.running, notUsed = (i) => !!who && !!i.model && i.model !== who && !i.in_use;
          const mine = items.filter((i) => !notUsed(i)), others = items.filter(notUsed);
          const list = (mine.length ? ul(mine) : "") + (others.length ? '<h4 class="sother">Not used by ' + esc(who) + "</h4>" + ul(others) : "");
          return '<div class="sdrive">' + head + bar + "</div>";
        }
        function showStorage() {
          const m = storageMap; if (!m) return;
          const items = m.drives.flatMap(d => (d.items || []).map(i => ({ ...i, drive: d.name })));
          const groups = new Map();
          for (const i of items) { const key = i.model || "Other files"; if (!groups.has(key)) groups.set(key, []); groups.get(key).push(i); }
          const button = (i, act, label) => '<button type="button" data-sact="' + act + '" data-path="' + esc(i.path) + '">' + label + '</button>';
          const file = i => {
            const acts = i.actions || [];
            const used = i.state || (i.in_use ? "in use" : i.unused ? "not used" : "");
            return '<li class="frow"><div><div class="file-title">' + esc(roleName(i)) + ' <span class="tag">' + esc(used) + '</span></div><div class="file-location">' + esc(driveLabel(i.drive)) + ' · ' + disk(i.allocated_bytes == null ? i.bytes : i.allocated_bytes) + '</div><details class="file-path" data-storage-key="file:' + esc(i.path) + '"><summary>Path</summary><p>' + wpath(i.path) + '</p>' + (i.blocked ? '<p>' + esc(i.blocked) + '</p>' : '') + '</details></div><div class="sact">' +
              (acts.includes("move") ? button(i,"move","Move…") : '') +
              (acts.includes("delete") ? button(i,"delete","Delete…") : acts.includes("remove") ? button(i,"remove","Delete…") : '') +
              (acts.includes("relink") ? button(i,"relink","Locate…") : '') +
              (acts.includes("forget") ? button(i,"forget","Unlist…") : '') + '</div></li>';
          };
          const html = '<div class="drive-grid">' + m.drives.map(storageDrive).join('') + '</div>' + [...groups.entries()].map(([name, files]) => '<details class="model-files" data-storage-key="model:' + esc(name) + '" open><summary><h3>' + esc(name) + '</h3><span>' + files.length + (files.length === 1 ? ' item · ' : ' items · ') + disk(files.reduce((n,i) => n + (i.allocated_bytes == null ? i.bytes : i.allocated_bytes),0)) + '</span></summary><ul class="sitems">' + files.map(file).join('') + '</ul></details>').join('');
          if (html !== showStorage.html) {
            const expanded = new Map([...$("storage").querySelectorAll("details[data-storage-key]")].map(node => [node.dataset.storageKey, node.open]));
            const focused = document.activeElement && document.activeElement.dataset, focusPath = focused && focused.path, focusAction = focused && focused.sact;
            $("storage").innerHTML = html; showStorage.html = html;
            $("storage").querySelectorAll("details[data-storage-key]").forEach(node => { if (expanded.has(node.dataset.storageKey)) node.open = expanded.get(node.dataset.storageKey); });
            if (focusPath && focusAction) [...$("storage").querySelectorAll("button[data-sact]")].find(node => node.dataset.path === focusPath && node.dataset.sact === focusAction)?.focus();
          }
          swap("disk-budget", diskBudget(status));
          $("storage-total").textContent = disk(items.reduce((n,i) => n + (i.allocated_bytes == null ? i.bytes : i.allocated_bytes),0)) + " on disk";
          modelList();
          $("storage-recovery").innerHTML = (m.operations || []).map(o => '<p>Interrupted move · ' + esc(o.name) + ' <button type="button" data-recover="' + esc(o.id) + '">Recover…</button></p>').join('') +
            (m.relinks || []).map(r => '<p>Drive renamed or remounted · ' + esc(r.setting) + ': ' + wpath(r.from) + ' is at ' + wpath(r.to) + ' (' + esc(r.how) + '). Quit the server and run <code>origami relink</code> to use it there.</p>').join('');
          gate();
        }
        let storageConfirmResolve = null;
        function confirmStorage(plan) {
          return new Promise(resolve => {
            storageConfirmResolve = resolve;
            const action = { move: "Move", delete: "Delete", remove: "Delete", forget: "Remove from library", relink: "Use the new location" }[plan.kind];
            $("storage-confirm-title").textContent = action + " · " + plan.name;
            $("storage-accept").textContent = action;
            const moving = plan.kind === "move", deleting = ["delete","remove"].includes(plan.kind);
            $("storage-confirm-body").innerHTML = '<p>' + wpath(plan.path) + (plan.to ? '<br>→ ' + wpath(plan.to) : '') + '</p>' +
              (moving ? '<p>' + (plan.copy_bytes ? disk(plan.required_bytes) + ' required · ' + disk(plan.free_bytes) + ' free. Copy, verify, then remove the original.' : 'Same disk · no additional copy.') + '</p>' : '') +
              (deleting ? '<p>Delete ' + disk(plan.allocated_bytes || plan.bytes) + ' permanently.' + (plan.disables_cache ? ' Automatic hot copying will be turned off for this model.' : '')
                + (plan.role === "download" ? ' This is a whole downloaded model: to use it again, all ' + disk(plan.allocated_bytes || plan.bytes) + ' must be downloaded again.' : '') + '</p>' : '') +
              (plan.kind === "forget" ? '<p>Files stay on disk.</p>' : '') +
              (plan.in_use ? '<p>' + (moving ? 'The model pauses during the move and loads from the destination.' : 'The loaded model must stop for this operation.') + '</p>' : '');
            $("storage-confirm").showModal();
          });
        }
        async function storageAction(act, path, button) {
          const note = $("storage-note"), item = storageMap && storageMap.drives.flatMap(d => d.items || []).find(i => i.path === path);
          if (!item || running() || !item.actions.includes(act)) return;
          try {
            let to;
            if (["move", "relink"].includes(act)) {
              const chosen = await browse(act === "move" ? "Move " + item.name : "Locate " + item.name, { models: act === "relink" });
              if (!chosen) return; to = chosen.path;
            }
            const body = { kind: act, path, ...(to ? { to } : {}) };
            let plan;
            if (act === "relink") plan = { ...body, name: item.name, in_use: item.in_use, revision };
            else {
              const response = await post("v1/storage/plan", body); plan = await response.json();
              if (!response.ok) throw Error(plan.error && plan.error.message || "Storage changed; refresh and try again.");
            }
            if (!await confirmStorage(plan)) return;
            if (plan.revision && plan.revision !== revision) throw Error("Settings changed. Review the operation again.");
            storagePath = path; await startJob(body, button, note); storageAt = 0;
          } catch (error) { note.textContent = error.message || "Could not start"; }
        }

        $("storage").addEventListener("click", (event) => {
          const button = event.target.closest && event.target.closest("button[data-sact]");
          if (button) storageAction(button.dataset.sact, button.dataset.path, button);
        });
        $("storage-refresh").addEventListener("click", loadStorage);
        // All workspaces share one observed server state and one draft.
        function acceptSnapshot(next) {
          const revised = revision && revision !== next.revision;
          const restarted = generation && generation !== next.generation;
          // A draft based on older settings stays editable (the save names any key changed since); access edits and a
          // restarted server, which no longer knows the draft's base, still ask for a review.
          if ((restarted && dirty() || (revised || restarted) && Object.keys(accessChanges()).length > 0) && !applying && !reloading) conflict = true;
          if (restarted) {
            ++setupSeq; ++planSeq; base = lastPlan = null; setupFor = null; ready = false;
            showSpeed.last = null; smSeen = null; xFrom = xHot = null; expertsCard.acts = null;
            $("experts").hidden = true; $("speed").textContent = "—";
          }
          if (revised || restarted) { syncedRevision = null; auxAt = storageAt = 0; }
          dashboard = next; revision = next.revision; generation = next.generation;
          showWiredMemory();
          $("restore").hidden = !next.can_restore;
        }
        function syncSaved() {
          if (!dashboard || dashboard.revision !== revision || !initial || conflict || syncedRevision === revision || dirty()) return;
          const saved = dashboard.saved_settings || {}, value = k => saved[k] && saved[k].value;
          const context = value("context");
          if (context) { const n = parseFloat(context) * (/k$/i.test(context) ? 1024 : 1); $("context").value = String(Math.max(0, Math.min(+$("context").max, CONTEXTS.findIndex(x => x[0] >= n)))); }
          if (value("quality")) $("quality").value = String(levelAt(value("quality")));
          if (value("reasoning_effort")) $("reasoning").value = String(effortIndex(value("reasoning_effort")));
          const memory = value("memory_limit");
          if (memory) {
            autoMemory = memory === "auto" || memory.startsWith("(");
            if (!autoMemory && memoryStops.length) {
              storedMemory = parseFloat(memory) * (/tb$/i.test(memory) ? 1e12 : /mb$/i.test(memory) ? 1e6 : /gb$/i.test(memory) ? 1e9 : 1);
              const i = memoryStops.findIndex(x => x * GB >= storedMemory); $("memory").value = String(i < 0 ? memoryStops.length - 1 : i);
            }
          }
          modelInitial = dashboard.configured_model; modelChoice = modelInitial; fillPicker();
          rebase();
          if (smSaved) {
            const custom = value("sampling") === "custom", values = { ...smSaved.rec.non_thinking };
            if (custom) for (const k of SM_KEYS) { const n = Number(value(k)); if (value(k) && Number.isFinite(n)) values[k] = n; }
            smSaved = { ...smSaved, custom, override: value("sampling_override") === "true", values };
            $("sm-custom").checked = custom; $("sm-recommended").checked = !custom; $("sm-override").checked = smSaved.override;
            smShow(values);
          }
          syncedRevision = revision; drawDials(lastPlan); showEffects();
        }
        // Saved settings the loaded model does not run with yet, in the same words wherever it shows.
        function pendingText() {
          const lines = (dashboard && dashboard.pending_changes || []).map((c) => c.label + ": saved " + memoryText(c.saved) + ", running " + memoryText(c.running));
          return STATE.reload + ": saved settings differ from the running model" + (lines.length ? ". " + lines.join("; ") + "." : ".");
        }
        // Why tuning and measuring cannot start now ("" when they can); the server's refusal of a tune says the same.
        const RELOAD_FIRST = "Reload to apply saved settings first", FAILED_START = "The model failed to start: Try again first";
        function tuneBlock(edits, switching, needsReload) {
          return offline ? "Not connected to Origami" : running() || jobPost ? "Another job is running" + (job && job.title ? ": " + job.title : "")
            : liveState === "stopped" || liveState === "stopping" ? "The model is stopped: Start it first" : liveState === "failed" ? FAILED_START : liveState === "down" ? "No model is running"
            : applying || reloading || ["loading", "reloading", "unavailable"].includes(liveState) ? "The model is loading"
            : conflict ? "Settings changed elsewhere: review them first" : edits || switching ? "Save or discard your changes first"
            // The server's reason (`DashboardState.tuneBlock`), the one its job start and `origami tune` refuse with.
            : dashboard && typeof dashboard.tune_block === "string" ? dashboard.tune_block
            : needsReload ? RELOAD_FIRST : liveState === "busy" || dashboard && !dashboard.can_tune ? "A request is running: this starts when the model is idle" : liveState !== "ready" ? "The model is not ready" : "";
        }
        async function reloadSaved(by) {
          applying = by; gate();
          try {
            const r = await post(liveState === "stopped" ? "v1/start" : "v1/reload", {}), result = await r.json();
            if (!r.ok) throw Error(result.error && result.error.message || "Reload failed");
            reloadStarted("reload-saved"); await refresh();
          } catch (error) { $("apply-note").textContent = error.message; }
          finally { applying = null; gate(); }
        }
        ["reload-saved", "pending-reload", "st-reload", "health-reload"].forEach((id) => $(id).addEventListener("click", () => reloadSaved(id)));
        // Discard the saved change: the running values are saved again, so nothing waits for a reload.
        $("pending-discard").addEventListener("click", async () => {
          const body = dashboard && dashboard.pending_revert || {};
          if (!Object.keys(body).length) return;
          applying = "pending-discard"; gate();
          try {
            const r = await post("memory_limit" in body ? "v1/memory/wired" : "v1/config", body), result = await r.json();
            if (!r.ok) { if (r.status === 409) conflict = true; throw Error(result.error && result.error.message || "Not saved"); }
            revision = result.revision || revision; syncedRevision = null; await refresh();
          } catch (error) { $("apply-note").textContent = error.message; }
          finally { applying = null; gate(); }
        });
        function gate() {
          coreGate();
          const doing = id => applying === id || reloading && reloadBy === id;
          const busy = running() || !!jobPost || !!applying || reloading || ["loading", "stopping", "unavailable"].includes(liveState);
          if (noticeKind === "missing") $("power").hidden = true;
          const switching = modelChoice !== modelInitial || dashboard && dashboard.active_model && dashboard.active_model !== settingName(modelInitial) && dashboard.reload_required;
          const edits = dirty(), canSave = !offline && !busy && !conflict && ready;
          // A tuning job runs the model at settings under test (the memory search, at each limit): no saved draft waits for a reload then.
          const needsReload = !!dashboard && !!dashboard.reload_required && !(running() && job.kind === "tune");
          $("dials").disabled = offline || busy || !!switching || conflict;
          $("sm-group").disabled = offline || busy || !!switching || conflict || !smSaved || smSaved.launched;
          // The dials say why they do not move, as the buttons do.
          const dialsWhy = waitReason() || (conflict ? "Settings changed elsewhere: review them first" : switching ? "Load the chosen model first" : "");
          $("dials").title = $("dials").disabled ? dialsWhy : "";
          $("sm-group").title = $("sm-group").disabled ? dialsWhy || (smSaved && smSaved.launched ? "Set by the command that started Origami" : "The model's sampling is not loaded yet") : "";
          hold($("model"), offline || busy);
          const saveWhy = busy || offline ? "" : conflict ? "Settings changed elsewhere: review them first" : !ready ? "The setup is still loading" : !edits ? "Nothing changed" : planRefused ? "These settings cannot run on this Mac" : "";
          hold($("apply"), !canSave || !edits || planRefused, doing("apply"), saveWhy);
          hold($("save"), !canSave || !edits || planRefused, doing("save"), saveWhy);
          hold($("discard"), offline || busy || !(edits || conflict), false, busy || offline ? "" : "Nothing to discard");
          if (planRefused && edits && !applying) $("apply-note").textContent = "These settings cannot run on this Mac.";
          const chosen = models.find(m => (m.origami ? m.origami.setting : m.id) === modelChoice);
          const missing = chosen && chosen.origami && ["not_connected", "no_permission", "missing", "checking", "slow"].includes(chosen.origami.availability);
          $("use-model").hidden = !modelChoice || modelChoice === modelInitial || noticeKind === "missing";
          const modelWhy = !modelInitial ? "Choose a model under Models first" : missing ? "The model's files are not found: see Models" : "";
          hold($("use-model"), !canSave || !!missing, doing("use-model"), busy || offline ? "" : conflict ? "Settings changed elsewhere: review them first" : !ready ? "The setup is still loading" : modelWhy);
          hold($("power"), offline || running() || !!jobPost || !!applying || liveState === "stopping" || reloading && reloadBy === "power" || liveState === "stopped" && (!modelInitial || !!missing), doing("power") || liveState === "stopping", busy || offline ? "" : modelWhy);
          if (noticeKind === "stopped") { $("notice").hidden = true; hold($("notice-retry"), offline || busy || !modelInitial || !!missing, doing("notice-retry"), busy || offline ? "" : modelWhy); }
          const idle = !offline && !busy && !edits && !conflict && !switching && (dashboard ? dashboard.can_tune && !dashboard.reload_required : liveState === "ready");
          // No silently disabled action: the reason stands next to tuning and the memory search, with Reload when that is it.
          const why = idle ? "" : tuneBlock(edits, switching, needsReload) || "Available when the loaded model is idle", reloadFirst = why === RELOAD_FIRST;
          ["st-go", "st-check", "st-undo", "measure"].forEach(id => { hold($(id), !idle || measuring, id === "measure" ? measuring : running() && job.kind === "tune" && tuneBy === id); $(id).title = why; });
          $("st-why").textContent = why ? why + "." : ""; $("st-reload").hidden = !reloadFirst;
          $("pending").hidden = !needsReload || offline; $("pending-text").textContent = needsReload ? pendingText() : "";
          $("pending-discard").hidden = !(dashboard && Object.keys(dashboard.pending_revert || {}).length);
          const draftWhy = busy || offline ? "" : conflict ? "Settings changed elsewhere: review them first" : edits ? "Save or discard your changes first" : "";
          ["pending-reload", "pending-discard", "st-reload", "health-reload"].forEach(id => hold($(id), offline || busy || edits || conflict, doing(id), draftWhy));
          $("st-result").querySelectorAll("button[data-tune-apply]").forEach(b => hold(b, !idle || b.hasAttribute("data-blocked"), false, b.hasAttribute("data-blocked") ? "" : why));
          const staleMap = !!(storageMap && storageMap.revision && revision && storageMap.revision !== revision);
          $("storage").querySelectorAll("button[data-sact]").forEach(b => hold(b, offline || busy || staleMap, false, busy || offline ? "" : "The list is out of date: it refreshes in a moment"));
          $("storage-recovery").querySelectorAll("button").forEach(b => hold(b, offline || busy || liveState !== "stopped", false, busy || offline ? "" : "Stop the model before recovering a move"));
          listWhy("storage-why", [...$("storage").querySelectorAll("button[data-sact]"), ...$("storage-recovery").querySelectorAll("button")]);
          ["x-use", "restore", "acc-restart"].forEach(id => { if (offline || busy) hold($(id), true, $(id).getAttribute("aria-busy") === "true"); });
          $("x-acts").querySelectorAll("button").forEach(b => { if (offline || busy) hold(b, true); });
          listWhy("x-acts-why", $("x-acts").querySelectorAll("button"));
          hold($("restore"), offline || busy || edits, false, busy || offline ? "" : "Save or discard your changes first");
          const wired = wiredMemory();
          hold($("wired-restore"), offline || busy || edits || conflict || !wired || !wired.supported || !wired.requires_raise || launched.includes("memory"), doing("wired-restore"),
            busy || offline ? "" : draftWhy || (launched.includes("memory") ? "Set by the command that started Origami" : !wired || !wired.supported ? "macOS does not report its memory ceiling here" : "macOS already allows this limit"));
          if (conflict || busy) hold($("acc-save"), true, applying === "acc-save", busy ? "" : "Settings changed elsewhere: review them first");
          $("reload-saved").hidden = !needsReload;
          // A saved draft has one action, Reload: the three that edit give way until there is an edit again.
          ["discard", "save", "apply"].forEach(id => { $(id).hidden = needsReload && !edits && !conflict && !applying; });
          if ($("save").hidden) $("save-why").textContent = "";
          hold($("reload-saved"), offline || busy || edits || conflict, false, draftWhy);
          const message = offline ? "Disconnected · values may be out of date" : review ? "Changed elsewhere · keep yours or take theirs" : conflict ? "Changed elsewhere · discard to load current settings"
            : edits && draftBase && revision && draftBase !== revision ? "Unsaved changes · based on older settings" : edits ? "Unsaved changes" : needsReload ? STATE.reload + " · settings saved" : liveState === "stopped" ? STATE.stopped + " · settings saved" : "Settings saved";
          $("save-state").textContent = message; $("save-state").dataset.pending = String(edits || conflict || needsReload);
          $("apply").textContent = liveState === "stopped" ? "Save & load" : "Save & reload";
          $("st-floor-label").hidden = $("st-mode").value !== "tradeoffs"; $("st-budget-label").hidden = $("st-mode").value === "memory";
          // The save bar shows only while there is something to save, reload or review; a model chosen and nothing else
          // has its own one button, Load model.
          const drafted = edits && !(Object.keys(changes()).join() === "model" && !Object.keys(samplingChanges()).length);
          $("savebar").hidden = !(drafted || conflict || applying === "apply" || applying === "save" || needsReload || reloading && ["apply", "save", "reload-saved"].includes(reloadBy));
          $("setup-dot").textContent = drafted || conflict ? "●" : "";
          $("review").hidden = !review;
          if (review) $("review-text").textContent = "Changed elsewhere since you started: " + review.changes.map(c => c.key.replace(/_/g, " ") + " " + memoryText(c.base) + " → " + memoryText(c.current)).join("; ") + ".";
          ["review-keep", "review-take"].forEach(id => hold($(id), offline || busy));
          $("model-list").querySelectorAll("button").forEach(b => hold(b, offline || busy || b.hasAttribute("data-blocked")));
          listWhy("model-list-why", $("model-list").querySelectorAll("button"));
          $("first").querySelectorAll("button").forEach(b => hold(b, offline || (b.dataset.first === "start" ? $("power").disabled : busy), false, b.dataset.first === "start" ? $("power").title : ""));
          let held = false;
          $("health").querySelectorAll("button[data-hact]").forEach(b => { hold(b, !idle, false, why); held = true; });
          $("health-why").textContent = held && why ? "The memory actions wait. " + why + "." : ""; $("health-reload").hidden = !(held && reloadFirst);
        }
        function chooseWorkspace(view) {
          workspace = view;
          ["run", "setup", "library", "settings"].forEach(v => $("view-" + v).hidden = v !== view);
          document.querySelectorAll("button[data-view]").forEach(b => { if (b.dataset.view === view) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current"); });
          if (view === "library") loadStorage();
          if (view === "settings") { loadAccess(); loadConnect(); }
          requestAnimationFrame(drawStrip);
        }
        $("x-chip").addEventListener("click", event => { event.preventDefault(); chooseWorkspace("library"); $("experts").scrollIntoView({ block: "start" }); });
        document.querySelectorAll("button[data-view]").forEach(b => b.addEventListener("click", () => chooseWorkspace(b.dataset.view)));
        $("save").addEventListener("click", () => apply(false));
        $("wired-restore").addEventListener("click", () => apply(true, true));
        $("discard").addEventListener("click", async () => {
          conflict = false; review = null; syncedRevision = null; rebase(); modelChoice = modelInitial; access = null; loadAccess();
          if (smSaved) { $("sm-custom").checked = smSaved.custom; $("sm-recommended").checked = !smSaved.custom; $("sm-override").checked = smSaved.override; smShow(smSaved.values); }
          syncSaved(); $("apply-note").textContent = "";
          if (status && ["ready", "busy"].includes(status.status)) await loadSetup(status);
          gate();
        });
        // Keep mine: the draft is now based on the current settings, and saved again; Take theirs is Discard.
        $("review-keep").addEventListener("click", () => { const again = review.reload; review = null; conflict = false; draftBase = revision; apply(again); });
        $("review-take").addEventListener("click", () => $("discard").click());
        $("restore").addEventListener("click", async () => {
          applying = "restore"; gate();
          try {
            const r = await api("v1/config/restore", { method: "POST", headers: revision ? { "If-Match": revision } : {} }), result = await r.json();
            if (!r.ok) throw Error(result.error && result.error.message || "No saved working settings");
            $("restore-note").textContent = "Restored · reload to use these settings"; syncedRevision = null; await refresh();
          } catch (error) { $("restore-note").textContent = error.message; }
          finally { applying = null; gate(); }
        });
        $("st-mode").addEventListener("change", () => {
          $("st-scope").textContent = $("st-mode").value === "tradeoffs" ? "Times each quality level. Speed only: answer quality is not scored. Nothing changes until you apply." : $("st-mode").value === "memory" ? "Reloads the model at a few memory limits (10 to 20 minutes) and suggests the highest this Mac keeps without cutting or swapping. Your setting is not changed." : "Tries timing settings against the current ones; keeps one only if it is faster with the same test replies.";
          gate();
        });
        $("st-result").addEventListener("click", async event => {
          const button = event.target.closest && event.target.closest("button[data-tune-apply]"); if (!button || button.disabled) return;
          applying = "st-go"; gate();
          try {
            const r = await post("v1/tuning/apply", { quality: button.dataset.tuneApply }), result = await r.json();
            if (!r.ok) throw Error(result.error && result.error.message || "Could not apply result");
            reloadStarted("st-go"); $("st-result").innerHTML = ""; await refresh();
          } catch (error) { $("st-note").textContent = error.message; }
          finally { applying = null; gate(); }
        });
        function finishStorageConfirm(accepted) {
          if (!storageConfirmResolve) return;
          const resolve = storageConfirmResolve; storageConfirmResolve = null;
          $("storage-confirm").close(); if (resolve) resolve(accepted);
        }
        $("storage-cancel").addEventListener("click", () => finishStorageConfirm(false));
        $("storage-accept").addEventListener("click", () => finishStorageConfirm(true));
        $("storage-confirm").addEventListener("close", () => finishStorageConfirm(false));
        $("storage-recovery").addEventListener("click", async event => {
          const button = event.target.closest && event.target.closest("button[data-recover]");
          if (button && !button.disabled && confirm("Verify the interrupted move and restore a consistent location?")) { await startJob({ kind: "recover", id: button.dataset.recover }, button, $("storage-note")); storageAt = 0; }
        });
        $("fs-go").addEventListener("click", () => { $("fs-search").value = ""; fsOpen($("fs-path").value.trim()); });
        $("fs-path").addEventListener("keydown", event => { if (event.key === "Enter") { event.preventDefault(); $("fs-search").value = ""; fsOpen($("fs-path").value.trim()); } });
        let fsSearchTimer;
        $("fs-search").addEventListener("input", () => { clearTimeout(fsSearchTimer); fsSearchTimer = setTimeout(() => { if (fsState && fsState.listing) fsOpen(fsState.listing.path); }, 200); });
        $("fs-more").addEventListener("click", () => { if (fsState && fsState.listing) fsOpen(fsState.listing.path, false, fsState.listing.next_offset || 0); });
        setInterval(() => {
          if (observed && Date.now() - observed > 12000 && !document.hidden) {
            offline = true; $("chip-text").textContent = "Reconnecting"; $("speed").textContent = "—"; gate();
          }
        }, 2000);
        window.addEventListener("beforeunload", event => { if (dirty()) { event.preventDefault(); event.returnValue = ""; } });

        layoutArt(); drawArt(null); refresh();
        """#
}
