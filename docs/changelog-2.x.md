# Origami 1.x and 2.x changelog

This is the history of Origami before 3.0, kept as it was written at the time, from 1.0.0 (24 October 2026) to
2.4.3 (6 November 2026). Version 3.0 started a fresh public repository from the 2.4.3 sources, so the commits behind
these entries are not in this repository's history. Several entries mention measurement reports ("see the release
measurements"); those reports lived in a development evidence folder that was not brought over, so they are named
but not linked. Numbers here were measured on one M2 Pro Mac with 32 GB. The current changelog is
[CHANGELOG.md](../CHANGELOG.md).

Each release says how its replies compare with the previous release's at temperature 0.

## 2.4.3 (6 November 2026)

- Tuning on Qwen, GLM and DeepSeek accepts ordinary background activity, stable Low Power Mode and modest
  memory-allowance recovery. Critical pressure still stops it; substantial condition changes discard the affected
  comparison and allow a bounded retry. The governor and saved memory limit remain active and unchanged.
- Quality comparisons use discarded warm-ups and two balanced rounds of 16-token replies. Results show sample
  counts and variation, and users can explicitly select a measured mode even when timings are noisy. Automatic
  performance tuning retains its confidence checks. Speed comparisons do not score answer quality.
- The dashboard distinguishes completed, incomplete and interrupted tuning, displays the actual reason, and no
  longer reports an interrupted comparison as successful. Session-wide request tracking detects other apps' traffic
  without confusing fast adjacent tuning replies. Final model and settings checks prevent stale proposals.

This release changes shared tuning, status accounting and dashboard behavior; inference arithmetic is unchanged.
Validation: the release build, host-condition policy checks, 181 dashboard checks, 19 job-subprocess checks and
10 model-free CLI scenarios passed. The installed DeepSeek model matched the prior four-token exact-output
reference. One real six-request quality comparison completed in 191 seconds, produced two samples per mode and
left the selected model, configuration and saved tuning unchanged. Qwen/GLM model runs were not repeated for this
shared control-flow fix; these short measurements are not a sustained performance or answer-quality assessment.

## 2.4.2 (5 November 2026)

- Qwen, GLM and DeepSeek reuse scratch storage after its last consumer, preserving each engine's arithmetic order.
  Captured diagnostics retain separate storage. Escaped Metal-buffer views keep their memory reservations alive.
- GLM and DeepSeek reclaim optional cache before growing mandatory scratch within the user's hard limit. After
  actual owner retirement, a bounded physical-footprint recheck handles delayed reclamation without weakening
  allocation admission. Family prefill revalidates fallback slots to prevent reuse of a slot already chosen.
- Family dashboard status uses snapshots published at safe boundaries. Experimental cache-admission hysteresis
  and bounded delayed-expert contributions are implemented across all three engines but remain off: grouped
  screens did not establish repeatable speed gains.

The final batch passed 52 selected checks (545 available), scoped exact-output checks for all three families,
and repeated short requests at configured 128K capacity. Qwen/GLM results retain their measured binary identities
with a documented source-impact review. Host paging remained; these checks do not establish swap-free operation,
full-length 128K prompt quality or a speed gain from the new experimental policies. See the
2.4.2 qualification for release status and evidence.

This release also includes the earlier single-request efficiency work:

- Qwen, GLM and DeepSeek share temporary recovery backoff after repeated memory-pressure reversals. Small isolated
  system-wide swap changes pause recovery; repeated changes or a substantial burst still trigger relief. The saved
  explicit memory target and physical ceiling remain unchanged. Qwen cache growth is paced, deferred trims complete
  at safe boundaries, and slab recycling avoids repeated metadata scans.
- All three engines reuse bounded, ledger-charged prompt preparation data. Converted template values use typed,
  byte-exact keys; tokenization reuse is restricted to added-token boundaries with compatible tokenizer processing.
  Full template rendering still sees every current message, tool schema and reasoning setting.
- Qwen checkpoint saves reuse immutable resident-page digests and copy payload only for changed pages. Prompt
  encoding overlaps the previous durable save, with the save barrier retained before model-state admission.
  GLM and DeepSeek checkpoint reads/writes avoid temporary chunk copies and bypass the filesystem data cache.
- Expert retention accounts for storage cost across all engines. Source scheduling uses observed read service and
  queued work; Qwen mirror reads can use either identical copy and hedge in either direction. Material routing
  changes can refresh same-sized hot copies at the appropriate setup/idle boundaries.
- Qwen gains explicitly admitted device-resident prefill for short cached suffixes, preserving actual row counts
  and existing segmentation, plus fused long-context selected attention through 262K using charged partial buffers.
  GLM and DeepSeek already keep short-prompt intermediates on the GPU; their distinct attention arithmetic is retained.
- Completed requests expose coarse `stage_seconds` for preparation, restore, prefill, generation and finalization.
  These are wall times, not GPU-only measurements; transport, image expansion and deferred cleanup are outside them.

The release `origami` product compiled and linked successfully on 5 November 2026, without compiler warnings.
Runtime validation and deployment were subsequently authorized with the normal Qwen model replacement.
Qualification comprises 501 effective engine-check passes and four optional skips, including three fixture-only
corrections after the complete run, plus successful native 4-bit Qwen instruction, arithmetic and tool-call checks.
The initial long-attention check passed at 65,537, 131,072 and 262,144 positions, but a later audit found that
both arms used the reference host path; it did not validate the fused kernel. A corrected paged, dispatch-asserting
check now passes at 65,536, 65,537, 131,072 and 262,144 positions. Five additional focused follow-up checks pass,
including one fixture-only correction for autoreleased Metal-buffer ownership. After macOS removable-volume consent
and a successful launchd access preflight, the follow-up build and normal checkpoint were deployed; the final
switch took 7.18 seconds and production traffic resumed. Four long-context production requests subsequently completed.
Fresh short GLM/DeepSeek runs completed; the DeepSeek output matches the adapted InferSSD baseline. System paging
qualifies timings. A subsequent coordinated Simulator-off window completed all six current-checkpoint arms;
Qwen/DeepSeek token IDs match native, while GLM differs. Qwen has faster decode but slower first-token and
short-request latency. See the coordinated comparison.
Sustained performance remains unverified. See resumed validation.
See the follow-up qualification for failures, recovery and limits.

## 2.4.1 (4 November 2026)

- An explicit memory limit is now the utilization target for Qwen, GLM and DeepSeek. Startup and cache growth use
  that target without another free-memory or learned-budget deduction. Automatic mode retains adaptive budgeting.
- Actual macOS memory pressure and fresh swapping temporarily reduce the expert allowance. Once the system is
  stable, the governor recovers toward the user's target. The limit includes transient framework memory; physical
  footprint overruns also release cache. Machine/GPU ceilings and allocation accounting still apply.
- Wired RAM shows the loaded limit, current allowance and process footprint separately, and explains temporary
  pressure relief and recovery.
- Cache recovery works within the current reply, including a reply that starts with a reduced allowance and
  file-backed Qwen context. Mandatory family state and cache allocations require successful reservations;
  replacement state retains its old buffers if admission or allocation fails.
- GLM and DeepSeek admit scratch-buffer growth before GPU encoding, account for live replacements and release
  completed command references promptly. Existing scratch capacity is reused without extra physical-memory probes.
- Qwen gains optional execution of cold experts as their reads finish, retaining the trained summation order.
  `qwen_arriving_first` defaults off; it is available for the same short tuning comparisons as the family paths.

Validation: 500 effective engine-check passes and four optional skips, composed from the full run and a fixture-only
correction with focused retries; 429 CLI checks passed. All three current 64-token outputs match the archived release
outputs. GLM and DeepSeek grouped-prefill regressions fit their 25.77 GB process caps. Short same-checkpoint external
comparisons and their paging, timing and numerical limits are recorded in the 2.4.1 measurements.
Optional overlap features remain off by default because the feature screens did not establish a reliable performance gain.

## 2.4.0 (3 November 2026)

- Wired RAM now coordinates Origami's saved budget with the macOS GPU wired-memory ceiling. Limits are validated
  against each Mac's RAM and OS headroom; raising the ceiling uses macOS administrator authorization. The dashboard
  distinguishes saved, active and system limits, model reload requirements, and restoration after reboot.
- Larger ASCII crane in the dashboard header, with a separate size for narrow screens.
- Correct the shared macOS available-memory estimate: speculative file-cache pages count once, rather than three
  times. This prevents a warm disk cache from inflating Qwen, GLM or DeepSeek's initial expert-cache allowance and
  provoking avoidable swapping. Invalid counter arithmetic fails closed (BITWISE).
- GLM and DeepSeek retain only an unfinished pooling group instead of full raw key/gate histories (BITWISE).
  At 128K allocated positions this removes about 1.48 GB / 1.61 GB of capacity respectively; file-backed capacity is
  not equivalent to resident RAM. Checkpoint carry remains recurrent state.
- Family engines release prompt scratch once before decode and lend only the extra prompt-slice reservation to expert
  caches. They reclaim it before another prompt and preserve explicit expert-memory caps (BITWISE in exact mode).
- Optional completion-driven expert execution for both family engines, with canonical summation and joined read/error
  handling. `family_arriving_first` is available to the existing tuner; it defaults off pending measurements (BITWISE).
- DeepSeek hashes Engram rows from bounded token history and can overlap their reads with earlier layers. The optional
  `family_engram_overlap` tuner arm defaults off; every started read joins before state or scratch reuse (BITWISE).
- Qwen's MTP verifier backs off repeatedly failed resident-only pair attempts and overlaps shared-expert work with
  the streamed expert union. Ordinary decode is unchanged; MTP remains opt-in and NUMERIC.
- A tokenizer-only benchmark export and sequential external-engine collectors support same-checkpoint comparisons;
  benchmark instrumentation is not installed in the serving path.
- In-memory checkpoints at the first tool-call marker reuse an unchanged reasoning prefix when clients rewrite arguments;
  bounded to one anchor, with no full-context write during decode (NUMERIC, like prompt reuse).
- Optional prefill-group tuning within existing allocations: Qwen read scope (NUMERIC, separate saved-cache epoch)
  and family slice groups (BITWISE), with prompt-speed scoring, output checks and a decode regression guard.
- Mixed-format family expert pools use observed reuse and read cost per byte at safe boundaries; profiling is amortized
  and small reallocations are ignored (BITWISE in exact mode).
- Dashboard reorganized into Run, Models & Storage and Settings, retaining the original palette and ASCII crane.
- Versioned shared state, preserved drafts, conflict detection, visible reload requirements and disabled unavailable actions.
- Per-model context, quality, reasoning, memory, expert width, sampling and hot-copy settings.
- Independent file inventory, move/delete previews, model unlisting/relinking and recoverable move journals.
- Host folder browser with pasted paths, search, paging and read-only destination checks.
- Reuse healthy hot copies across routing changes and model switches; retain disabled copies and separate family expert formats.
- Bounded automatic tuning with paired controls, real elapsed timings, review before applying, and an explicit quality tradeoff mode.

Validation includes engine and dashboard checks plus bounded runs of all three actual model families.
See the release measurements for matched-checkpoint comparisons,
output-parity results, failed attempts and limitations. Short samples do not establish long-context performance.
Optional overlap features remain off by default; MTP remains opt-in.

## 2.3.4 (2 November 2026)

**Replies:** Qwen's byte-identical to 2.3.3 in exact mode, except in a chat whose earlier tool calls have arguments
out of alphabetical order, which now render in the model's order (NUMERIC from that turn on). GLM-5.3-Flash and DeepSeek-V4.1-Flash keep their reasoning in the conversation history (their templates' `clear_thinking` / `drop_thinking` off), so later turns differ from 2.3.3 (NUMERIC by design).

### New
- **`origami tune`, and Tune / Re-check / Undo on the dashboard's speed readout**: finds this Mac's fastest timing-only
  settings on the running model, about 8 minutes (`--minutes 3…30`), replies unchanged. It compares, request by request
  in the one loaded model (`origami_tune`, a request field only this Mac's tuner may send, on exact requests), the
  current settings, a repeat of them as the noise control, and each candidate: Qwen's `gpu_keepalive` flipped; GLM's and
  DeepSeek's next-layer guesses flipped, a guess depth from their per-rank use, and spinning flipped. Three rotated
  rounds; a candidate is kept only if it writes faster by at least max(3%, twice the control's gap) in every round, reads
  at most 2% slower and replies identically, and the single winner is confirmed in two more rounds. It stops, changing
  nothing, on Low Power Mode, a serious thermal state (after 2 min), memory pressure or fresh swap, or a request from an
  app. Results go to `<data_dir>/tuning.json` per model and context, bound to this Mac (model, memory, GPU), the drives,
  the model and the tuning version (`SelfTuning.version`, not the build): another Mac, model, drive set or tuning version
  is not applied; a macOS update or 90 days ask for a re-check. At start your own setting beats a tuned one, which beats
  the default; a missing, broken or stale file only means the defaults. `/v1/status`, `origami report` and the
  dashboard ("Tuned on 1 Nov: writing +12% (GPU keep-alive off)") show it; each run is logged to
  `tuning-history.jsonl`. It warms up until the speed repeats within 5%, leaves out the stalls a Mac adds to a reply
  now and then (DeepSeek on an M2 Pro: about 8.5 s every 90 s), and holds a family model's hot copy builder while it
  runs.
- **A memory budget learned from this Mac's other apps**: the governor records, per 10 minutes, the budget that would
  have kept free memory at its floor, and every trim, in `<data_dir>/memory-history.json` (14 days, per Mac). With an
  hour or more of history the start budget, and every regrowth, is capped at the 10th percentile of those 10-minute
  figures (and with three trims or more at what four in five trims cut to), never above today's ceiling nor below the
  least the model runs with; `memory_limit` still wins. The dashboard shows "Memory learned from 3 days: 20.0 GB" with a
  Reset (`POST v1/memory/learned`). Replies are unchanged; only how many experts stay in memory.
- **DeepSeek-V4.1-Flash GGUFs with an Engram table per layer load** where they were refused: vcruz305's converter's
  (`blk.N.engram_embd` in a ggml type, as in mrfakename's GGUFs, whose headers also leave out the source-layer keys,
  now read from the tensors) and DwarfStar's I8 rows (E4M3 codes with an E8M0 scale per 32, decoded as DwarfStar's
  reader does). Checked against the published files' headers, not yet run with either model. Unsloth's single table
  reads as before (bitwise).

- **Qwen's expert cache grows back during a reply** once memory has been free for 20 s after a trim, instead of at the
  next request: the slabs are allocated and faulted (or wired) on a background thread and join the cache between two
  tokens; a new trim meanwhile releases them. Replies are bitwise the same; only how many experts stay in memory.

### Fixed
- **Qwen no longer fails a prompt of about 512 to 1,535 tokens now and then with "Insufficient memory for this
  request"** (seen in `origami tune` on a ~1,460-token document, the 8th such read of the run). Such a prompt's first
  scope reads its experts layer by layer on the GPU under an exact escrow with room for one batch's maps; a finished
  command's buffers retire when Metal lets go of the command, which can trail its completion by a moment, and the next
  batch then found the previous one's maps still charged. A charge now waits up to 2 s for buffers being released and
  refuses at once only what can never fit. Replies are bitwise the same. Model-free check:
  `device_moe_charge_waits_for_retirement`.
- **Qwen no longer fails a reply with `expert_slots_busy` when the expert drive is slow** (seen with the pack drive
  saturated by a download and a hash pass: the reply stopped at its third token and the client got a 500 after about
  170 s). Replies are bitwise the same; only which slot a read takes, and how long a demand waits, changed:
  - the ten cheapest eviction candidates a decode row prepares now skip records held for the GPU, as the two places that
    use them already did; the current layer (held while its own experts are requested) could fill all ten, so a miss
    found no victim while other records could be evicted, and every miss at that layer went to a transient slot that
    was never kept;
  - a demand waiting for a busy slot fails only after 30 s in which no read lands on either drive (a stuck cache or a
    hung read), counted in wall time (it counted only its sleeps, about 75 s of real time); a slow drive makes a reply
    slower, never fails it;
  - when the model drive's hedge read wins while the external read is still queued, the queued read is cancelled,
    which frees its cache slot and staging slot; a started read finishes as before;
  - the memory governor's log no longer hides budget changes behind its one line per 10 s: held changes print merged
    (`budget 24.1 GB -> 17.1 GB in 2 steps`).
  Model-free checks: `expert_reuse_demand_skips_gpu_held_victims`, `slot_wait_counts_only_idle_time`,
  `expert_replica_win_cancels_queued_primary`.
- **DeepSeek-V4.1 and GLM-5.3 decode faster on the GPU, bitwise the same.** Measured with the new model-free
  `engine-checks --bench-deepseek-decode <shard 1>` (one DeepSeek token with the GGUF's shapes and types, random
  weights, GPU time by kind of dispatch) on an M2 Pro:
  - the indexers' top-k (`index_select`) picks its threshold by a radix select and writes its rows from every thread,
    not one: 3.6 ms to 0.12 ms a DeepSeek indexer layer at 20K tokens, 29 ms less a token (DeepSeek's eight indexer
    layers), and faster in prompts too;
  - the indexers' scores (`index_scores`) sum eight heads at a time as independent sums (each head's terms in the
    same order): 1.2 ms to 0.2 ms a DeepSeek indexer layer at 20K tokens, about 8 ms less a token; DeepSeek's prompts,
    whose indexer dominated their attention, gain the same factor on that kernel (not timed with a model);
  - decode attention takes eight heads a thread, so each selected row is read once for eight heads, not once a head:
    DeepSeek's scores 0.31 to 0.05 ms and values 0.31 to 0.16 ms a layer, about 15 ms less a token; GLM's the same;
  - the hyper-connection mixes' Sinkhorn step (`hc_prepare`) runs sixteen threads a row instead of one: 43 to about
    14 µs a call, about 3.5 ms less a DeepSeek token (119 calls).
- **GLM-5.3 and DeepSeek-V4.1 keep the GPU busier while the host routes, bitwise the same.** Each MoE layer queues its
  shared expert behind the router before waiting for it, and decode commits the routed experts as soon as they are
  encoded, so the GPU runs them while the next layer is encoded. On the model-free DeepSeek bench (M2 Pro) the GPU's
  idle time fell from 8.2 to 0.9 ms a token with every expert in memory (wall 110.5 to 102.9 ms at 20K) and by about
  2.5-4 ms with simulated reads; not timed with a model.
- **GLM-5.3 and DeepSeek-V4.1 decode faster while experts are read from the drive, bitwise the same.** The serving
  thread now spins while a token's missing experts are read, for up to twice the time the reads should take, where it
  slept: with the whole process asleep in the drive, macOS ran the GPU's next work far slower. On the model-free DeepSeek
  bench with real uncached reads from an external SSD (M2 Pro), two to four experts read a layer: 333-450 ms a token to
  231-339 ms (GPU 185-208 to 101-104 ms) at 1K-20K tokens, for about one core busy during the reads. Not yet timed with
  a model; ORIGAMI_FAMILY_SPIN=0 or Low Power Mode restores the old wait.
- **GLM-5.3 and DeepSeek-V4.1 decode runs the experts already in memory while the missing ones are read, bitwise the
  same.** Each layer's resident experts are encoded and committed at once, the rest once read, and the weighted add
  still sums every expert in the same order. On the model-free DeepSeek bench with real uncached reads from an external
  SSD (M2 Pro), one, two and four of six experts read a layer: 192-211 to 144-145 ms a token, 232-234 to 205-209 and
  336-342 to 324-329 at 1K-20K tokens. Not yet timed with a model; prompts are unchanged.
- **GLM-5.3 and DeepSeek-V4.1 choose each prompt row's experts faster:** a partial top-k in place of a full sort of
  every expert's score (about 3.5 µs a row with the router's weights where it took 15–19 µs on an M2 Pro), and a
  prompt's rows split over the performance cores (under 1 µs a row): about 5 s less in a cold 7.4K-token DeepSeek
  prompt (estimated from the per-row cost, not timed). Decode reads the next layer's guessed GGUF records as three
  parts at once, as a miss's read does. Replies are bitwise the same.
- **Decode guess counters by rank:** reply notes carry `expert_prefetch_issued_by_rank` and
  `expert_prefetch_used_by_rank`, and `/v1/status` `expert_cache` the same as `prefetch_issued_by_rank` /
  `prefetch_used_by_rank`; `ORIGAMI_FAMILY_PREFETCH_DEPTH=N` (internal) reads only the best N guesses not in memory.
- **A conversation continues where the reply ended, on every family.** The next turn's prompt now renders the reply as
  the model generated it, so the engine reads only what is new instead of the whole reply again:
  - a tool call's arguments render in the order the model wrote them (the client's JSON string is read with its key
    order kept); before, the template saw them sorted, so a call written out of alphabetical order (`path` before
    `edits`) diverged at its first parameter and the next turn re-read the whole reply;
  - DeepSeek-V4.1's content before a `<｜DSML｜ calls>` block leaves out the `\n\n` the template writes there itself,
    which it rendered twice (the cause of a 202 s re-read of a 5,417-token reply, where continuing takes about 27 s);
  - GLM-5.3 and DeepSeek-V4.1 keep past reasoning in the history (`clear_thinking` / `drop_thinking` off), so a user's
    next message no longer rewrites the earlier replies (a GLM turn re-read 4,765 tokens in 230 s where 78 were new).
  Nested values (an array or object argument) still render through the template's own `tojson`, sorted and compact:
  where the model writes them otherwise, the next turn still re-reads from there.
  A key the client's string repeats keeps its first value, as before.
- **GLM-5.3 and DeepSeek-V4.1 save a conversation when a request leaves it** (a short side request such as a title,
  another chat, or a prompt that does not continue it): its state, reply included, takes the slot of its saved prompt
  end, so coming back to it restores the reply too instead of re-reading it; 0.3 to 0.7 s at 15 to 25K tokens. A
  regenerated or edited turn (which branches from that saved end) and `no_store` chats save nothing.

## 2.3.3 (1 November 2026)

**Replies:** byte-identical to 2.3.2 in exact mode.

### New
- **Stop and unload, and Start** (the dashboard's header, or `origami stop` and `origami start`): the model's process
  stops and all its memory goes back to macOS, while the server keeps answering the dashboard. A running request gets
  30 s to finish, then it is cancelled (the dashboard asks first, and says so). While stopped, every request gets a 503
  `server_unavailable` / `stopped`, "Origami is stopped; press Start on its dashboard (this server's address in
  a browser) or run origami start on its Mac", with `X-Should-Retry: false` so the OpenAI SDKs do not retry a stop that
  will not end by itself, and `/health` says `stopped`
  (`stopping` while it unloads). It is saved as `stopped: true`, so a restart of the server or the Mac stays stopped
  until Start, which loads the same model again with the saved settings. `POST /v1/stop` and `/v1/start` take the
  API key and only this server's own page or the CLI, as the dashboard's other changes do; every model and engine.
- **The dashboard while stopped**: Stop and unload turns while the model unloads, the chip says Unloading, then
  Stopped with how much memory came back; the setup greys out, the memory strip shows the freed part where Origami's
  was, and a notice offers Start beside the header's. On a phone the header keeps "Stop", and the hot copy chip its
  mark and one word.
- **Sampling per model, from the model's own recommendation**: a request that sends no temperature, top_p, top_k,
  min_p or presence_penalty gets what the model recommends, from its `generation_config.json` (Qwen MLX) or its GGUF's
  `general.sampling.*` (llama.cpp's keys), else Qwen's values as before. Qwen's model card splits thinking from
  not thinking, so there a single set from the files is the thinking one; a GGUF's set is for every mode. `sampling
  custom` with `temperature`, `top_p`, `top_k`, `min_p` and `presence_penalty` sets your own for every mode, and
  `sampling_override true` uses them even when an app sends its own. They are kept per model, like `expert_bits`.
  `/v1/status`, `/health` and `/v1/models` give the defaults per mode (`defaults.sampling`, `sampling_recommended`,
  each model's `origami.sampling` in the front's listing). Qwen's `generation_config.json` holds its thinking values,
  so Qwen's replies keep 2.3.2's sampling; a temperature-0 request is greedy and byte-identical as before.
- **The dashboard's Sampling group**: the model's recommendation with its values per mode, or Custom with plain dials
  (Creativity, Word pool, Shortlist, Minimum chance, Repetition brake, each with its technical name), "Use these even
  when an app sends its own", and Reset to the recommendation. Apply saves and reloads.
- **Other devices on your network can use Origami** (`listen lan`, or Access in the dashboard; applies at the next
  start, which the dashboard does for a launchd service). The front then listens on every interface and answers this
  Mac's own `<name>.local`, its bare name and its addresses as a `Host` (no other `.local` name, so DNS rebinding
  still fails); model servers stay on 127.0.0.1. Other devices need a key: `api_key` when set, else `lan_key`, made
  when `listen lan` is set, so programs on this Mac keep working without one. `lan_key off` opens the network without
  a key (less safe). `GET /v1/lan` gives the addresses, and to this Mac only the key and a QR code.
- **The dashboard's Access section** switches between this Mac only and its network, the key for other devices on or
  off, and restarts Origami to apply it when launchd runs it (otherwise it says to restart `origami serve`). On the
  network it shows the address with the key and a QR code to scan with an iPad or a phone, on this Mac only.
- **A page without its key asks for it**: the 401 state has a "Paste the key" field (no command to run on an iPad),
  and the key, pasted or brought by `#key=`, is kept in that browser rather than the tab.
- **The dashboard installs as a home-screen app**: Add to Home Screen in Safari (or Install in Chrome) gives an
  Origami app with the crane icon that opens full screen, instead of a bookmark; the page keeps clear of the status
  bar, the notch and the home indicator there. Origami serves a web app manifest and PNG icons (180, 192, 512) without
  the key.
- **The folder browser and the saved-prompt viewer open on iPadOS 15.0 to 15.3** (Safari without `<dialog>`): as a
  fixed overlay that closes with Escape or Close; before, they showed at the bottom of the page and did not open. The
  rest of the page needs nothing newer than Safari 15.0.
- **Changes from the dashboard work behind a proxy or a Tailscale https name**: the front and the model's server take
  one rule for writes (no `Origin`, or `http(s)://` plus the request's `Host`), where the front accepted only a
  loopback `Origin` and the model's server only `http://`.
- **GLM and DeepSeek build their hot copy while serving.** No restart and no `origami setup`: after the first reply,
  Origami copies the most-used experts to the second drive in the background, pausing while a reply runs, and attaches
  the copy between replies; it grows as replies use other experts, and a new drive or size from the dashboard applies at
  once. The start no longer waits for a copy. `/v1/status` gains `experts_from` (each place the experts are read from,
  its state and the share of reads it served) and `events` for these models; `POST /v1/hot-copy` builds, moves or
  resizes the copy (for Qwen it runs `origami setup` as a job and reloads).
- **Settings that do not apply say so.** An `expert_bits` a model does not take is refused by the dashboard, its jobs
  and `origami config set` (when the model can be read), and ignored with a note at start (DeepSeek's IQ1_S has no
  smaller width); a named model's width is kept in its own entry. A family hot copy on the model's own drive is refused
  (it gains nothing). Saving `stores` drops the data folder's own store, which is always used.

- **The dashboard's Experts card** replaces the Drives section: one row per place the running model's experts are
  read from, with its role, size, live state (the hot copy learning, building with its progress, paused while a reply
  runs, ready) and share of reads, a line splitting the reads between memory, the hot copy and the model's drive, and
  **Change…** (a drive and a size, no folder to pick) and **Build now**. Storage and Folders move into one closed
  "Storage and folders" section: the running model's items first, another model's under their own heading, support
  folders named after their model, and no store or hot-copy buttons per drive; the expert folders show only for a
  model with a smaller width, and never the data folder's own. The Expert width dial shows only where there is a
  choice of width.

- **Qwen's hot copy and copies go on a drive from the dashboard again** (2.3.2 had it under Folders): its Experts
  card rows get the same **Change…** (a drive and a size) as DeepSeek's, show where the copy will go before it is built
  (not the model's folder), and **Copy to <drive>** writes all experts, or a further copy beside them, on a drive with
  room (`POST /v1/hot-copy` `{"copy": …}`, then `origami setup` as a job). A converted or packed copy of all experts
  (Qwen's 4-bit, 3- or 2-bit, GLM's) gets **Move N-bit experts…**: a drive, then a move job to that drive's store.
- **Is the hot copy faster?** With three recent replies or more on each side, the Experts card compares their median
  writing speed with and without the hot copy; each request's record says whether the copy served reads to it. Qwen's
  read split now credits the hot copy, the 4-bit copy and the model's folder each with their own reads.
- **The hot copy's state in the header**: a small chip (learning, building with its percentage, paused, ready) next to
  the status chip, linking to the Experts card.
- **Leftover settings show with Clear**: an expert width the running model does not take (`expert_bits 4` on
  DeepSeek) and expert folders it never writes that hold nothing (one holding another model's pack stays) are named in
  the Experts card, and one click removes them from the settings file.

### Fixed
- **Re-walk 3 fixes on the dashboard and the CLI.** Sampling's heading names the running model; its source line says
  what the model's files name and that the rest are Origami's general defaults; the table speaks the dials' words
  ("20 words", "Off"); the dials say they start from the Not thinking column and Custom that it replaces all three;
  with Custom each slider marks the model's own value. "Saved. Reloading" becomes "Reloaded with the new sampling"
  once it is back, and the reload notes go when Origami stops. Stopped, the speed readout says Stopped with the speed
  last measured, and the Experts card hides. Recent requests' times show the clock time on hover, as Events does. On a
  touch screen a tapped button no longer keeps its hover fill (a tapped Start looked filled as Stop). `origami stop`
  and `origami start` point to `origami doctor`, which asks the server, and their `--help` lists only `--port` and
  `--config`.
- **A request macOS interrupts on the GPU is read again instead of failing.** When the display or another app needs the
  GPU, macOS can stop a command buffer (Impacting Interactivity, a GPU timeout, or a victim of another app's GPU
  recovery); a long prompt read then failed with a 500 "Inference request failed" that agents such as Pi did not retry.
  Every engine now tags it `gpu_interrupted` and reads the request again from its last saved state (a saved prompt, or
  the start), at most twice more, replaying the reply tokens already sent: the reply is the uninterrupted one, bit for
  bit, and the client gets no text twice. The server log and the dashboard's Events say "macOS needed the GPU; reading
  again from …, retry 1 of 2 (the step that failed)", and Recent requests marks the request "retried after macOS
  needed the GPU" (`gpu_retries` in its row). If it fails a third time the client gets a 503 (`server_busy`,
  `gpu_interrupted`, "… please retry your request"; in an open stream, an error event), and Recent requests shows
  "interrupted by macOS (GPU busy)", on a phone or an iPad too, under the prompt.
- **DeepSeek and GLM keep their GPU command buffers short at any context**, so macOS has less reason to stop one. A
  long prompt's indexer work grew with the context and was not counted toward a buffer's limit: DeepSeek's indexer
  layer (layer 20) ran one buffer of 0.5 s at 13K tokens and 0.9 s at 21K (it ran its candidate blocks as 256
  one-row dispatches), GLM's DSA layers 0.14 s at 7K. The indexer's scoring and selection now run in slices of rows,
  each counted and committed before a buffer passes about 25 ms of work, and the candidate blocks in one dispatch.
  The replies are bit for bit the same.
- **A DeepSeek prompt slice macOS stopped on the GPU no longer goes unnoticed.** A slice that only feeds the next
  layers' state read no logits, so a stopped buffer at its end was never checked, and since the retry above the next
  slice no longer saw it either: the reply went on from half-written rows. Every waited family buffer now checks each
  buffer committed before it by its own status, and a fed slice waits for its last one, so it is retried like any
  other. A stop while a model is starting answers "Origami is stopped" rather than "did not start", and one more Qwen
  decode wait names an interruption as one.
- **With `api_key` set, the front checks it before what it does itself**: a program on this Mac without the key could
  save settings, start or cancel jobs, make folders, change the hot copy, reload and restart Origami (only the reply
  was refused).
- **`listen lan`: a request from this Mac that names its network name or address is another device's** and needs the
  key, so a proxy forwarding the network to 127.0.0.1, or that name rebound to 127.0.0.1, never gets in without it or
  sees the key; and `origami serve` refuses to start beside another server on 127.0.0.1 at its port (the network socket
  bound beside it, and this Mac's apps kept talking to the other one).
- The memory dial shows a saved limit as saved (25.8 GB, not 24 GB) until it is moved.
- The model menu's size of a split GGUF counts every part (DeepSeek showed 0 GB); a missing .gguf is named after its
  file and says `The model file is missing at <path>.`
- GLM and DeepSeek hot copy: a record it cannot read (its drive pulled) is read from the model's drive instead of
  failing the reply, and a drive gone is no longer read, its copy kept and read again when it is back; a copy that filled
  its drive is no longer removed at each start; a build stops if the drive falls under 10 GB free; a failed build waits
  for Build now or a new place instead of starting again after every reply; a pause names the warm-up too.
- `origami config set|unset expert_bits` for a named model changes its entry's width (the file's key was ignored).
- The hot copy's **Change…** lists each drive once: a hot copy folder set by hand no longer adds an "Automatic" row for
  the same drive (which rebuilt 40 GB elsewhere on it); the drive in use shows the copy's own folder and is marked, and
  choosing it again changes nothing.
- **Build now** before the first reply says "after the first reply" beside it, not only in a tooltip touch never shows.
- Wording: the internal drive is "This Mac" everywhere (the conversion block and the folders said "this Mac's drive");
  the model's own folder reads "all experts, as released" and a converted or packed copy "all experts, N-bit"; the
  storage map names a hot copy once; the expert folders show only while a model that writes there runs (also Qwen's
  one-width builds), with the same words as the card.
- Phone: the read shares keep a "reads" label, the role and model lines wrap instead of being clipped, and paths wrap
  after their slashes; a missing model set by its path is listed once in the model menu (by its registered name).
- A Qwen hot copy or copy job ends with "Done. Origami reloads the model to use it", not "Apply and reload to use the
  smaller experts" (there was nothing to apply; Origami reloads by itself), and the card's "Saved. Copying the
  experts…" note goes when the job ends.
- **Copy all experts, 4-bit, to <drive>** names what it writes and asks first, with its size and about how long it
  takes; it is offered only on a drive with room for the whole copy plus the 10 GB setup keeps free (setup skipped a
  smaller one and the job ended "done"), and not beside 3- or 2-bit experts, which are read alone.
- A conversion's percent and time left also show under the turning **Convert now**, where a phone has no header text;
  the header's job text ends in the same poll as the job band, not two seconds later.
- A page waiting for its key shows the key box alone, not an empty Model menu and dials behind it.
- **Access on another device** shows the key it got in with, with Copy, to paste once into the home-screen app (which
  keeps its own storage), under "This device"; "This Mac only" says that device would lose its connection before Save;
  a Tailscale address among the "Or" addresses is named.
- The Experts card's header says "released as Q8_0", so it no longer reads as the width in use beside a 4-bit set.
- The web app manifest lists its icons once as `any` and once as `maskable` (one `any maskable` entry made Chrome draw
  the padded icon small).
- `origami help` names the model families it runs, and `serve` mentions `listen lan`.
- A server stopping while a hot copy builds waits briefly for the build to remove its partial `experts.next` (the next
  build in that folder overwrites one left anyway), and while a new copy is written on another drive the old one, still
  read, is not offered for removal.

## 2.3.2 (31 October 2026)

**Replies:** byte-identical to 2.3.1.

### Fixed
- **Folders with a space** (such as "Application Support") opened from the dashboard's folder browser, model check or
  plan preview: the page sends a space as + and Origami read it literally ("No folder at …/Application+Support/…").

## 2.3.1 (31 October 2026)

**Replies:** byte-identical to 2.3.0 in exact mode.

### New
- **Storage on the dashboard: where Origami's files are, and moving them.** A Storage section shows each drive as a
  bar (models, expert packs, hot copies, saved prompts, other files, free) and every item Origami keeps there, for
  every model and engine: what the running model uses, and leftovers nothing reads (copies whose model is gone or in a
  folder no longer used, interrupted builds and moves). A model, a pack or the saved prompts move to any folder: copied,
  read back and compared by SHA-256, recorded, and only then deleted at the old place (a rename on the same drive; a
  model in use stops and starts again); a hot copy is rebuilt where you set its folder; a leftover is removed after a
  confirmation; nothing a model reads, nor a folder something in use was moved into, is offered for removal, and a
  folder holding a store, the saved prompts or another model does not move. `GET /v1/storage`; moves and removals are jobs with
  progress and Cancel.
- **Tools work with GLM-5.3-Flash and DeepSeek-V4.1-Flash.** Their tool calls (GLM's `<tool_call>NAME<arg_key>…`,
  DeepSeek's `<｜DSML｜ calls>` invokes) reached apps as plain text; the server now reads each family's own syntax into
  the same tool calls Qwen's give, streamed and whole, with `finish_reason: tool_calls`, on every API (OpenAI chat and
  Responses, Anthropic, Ollama), arguments typed as JSON (strings stay strings). Qwen's calls are unchanged.
- **The model menu is reliable.** The server keeps the one list of added and found models and checks each in the
  background, so a sleeping USB drive never slows a listing or makes models vanish: each entry shows its state
  (checking, drive slow to answer, drive not connected, needs permission to read the drive, not found), a drive macOS
  refuses is named with how to allow it instead of the model being called missing, the model search answers only when
  complete and keeps what it found on a drive that does not answer, and the page asks again while a check runs.
- **Apps can discover the thinking levels.** Every `/v1/models` entry (and the new `/v1/models/<id>`) lists
  `reasoning_efforts`, OpenRouter-style `supported_parameters`, `context_length` and Ollama-style `capabilities`
  (`thinking`, `tools`, `vision` where the model reads images), also in `/api/tags` and `/api/show`, for every
  model; `include_reasoning: false` hides the thinking (and status lines), `true` turns it on unless another field sets it. The dashboard's **Connect an app** gives the base URL, the
  model id and a ready Pi `models.json` block mapping every Pi thinking level, with copy buttons.
- **Saved prompts open on the dashboard.** Every saved prompt is listed; opening one shows its text (its tokens
  decoded), kind, tokens, size on disk and last use, and lets you rename it, keep or release it, delete it after a
  confirmation, or edit a system prompt's text into a new saved one that replaces it (a save that holds only the
  prompt's start says so). For every model, through
  `GET`/`PATCH`/`DELETE /v1/prompts/<id>` and `POST /v1/prompts`.
- **The dashboard shows how much of a prompt came from a saved one.** While a prompt is read, the strip gives the
  reused share next to its progress, keyed to the bar's grey part (and in the bar's screen-reader value); recent
  requests show each one's share beside its reused tokens. GLM-5.3-Flash and DeepSeek-V4.1-Flash now report their
  prompt progress too (the reused part, then each slice read), so their bar and stream `prompt_progress` frames move.
- **A finished job no longer picks the model on each page load.** The dashboard chose the model of the last "add a
  model" job every time it opened (`v1/jobs` keeps that job until dismissed), so the next Apply of any dial also
  switched the model. A job's end now acts only on the page that saw it run (jobs report a `started` identity), Apply's
  summary names a model switch in full, and a model an add job chose goes with other dial moves only after a confirm.
- **Choose folders on the dashboard.** One in-page folder browser (drives with free space, home, Origami's folders,
  New folder) sets the model to add, where a download or conversion goes (`model add --to`), which store a smaller-
  expert conversion is built in (`setup --store`, tried first), the stores, the hot copy's folder (new `hot_copy_dir`,
  for Qwen's mirror and the GLM/DeepSeek hot copy), the saved prompts' folder and more model-search folders (new
  `models_dirs`). The front serves it read-only (`GET /v1/fs`, names only, inside those places) plus `POST
  /v1/fs/mkdir`, guarded like `/v1/config`; macOS permission refusals say how to allow access. Replies are unchanged.
- **The dashboard shows what Origami is doing, without words.** Whatever works (a model loading or starting with a
  new setup, Apply, Start with this model, a conversion or download, a check, Measure, a folder or saved-prompt action)
  turns a wheel on its own button and on the status chip (a still mark with reduced motion), and takes no second
  press until it ends; controls that cannot act now (dials and the model menu while a model loads, the whole page
  while Origami does not answer, Convert and Add while a job runs) are greyed out with `aria-disabled`; a control that
  failed is marked in orange, and the one that needs you is outlined. A reload now ends when the new model answers,
  not at the old one's last ready answer, and a failed start no longer leaves the page saying it reloads.
- **The dashboard has an icon.** A white paper crane on black (one small SVG) marks its browser tab; the page carries
  it inline, and the front and each model's server answer `/favicon.svg` and `/favicon.ico` with it, without the API key.
- **Tool calls decode exact at every quality level.** `dynamic`, `faster` and `fastest` could misspell a call's name
  or arguments; once a reply opens a tool call (Qwen's and GLM's `<tool_call>`, DeepSeek's DSML marker, one token
  compare a step), the rest of that reply decodes exact, on Qwen and the family engines alike. Prose before the call
  keeps the level's speed. The request log gives the first exact token (`exact_from`) and the dashboard tags the row.
- **The dashboard's Check counts what a conversion writes.** Adding a GLM-5.3-Flash checkpoint (or a Qwen GGUF) said
  it needed only the 10 GB kept free and that it fit; the check now counts the converted model's size as `model add`
  does (one computation, `Models.conversionBytes`), for GLM's GGUF and Qwen's 4-bit layout alike, and refuses a drive
  without room for it; a Hugging Face checkpoint that converts needs room for the download and the conversion both.

## 2.3.0 (31 October 2026)

**Replies:** Qwen's are byte-identical to 2.2.2 (its engine changes only which experts stay in memory and which
unread rows run). GLM-5.3-Flash and DeepSeek-V4.1-Flash replies are NUMERIC: a prompt long enough to save is read in
slices that end where a state is saved, and a long DeepSeek prompt's early rows skip its late layers.

### New
- **Qwen's prompt reading skips the last layer's unread rows.** Layer 47's routed experts reach only the row whose
  logits start the reply (and the rows MTP drafts from), so a prompt scope now runs only those rows' experts there,
  each over all its rows as before, and the expert read-ahead stops at layer 47: about 2% fewer cold expert reads on
  long prompts. The counters `lastLayerRowSkips` and `lastLayerSkipReplays` (expected 0) report it. Replies are the same.
- **Qwen wires its resident weights too.** On Macs that wire the decode slabs (`wire_decode_slabs`), the about 3 GB of
  dense weights every token reads are now wired (mlock) as well, so another app's memory pressure can no longer
  compress them and stall each GPU command on decompression. Replies are the same.
- **Qwen's mirror holds the experts that reach the drive.** Once a model's own routing is counted, `origami setup` now
  ranks the hot-expert mirror (and the copies on further drives) by how often each expert had to be read, then by use,
  instead of by use alone, which mostly copied experts already in memory. Replies are the same.
- **Prompt reading keeps what decode uses.** Qwen's prefill now evicts the records it has already passed before
  records still ahead or the ones decode was using, so a follow-up prompt reads fewer experts and the reply after it
  starts with its experts still in memory. Replies are the same.
- **`gpu_keepalive` setting** (true, false or auto; `--gpu-keepalive`): Qwen's GPU keep-alive spin can be turned off
  on a Mac where it slows replies; auto keeps it on for now. Replies are the same either way.
- **One feature layer for every model.** GLM-5.3-Flash and DeepSeek-V4.1-Flash get what Qwen had: saved prompts (the
  end of each system prompt and each whole prompt, in the same prompt cache, so a new chat, another conversation in
  between or a restart starts from there instead of reading it all again; `origami save`, `forget`, `prompts`,
  `X-Origami-Keep` and the dashboard's list), the dashboard's memory plan (the memory strip, the fits line, the memory
  dial, context up to the model's own maximum), `memory_limit` (it sizes their expert slots), `/v1/status` and
  `/v1/plan` with Qwen's fields, chat commands, and an automatic hot copy (of their most-missed experts on the data
  folder's drive when the GGUF is on another one; kept while it still holds most of what replies miss).
- **Smaller experts for every model that stores bigger ones.** A GLM or DeepSeek GGUF whose routed experts are in a
  bigger type (Q8_0, Q6_K, Q5_K, Q4_K, BF16…) converts them once to Q4_K or Q2_K (`expert_bits` 4 or 2; 8 for Q8_0):
  `origami setup` or the dashboard's Convert now builds the pack in a store with room, the dial lists only the widths
  the file converts to. An MLX Qwen checkpoint at 8 or 2 bits, or at 4 bits with another group size, converts once
  to Origami's 4-bit layout (`origami model add`; 3-, 5- and 6-bit MLX packing is refused by name for now).
- **Reasoning from the dashboard.** A Reasoning dial (none, low, medium, high, max) sets the server's default
  `reasoning_effort` for every model; a request that asks for its own level (`reasoning_effort`, `reasoning.effort`,
  Anthropic's `thinking`, Ollama's `think`, a thinking budget) still gets it.
- **Context on disk for GLM-5.3 and DeepSeek-V4.1**, for Macs with less memory: their per-position state goes to
  files on the context drive when it would take more than a quarter of what Origami may use (`context_on_disk`
  true, false or auto), paged by macOS like any file; replies are bitwise the same.
- **GLM-5.3 and DeepSeek-V4.1 give memory back when the Mac needs it**, as Qwen does: memory pressure, swap or free
  memory under the floor releases (and unwires) expert slots, so other apps no longer swap behind Origami's wired
  pool, and 20 s of free memory grow them back at the next layer, even mid-reply; a Mac busy at start starts with fewer slots
  instead of being refused or kept small. Replies are bitwise the same.

### Faster
- **GLM and DeepSeek prompts skip the last layer's unused experts**: only the head reads the last layer's routed
  experts, and only for the prompt's last row, so a prompt pass now reads just the experts that row picks there,
  about 2% fewer expert reads; replies are bitwise the same.
- **GLM and DeepSeek start warm**: after each start, while no request has come, the empty expert slots are filled
  with the experts earlier replies used most (`uses.json`), leaving two prompt waves' slots empty and evicting nothing,
  so the first reply after a model switch or restart no longer misses nearly every expert; the first request stops it
  after the batch being read. Replies are bitwise the same.
- **GLM and DeepSeek prompts read each layer's experts once per 2 to 4 slices**: a prompt runs up to four 2,048-row
  slices in one forward pass (as many as 1/32 of the Mac's memory, at most 2.5 GB, holds in working buffers: three on
  a 32 GB Mac for GLM), and every slice used nearly every expert, so long prompts read about that many times fewer
  expert bytes; each expert still runs once per slice, so replies are bitwise the same.
- **Less host work per GLM and DeepSeek layer**: a prompt's experts write their rows into one reused buffer instead of
  two new GPU buffers each, choosing the expert slot to reuse needs no exp2 per slot (the same slot as before), and the
  resident experts of a layer are kept instead of rebuilt from every slot; replies are bitwise the same.
- **A GLM and DeepSeek hot copy that serves the misses, read from both drives at once.** It holds the experts replies
  miss most (counted in `misses.json`, falling back to the most used), up to half the routed experts (at most 40 GB,
  leaving 30 GB free, at least 20 GB) when automatic, and each token's misses it holds are split between the copy's
  drive and the model's by which would finish first at their measured read rates; replies are bitwise the same.
- **GLM and DeepSeek prompts read experts while the GPU runs them**: each wave of a prompt slice's experts is read on
  another thread while the wave before is encoded and runs, into slots neither wave holds; replies are bitwise the same.
- **DeepSeek decode reads the next layer's experts early**: each layer also runs the next layer's router on
  its own input and starts reading the experts it picks while the GPU runs this layer, into slots no running work
  holds; a guess the next layer does not ask for is cancelled or the first to be evicted; 17-29% faster DeepSeek decode on an M2 Pro with one external SSD. GLM-5.3 keeps it off:
  with fewer hits a third of its guesses go unused and the wasted reads slowed its decode after long prompts
  (`ORIGAMI_FAMILY_PREFETCH=1` or `0` overrides either default). Replies are bitwise the same.
- **The decode guess uses the next layer's own mix and norm**: the guess router
  reads the next layer's hyper-connection mix and `ffn_norm` of this layer's post-attention streams (about five
  one-row dispatches) instead of this layer's own input, closer to what the next router sees; replies are bitwise the same.
- **GLM and DeepSeek long prompts read the next layer's experts during this layer's last GPU work, and keep what
  decode and the next slices need**: a prompt call of 1,024 rows or more starts reading the next layer's first wave
  of experts once only its own last wave runs, so the drive no longer idles through the next attention; prompt reads
  now reuse the slot of the prompt record passed most recently before any of decode's, so a follow-up prompt no
  longer flushes the experts the reply was using and later super-slices find the start of the sweep still resident.
  Replies are bitwise the same.
- **GLM and DeepSeek lend the unused context to the expert slots**: the per-position state now holds what a request
  can reach (4,096 positions for an empty conversation, the prompt and up to 1,024 reply tokens, doubling as a reply
  grows), and the rest of the context's state goes to the expert slots, given back before the conversation needs it.
  At a 32K context on a 32 GB Mac about 1 GB (GLM) more experts stay in memory for a short chat, at 128K about 4.5 GB;
  `expert_memory` and context on disk lend nothing. Replies are bitwise the same.
- **Long DeepSeek prompts skip the late layers for their early rows**: layers 21 to 39 read earlier rows only through
  layer 20's compressed rows and their own 128-row windows, so a prompt slice that ends more than 2,414 rows before
  the prompt's end (or a saved prompt's end) runs only layers 0 to 19 and layer 20's window and compressed rows; long
  prompts read about half the experts per early slice. Replies are NUMERIC (prompts shorter than 4,462 rows past a
  saved point are unchanged, bitwise).
- **Qwen lends the unused key/value room to the expert cache.** The in-memory context reserves its whole key/value
  arena (3.3 GB at 128K, 6.6 GB at 256K, 0.8 GB at 32K), but a conversation fills it only as it grows: until it does,
  decode keeps that many more experts in memory (about 1,200 more at 128K). Before each prompt is read the cache gives
  back what the conversation and its `max_tokens` can reach; replies are bitwise the same.
- **Faster expert-slot choice for Qwen.** A full cache no longer scans every slot for a free one on each miss, and a
  pre-routed read finds its victim from per-layer minima instead of two scans of every slot (about 3.5 ms of host
  time per token at 5,500 slots, more on 64-96 GB Macs). The same slots are chosen, so replies are the same.

## 2.2.2 (30 October 2026)

**Replies:** byte-identical to 2.2.1 (the dashboard and the GLM/DeepSeek status changed).

### Fixed
- **Dashboard with GLM-5.3-Flash or DeepSeek-V4.1-Flash**: every dial was greyed out ("set from Terminal") and the
  speed read "Not measured yet" after measuring. Their engine has no plan previews; it now reports its settings, model
  and Mac in `/v1/status`, and the page sets context and quality from them (memory and expert width say why they are
  fixed). The measured speed counts replies at the server's own quality level, not exact alone.

## 2.2.1 (30 October 2026)

**Replies:** byte-identical to 2.2.0 (only the dashboard's model search changed).

### Fixed
- **Found models listed once**: on a drive that ignores letter case `~/models` is `~/Models`, so the running model came
  back as "found" under the other spelling. Folders now compare by file identity.

## 2.2.0 (30 October 2026)

**Replies:** unchanged from 2.1.1 (the server front and the dashboard changed, not the engine).

### New
- **The dashboard fixes what is in the way.** A drive macOS has not let Origami read (the model's, or a store with its
  copy of the experts) gets a note naming it, with the steps to allow it and Check again; a model that did not start
  says why, from the model process's own error. `origami serve` stays up when its model cannot start, so the page can
  fix it, and does not start a model on a drive that does not answer (it would wait on an invisible prompt).
- **Conversions from the dashboard.** Choosing 3- or 2-bit experts shows the one-time conversion it needs, its size,
  which drive has room (10 GB to spare) and Convert now, which runs `origami setup` in the background with its
  progress, time left and Cancel; Apply waits until the pack is ready.
- **Any model from the dashboard.** The model menu lists the models Origami found on this Mac (Hugging Face's and
  LM Studio's downloads, `~/Models`, the Models folder of a drive it reads), judged from their headers, and Add a
  model takes a Hugging Face repo id or a folder or `.gguf` path: it says whether the model runs as it is, converts
  once or downloads, and whether there is room, then adds it with `origami model add` in the background. With no model
  yet the page shows only the menu and Start with this model.
- Commands print whole lines into a pipe (`origami setup | tee` shows progress as it comes).

## 2.1.1 (30 October 2026)

**Replies:** byte-identical to 2.1.0 (only the dashboard page changed).

### Fixed
- **Dashboard with one model**: the empty model menu stayed on screen next to the model's name (a hidden element
  kept its own display rule); hidden parts of the page now stay hidden. The live line says "No timed reply yet",
  not "No requests yet", when the only replies were too short to time.

## 2.1.0 (30 October 2026)

**Replies:** in exact mode, Qwen's replies at temperature 0 are meant to stay byte-identical to 2.0.5 (the eight
ballpark cases are the check); the quality dial's other levels and the new families are NUMERIC.

### New
- **Quality dial**: one setting from `exact` (the default, the model's own replies) through `faster` to a
  deliberately aggressive `fastest`, per request (`"origami_mode"`, the `-<level>` model ids), per chat
  (`origami quality <level>`) or for every request (`origami config set quality <level>`). Its four knobs are settings
  of their own: `experts_per_token`, `tail_skip`, `tail_skip_misses_only` and `late_miss` (`wait`, `skip`,
  `substitute`). On the 32 GB M2 Pro without a pack: `faster` decodes 1.7× as fast as exact and picks exact's next
  token 94.8% of the time, with every ballpark answer right; `fastest` 2.1× and 90.0%, slipping on one strict
  instruction. `fast` (the `-fast` model id, `"origami_mode": "fast"`, `origami fast on`, a stored `quality fast`) now
  means `faster`; 1.x's fast mode (1.3× and 91.6% here) stays reachable as the knob `late_miss skip`. Prompts are
  always read exactly; the dashboard's trade-off card, `/v1/status` and `/v1/plan` show the levels and their cost.
  `/v1/models` and `/api/tags` list a `-<level>` id per level when non-exact levels are allowed, and `/health` and
  each model entry list the levels (`quality_levels`). Two request fields measure a level against exact:
  `origami_return_tokens` puts the reply's token IDs in its usage, and `origami_teacher_tokens` forces the reply along
  given raw token IDs and reports top-1 agreement; a teacher-forced request is never saved.
- **Fewer GPU dispatches for GLM-5.3 and DeepSeek-V4.1**: decode runs a layer's routed experts in four dispatches
  instead of six per expert (on GLM about 2,200 small dispatches and 720 buffer allocations a token become 180),
  GLM's 64 per-head products and DeepSeek's grouped output projection in one each, and the shared expert while the
  routed ones are read. Replies are byte-identical. On the 32 GB M2 Pro, where GLM decode waits for the drive most of
  the time, the GPU wait per token fell about 10% and decode moved within run-to-run noise (about +5%).
- **The dashboard is the setup tool**: a first-time user picks the model and moves four dials (context, quality,
  memory, expert width); each moved dial shows its effect on this Mac, from much slower to much faster, judged from
  its own miss rate, memory and context tier (no predicted numbers: the speed shown is this Mac's, from its last
  replies or a one-reply Measure), with this Mac's memory drawn as an ASCII strip. Apply saves the setup and Origami
  reloads itself (`POST v1/reload` on the serve front), no terminal needed; a dial given when Origami was started (a
  flag such as `--context 8k`, or an `ORIGAMI_*` variable) stays put and says so, since it would win again after the
  reload. Live state, recent requests, saved prompts, drives and events stay on the page.
- **One model, `origami`, the setup chosen in the dashboard**: `origami serve` lists `origami` at `/v1/models` and
  `/api/tags`, and every request gets that setup whatever model name it sends. With `list_models true` it lists every
  model instead, as with Ollama: the configured one and each `origami model list` name (with their `-<level>` ids),
  read at each request, a request naming another listed model loading it in place of the running one after the
  running requests finish.
  Each model runs in its own process (`origami serve --single-model`, which serves one model as before), so a switch
  returns all of the previous model's memory; a name not listed goes to the running model. `/api/show` reports each
  model's own family (GLM and DeepSeek no longer claim Qwen's architecture or images), and `origami model list` no
  longer calls a registered GGUF file a missing folder.
- **`dynamic` quality level**: `faster`'s shortcut only on the decode steps after a token the model gave at least
  0.9 probability (its confidence in the last token stands in for how hard the next one is), exact after the others,
  within a sentence too. Measured with the others on 30 Oct: 1.4× exact's decode, 96.3% of exact's next tokens
  (`faster` that day: 1.8×, 93.8%), every ballpark answer right. The threshold is a knob of its own,
  `confident_above` (`--confident-above`, `off` by default), on top of any level; `dynamic` is also a model id
  (`-dynamic`), an `origami_mode` and a chat command level, and the families apply it per step too.
- **Other model families (experimental)**: GLM-5.3-Flash and DeepSeek-V4.1-Flash GGUF files served as they are, with
  their own tokenizer and chat template (text only); a GLM-5.3 safetensors checkpoint converts once into such a GGUF.
  `origami inspect --family <file, folder or owner/repo>` judges any checkpoint from its headers. GGUF files also read
  the IQ1, IQ2 and IQ3 types.
- **Optional API key** (`origami config set api_key <16–256 characters>`, or `ORIGAMI_API_KEY`; off by default): every
  request must then carry `Authorization: Bearer <key>` or `x-api-key: <key>`, or gets a 401 (`invalid_api_key`) before
  its body is read. The CLI's server commands send it, `origami dashboard` brings it to the page, and the key is never
  shown. The settings file is now written readable by its owner only.
- **`prompt_cache_entries`** (3 to 8): how many saved prompts and conversations a context above 8K keeps, with the
  disk each choice takes in its description, the README and `origami doctor`.
- **LICENSE**: Origami is licensed under the Apache License 2.0.
- **GGUFs with smaller experts, served as they are** (the `gguf-experts` scheme): `origami model add` of a Qwen GGUF
  whose routed experts average under 4.5 bits per weight converts the rest once and reads the experts from the GGUF in
  each layer's own types (IQ1_M to Q6_K, Q2_0 included) on llama.cpp's kernels, for ISTA-DASLab's GSQ-RCO releases
  (IQ3_XXS: 3.0 bits per weight, 2.33 MB cache slots against 4-bit's 2.76 MB). A pruned GGUF runs too: the GSQ-RCO
  Coder keeps 256 of 512 experts per layer, and the router never chooses the others. Such a model decodes without
  queued experts and without packs, the hot copy, `expert_bits` or MTP; its replies are NUMERIC against every other
  build. On the 32 GB M2 Pro the IQ3_XXS build decodes 1.33× as fast as 4-bit at the same memory limit (7.8 against
  5.8 tok/s), with every scored ballpark answer right; the Coder too. Other models are unchanged (their kernels,
  routing and records are untouched; 4-bit replies byte-identical).

### Changed
- GLM-5.3 and DeepSeek-V4.1: long prompts no longer fail with "Impacting Interactivity" (macOS stopped a long GPU
  command buffer while the display needed the GPU): prompt work is cut into short buffers, GLM's attention in batches
  of 64 rows; a GPU failure names the step it happened in. Replies are unchanged.
- GLM-5.3 and DeepSeek-V4.1: expert slots come in one pool per record size (DeepSeek's mixed types now hold about a
  quarter more experts in the same memory), and the slots experts keep coming back to are kept over ones a prompt used
  once (DeepSeek decode about 3% faster in a short test). `hot_copy` sets a copy of their most-used experts (counted
  while serving) in the data folder, read alongside the model's drive. Replies are unchanged.
- GLM-5.3 and DeepSeek-V4.1 decode their attention with a thread per head (one decode row had left the GPU mostly
  idle): DeepSeek decodes about 13% faster, with bitwise the same replies.
- A request no longer fails with "Expert cache phase change requires joined windows and reads": a read that was
  still running when the reply ended or its cache changed phase is now waited for (up to 5 s).
- A reply no longer fails with "Tokenizer streaming prefix changed" when a token adds a combining mark or variation
  selector to the character streamed last (or "\n" after "\r").
- A stored non-exact `quality` with `allow_fast false` now serves exact and says so in the log, instead of stopping
  the server from starting; `--quality` and `ORIGAMI_QUALITY` win over knobs stored in the settings file; knobs that
  change nothing (`experts_per_token 10`, `tail_skip_misses_only` without a tail skip) are exact. The knobs are flags
  too (`--experts-per-token`, `--tail-skip`, `--tail-skip-misses-only`, `--late-miss`).
- `origami bench` and `origami report` time exact replies whatever the server's default level.
- The GGUF matrix kernels compile with safe math instead of ggml's fast math: as fast on GLM-5.3, and closer to the
  reference implementation (the families' replies at temperature 0 can differ from earlier 2.1.0 builds).
- Chat templates: `dict(mapping, key=value)` keeps the mapping's entries, as in Python (the template engine kept only
  the keywords, so DeepSeek-V4.1's template lost every message's role); `reasoning_effort` `high` and `max` reach a
  template as `xhigh` only when the template names `xhigh` (Qwen's does, so its prompts are unchanged).
- Image links to this Mac or a private, shared, link-local or multicast network address (IPv4 or IPv6, after name
  resolution, on every redirect and by the address connected to) are refused with a 400 (`image_link_refused`); send
  such an image as a `data:` URL.
- The 32-image limit counts only the images a request still reads: a long chat whose earlier images are in its saved
  state may hold more (up to what its context fits, 768 MB of image files in all), and only those images are decoded.
- Up to 512 tools a request (was 128).
- A call to a tool the request did not declare comes back as the reply's text instead of failing the reply (500).
- Above 8K the prompt cache keeps three saved prompts by default (2.0.5: five), configurable with
  `prompt_cache_entries`; a reply saves its prompt last, so a kept session (`X-Origami-Keep`) still advances with a
  single free entry.
- Group-32 builds (mlx-community): the host-joined path (scopes too short for the whole-prefill layout, replays)
  dequantizes the 8-bit router and gates and the 4-bit float matrices as the whole schedule does, instead of reading
  them as 4-bit projections, which failed such requests.
- Group-32 builds (mlx-community) reply correctly: they store the zero-centred norms (attention q/k, the indexer's,
  every hyper-connection norm and PLE's) as w where Origami computes with 1 + w, so every reply was garbage from its
  first token; Origami now tells the two apart from the attention norms when a model loads and adds the 1 as its
  converter does (checkpoints that store 1 + w, such as carloslfu's, read exactly as before). Their whole-prefill
  scopes no longer fail with "Embedding row geometry differs".

- **Speed** (exact replies unchanged; figures from the M2 Pro):
  - the fused compact prefill attention, bit-identical to the split kernels, runs on every in-memory context from 32K
    up (2.0.5: 32K only): a cold 42,052-token prompt at 64K read in 363 s instead of 441 s, the same reply;
  - a whole-prefill scope allocates its MoE contribution buffer (839 MB at 8K) once instead of once per layer: about
    1.1 s less per 8K scope in isolation (48 fresh buffers 1.35–1.42 s, one reused 0.23–0.24 s); a cold 6,173-token
    prompt at 8K read in 45.3 s instead of 49.9 s, the same reply;
  - each streamed token decodes only the reply's tokens since its text last ended in a whole character, instead of the
    whole reply: an 8,192-token reply's streaming decodes took 0.02 s in all instead of 26–29 s (up to 7 ms a token
    near its end), the same text byte for byte;
  - a prompt-cache save clones the image saved before it and writes only the pages that differ: a 7,680-token prompt
    image after its 8,000-token conversation writes 2.9 MB instead of 311 MB (0.17 s instead of 0.44 s in the check);
    at 256K a reply's saves no longer rewrite the shared key/value pages (up to 6.6 GB each);
  - the file-backed context appends each decoded token's key and value rows in place (1 KiB per vector) instead of
    rewriting the 256 KiB tail page: filling a page writes 0.26 MB instead of 33.7 MB per vector (about 30 GB of SSD
    writes per 10,000 tokens before); a 384-token reply at 256K in a file had the server write 304 KB a token instead
    of 555 KB (the rest is frontiers and other files) and decode at 2.95 tok/s instead of 2.88, the same reply.

### Removed
- `ORIGAMI_DIAGNOSTICS` and the per-request snapshots it recorded (timing breakdowns, owner counters, the handoff
  window, path-proof counters).
- Code only checks reached: the stream-v1 checkpoint codec (the prompt cache uses the paged format), the
  conversation-less prefix path, and `ResidentGDN`.

## 2.0.5 (unreleased)

**Replies:** at temperature 0 with the same settings, byte-identical to 2.0.4 on the eight ballpark cases; an image
request and a tool-call request give the same replies. No fast-mode arithmetic changed.

**Speed:** the same as 2.0.4 within run-to-run noise. M2 Pro 32 GB, a 7.7K-token prompt and 512 tokens, one run each
in A B B A order: at 8K, first token 70.1 s cold (2.0.4: 71.4 s), decode 8.70 tok/s cold (8.87) and 8.57 warm (8.80);
at 32K, 68.1 s (67.3 s), 8.66 (8.65) and 8.51 (8.37).

### Fixed
- **Crashes**: a client that disconnected just as its stream started could stop the server; a size such as `-1` or
  `nan` in `GET /v1/plan`, `POST /v1/config`, a flag or a setting stopped it too; a malformed model file, GGUF, MTP head,
  vision tower or expert-store header is refused with a message instead of crashing, and can no longer make Origami read
  or write out of bounds; with the dashboard open during a prompt read, a rare race could crash the server.
- **Lost replies**: a long prompt after an unrelated long chat (32K and above) failed after minutes of reading; a
  prediction read still running at the end of a reply turned the finished reply into an error; with MTP speculation on,
  a reply ending on a drafted end-of-turn token failed after it was streamed, and a forced think-closing step could emit
  one wrong token.
- **Saved prompts**: a valid saved prompt could be deleted as damaged when memory or a drive was briefly short; kept
  sessions (`X-Origami-Keep`) now move to each newer turn on every context size, and no save is skipped silently (tiers
  above 8K keep five regular entries, up from two above 32K and four at 32K; saves still leave 10 GB of the drive free,
  now counted in the blocks really written, plus room for a context kept in a file); a folder that cannot be deleted no
  longer stalls saving; a data folder on a drive without file permissions (exFAT) serves without saved prompts and says
  so at start instead of failing every save.
- **Memory**: an idle server gives memory back when macOS runs short (released experts stayed in the GPU's residency
  set), and a short burst of swap no longer cuts the expert cache many times a second.
- **Group-32 builds** (mlx-community) no longer fail every request at the final mixer.
- **Security**: requests from web pages on other sites are refused (a page could send chat commands such as
  `origami forget all` to `/api/chat`); JSON structure is bounded before it is parsed on every API; a request carries at
  most 32 images, in PNG, JPEG, HEIC, GIF, WebP or TIFF and up to about 67 megapixels, and an image link may not redirect
  from https to http; the converter never reads or writes outside its folders and never deletes a folder it did not make.
- **Expert copies**: a record that fails its checksum is no longer read from that copy (the whole copy was dropped for
  30 s and then trusted again); a copy with a missing or short PLE table is rebuilt; a drive that comes back is reopened
  in the background, not during a request; an interrupted `origami setup` no longer blocks the next one.
- **CLI**: `origami uninstall`, `cache prune` and `store remove` delete only Origami's own folders, also in a shared
  `context_dir`, and remove prepared 3- and 2-bit packs; `config set speculation <file>|off` and `request_log off` work;
  `setup --model <name>` keeps the name and its expert width; every setting is also a flag; `doctor` counts a full disk
  and stops reporting copies that are not needed; `bench` and `report` no longer fill the prompt cache; downloads stop at
  once on 401/403/404 and say why; a second Ctrl-C quits at once, and saves are drained on every shutdown path.
- **Converting**: Hugging Face cache snapshots (symlinked files) convert; F16 and F32 checkpoints convert correctly; a
  fine-tune saved again in place converts again (so a model converted by 2.0.4 is converted once more on its next
  `origami model add`); a chat template kept only in `tokenizer_config.json` is read.
- **APIs**: OpenAI Responses stream errors use Responses events; Anthropic errors carry Anthropic's error types; a
  `<tool_call>` with no tools declared, or not in function form, comes back as text instead of an error or nothing; above
  the connection limit a request gets a 503; `HEAD` gets no body; `/api/tags` and `/api/show` report the real
  quantization and context; the old request field `qfs_mode` is refused with a message (use `origami_mode`).
- The sampler no longer sorts the vocabulary for top-k and top-p (the same draws; in a CPU
  microbenchmark 27 ms → 1.8 ms a token at the full vocabulary, at temperature above 0); a NaN logit now fails the request instead of emitting a token.

### Removed
- Research diagnostics that a release plan never ran: the decode GPU profiler and layer-phase profiler
  (`profile_decode_gpu`, `gpu_profile_phase`, `profile_layer_phases`), the QSA capture (`qsa_decode_capture_directory`),
  `ORIGAMI_DUMP_HIDDEN`, the requantizing switch `expert_requantize_bits` (prepared 3- and 2-bit packs replace it) and
  `ORIGAMI_FIXTURES` with the frozen-reference checks that needed it. `ORIGAMI_EXECUTION` and `ORIGAMI_DIAGNOSTICS` stay.

## 2.0.4 (28 October 2026)

The first 2.0 release; 2.0.0 to 2.0.3 were development builds and are included here.

**Replies:** at temperature 0 with the same settings, byte-identical to 1.0.0 on the eight ballpark cases, also with the M1 kernels forced on an M2 Pro. Fast mode is unchanged.

**Speed:** the same as 1.0.0. M2 Pro 32 GB, a 7.7K-token prompt and 512 tokens, two runs each in ABBA order:
first token after 69.9 s cold (1.0.0: 69.6 s) and 0.20 s warm (0.19 s); decode 7.28 tok/s cold (7.21) and
6.99 warm (7.01).

### New
- **MTP speculation, opt-in** (`origami config set speculation <file>`): the model's multi-token-prediction head
  (shipped in ARC4NUM's MLX-Serve build; `scripts/fetch-mtp-head.py` fetches just the head, 1.5 GB) drafts the next
  token and one two-token step checks it. A potential speed upgrade for Macs that hold every expert in memory (96 GB
  or more). The two-token step shares every projection's weight reads (a two-row decode kernel) and, when every
  routed expert is in memory, runs without host round trips. Each reply measures the step against plain decode and
  decodes plainly where it does not pay; sampling keeps its distribution, and replies at temperature 0 can differ
  slightly from plain decode (the step's arithmetic differs; the ballpark validators all pass). On the 32 GB M2 Pro a
  checking step costs about 2.5 plain steps (experts outside memory), so it turns itself off.
- **M5 matrix units for prompts**: on M5 Macs (Metal family Apple10) the 64 × 64 prefill matrix kernel, which reads
  dense projections and large expert tiles, multiplies through Metal 4 TensorOps instead of simdgroup MMAs. Chosen when
  the kernels are built; earlier Macs run exactly as before. Checked against the current kernel on an M2 Pro (TensorOps
  run there without matrix units); its speed on an M5 is not measured yet.
- **NVIDIA's NVFP4 release** (`nvidia/Qwen3.8-Flash-Next-NVFP4`) converts with `origami model add` and keeps its
  experts as NVFP4, read natively by the expert kernels: the same bytes per expert as 4-bit, closer to the original
  weights (8.1% RMS error against bf16 on two layers; 4-bit group 64: 9.5%). Expert kernels cost the same within
  noise (ten-expert decode 759 vs 727 µs, grouped prefill 1.96 vs 1.95 ms).
- **More APIs**: Anthropic Messages (`/v1/messages`), OpenAI Responses (`/v1/responses`) and Ollama (`/api/chat`,
  `/api/generate`, `/api/tags`) beside OpenAI chat, with tools, images, thinking and streaming; `/health` lists the
  endpoints.
- **Claude Code and Codex** work against Origami (tested: Claude Code 2.1, Codex 0.159, 32K). A system or
  developer message after the conversation began is now a user turn where it stands instead of a 400, and a
  budget past the context is refused in Anthropic's words, which Claude Code reads to retry with less.
- **Request records** (`request_log`, off by default): one JSON line per request, system prompts and tools kept once.
- **Images**: send them as OpenAI `image_url` parts (base64 `data:` URLs or http(s) links) in user messages. The
  checkpoint's vision tower runs on the GPU, 1 to 4 s per image on an M2 Pro, and an image becomes up to
  1,024 tokens. A chat's next turn continues after its images from saved state instead of reading and
  encoding them again. Text-only replies are unchanged.
- **Dashboard** at `http://127.0.0.1:11439/dashboard` (`origami dashboard`): whether Origami is up, how
  fast it writes, memory, recent requests, saved prompts and drives. `GET /v1/status` returns the same
  data as JSON.
- **`origami doctor`**, **`origami inspect <folder>`**, **`origami bench`** and **`origami report`**
  (a paste-ready block for issues).
- **Chat commands**: a whole message `origami help`, `status`, `save`, `prompts`, `forget` or
  `fast on|off` is answered without the model; earlier commands are left out of what the model sees.
- **System prompts are saved on their own** (4 slots that one-off chats cannot evict); a changed system
  prompt reuses the part it shares with an earlier one. The point where each reply begins is saved too,
  so a follow-up after a restart is reused even when the client drops the reasoning.
- **`X-Origami-Keep: <name>`** keeps a long session's saved prompt pinned under a name so other chats cannot
  push it out (two sessions at most; `origami prompts` lists them, `origami forget` releases them).
- `"prompt_cache_policy": "no_store"` for title generators and summarizers; each engine build keeps its
  own cache folder; `origami cache` lists them and `origami cache prune` removes old ones.
- **Thinking**: a `reasoning_effort` default setting, `max_thinking_tokens`, and the fields harnesses
  send (`think`, OpenRouter `reasoning`, Anthropic `thinking`, `chat_template_kwargs.enable_thinking`,
  `thinking_budget`). When the answer would have less than max(1024, 15% of `max_tokens`) left,
  Origami closes the think block so the reply still ends with an answer.
- Streams send `: keepalive` every 10 s while a request waits or reads a long prompt. Requests queue with
  a person's message ahead of an agent's tool turn; `/health` shows the queue.
- Opt-in progress lines in the thinking panel (`status_messages`) and llama.cpp-style
  `prompt_progress` frames (`return_progress`).
- **Any context up to 262,144 tokens** (`origami config set context 100000`, `--context 64k`): Origami serves it on the
  smallest tier that holds it, and a request longer than the setting is refused in the client's own wording. The context's
  key/value cache stays in memory up to the size `origami setup`, `origami status` and the dashboard show for this Mac
  (all 256K with 32 GB and the default settings: 64K, 128K and 256K keep it in memory with every decode optimization, 5.2
  tok/s at 46K tokens, 4.5 at 101K and 4.3 at 157K, against 1.5 in a file at 31K); a context that does not fit keeps it
  in a file in the data folder, and Origami says what
  that costs when it starts. Measured on an M2 Pro 32 GB: a 30K-token prompt took 6.5 minutes and its follow-up 5 s; a
  259K-token prompt took 67 minutes, recalled facts hidden at 1% and 55% of it, and its follow-ups started in 20–30 s;
  replies come at about 2.6 tok/s after 30K tokens (32K in memory: 6.3) and 1.8 at full context, since this tier runs
  without the 8K/32K-only decode optimizations and reads its attention through the file. It needs 32 GB of RAM and about
  37 GB of free disk (a full conversation used up to 52 GB while the server ran), and at full context the conversation is
  saved for 1–2 minutes after each reply (a message sent sooner waits).
- **Memory and disk** in the dashboard: the plan for another expert width, context or memory for Origami
  (`GET /v1/plan`) before it is applied: how many experts stay in memory, whether the context pages to disk, which drive
  holds what, what each choice costs or gains (measured on an M2 Pro 32 GB, or marked estimate), and the commands that
  apply it.
- **Macs with 64 GB or more** use their memory: the expert cache is no longer capped at 8,192 experts.
  96 GB decodes with every expert in memory; from 128 GB every expert stays in memory once read and
  `origami setup` builds no expert copies. Planned and checked by simulation; not yet run on such a Mac.
- `ORIGAMI_NO_BFLOAT_MATRICES` (set to any value) runs the M1 kernels on a newer Mac.
- **Group-32 MLX builds** such as `mlx-community/Qwen3.8-Flash-Next-4bit` are recognised; their 8-bit router and other
  quantized float matrices are dequantized at load (checked kernel by kernel; the full build was not run yet).
- **The hot-expert mirror follows the model**: Origami counts which experts each model needs; after 20,000 tokens,
  `origami setup` ranks (and when needed rebuilds) the mirror from those counts instead of the shipped list.
- **bf16, FP8 and GGUF checkpoints**: `origami model add <folder>` converts an unquantized checkpoint (the official
  release or a fine-tune), the official FP8 release or a llama.cpp GGUF (Q8_0, the K-quants, IQ4 and more) once into the
  4-bit layout; from bf16 its tensors are byte-identical to the MLX conversion of the same weights.
- **`origami model list|add|remove|default`**: several models kept under names, each with an optional expert width;
  `--model <name>` serves one.
- **`expert_bits` 3 or 2**, on any Mac: `origami setup` prepares a pack of smaller experts from the model, so more of
  them stay in memory and there is more room for context. On 32 GB, 3-bit decode tied 4-bit and a long prompt's first
  token came 15–20% later; with less memory for experts (a 16.8 GB budget) 3-bit decoded 12–14% faster. Replies differ
  slightly from 4-bit.

### Changed
- A request no longer fails with "All expert slots are reading, reserved or GPU-pinned" when the Mac runs short of
  memory at the end of a prompt or during the reply: the demand path waits for a slot, as the prompt read-ahead
  already did (2.0.3 failed one of three squeezed requests; this build none).
- `--help` and `--version` never run a command; usage errors exit with status 2.
- Drive measurements are keyed by the drives' identity, not their paths.
- Chat markers typed into a message (`<|im_start|>` and the like) stay text.
- A reply's stream ends with its last token: the conversation is saved for reuse after it (8K, a 4K-token prompt: the
  stream ended 0.25 s after the last token instead of 2.1 s), and a request that arrives before the save is done waits.
- The prompt cache keeps 10 GB of disk free after each save (it checked only before writing, so a large save could leave
  less), and resume points stop early enough to leave room for that save.
- Decode is about 4% faster at 8K: each layer's next attention is prepared while the GPU computes its experts (replies
  are byte-identical).
- Memory, context and drives are one choice in the dashboard: a bar shows how memory is split, a preview shows another
  split (expert width, experts in memory, memory for Origami, context and its drive, the hot copy) with what it costs,
  and **Apply** saves it (`POST /v1/config`, from the page itself only). New settings `expert_memory`, `hot_copy` and
  `context_dir`; `memory_limit` may now go above the automatic share, up to Metal's limit; `origami config set expert_bits` works (it was refused), and `config set stores`
  keeps each store's size.
- Several drives: `origami setup` puts a copy of the experts on every drive in `stores` (the whole pack where it fits,
  otherwise the most-used experts after the hot copy's), and Origami reads them all at once, each read going to the least
  busy drive holding that expert; a damaged or unplugged copy is dropped and the others keep serving.
- Decode is about 6% faster again at 8K and 32K: each layer's experts are queued on the GPU before the router has chosen
  them, the GPU looks up the cached ones itself and starts them at once, and the host only reads and hands over the
  missing ones (replies are byte-identical; `decode_preencoded_layers`, off at 256K).
- Long-context attention reads its index where it is stored instead of copying it every token: about 3 ms less a token
  after a 25K-token prompt and 32 ms less at full 256K context than 1.0.0 (about 2% and 6% of decode, timed on that
  step); replies are byte-identical, and decode plans 64 MiB less memory.
- Memory left to other apps is at most 24 GB (still 30% of RAM below 80 GB).
- A request no longer fails ("Inference request failed") when the Mac runs short of memory while a long prompt is read:
  the prompt read-ahead now follows the expert cache as Origami shrinks it, where it kept asking for the slots it had at
  the start (2.0.2 failed one of three such prompts with another app taking 18 GB; this build none of three, with up to
  24 GB taken). The cause of a failed request is now logged.
- Free space on volumes that report none for "important" use (disk images among them) is read correctly: `origami setup`
  built no copy there and the prompt cache refused to save to them.

## 1.0.0 (24 October 2026)

First release: Qwen3.8-Flash-Next 4-bit (group 64) on Apple-silicon Macs with 24 GB or more, streaming
experts from one or two drives; OpenAI-compatible server; prompt cache that survives restarts; opt-in
fast mode.
