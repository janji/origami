# Changelog

Origami's releases, newest first. Each entry says whether replies at temperature 0 changed.

Earlier releases (1.0.0 to 2.4.3) are in [docs/changelog-2.x.md](docs/changelog-2.x.md).

## 3.1.3 (2026-11-21)

**Replies:** unchanged.

- **Qwen: the next turn of a conversation no longer waits for the last reply's saved prompts to be written.** A
  request that arrived within seconds of the previous reply (an agent's next step, a regenerated reply, a repeated
  prompt) waited for up to three prompt-cache images to reach the drive: 1.2 to 4.0 s before its first token on an
  M2 Pro with the data folder on an external drive. A request that continues the conversation from memory now puts
  those saves off until it has answered; any other request waits as before. Measured on the 4-bit model, old and new
  build in turn: a repeated 1K or 8K prompt's first token after 0.15 s, against 1.5 to 9.8 s. New check:
  `prompt_cache_saves_deferred`.

## 3.1.2 (2026-11-20)

**Replies:** unchanged.

- **Qwen with GGUF experts gets a hot copy**: a low-bit variant (`origami model add <experts.gguf> --base <model>`) and
  a converted low-bit GGUF had none. It is built and kept while serving like the 4-bit model's, in its own folder of
  `hot_copy_dir` (else the data folder's store), with the same settings (`hot_copy`, `hot_copy_more`, `hot_copy_keep`)
  and the same dashboard control. A record in the copy is its layer's own size, so a file that mixes types per layer
  takes no padding there. It is built only on another drive than the GGUF's. Packs, further copies and MTP still do
  not apply. On a real IQ3_S experts file the reply was the same with and without the copy; its speed is not measured
  yet. New check: `gguf_experts_hot_copy_matches_gguf`.

## 3.1.1 (2026-11-20)

**Replies:** unchanged.

The first release run on a second Mac: an M1 MacBook Air with 16 GB on macOS 27, Qwen on an exFAT portable SSD, built
there with the macOS 27 tools. It fixes what that Mac showed.

- **A build made with Swift 6.4 (the macOS 27 tools) crashed on the first request.** A copy closure passed through a
  rethrowing helper was compiled to a tail call that left a buffer's address in Swift's error register, and the caller
  took it for a thrown error. That copy now goes through the non-inlined helper its neighbours already used, and a new
  check, `Tests/error-register-check.py`, reads a built binary for this fault (CI runs it on every build).
- The hot copy builds on exFAT and other drives that are not APFS: after its first gigabyte the build took such a
  drive for full ("under 10 GB free") and stopped.
- **A data folder on a drive without file ownership (exFAT, FAT):** saved prompts now default to this Mac's own data
  folder. On such a drive they were off, and a follow-up read its whole conversation again. `origami setup` says where
  they go; `origami doctor` says so when `context_dir` itself is on such a drive.
- When the model's process dies under a request, the client gets a 502 with the reason instead of an empty reply, and
  the log names the signal. The next request starts the model again.
- `origami doctor`, `origami status` and the dashboard advise `origami refresh` only once a terabyte has been read from
  the files (or a slowdown is on record for them); before that, slow reading in a first hour is reported without
  blaming the drive.
- The scan for another process holding a hot copy open is bounded at two seconds (no answer counts as open): it could
  wait for minutes while any process was blocked on a slow drive.
- Downloads: the rate shown after a resume no longer counts what was already on disk, and a drive that writes under
  60 MB/s (a USB 2 cable or hub) is named before the download starts.
- Checks: the command-line, dashboard and ingress checks pass on a 16 GB Mac, with the macOS firewall on and without
  node (those parts are skipped and say so); the hedge check that staged more records than its arena holds no longer
  depends on the Mac's drive; the ingress checks run in CI. CI ran on GitHub for the first time: its virtual Macs have
  no GPU Origami accepts, so the four scripts that need one throughout are skipped there as steps.
- Removed: the conversion scripts of the research draft model that 3.1.0 deleted (`scripts/convert_dspark.py`,
  `scripts/dspark_codec.py` and their check), and with them the checks' need for numpy.
- Docs: the README has a list of what Origami does, pictures of the dashboard and its crane, the numbers measured
  on the M1 with 16 GB, what makes speed vary from one request to the next, and the clone address.

## 3.1.0 (2026-11-18, not published)

**Replies:** these changed; everything else is unchanged and exact.

- GLM and DeepSeek replies that ran out of tokens while thinking now close the think block before the limit and
  answer, as Qwen's do.
- A turn that ended with neither text nor a tool call (inside an unclosed think block, or after an empty tool-call
  block) used to return empty content; it now answers with its reasoning. A reply cut by `max_tokens` is unchanged.
- Requests with `tool_choice` `required` or a named tool get a different prompt: the instruction now ends the last
  user message instead of the system prompt, a named choice keeps every tool listed, and the reply is made to open a
  tool call.

### Correctness
- GLM and DeepSeek: an exact request could reuse state computed in a fast mode (since at least 2.4.3). A fast turn
  that continued an earlier fast turn saved its prompt over that fast reply, and a later exact request restored it,
  in the same server or after a restart. Only exact state is saved now; `keep` on a fast turn keeps nothing, and the
  log says why.
- Qwen: with `allow_fast` off, an exact request could reuse state computed in a fast mode (since at least 2.4.3),
  from a saved prompt made while `allow_fast` was on. Every exact request now skips such state.
- GLM and DeepSeek: saved prompts made with `ORIGAMI_FAMILY_GROUPED_DECODE=0` are kept apart from the default
  path's, since no check shows the two give the same bits.
- GLM and DeepSeek: a follow-up continued live is saved even when its prompt was saved before, so a restart gives the
  same reply in exact mode. DeepSeek's saved prompt end after a cancel and reset equals a cold run's.
- JSON mode: a strict schema accepted three kinds of answer it forbids (a value of the wrong type from an `enum`
  beside `type`, a declared key spelled with an escape, a property set to `false` when extra keys were allowed). Now
  `type` narrows an `enum`, a key cannot use an escape, and a forbidden property is never taken; an `enum` with no
  value of its `type`, or beside another constraining keyword, is a 400 that names it.

### Speed

Measured against 3.0.1 on 2026-11-16 on the development Mac (M2 Pro, 32 GB, models on an external Thunderbolt NVMe
drive, freshly rewritten; exact mode; memory limits 24 GB for Qwen, 22.5 GB for GLM and DeepSeek; single runs), with
the same replies. Time the client waited for the first content:

| Prompt | GLM | DeepSeek | Qwen |
|---|---|---|---|
| 2,471 tokens | 88.6 → 46.0 s | 126.2 → 70.6 s | 34.7 → 37.7 s (33.9–37.7 over the morning's runs) |
| 7,280 tokens | 261.0 → 123.6 s | 222.5 → 95.9 s | not comparable |

Writing: GLM 1.19 → 1.37 tok/s, DeepSeek 1.52 → 1.82. Qwen's 3.1 run built an 8 GB hot copy during the run (3.0.1's
had none), so its writing and long-prompt figures are left out. Current figures: the README's "What to expect".

- GLM and DeepSeek read each layer's experts once per long prompt: the prompt borrows working memory from the expert
  cache, and DeepSeek runs its feed-only and full slices in one pass. A 7,700-token prompt reads 180 GB instead of
  356 GB on GLM, 266 GB instead of 402 GB on DeepSeek.
- GLM and DeepSeek prompts read ahead under memory pressure too (a limit close to the Mac's memory used to switch it
  off for the whole prompt), and keep reading into the slots the cache has after a cut; only a new slab waits.
  Read-ahead starts by Qwen's rule (GLM from 246 rows, DeepSeek from 439, not 1,024).
- GLM and DeepSeek prompt reading uses 14–19% less GPU time: expert runs are grouped per GPU dispatch and the
  attention-value kernel reads its inputs once per thread group.
- GLM and DeepSeek read the next layer's likely experts early whenever the read can start before the next layer's
  router. DeepSeek decodes about 12% faster than 3.0.1; on GLM the guess is now on by default (a third fewer demand
  misses, about 12% less decode time).
- GLM and DeepSeek keep the GPU at speed through decode (`gpu_keepalive` auto: on except in Low Power Mode): GPU time
  per token falls by a third, and DeepSeek decodes 6% faster.
- GLM and DeepSeek: with a hot copy attached, a decode miss costs a third of the CPU time (46 → 15 µs on a 600-slot
  pool).
- Qwen reads 12% fewer bytes per token from the model drive: a next-layer guess is read early only when it clears a
  0.1 margin.
- Qwen prompts: the read-ahead that overlaps expert reads with the GPU starts at 351 rows instead of 1,024; after a
  memory cut it keeps reading ahead inside the smaller cache and pauses only while pressure is live (2–8 s).
- Qwen prompts: the expert cache gives back only what the prompt's own new rows need (in 256-token steps), not the
  room of an 8,192-token prompt, so a 600-token turn keeps most of the decode cache. The 4,416-expert limit measured
  on one Mac is gone.
- Qwen prompt reading runs the grouped expert gate, up and SiLU product as one GPU dispatch (a 2–4% faster expert
  step).
- Greedy sampling picks the token directly (about 0.15 ms less per reply token on Qwen).
- GLM and DeepSeek requests count, per drive, how long it was busy or idle while the model waited, the guessed reads
  and their timing (`decode.*`, `family_overlap`).

### Memory

- **Automatic memory uses the measured limit.** With no `memory_limit`, a model starts at the limit
  `origami tune --memory` measured for it on this Mac and never goes above it, until the measurement goes stale
  (another Mac, model or context; cuts on three different days; or 90 days). One line in the log, `origami status`
  and the dashboard says which rule is in force.
- **New: `origami tune --memory`** reloads the model at a few limits and suggests the highest one macOS kept without
  cutting or swapping (speed overrules it only when a limit is over 15% slower on both measures, after a retry). It
  never changes the setting. About 4 minutes for Qwen, 6 for GLM, 10 for DeepSeek. Finished trials are kept, so a run
  stopped by a request, Cancel or the time limit continues within a day. When even the lowest limit is cut, it goes
  on downwards (at most three more) and names the other processes holding memory.
- **New, opt-in: `memory_auto_measure`.** With no limit set, an idle server (30 minutes without a request, on power,
  at most once a day per model and context) measures the limit itself. A request ends the trial before it is
  admitted and is answered at your saved limit. The status says why it has not run and since when.
- **An explicit `memory_limit` is a ceiling.** After a cut Origami holds the level that held, then climbs back by half
  the gap to the limit (at least 1 GB) every 40 s while memory is free (14.7 → 24 GB in 200 s instead of 360 s). A
  model always starts at its limit; a limit cut on three different days gets a suggestion of the level it kept.
- A cut for a swap increase follows its size: the budget drops by the swap plus 1 GB (0.35 GB swapped at 24 GB now
  costs 1.35 GB, not 9.3 GB).
- Memory cuts are kept per model and context, so one model's cuts no longer affect another's measured limit.
- A cut shows in automatic mode too: `/v1/status` `temporarily_reduced` and `last_cut`, and the dashboard's "held to
  18.0 GB for now (…, 5 min ago)".
- The least memory limit `/v1/status` reports (`memory_learned.minimum_bytes`) is now one the model really starts at;
  a limit too small says what is short and the smallest limit that works.
- A Mac too small for the model (8 GB, or 32K on 16 GB for Qwen) says "this model does not fit this Mac at this
  context", with the most this Mac allows and what would fit, instead of advising a higher `memory_limit`. The
  out-of-memory hint names this Mac's largest limit.
- `origami status` and the idle measurement use the measured limit the running model uses, also after a move or
  refresh.

### Hot copy and drives

- **The hot copy follows usage in place** (all three models): between requests it swaps its least-used experts for
  clearly more-used ones, at most 1 GB read from the model drive per hour, with no second copy and no free space
  needed. Usage counts halve once they cover 80,000 tokens, so the copy follows a workload that changed.
- **Qwen's hot copy is built while serving**, as GLM's and DeepSeek's are: from the first reply, attached without a
  reload, with the same format, checks and `origami verify`. `origami setup` no longer builds it.
- **The automatic size grows while it pays** (each further tenth of the experts must add 2% of uses), past 40 GB, up
  to the drive's free space less 20 GB. A size you set is never changed. GLM's and DeepSeek's first copy leaves 20 GB
  free, not 30 GB.
- **A working copy is never deleted or shrunk because free space dips**; falling space only stops it growing. A new
  size is written beside the copy in use and attached once complete; a lower size with no room beside it shrinks the
  copy in place.
- **Hot copies on several drives** (`hot_copy_more`, all three models): the fastest drive holds the most-used experts,
  the next drive the next ones, spare room the most-used again; each read goes to the drive expected to finish first.
  A drive unplugged mid-request fails nothing.
- **Several models on one drive:** the loaded model's copy grows by taking the least-used experts of idle models'
  copies on that drive (each keeps what covers half its uses), and they grow back when their model loads. The new
  per-model `hot_copy_keep` keeps a copy whole.
- **One disk budget per drive:** a saved-prompt save short of room makes the hot copy on that drive give back just the
  least-used experts it lacks (at most twice the shortfall) instead of the save being skipped.
- Drive speeds are measured on each Mac (at first start, `setup` and `tune`, about 3 s a drive), not taken from the
  development Mac; a drive off by 3x for 10 minutes, or moved to another port, is measured again, and a new copy
  place when it is added. GLM and
  DeepSeek plan copies on several drives at the sum of their speeds. The old `calibration.json` is no longer read.
- A drive avoided after it slowed is judged on its live speed again within seconds.
- GLM and DeepSeek demand reads of a record the hot copy holds are hedged to the other drive when the read fails or
  has waited 50 ms and the other drive is expected to be faster. A hot copy that fails a read is skipped for 30 s.
- A hot-copy record that fails its checksum is read from the model's drive and rewritten after the next reply
  (`repaired_records`).
- A hot copy build that failed is planned again five minutes later; each detach, re-attach and failure is logged.
- **Origami counts what it reads from each drive** since the files were last written, and how much was slow; the
  dashboard, `origami doctor` and `/v1/status` (`since_rewrite`) show it and warn ahead of the next slowdown. A
  sustained slowdown keeps counting as slow.
- **New: `origami refresh`** rewrites a model's files in place, checked, when its drive reads them far below its own
  speed, and keeps its hot copy, counts and GLM/DeepSeek saved prompts (Qwen's are deleted). It checks it can write
  every file first, and refuses to rerun after a file read back differently.
- When the model's own drive goes away, the dashboard and `origami status` say **Drive gone** (`degraded` in
  `/health` and `/v1/status`, "reconnect it, then Reload"); the dashboard offers Reload once it is back.
- Opt-in experiment `ORIGAMI_SPLIT_READS=1`: a lone expert miss that two idle drives hold is read in parts from both
  (1.09 → 0.82 ms for a 2.76 MB record). Off by default: on one fast drive it gained nothing (9.72 against 9.75
  tok/s).
- The model's drive dropping off and coming back no longer needs a Reload (all three models): while it is gone
  requests fail at once with `drive_stalled` (instead of each waiting up to a minute) and the status says `degraded`;
  once it is back and its files are the same (size, modification time, inode, 64 KiB of content), the open files are
  replaced in place within about 5 s and requests answer again with nothing reloaded. A different file at the same
  path is refused and named.

### Saved prompts

- A start or a save short of disk no longer removes another model's saved prompts. Only this model's other builds go,
  after 14 days unused or when disk runs short, and never a folder holding a kept session or a saved system prompt.
- Qwen's saved prompts of a model file no model has any more are removed after 14 days unused, and listed on their
  own.
- `origami cache prune` covers GLM and DeepSeek folders and a context left on disk, runs while a server is stopped,
  lists what it removes with sizes, and keeps a build with pinned or named conversations unless `--yes`.
- GLM and DeepSeek: `origami stop`, an unload and a model switch now save the conversation in memory, so the next
  start does not read its last turn again.
- GLM and DeepSeek: alternating conversations no longer crowd each other's saves out. When no slot is free, a
  conversation's own earlier save is the first to go; while one is free both stay.
- A second `origami serve` on the same data folder no longer overwrites the first one's usage counts.
- A data folder that is not private (saves off) no longer gets the system prompt's token IDs.
- Saved prompts are off when the data folder (`context_dir`) is on an exFAT or FAT drive: such a drive ignores
  ownership, so every user of the Mac can read the saved conversations, although its folders show as private. The
  start-up line says so and how to move it. On an external APFS or Mac OS Extended drive mounted with ownership
  ignored (macOS's default) prompts are saved as before, and the start says once who can read them.

### Models and storage

- **New, opt-in: published low-bit experts.** `origami model add <experts.gguf> --base <model>` registers a variant
  that reads its routed experts from a published 2- to 3-bit GGUF and the rest from your exact model, after checking
  both are the same checkpoint. Replies differ, and status, the dashboard and the request log say so. Measured: Qwen
  with ISTA's GSQ-RCO Q2_0 experts, 12 of 14 checkable tasks (exact: 12), 91% next-token agreement; GLM with the
  Strata Q2_K/Q3_K experts, 12 of 14 (exact: 10), 92%. Speeds: the README. DeepSeek is verified on synthetic models
  only.
- **Moving keeps what is keyed on a model:** **Move…** and **Locate…** in the dashboard, and the new
  `origami model move <name> <folder>`, carry its hot copy, counts, tuning and GLM/DeepSeek saved prompts.
  `origami move list` and `origami move recover` finish or undo a move a stop interrupted. A move counts as done only
  once every part is in place, and a move across drives flushes the copy before deleting the original.
  A failed or cancelled move leaves nothing behind; one whose model cannot be read when it ends keeps its journal,
  and the copy waits as "belongs to <model> before its move" until `origami move recover`. **Locate…** accepts a
  moved Qwen model.
- **Move…** and **Delete…** on a hot copy in a further copy place change that place; deleting a copy in a folder no
  longer used changes no setting.
- **New: `origami relink`** for a drive that was renamed or mounted under another name: Origami notes each drive's
  volume UUID; `doctor`, `model list` and the dashboard name the folders found there, and `relink` uses them once
  confirmed.
- Moves onto exFAT work (exFAT refuses an exclusive rename).
- Unfinished conversions (`*.partial`) are listed with their size; Ctrl-C deletes a GLM conversion's partial (it
  cannot resume) and keeps a Qwen one to resume.
- `model add owner/repo` and `setup --model owner/repo` refuse a repo of GGUF files before downloading, and
  `model add` checks room for the download and its conversion first.
- `setup` checks the Mac before any download: it refuses under 16 GB and warns under 24 GB.
- `model add` of a model already added under that name keeps its settings; another file under the name needs
  `--name`. `setup --model x.gguf` makes that model the default only once its setup succeeded.
- `model list` and `setup` say "drive … not connected" for a model on an unplugged drive; a GGUF shorter than its
  header says "the download is incomplete".
- A relative `owner/repo` that is also a folder here is not downloaded (both readings are printed); a mistyped
  `.gguf` path gets a GGUF message.
- A low-bit variant starts with a copy of its base's usage counts; `refresh` carries them and rewrites its experts
  file; `status`, `doctor` and `model list` show the experts file, its drive and size.
- Download resume keeps a file only when the last download recorded it at this revision and size.
- `origami uninstall` also removes the copies in every copy place and in folders used before, and names what it
  leaves.
- `origami store remove` is refused while a server runs.
- A move across drives (or of a folder with links) deleted the original even when one of its files was written during
  the copy; such a move now fails, keeps the original, removes the copy and names the file, and a file written between
  the copy and the deletion is kept.

### Settings and tuning

- New settings: `start_on_request` (a request after `origami stop` starts the model), `idle_unload_minutes` (unload
  after that long idle), `memory_auto_measure`, `hot_copy_more`, `hot_copy_keep` and `trusted_proxy`.
- The quality knobs (`experts_per_token`, `tail_skip`, `tail_skip_misses_only`, `late_miss`, `confident_above`) are
  kept per model.
- `origami config set` says when a change applies (now, at the next load, or when `serve` starts again), whether it
  was saved for one model, and when an option or variable overrides it. It no longer prints `api_key`.
- `config unset` works when config.json holds a refused value; errors name the file; `config` and `doctor` name keys
  no setting reads.
- `config set data_dir` refuses while the old folder holds models.json, unless `--yes`.
- Writers of `config.json`, `models.json` and `tuning.json` in other processes no longer lose each other's changes.
- `origami tune` sizes its time limit from its first measurement (up to 30 minutes; your `--minutes` is kept), keeps
  every finished comparison so an interrupted run continues, records "the defaults are fastest" as a result, and says
  at every ending what was measured, what was kept and what to do next.
- `origami tune` refuses a setting option (it would only mislabel the result) and bad `--review`, `--tradeoffs` or
  `--minutes` values. A faster Qwen prompt group says before Apply that it sets saved prompts aside.

### Dashboard

- **Redesigned** in four tabs (Now, Setup, Models, Connect), state and next action first, one name per state; light
  and dark, phone width, no network requests.
- Now says why the last request was slow, with its prompt read, GPU time and waits per drive (`origami status` gives
  the same line).
- `origami status` with a server running adds what runs now: model, queue, memory against its target, cuts, where
  experts were read from and the last prompt restore.
- The hot copy control: a size slider over a popularity graph, in GB or % of experts; every drive with its speed and
  free space; every model's copy on a shared drive with **Keep when idle**; the swap backlog. Drives and files shows
  each drive's disk budget.
- The memory line: the limit in force, cuts, **Find the fastest** and **Measure when idle**, in the same decimal GB as
  `memory_limit`.
- "Reload required" covers every setting a model reads once. A page open on old settings cannot overwrite newer ones
  (Keep mine / Take theirs); saved settings that differ from the running model show **Reload to apply** and
  **Discard the saved change**; only the settings you moved are saved; every disabled button says why.
- Loading shows the seconds since start and the model's last log line; a failed start offers Try again.
- Delete on a model removes only its own files; deleting a downloaded model says it would have to be downloaded
  again; Delete, Move and Locate wait for a model that is starting; deleting or unlisting a low-bit variant never
  touches its base.

### Serving and clients

- New: `POST /v1/messages/count_tokens` (Anthropic's token count).
- New: JSON mode (`response_format` `json_object` or `json_schema`, Responses `text.format`, Ollama `format`) on every
  model; unsupported schema keywords are a 400 that names them.
- New: `frequency_penalty` on all three models; Ollama's `options` carry it and `presence_penalty`.
- New: the request field `origami_time_limit`, counted from the request's arrival.
- Each model lists its distinct reasoning levels once in `/v1/models` and `/health`; every name is still accepted.
  `xhigh` on DeepSeek is now `max`. The dashboard's Thinking dial offers the running model's levels.
- Responses `reasoning` input items reach the model; a Responses `custom` tool is a 400 naming it; an Anthropic
  `tool_result`'s images reach a model that reads images; a stop-string hit is `stop_sequence` on the Anthropic wire.
- A reply that answers with its reasoning because the turn had no answer says so in usage
  (`completion_tokens_details.empty_answer`) and the request log.
- A tool choice the model does not meet is a 422 `tool_choice_unmet` naming the tool, not a 500 after the whole reply.
- A model name selects a quality level only as the model's own id (or `origami`) plus `-<level>`.
- With `list_models` on, an unlisted model name is a 404 `model_not_found`, and requests for the loaded model that
  wait together go first (up to four; none once another model's request has waited 15 minutes).
- A request for a model that cannot start no longer stops the model that serves; a failed switch brings the previous
  model back.
- A streamed request that waits on a switch, a load or the queue gets its headers and a keepalive every 10 s.
- A streamed request whose client shuts its sending side gets its whole reply. A non-streamed one is still treated as
  gone: use streaming.
- A prompt that leaves less room than the reply asks for (at most 512 tokens) is refused as "prompt is too long".
- A streamed tool call is sent once its block is read whole.
- A client that sends a reply's thinking back under both `reasoning` and `reasoning_content` is no longer refused.
- A reply the client closed right after receiving it is no longer recorded as cancelled.
- New in `/v1/status`: `disk_budget` per drive, the hot copy's `popularity` and each drive's `room_bytes`,
  `split_reads`, and per request `prompt_restore` and the memory-pressure counters.
- `origami service install|remove` writes and loads the LaunchAgent (with the setting options given, such as
  `--port`), or stops and deletes it.
- Errors on `/v1/messages` (a stopped server, a busy queue, an unknown model, a wrong key) are in Anthropic's shape.
- Safe LAN mode: header and body deadlines, a minimum upload rate and per-device connection caps; 10 wrong keys a
  minute lock a device out for that minute; new `trusted_proxy` and a TLS proxy recipe; `doctor` warns when the key
  travels in clear.

### Reliability

- A GPU that stops answering no longer hangs the server: waits are bounded and a late finish retries the request.
- A model drive that stops answering fails the request with `drive_stalled` (503) after a minute, on all three
  models, also while Qwen reads a prompt's embedding rows.
- GLM prompts no longer fail after about 70 s when another app uses the GPU: command buffers stay under 25 ms (the
  longest was 97 ms), and Qwen's prompt buffers too.
- GLM and DeepSeek: a second long conversation is no longer refused with "Insufficient memory"; a memory refusal no
  longer blocks later requests; memory a cancelled prompt borrowed goes back at once; a cancelled prompt keeps the
  conversation at the last point read whole.
- A memory search that is stopped, cancelled or killed always ends with your saved limit in force.
- Unreadable `tuning.json`, `drive-reads.json` or `memory-history.json` files are set aside, never rewritten from
  empty.
- A request holding a number beyond a double's range, a tokenizer missing symbols, a GGUF with overlapping or repeated
  tensors, a corrupt GGUF length or a template `range()` of millions no longer crash or stall the server.
- GLM's `[gMASK]`, `<sop>` and argument tags typed into a message, and chat markers in a tool's name or schema, stay
  text.
- A client that stops reading a stream can no longer block the model.
- GLM and DeepSeek reads run on Qwen's shared scheduler and slot core; a guess a fetch waited for and the scheduler
  cancelled could be returned unread, and is now read by the fetch.
- `origami start` after a failed start starts the model again.
- An unwritable or full data folder gives one line naming the folder and the fix.
- Clearer messages for a missing model, a disconnected drive, a port in use, a running server and a too-small limit;
  `origami` alone shows the three steps to get running; `tune` and `bench` name the state of a server that is not
  ready; `doctor` and `status` show the same rows for GLM and DeepSeek as for Qwen, and name a Qwen hot copy in its
  store; the log says when a client cancels and when a hot-copy record is rejected or repaired.
- Checks: `engine-checks --fuzz`, Thread Sanitizer runs, `--no-gpu`, tiny synthetic GLM and DeepSeek models end to
  end, simulated 8, 16 and 24 GB Macs, and model-free profiles (`--bench-family-prefill`, `--bench-qwen`); CI runs
  them and fails when a build piped through `tee` fails. Building with an older Swift fails with a clear message (tools
  version 6.2). The `origami` binary no longer ships the engine's self-checks (1.5 MB smaller).
- A GPU step that never ended was taken for lost only at the deadline of an earlier, longer step when the watchdog was
  already asleep toward it (up to that step's bound late); it is now taken for lost at its own.

### Removed

- `expert_bits` (smaller experts Origami made itself): published low-bit experts do better on every count measured.
  A settings file that still sets it starts with the model's own experts and says so; `--expert-bits` stays only for
  converting GLM-5.3 safetensors.
- Qwen's offline hot-expert mirror built by `setup`: the hot copy is built while serving. A 3.0 mirror is removed when
  the new copy is first built.
- `ORIGAMI_FAMILY_INCREMENTAL_HOT_COPY`: following usage in place is the default.
- `ORIGAMI_STARTUP_PREPARE`: preparing Qwen's fixed layers at load is always on.
- `ORIGAMI_FAMILY_BUFFERED_SOURCE` and `ORIGAMI_FAMILY_BUFFERED_HOT`: GLM and DeepSeek always bypass the page cache.
- `ORIGAMI_FAMILY_SOURCE_READ_LANES` and the read-worker switches: reads are fixed at 64 per drive.
- `ORIGAMI_FAMILY_ASYNC_DEMAND`: every demand read is asynchronous.
- GLM and DeepSeek experiments a real-model screen did not keep (early and allocated-slot prefetch, arriving-first,
  buffered reads, the protected cache, compact shrink, admission variants, warm-after-restore, Engram overlap and
  memo, grouped prompt gate/up, turning adaptive prefetch off); their `origami_tune` fields are accepted and ignored.
- Qwen experiments a real-model screen did not keep (`ORIGAMI_QWEN_COMPACT_SHRINK`, the `expert_admission` variants,
  `qwen_arriving_first`), the batched sparse-attention path and the `slabOriginal` expert mode.
- Early guesses and the split by drive curves, tried during 3.1: 8% slower, and no change.
- Keeping layer 0's most-used Qwen experts resident, tried during 3.1: no gain, slightly slower decode.
- The research-only DeepSeek draft model (`ORIGAMI_TEST_DSPARK_PACK`); drafting stays parked.
- `ORIGAMI_QWEN_LONGEST_SAVED_PREFIX` and `ORIGAMI_FAMILY_LONGEST_SAVED_PREFIX` (off, unmeasured, not exact) and
  `ORIGAMI_FAMILY_ACTIVATION_OBSERVATIONS` (measured for a reuse rule that is closed).

### Docs

- A guide for new users ([docs/guide.md](docs/guide.md)), the entry point from the README.
- How to download GLM and DeepSeek, install with a symlink, keep downloads on another drive, run as a service, the
  free-space rules and model identity.
- The README lists each model's and each published file's license and what Origami derives from the weights (all
  kept on your Mac); its speed numbers were measured again on a healthy drive.

## 3.0.1

**Replies:** unchanged.

- **No false "loading".** A reload (Apply, a new hot copy, `/v1/reload`) waits for the reply being written; meanwhile
  the running model keeps answering `/health`, `/v1/status` and the dashboard, which shows "The model will reload after
  the current reply" instead of a loading screen.
- **A job's outcome stays** on the dashboard and in `/v1/jobs` (state, message, end time, last lines) until the next
  job or Dismiss, and the server log gets one line as each job ends.
- **A hot copy with no room fails and says why**: what it needs, what is kept free, what is free, and what old copies
  hold there. The model reloads after a hot-copy job only if the copy changed.
- **Copies made by Origami 2.x** are listed as "made by an earlier Origami version for <model>; safe to delete".
- **Paused conversation saves are visible** (log and dashboard, with the numbers), and a save that cannot fit no
  longer deletes the saved prompts first.

## 3.0.0 (unreleased)

Origami 3.0 is the first public release and a fresh starting point. It runs three mixture-of-experts models on
Apple Silicon Macs by keeping the always-used weights in memory and reading each token's experts from the SSD:
Qwen3.8-Flash-Next (the mature path), GLM-5.3-Flash and DeepSeek-V4.1-Flash (newer, text only). It starts from the
2.4.3 sources in a new repository; the 2.x history stays in [docs/changelog-2.x.md](docs/changelog-2.x.md).

**Replies:** the engine arithmetic is the same as 2.4.3, so replies in exact mode are unchanged, except: GLM and
DeepSeek replies that a cancelled read-ahead had made wrong; prompts that type chat control markers; thinking under a
small `max_tokens`; and images, whose vision step rounds slightly differently (largest difference 1.5e-5).

### Changed

- **Each model offers only the fast modes that help it.** Besides `exact`: Qwen offers `faster` and `fastest`, GLM
  offers `faster`, DeepSeek offers the new `dynamic-fastest` (fast skipping only on confident steps). A request for a
  mode the model does not offer runs its mildest fast mode and reports which one ran.
- **Many requests at once.** Up to 64 open requests wait in the queue (the fifth used to get a 503). Request-body
  memory is still sized for 4 (2 at 256K); when it is full the reply is a 503 "Request buffers are busy; retry in a
  moment".
- **Thinking under a small output cap.** Without a thinking budget, the room kept for the answer is at most half of
  `max_tokens`, so with `max_tokens` of 1024 or less thinking no longer closes at its first token.
- **A tool call the model wrote badly** becomes reply text with the normal finish reason, instead of failing the
  reply. A streaming client has already received the start of that call.
- **More requests accepted.** `strict` and unknown keys on a tool are ignored; `top_p: 0` means greedy; a
  `max_tokens` larger than the context allows is lowered to fit instead of refused. `response_format` types that need
  constrained decoding are still refused.
- **Saved prompts are safer.** Their index is flushed to disk on every write. An unreadable index is kept as
  `index.json.unreadable` and rebuilt from the saved slots (names and pins are lost) instead of deleting every saved
  prompt. A second server on the same prompt-cache folder saves no prompts and says so at start.
- **A new repository.** Origami 3.0 starts a fresh public history. Nothing from the old development history is
  needed to build, run or check it.
- **Sources organised by subsystem.** The engine lives in `Sources/OrigamiEngine/`, one folder per subsystem:
  `Core`, `Models`, `Formats`, `Memory`, `Streaming`, `Qwen`, `Family` (GLM and DeepSeek), `Conversation`,
  `Serving`, `Planning` and `SelfChecks`. The command-line tool stays in `Sources/origami/`.
- **One code style.** All Swift sources are formatted with `swift format`; the configuration is in `.swift-format`.
- **Documentation rewritten.** The README is a short quick start. The reference material moved to `docs/`:
  [settings](docs/settings.md), [models](docs/models.md), [integration](docs/integration.md),
  [dashboard](docs/dashboard.md), [troubleshooting](docs/troubleshooting.md) and every
  [`ORIGAMI_*` switch](docs/switches.md). [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md) describe
  how to work on the code; [SECURITY.md](SECURITY.md) how to report a vulnerability.
- **Benchmark evidence folder removed.** The 2.x measurement reports were tied to one development Mac and are
  not part of this repository. Generic benchmark tools live in `Tools/bench/`.

### Fixed

- **Qwen: any image request stopped the server.** Apple's graph compiler turned the vision attention into a built-in
  kernel that fails under 512 patches; it is now written so that does not happen, with the same arithmetic.
- **GLM and DeepSeek: silently wrong experts.** A read-ahead that cancelled itself could leave a slot marked as
  loaded with another expert's bytes, which then served later requests too. It is now dropped like a failed read.
- **GLM and DeepSeek: the hot copy is checked.** Each record carries a CRC32, checked on its first read; a bad one
  is read from the model file instead (counted as `rejected_records`). The copy is flushed to disk before use. Copies
  made by 2.x are read unchecked until rebuilt.
- **GLM and DeepSeek: a model file with wrong expert shapes** is refused at load with the tensor's name, instead of
  reading neighbouring memory.
- **GLM and DeepSeek: cancelling in the middle of a token lost the conversation.** That token now finishes and the
  reply stops after it, so the next turn continues the conversation.
- **Chat control markers typed into messages** stay text in every model's format, including DeepSeek's fullwidth
  markers.
- **The front** gives up on a model that has not started after an hour (it stops it and answers a 503), and a
  number JSON cannot hold no longer stops it.
- **Qwen:** a model file with an out-of-range date is refused instead of crashing; a request joining an expert read
  that had just finished no longer fails with "discarded before publication".
- **Checks:** on a machine without a usable GPU, only checks that used the GPU are skipped (every failure was).
  CI lists skipped checks, fails on unexplained ones, and also runs the Python and standalone Swift checks.

### Known limits

- Only one Mac has run Origami so far: an M2 Pro with 32 GB on macOS 26. Other Apple Silicon Macs are expected to
  work but are untested. Please send `origami report` output from yours.
- GLM-5.3-Flash and DeepSeek-V4.1-Flash are slow on a 32 GB Mac (about half a token per second when read from an
  external drive) and read text only.
- No signed binary yet: build from source.
- The plan for what comes next, and the rules for working on it, are in [ROADMAP.md](ROADMAP.md).
