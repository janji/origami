# Settings

Nothing needs configuring: Origami fits itself to the Mac it runs on. This page lists every setting, explains the
memory, context and drive trade-offs, and says where Origami keeps its files.

Measurements on this page come from one Mac: an M2 Pro with 32 GB of memory on macOS 26, mostly with
Qwen3.8-Flash-Next.

## Changing a setting

```sh
origami config                        # every setting, its value and where the value comes from
origami config get context
origami config set context 32k
origami config unset context          # back to the default
```

The dashboard's **Setup** tab changes the same settings, and **Connect** the access ones ([dashboard.md](dashboard.md)).

`config set` says when a change applies to a running server: at once (`memory_auto_measure`, `lan_key`,
`models_dirs`, `hot_copy_keep`), when `origami serve` starts again (`port`, `listen`, `data_dir`, `trusted_proxy`), or,
for every other setting, when the model next loads (the dashboard's Reload, or `origami stop` then `origami start`;
the dashboard says "reload required" for each of them). It also says when the value was saved for one model, and
when an option or variable, here or the one the running server was started with, overrides it.

Most settings can also be given as a flag (not `api_key`, `lan_key`, `trusted_proxy`, the sampling settings,
`memory_auto_measure`, `start_on_request`, `idle_unload_minutes`, `hot_copy_keep` or `stopped`), and every one as an environment variable:

- as a flag: `--context 32k`, `--context-dir …` (underscores become dashes);
- as an environment variable: `ORIGAMI_` plus the name in capitals, such as `ORIGAMI_CONTEXT=32k`. Stores are
  `ORIGAMI_STORES=/Volumes/A/Origami,/Volumes/B/Origami`. See [switches.md](switches.md).

A flag wins over a variable, a variable over the settings file. The settings file is
`~/Library/Application Support/Origami/config.json`; `--config <file>` or `ORIGAMI_CONFIG` names another, which is
how two llama-swap entries keep separate settings. A flag or variable that sets something keeps its dashboard
control disabled.

Some settings are kept per model when the model has a name (`origami model add … --name`): `context`, `quality`,
`reasoning_effort`, `memory_limit`, `hot_copy`, `hot_copy_dir`, `hot_copy_more`, `hot_copy_keep`, the five quality knobs and the sampling settings. A named
model's own value beats the settings file; a flag or variable beats both.

`origami <command> --help` shows a command's options without running it. Exit status: 0 success, 1 failure,
2 usage error. Without a terminal (launchd, llama-swap), Origami never asks questions: commands that would ask fail
unless `--yes` is given.

## All settings

### Model and files

| Setting | Default | What it does |
|---|---|---|
| `model` | found automatically | model folder, GGUF, Hugging Face repo id, or a name from `origami model list` |
| `data_dir` | `~/Library/Application Support/Origami` | settings, usage counts, hot copy, saved prompts and downloads |
| `stores` | none | extra folders, on any drive, where expert copies may be written (see [Drives and stores](#drives-and-stores)); the data folder is always used too |
| `context_dir` | `<data_dir>/prompt-cache`; this Mac's own data folder when `data_dir` is on an exFAT or FAT drive | where a context kept on disk and the saved prompts go; the fastest drive with room is best |
| `hot_copy` | automatic | copy of the most-used experts, as a size (`20GB`) or `off`; presets `save-space` (off), `balanced` (automatic), `fastest` (as large as the drive allows). Automatic starts at the old size (GLM and DeepSeek: 20–40 GB; Qwen: its shipped list, about 18 GB) and, once replies are counted, grows while it still pays: each further tenth of the model's expert bytes must add at least 2% of the expert uses counted. It never leaves less than 20 GB free on the copy's drive and never exceeds the model's experts; a copy already built never shrinks because the counts change or free space dips (falling space only stops it growing); it follows usage by swapping experts in place. On a Mac that keeps every expert in memory once it is read (Qwen from 128 GB) the automatic copy is not built: each expert is read once after a start, so a copy would only shorten those first minutes; a size you set is still built. A size you lower by 10% (at least 1 GB) is written beside the copy in use and attached once complete. A lower size with no room for a compact copy beside the current one shrinks that copy in place (its least-used experts are given back to the drive); `off` only stops using the copy and keeps its files. The dashboard's Models tab (Where experts are read from › Change…) shows what each size would cover and why the automatic size is what it is |
| `hot_copy_dir` | automatic | folder for that copy, on the fastest drive other than the pack's (Qwen: a store on the model drive; GLM and DeepSeek: the data folder) |
| `hot_copy_more` | none | more places for the hot copy, one per drive: `folder=size` pairs, comma-separated, such as `/Volumes/Fast/Origami/hot-copy=200GB` (a size as `hot_copy` takes it; `auto` when none), or `off`. Stored as a JSON list with each size in bytes. All places share one ranking: the fastest measured drive holds the most-used experts, the next drive the following ones, no expert twice until a place has room left after every expert no faster place holds, which then holds the most-used again (dropped first when its size shrinks). Each keeps 20 GB of its drive free; a second place on one drive is refused. A place whose drive is not connected, not permitted or full is skipped with its reason and used again when it is back. Adding, removing or moving a place applies when the model restarts; a size at once. `origami config set hot_copy_more … --yes` (or `unset … --yes`) also deletes the copy in each place it drops; without `--yes` the copy is kept |
| `hot_copy_keep` | false | with a `hot_copy` size set, this model's copy keeps that size while another model is loaded. Off, the hot copies of models not loaded share their drive with the loaded model's: its copy grows to its size (or its automatic size) by taking their least-used experts, the model used longest ago first, each keeping at least the experts that cover half of its own uses; a copy another running server has open is never touched. The copy grows back when its model is loaded again. The dashboard's Change… has it as a checkbox |
| `models_dirs` | none | more folders the dashboard's model search looks in, comma-separated (the usual download places are always searched) |

### Memory and context

| Setting | Default | What it does |
|---|---|---|
| `context` | `8k` | context length in tokens, up to 262144: `8k`, `32k`, `100000`, `256k`… Coding agents need `32k` or more |
| `memory_limit` | fitted to this Mac | Origami's total memory ceiling, such as `20GB` (see [Memory](#memory)) |
| `memory_auto_measure` | `false` | without `memory_limit`: the server measures the fastest limit by itself while idle (see [Memory](#memory)) |
| `start_on_request` | `false` | after `origami stop` or an idle unload, a request starts the model instead of getting 503 `stopped` ([integration.md](integration.md#stop-and-start)) |
| `idle_unload_minutes` | off | unload the model after this many minutes without a request, as `origami stop` does (saved as stopped) |
| `expert_memory` | what `memory_limit` leaves | memory for cached experts, such as `12GB`; less leaves more to other apps and makes replies slower |
| `context_on_disk` | `auto` | GLM and DeepSeek: context in files on the context drive (`true`), in memory (`false`), or on disk when it would take more than a quarter of Origami's memory (`auto`) |
| `prompt_cache_entries` | 3 | saved prompts and conversations kept on disk above an 8K context, 3 to 8 (8K always keeps 8) |
| `gpu_keepalive` | `auto` (on) | A tiny spin command keeps the GPU clocked up while a reply is made, also while GLM and DeepSeek wait for expert reads (auto leaves it off for those two in Low Power Mode). Replies are the same either way; if replies are unusually slow on your Mac, try `false` (another engine's keep-alive made decode about 19 times slower on a MacBook Air M5 and a Mac Studio M1 Max) |

### Replies

| Setting | Default | What it does |
|---|---|---|
| `quality` | `exact` | quality dial for requests that choose none: `exact` or one of the fast modes the served model offers (Qwen: `faster`, `fastest`; GLM: `faster`; DeepSeek: `dynamic-fastest`; see [models.md](models.md#quality-dial)); setting it resets the five knobs |
| `experts_per_token`, `tail_skip`, `tail_skip_misses_only`, `late_miss`, `confident_above` | the level's | the quality dial's knobs (see [models.md](models.md#knobs)) |
| `allow_fast` | `true` | allow levels other than exact |
| `reasoning_effort` | `none` | thinking for requests that ask for none |
| `max_thinking_tokens` | none | close the think block after this many tokens when a request sets no budget |
| `sampling` | `recommended` | `recommended` (the model's own values) or `custom` (the five values below on top of them); kept per model |
| `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty` | the model's | custom sampling values |
| `sampling_override` | `false` | use these values even when an app sends its own |
| `speculation` | `off` | Qwen only: MTP speculative decoding, a safetensors file with the model's MTP head (see [models.md](models.md#mtp-speculation-opt-in)) |

**Sampling.** A request that sends no `temperature`, `top_p`, `top_k`, `min_p` or `presence_penalty` gets the
model's recommendation: its `generation_config.json`, or its GGUF's `general.sampling.*`, else Qwen's
(temperature 0.7, top_p 0.8, presence_penalty 1.5 without thinking; 1.0 and 0.95 with thinking; 0.2 and 0.9 with
tools; top_k 20). Qwen's recommendation differs between thinking and not, so a single set from its files is the
thinking one. `/v1/status` and `/v1/models` show the defaults per mode. A request's `top_p: 0` means greedy. There
is no repetition penalty, so `repetition_penalty` in a model's files is not read.

### Server and access

| Setting | Default | What it does |
|---|---|---|
| `port` | 11439 | HTTP port |
| `listen` | `local` | `local`: this Mac only (127.0.0.1); `lan`: also other devices on the network (see [integration.md](integration.md#other-devices-on-your-network)) |
| `api_key` | off | a key every request must carry, 16 to 256 characters (see [integration.md](integration.md#security)) |
| `lan_key` | made when `listen lan` is set | the key other devices send when no `api_key` is set; `off` lets every device in without one (less safe). It travels in clear over plain HTTP: for TLS, see [integration.md](integration.md#tls) |
| `trusted_proxy` | `off` | `loopback`: a TLS proxy on this Mac forwards other devices' requests, and `X-Forwarded-For` from 127.0.0.1 names each client, which then needs `api_key` (see [integration.md](integration.md#tls)); `off`: that header is ignored |
| `list_models` | `false` | list every model and switch between them by name, instead of one model called `origami` |
| `commands` | `true` | answer `origami <verb>` chat messages |
| `status_messages` | `false` | show progress of long prompts as thinking lines in streamed replies |
| `request_log` | off | a folder for one JSON line per request (see [integration.md](integration.md#request-records)) |
| `stopped` | `false` | set by `origami stop` and `origami start`: while true, the server keeps the dashboard up but loads no model |

`HF_TOKEN` is used for gated Hugging Face downloads.

## Memory, context and drives: the trade-off

Three things share a Mac's memory and drives:

- the experts: which file they come from (the model's own, or published low-bit ones: a quality choice) and how many
  stay in memory (a speed choice);
- the context: how long, and how much of it stays in memory;
- the drives the model is read from.

The dashboard's **Setup** tab shows these as dials, with this Mac's memory answering each move, and keeps them with
**Save** (a draft) or **Save & reload** (loads them now).

| Setting | What it trades |
|---|---|
| `memory_limit` | Origami's memory against other apps'. |
| `expert_memory` | The experts' share of that memory. About a third of a reply's time waits for experts read from a drive, so less makes replies slower. |
| `context`, `context_dir` | Context length against speed and disk: above what this Mac keeps in memory, the context goes to a file in `context_dir`, so put that on the fastest drive with room. |
| `hot_copy` | Disk space for faster first reads of the most-used experts; `off` frees that space. |
| `stores` | Drives holding copies of the experts; Origami reads them all at once, each read going to the least busy drive that holds the expert. |
| `quality` | Replies for decode speed (see [models.md](models.md#quality-dial)). |

### Memory

Origami sizes itself from the memory that is free when it starts, leaves room for your other apps, and gives memory
back when macOS runs short: it shrinks its expert cache instead of letting the Mac swap, and grows again when memory
frees up, one step every few minutes once the squeeze has gone.

Without `memory_limit`, a limit measured by `origami tune --memory`
([dashboard.md](dashboard.md#the-fastest-memory-limit)) for this model and context on this Mac is used
as the ceiling and start, until it goes stale: cut on three different days since it was measured (under it or a lower limit), another memory size, or over
90 days old. The startup log, `origami status` and the dashboard say which is in force ("memory: automatic, using the
measured 22.5 GB", "memory: your limit 24.0 GB (ceiling)", or "not measured: run `origami tune --memory`").

With `origami config set memory_auto_measure true` (or **Measure when idle** on the dashboard's Memory line), a
running server does that measurement itself when no limit is measured for the loaded model and context, or the one
measured went stale or kept none. It starts only when all hold: no `memory_limit` is set; no request for 30 minutes;
no other dashboard job runs; the Mac is on power and not in low power mode; and no automatic try in the last day,
whatever its end. Each model and context has its own try and its own day of back-off, and it measures the model
that is loaded, whichever that is. A request from an app that arrives meanwhile (from the dashboard's own search or
`origami tune --memory` too) ends it before it is admitted: it waits for the measurement reply under way, the model
reloads at your saved limit, and only then is the request answered, so it is never served at a test limit and no
reload lands on it afterwards; the search then stops. The result is stored where `origami tune --memory` stores it,
so the reload the run ends with already serves with it. The log, `origami status`, `/v1/status`
(`memory_learned.auto_measure`) and the dashboard's Memory line say what happened in one line ("memory: measured
automatically while idle: 24.0 GB", "automatic measurement stopped: a request from an app arrived; will try again
tomorrow", and while its closing reload runs "automatic measurement cancelled; restoring your saved limit"). While a
measurement is due but cannot start, the line says why and since when ("memory: not measured yet: the Mac is on
battery since 14:02"; also low power mode, a request within 30 minutes, another job, the model busy or loading, or
"tried at 03:10; tries again after 03:10 tomorrow"). It never changes a setting.

An explicit `memory_limit` is the ceiling for the process: model state, cached experts and temporary memory,
within the machine's and the GPU's ceilings. Origami starts there and uses what the Mac can keep.

- Real memory pressure, fresh swapping or going over the target lowers the cache allowance. After such a cut it grows
  back to the level that held, then by half the remaining gap (at least 1 GB) every 40 s while there is no pressure
  and free memory stays 2 GB above macOS's share, up to the limit, so another app's squeeze does not cap it for long.
  The next start begins at the limit again: earlier cuts never lower it.
- A limit cut on three different days or more in the last 14 days (one bad evening does not count, however many cuts;
  only cuts under this limit or a lower one) gets a suggestion, in `origami status`, `/v1/status` (`memory_learned.limit_advice`)
  and the dashboard: the most this Mac kept, and the exact `origami config set memory_limit` to use it (or `unset` to let
  Origami choose). Origami never changes the setting itself.
- The maximum is derived from this Mac's RAM, leaving at least 3 GiB or one eighth for macOS.
- The dashboard's **Memory limit** dial (Setup) saves the target and, when needed, raises macOS's GPU memory ceiling through
  an administrator prompt. That ceiling resets on reboot; the saved limit stays and the dashboard can restore it.
  `origami config` changes never ask for administrator access.
- A loaded model needs a reload to use a changed limit.

What memory buys (Qwen 4-bit). Only 32 GB has been run; the other rows come from Origami's memory plan:

| Memory | Experts in memory |
|---|---|
| 16 GB | not recommended: the always-resident part needs about 11 GB, so only an 8K context fits |
| 24 GB | expected to work; fewer experts in memory than at 32 GB |
| 32 GB | tested; experts not in the cache are read from the drive as each token needs them |
| 64 GB | about 60% of the experts stay in memory, so fewer reads and faster decode |
| 96 GB | every expert stays in memory while writing, nearly all while reading a long prompt |
| 128 GB or more | every expert stays in memory once read; no expert copies needed |

### Context: what stays in memory

`context` can be any length up to 262,144 tokens; a longer request is refused in the client's own wording. Origami
keeps the context's key/value cache in memory up to the size `origami setup`, `origami status` and the dashboard
show for this Mac. With 32 GB and default settings, all of 256K fits in memory. On a smaller Mac, more memory for
Origami or low-bit experts leave more room.

A context that does not fit in memory is kept in a file on disk instead, and Origami says so when it starts.

Longer contexts write more slowly, and a context kept in a file is much slower than one in memory; reading a full
256K prompt takes over an hour. The README's [What to expect](../README.md#what-to-expect) gives the measured speeds
at the default context.

About the file:

- It needs about 37 GB free on the context drive; a full 256K conversation used up to 52 GB while the server ran.
- Follow-ups reuse a read prompt in seconds.
- Each reply's saved conversation is up to 6.8 GB at 256K; it is written after the reply (40–115 s for a full one;
  a save now clones the previous image and writes only what changed). A message sent sooner waits for it.

### Saved prompts

Above 8K, Origami keeps three saved prompts and conversations on disk by default (`prompt_cache_entries` 3 to 8;
8K always keeps eight), plus four system prompts. Each takes up to 1.0 GB at 32K, 3.5 GB at 128K and 6.8 GB at 256K
for a full context (shorter prompts take less). So three take up to 2.9, 10.4 and 20.4 GB, and eight up to 7.8, 27.7
and 54.3 GB. Saves always leave 10 GB of the drive free, dropping the oldest entries first. `origami doctor` shows
the figure for your context.

Saved prompts are kept across an update when the new build computes the same bits: their folder is keyed on the
engine's arithmetic, not on its build, and a new build checks a short fixed probe against the folder's before it reads
a save (the log says how long it took, or that the bits differ and it starts its own folder). An update that changes the
arithmetic, and the first update to a version that keeps them, start empty once.
`origami cache` lists saved prompts per build and `origami cache prune` removes other builds' ones. A start removes
saved prompts this build cannot read (another arithmetic, a canary that differed, or an earlier version that keyed them on its build) of the same
model and context once unused for 14 days, and a save short of disk removes them first; another model's saved prompts, and a folder holding a kept session or a saved system prompt, are never removed
automatically (a model file no model has any more: an earlier Origami version's identity, a replaced or rewritten model
file, goes with `origami cache prune`); the storage map and `origami cache` list those as
their own items, safe to delete. GLM's and DeepSeek's saved prompts are listed by the model, context and version each
folder was made for (`origami-epoch.json` in it, written at start); a folder of another build or of a model no longer
added is its own item, and `origami cache prune` removes it. Folders an earlier build made have no such file and stay with
the current ones.

One server owns a saved-prompt folder: a second server started on the same `context_dir` (same model, context and
build) saves and reuses no prompts, keeps no usage counts (only the first one writes them) and says so at start; give
each server its own `context_dir`. If the list of
saved prompts cannot be read (a crash while it was written), it is kept as `index.json.unreadable` and rebuilt from
the saved files, without their names and pins.

### Drives and stores

A store is any folder, on any drive, where Origami may keep copies of Qwen's experts (GLM and DeepSeek use more hot
copy places instead: `hot_copy_more`):

```sh
origami store add /Volumes/MySSD/Origami --size 150GB
origami store list
origami store remove /Volumes/MySSD/Origami
```

- `origami setup --store <folder>` adds a store and builds there first; `--store` on `serve` replaces the configured
  stores for that run.
- With one second drive, setup writes a packed copy of all experts there (100 GB for Qwen) and measures both drives,
  so each read goes to the drive that will finish first. With more drives, each gets the whole pack where it fits
  (`--size` caps it), otherwise the most-used experts.
- Each drive gets one copy. To rebuild one at another size, remove its folder and run `origami setup`.
- A store that is unplugged is skipped: reads go to the other copies, and finally to the model files. If it
  disappears while the server runs, it is used again as soon as it is back; one missing at start is used after the
  next start. Copying or moving a store keeps it valid. Moving the model folder by hand (Finder, `mv`) changes the
  model's identity, so its copies, usage counts, tuning and saved prompts are no longer found; move it with the
  dashboard's **Move…**, or with the server quit `origami model move <name> <folder>` (or point Origami at it with
  **Locate…**), which carries them to the new identity. A move a stop interrupted is listed by `origami move list` and
  finished or undone by `origami move recover` (the dashboard's **Recover…**); if the model cannot be read when a move
  ends, its hot copy and counts wait, listed as "belongs to <model> before its move", until `origami move recover`
  carries them over.
- A drive renamed or mounted under another name (`/Volumes/SSD 1`): Origami notes each drive's volume UUID while it
  is there, and `origami doctor`, `origami model list` and the dashboard's **Models** tab then name the folders found on
  it at their new place. `origami relink` shows them and, once confirmed, uses them: models, stores, hot copy places
  and `context_dir`, with what is keyed on each model carried over as a move does. Nothing is relinked without asking.
- A conversion stopped part way (`<name>.partial`) is listed with its size by `origami store list` and the **Models**
  view. A Qwen conversion resumes from its finished files when the same `origami model add` runs again; a GLM one
  cannot resume: Ctrl-C deletes it.
- Setup reuses compatible copies. Turning the hot copy off keeps its files; delete an unwanted copy in the
  dashboard's **Models** tab.

On the tested Mac a second drive added little, because its internal SSD is fast and the hot copy holds the most-used
experts. It should help more on a Mac whose internal SSD is slower or fuller.

## Where Origami keeps its files

| What | Where |
|---|---|
| Settings | `~/Library/Application Support/Origami/config.json` (or `--config`, `ORIGAMI_CONFIG`) |
| Data folder: named models (`models.json`), usage counts, tuning results, downloads | `~/Library/Application Support/Origami` (`data_dir` moves it) |
| Downloads and conversions | `<data_dir>/models` (`origami model add … --to <folder>` writes elsewhere) |
| Saved prompts, and a context kept on disk | `<data_dir>/prompt-cache`, or `~/Library/Application Support/Origami/prompt-cache` when `data_dir` is on an exFAT or FAT drive (`context_dir` moves it) |
| Expert copies | `<data_dir>/store`, your stores, and `hot_copy_dir` |
| Logs | wherever the launcher sends the output of `origami serve`; the last 50 lines are in `/v1/status` |

Saved prompts need a folder only you can write. On an exFAT or FAT drive every user of the Mac can read and write
every file (the drive ignores ownership, though its folders show as yours and private), so saved prompts are off
there, and so on a folder that is not yours or that others can write; Origami then serves without saving or reusing
prompts (every follow-up reads its whole conversation again) and says so when it starts. So with `data_dir` on such a
drive, saved prompts default to this Mac's own data folder, and `origami setup` says where they go. There is no setting
to save them on such a drive anyway: if you set `context_dir` yourself, use a folder of yours on APFS or Mac OS
Extended (`origami doctor` says so when it is not). An external APFS or Mac OS Extended drive is mounted with "Ignore ownership on this volume" on by
default: prompts are saved there as before, and the start says once that every user of the Mac can read and change
them until that box is turned off in the drive's Get Info window.

**Drive formats.** APFS (also case-sensitive) and exFAT were checked on macOS 26 with a model moved there and back,
stores and the file features Origami uses; Qwen was downloaded to, served from and given a hot copy on an exFAT drive
on macOS 27. What differs on
exFAT: no sparse files (a saved prompt and a context kept on disk take their
full size at once), no clones (each save writes a whole image instead of the changed pages), and no freeing part of a
file (the hot copy gives no room back for saves there). Files with extended attributes get `._` companions. Origami
works on it; APFS is faster for saves. Mac OS Extended (HFS+) is not checked yet.

The dashboard's **Models** tab shows every file Origami knows about and can move or delete them safely;
see [dashboard.md](dashboard.md#models-drives-and-files). `origami uninstall` deletes Origami's stores, caches and settings,
never the model.

## Which settings change replies

In exact mode at temperature 0 a reply is deterministic. **BITWISE** means byte-for-byte the same replies;
**NUMERIC** means replies may differ slightly.

| Setting | Replies |
|---|---|
| `memory_limit`, `expert_memory`, `hot_copy`, stores and drives, `context_dir`, `port`, `status_messages`, `commands`, `gpu_keepalive`, `allow_fast` | **BITWISE**, as long as the context stays in memory or stays in a file. Above 32K, a memory change that moves the context between the two changes how prompts are read: **NUMERIC** |
| `context` | **NUMERIC** between tiers (8K, 32K, 64K, 128K, 256K): each tier reads prompts in its own layout |
| `context_on_disk` | **BITWISE**: the same bytes, kept in files |
| `quality` other than `exact`, `experts_per_token` below 10, `tail_skip`, `late_miss` `skip` or `substitute`, `confident_above` with other knobs | **NUMERIC**: fewer or different experts per token; with `late_miss`, replies can also differ between runs |
| `quality exact`, `experts_per_token all`, `tail_skip off`, `late_miss wait`, `confident_above` alone | **BITWISE** |
| `speculation` (MTP) | **NUMERIC**: the two-token step's arithmetic differs |
| Prompt cache reuse (a saved prompt, a chat's next turn) | **NUMERIC**: the restored part was computed in other steps than a cold run, which can round differently. Replies agree until a near tie (in a test, 1 of 6 next turns was identical to a cold run and the first token matched in all 6) |
| Tuned `qwen_prefill_scope` | **NUMERIC**: prompt groups can use different grouped arithmetic; saved prompts are kept apart |
| Tuned `family_prefill_slices`, mixed-format pool sizing | **BITWISE** in exact mode: the same arithmetic in the same order |
| `ORIGAMI_NO_BFLOAT_MATRICES` | **BITWISE** on M1–M4; on M5 it also turns off the matrix-unit prompt kernel: **NUMERIC** |
| `reasoning_effort`, `max_thinking_tokens`, a request's thinking fields | change what is asked, so replies differ by design |
| `sampling`, `temperature`, `top_p`, `top_k`, `min_p` | **BITWISE** at a request's temperature 0 (greedy), unless `sampling_override` replaces it |
| `presence_penalty`, `sampling_override` | change the reply at temperature 0 too: the penalty applies before the greedy pick, and the override replaces the request's values |
