# Origami 3.1: a guide for new users

This page walks you through Origami from the first start. Each section is short and links to the page with the
details.

1. [What Origami is, and which Macs it suits](#what-origami-is-and-which-macs-it-suits)
2. [Install](#install)
3. [Get a model](#get-a-model)
4. [First run](#first-run)
5. [The dashboard](#the-dashboard)
6. [Connect an app](#connect-an-app)
7. [Memory](#memory)
8. [The hot copy](#the-hot-copy)
9. [Saved prompts](#saved-prompts)
10. [Exact replies and fast modes](#exact-replies-and-fast-modes)
11. [Tuning](#tuning)
12. [Moving and refreshing a model](#moving-and-refreshing-a-model)
13. [Running in the background](#running-in-the-background)
14. [When something is wrong](#when-something-is-wrong)
15. [Planned for later](#planned-for-later)

## What Origami is, and which Macs it suits

Origami runs three very large language models on an Apple Silicon Mac that cannot hold them in memory:
**Qwen3.8-Flash-Next**, **GLM-5.3-Flash** and **DeepSeek-V4.1-Flash**. In these "mixture-of-experts" models each
token uses only a few experts out of hundreds. Origami keeps the always-used part in memory and reads each token's
experts from the SSD when they are needed.

It is one program with a local server. Apps talk to it through the OpenAI, Anthropic or Ollama API. Your prompts and
replies stay on your Mac; Origami goes online only to download a model you ask for, or to fetch an image a request
links to.

Which Macs:

- Apple Silicon, macOS 15 or later. Origami has run on three Macs so far: an M2 Pro with 32 GB on macOS 26, an
  M1 MacBook Air with 16 GB on macOS 27 and, at a user's, an M4 Max with 128 GB ([which others should work](../README.md#will-it-run-on-my-mac)).
- 24 GB of memory or more, 32 GB recommended. `origami setup` checks the Mac first: under 16 GB it refuses, and
  under 24 GB it warns that only an 8K context fits.
- A fast SSD with room for the model: about 110 GB for Qwen, 199.7 GB for GLM-5.3-Flash and 500.7 GB for
  DeepSeek-V4.1-Flash (their UD-Q4_K_XL files; [Get the files](models.md#get-the-files)). Plus room for the hot copy
  (about 18 GB for Qwen, 20–40 GB for GLM and DeepSeek to start) and saved prompts.

Speed depends mostly on the SSD and on how much memory the Mac has. Qwen is the fastest of the three; GLM and
DeepSeek are much larger and slow on a 32 GB Mac. Measured figures: the README's
[What to expect](../README.md#what-to-expect).

## Install

There is no prebuilt release yet. Build from source with Xcode or the Command Line Tools, version 26 or 27:

```sh
git clone https://github.com/janji/origami.git
cd origami
swift build -c release
sudo mkdir -p /usr/local/bin && sudo ln -sf "$PWD/.build/release/origami" /usr/local/bin/origami
```

The link puts `origami` on your `PATH`; a rebuild updates it. Keep the binary where it is built (it needs the
`swift-transformers_Hub.bundle` beside it).

## Get a model

- **Qwen3.8-Flash-Next** (the main, most-tested model; text and images): `origami setup` finds it on this Mac (the
  Hugging Face cache, LM Studio, SlotStream) or offers to download it (104 GB, resumable). Other Qwen builds and
  checkpoints: `origami model add <folder, file.gguf or owner/repo>` ([models.md](models.md#qwen38-flash-next)).
- **GLM-5.3-Flash** and **DeepSeek-V4.1-Flash** (text only, slow on 32 GB): download a GGUF with the Hugging Face
  CLI, then `origami model add <first .gguf file> --name glm`. The exact commands and sizes are in
  [Get the files](models.md#get-the-files).

To keep a download on another drive, use `origami model add <owner/repo> --to /Volumes/MySSD/models`.

`origami model list` shows your models, `origami model default <name>` picks the one `serve` uses, and
`origami model remove <name>` unlists one (its files stay). `origami inspect <folder or file>` says whether Origami
runs a model.

## First run

```sh
origami setup        # check this Mac, find or download Qwen, measure the drives
origami serve        # the server at http://127.0.0.1:11439/v1; leave it running
origami dashboard    # open the dashboard (in another terminal)
```

For GLM or DeepSeek: `origami serve --model glm` (a name from `model list`, or the path of the `.gguf`).

The port answers at once and the model loads in the background; requests sent meanwhile wait. `origami` alone shows
these steps and every command; `origami help <command>` shows a command's options.

`origami stop` unloads the model and gives its memory back (the dashboard stays up, apps get "Origami is stopped");
`origami start` loads it again. Two opt-in settings change that: `start_on_request true` lets a request start a
stopped model, and `idle_unload_minutes 30` unloads it after 30 idle minutes.

## The dashboard

`origami dashboard` opens `http://127.0.0.1:11439/dashboard`. The header always shows one state: **Ready**,
**Loading**, **Stopping**, **Stopped**, **Failed**, **Reload needed** (saved settings the model does not use yet),
**Job running**, or **Drive gone** (the model drive stopped answering; the same in `origami status`, `degraded` in `/health`). It also has
**Stop and unload** / **Start**. Four tabs:

- **Now:** live status; the last request and why it was slow, if it was (memory cut, a prompt read again, a drive it
  waited on); where experts came from; one health line each for memory, each drive, the hot copy, tuning and access.
  The memory line has **Measure the best limit** and **Measure by itself when idle**.
- **Setup:** context, quality, thinking and memory as dials, with this Mac's memory answering each move;
  **Tune for this Mac**; sampling. **Save** keeps a draft; **Save & reload** applies it.
- **Models:** load, add or remove models; add low-bit experts; where experts are read from, with the hot copy control;
  every file Origami knows about, to move or delete.
- **Connect:** the address and model name an app needs, who can reach Origami, and the log.

Details: [dashboard.md](dashboard.md).

## Connect an app

| App | Setting |
|---|---|
| OpenAI clients (Chat Completions and Responses) | base URL `http://127.0.0.1:11439/v1`, any key |
| Anthropic clients, Claude Code | `ANTHROPIC_BASE_URL=http://127.0.0.1:11439` |
| Ollama clients | `OLLAMA_HOST=127.0.0.1:11439` |

The model name can be anything: by default every request gets the model chosen in the dashboard. Tool calls, JSON
mode, images (Qwen) and thinking controls work on every API. One request runs at a time; others wait.

**Use streaming.** A long prompt can take minutes before the first word. A streamed reply sends a keepalive every
10 s, so the client knows the server is working. A non-streamed request whose client closes its sending side is
treated as gone and cancelled. Set client timeouts to 10 minutes or more.

Coding agents (Claude Code, Codex) need `origami config set context 32k` or more. Recipes for Claude Code, Codex, Pi,
llama-swap, API keys and other devices on your network: [integration.md](integration.md).

## Memory

Memory is automatic: with no `memory_limit` set, Origami sizes itself to this Mac and leaves room for your other apps.

- **Measured.** `origami tune --memory` (or **Measure the best limit** on the dashboard) reloads the running model at a few
  limits and suggests the highest one macOS kept without cuts or swap: about 4 minutes for Qwen, 6 for GLM, 10 for
  DeepSeek. It changes no setting, but automatic memory then starts from that limit until the measurement goes stale.
  `memory_auto_measure true` lets an idle server measure by itself (30 minutes without requests, on power, at most
  once a day).
- **A cut.** When macOS runs short of memory, Origami shrinks its expert cache instead of letting the Mac swap. Replies
  slow down but stay the same. The dashboard's memory line says "macOS took memory back … because …" and that it grows back by itself.
- **Recovery.** Once memory is free again, the budget climbs back by half the gap (at least 1 GB) every 40 s, up to
  the limit. Past where it learned to start (two weeks of this Mac's memory use) it keeps growing while memory stays
  free: up to 2 GB a step, a minute apart. A limit cut on three different days gets a suggestion of the level this Mac kept.

`origami config set memory_limit 24GB` sets your own ceiling. Details: [settings.md](settings.md#memory).

## The hot copy

The hot copy is a copy of the model's most-used experts, packed so each one reads in one piece, on a fast drive. It
is automatic and replies are the same with or without it.

- **It follows usage.** Origami counts which experts each model uses. The copy is built while the model serves and
  swaps its least-used experts for more-used ones between requests (at most 1 GB read from the model drive per hour).
- **Size.** Automatic grows the copy while it still pays and always leaves 20 GB of the drive free. In the dashboard
  (Models › Where experts are read from › **Change…**) a slider over a graph of the experts' popularity shows what
  each size covers; switch it between **GB** and **% of experts**. `hot_copy` sets it from the command line
  (`20GB`, `off`, or `auto`).
- **Several models on one drive.** The loaded model's copy grows by taking the least-used experts of the idle
  models' copies on that drive (each keeps what covers half its uses) and gives them back when you switch.
  **Keep when idle** (`hot_copy_keep`, per model) keeps a model's copy whole.
- **More drives.** Tick more drives in **Change…** (or set `hot_copy_more`): the fastest drive holds the most-used
  experts, the next drive the next ones, and each read goes to the drive expected to finish first. For Qwen,
  `origami store add <folder>` also keeps a full copy of the experts on another drive.

Details: [settings.md](settings.md#model-and-files).

## Saved prompts

After each reply Origami saves the processed prompt on disk, so the next message of a chat, a repeated system prompt
or a restart reads only what is new.

- **What is kept:** above an 8K context, 3 saved conversations by default (`prompt_cache_entries`, 3 to 8; 8K keeps
  8), plus 4 system prompts. Each takes up to 1 GB at 32K and 6.8 GB at 256K. Saves always leave 10 GB of the drive
  free, dropping the oldest first.
- **Keeping one:** the header `X-Origami-Keep: <name>` pins a conversation; the chat command `origami save` pins a
  chat's system prompt.
- **After an update:** saved prompts are kept when the new build computes the same bits. When saved prompts exist,
  its first start with a model reads a short fixed probe (for Qwen about 600 tokens; for GLM and DeepSeek 18) and
  compares its bits with the folder's (written while the server was idle after the first save); equal, it restores the old saves, else it says so in the log and starts its own. An update
  that changes the engine's arithmetic, and the first update to a version that keeps them, start empty once.
- **When it is removed:** prompts this build cannot read go after 14 days unused, or sooner when disk runs short.
  Pinned or named ones, and other models' prompts, are never removed automatically.
- **Cleaning up:** `origami cache` lists them; `origami cache prune` removes old builds' prompts and those of model
  files no model uses (with the server quit or stopped).

Details: [settings.md](settings.md#saved-prompts).

## Exact replies and fast modes

**Exact by default.** Origami uses every expert the model picks, so replies are the model's own. Every option that
changes replies is opt-in and labelled wherever the model is shown.

- **Quality levels** leave out experts a reply would wait for, so it writes faster: `faster` and `fastest` for Qwen,
  `faster` for GLM, `dynamic-fastest` for DeepSeek. Choose per request (`"origami_mode": "faster"`, or the model
  name `qwen3.8-flash-next-faster`), per chat (`origami quality faster`), or as the default
  (`origami config set quality faster`).
- **`allow_fast`** (default `true`) allows those levels. `origami config set allow_fast false` makes every request
  exact.
- **An exact request never reuses fast-mode state.** It never restores work a fast mode computed, so fast turns
  never leak into exact replies, also after a restart.
- **Low-bit experts** (`origami model add <experts.gguf> --base <model>`) make a faster variant that reads its
  experts from a published 2- to 3-bit file. Its replies differ.

Measured quality and speed: [models.md](models.md#quality-dial) and the README's
[What to expect](../README.md#what-to-expect).

## Tuning

`origami tune` (or **Tune for this Mac** in Setup) tries timing settings that leave replies unchanged on the running
server: the GPU keep-alive, and for GLM and DeepSeek the next-layer guesses, their depth and spinning; also how prompts
are grouped. It keeps one only if it was faster every time. It takes about 8 minutes, longer when the model needs it.

- A request from an app stops it. Every finished measurement is kept, so the next run within a day continues.
- `origami tune --recheck` checks the result still wins; `origami tune --undo` removes it.
- `origami tune --memory` finds the fastest memory limit ([Memory](#memory)).

Details: [dashboard.md](dashboard.md#tuning-for-this-mac).

## Moving and refreshing a model

Origami knows a model by its files. Moving them by hand (Finder, `mv`) makes it a new model: its hot copy, usage
counts, tuning and saved prompts are no longer found. Move it with Origami instead, and they follow:

- **Move:** the dashboard's **Move…**, or with the server quit `origami model move <name> <folder>`. Across drives
  each file is copied, read back and compared before the original goes.
- **An interrupted move:** `origami move list` shows it; `origami move recover` (or **Recover…**) finishes it or
  puts the original back.
- **A renamed drive** (or one mounted as `/Volumes/<name> 1`): `origami doctor` names the new place, and
  `origami relink` (server quit) uses it.
- **A drive that slows down.** Some SSDs read often-read parts of a file at a fraction of their speed. The dashboard's
  drive line and `origami doctor` say so, and warn ahead of the next slowdown. Quit `origami serve`, then run
  `origami refresh`: it rewrites the model's files in place, byte for byte and checked, and keeps everything keyed on
  the model. It is safe to stop and run again. `origami refresh --copy` does the same for the hot copy when its
  drive is the slow one.

Details: [troubleshooting.md](troubleshooting.md#the-model-drive-got-slow).

## Running in the background

```sh
origami service install     # start `origami serve` now and at each login (a LaunchAgent)
origami service remove      # stop it and remove it
```

The service runs this binary with the `ORIGAMI_` variables of your shell and logs to `~/Library/Logs/Origami`. If
the model or a store is on an external drive, macOS may ask once to allow **Removable Volumes** for `origami` (System
Settings › Privacy & Security › Files and Folders); allow it, and again after a rebuild. Details:
[integration.md](integration.md#apps-and-harnesses).

## When something is wrong

| Command | What it does |
|---|---|
| `origami status` | the server, model, memory, drives, hot copy and saved prompts |
| `origami doctor` | checks everything without loading the model and says what to fix |
| `origami verify` | checks the expert copies against their checksums |
| `origami bench` | times a short question, a story and a long document |
| `origami report` | this Mac's details and a benchmark, to paste into a GitHub issue |
| `origami uninstall` | deletes Origami's copies, caches and settings, never your models |

Common problems and their fixes: [troubleshooting.md](troubleshooting.md). Every setting: [settings.md](settings.md).

## Planned for later

These are ideas waiting for a measurement, not promises. The full list is in [ROADMAP.md](../ROADMAP.md) section 7.

- Testing on other Macs: 16 and 24 GB under real load, M1, M5, 64 GB or more.
- Mixing exact and low-bit experts in one reply, and speculative decoding, where they pay.
- Deciding the four remaining experiment switches, such as split reads across two fast drives
  ([switches.md](switches.md)).
- Smaller speed-ups that still need measuring: kernel fusions, GPU keep-alive power use.

Also planned: a signed binary and a Homebrew tap, and built-in TLS
(parked in [ROADMAP.md](../ROADMAP.md) section 7, each with what reopens it).
