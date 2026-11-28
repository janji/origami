# Troubleshooting

## Tools

| Command | What it does |
|---|---|
| `origami status` | the model, the Mac, the memory plan, every store, the hot copy (and the share of expert uses other sizes would hold) and the server |
| `origami doctor` | checks settings, model, stores, hot copy, drive speeds, memory and disk without loading the model, and says what to fix; lines from a running server only when it serves the model asked about |
| `origami inspect <folder or file>` | what a model folder or GGUF holds (weights, sizes, quantization) and whether Origami runs it; `--family` for GLM and DeepSeek |
| `origami bench` | times a short question, a story and a long document against the running server |
| `origami verify` | re-reads the expert copies and checks every expert's checksum (`--deep` also compares them with the model) |
| `origami refresh` | rewrites the model's files in place when their drive reads them far below its own speed (below); `--copy` does the same for its hot copy |
| `origami report` | prints the Mac, the memory plan, the stores and a short benchmark, ready to paste into a GitHub issue (home-folder paths shortened to `~`) |

The server's log goes to the terminal or launcher that runs `origami serve`; the last 50 lines are also in
`/v1/status` (`events`).

## Why is it slow?

Look at the top of the dashboard's Now tab: while something slows Origami down, a **Slowed by** notice says what, one
line each, the costliest first, and what to do (or that it recovers by itself, and when). `origami status` prints the
same as "Slowed by" lines, and apps read it as `slowed_by` in `/v1/status` and `/health`. It covers memory below its
target (with the apps that hold the most), a drive that reads slowly or is gone or not allowed, the hot copy (building,
no room, failed), heat and Low Power Mode, conversation saves paused, a prompt read again, a job or waiting requests,
and, last, settings this Mac measured could be better. Memory under what Origami may use right after a start is
usually the start it learned from this Mac's memory use: it grows past it by itself, a step of up to 2 GB a minute,
while memory stays free; quitting the apps the notice names makes room. Under the last request's bar, a line ("Slower than usual: …") says the same
for that request alone.

## Common problems

| Symptom | What to do |
|---|---|
| The server hangs at start when run from launchd, llama-swap or another app | macOS is asking, invisibly, for permission to read a removable volume. The dashboard says which drive. Allow **Removable Volumes** for `origami` (or for your launcher) in System Settings → Privacy & Security → Files & Folders, or run `origami setup` once in Terminal and allow access. Origami gives up on a store after 10 s and serves without it, and does not start a model on a drive it cannot read (it gives a sleeping drive 30 s to wake). A grant given while `origami` runs in Terminal covers only Terminal runs. For launchd or another launcher, allow `origami` itself, and again after replacing or rebuilding the binary. |
| The first answer to a long prompt takes a minute or more | Normal when experts are read from the drive. Later turns reuse saved prompts; a second drive and the hot copy help most. `origami bench` shows this Mac's speed. |
| GLM or DeepSeek are very slow | On a 32 GB Mac most of their experts come from the drive: 1 to 2 tokens a second on the tested Mac (the README's [What to expect](../README.md#what-to-expect)). Much less means the drive is slow or has slowed down on the model's files ([below](#the-model-drive-got-slow)); a faster drive for the model or its hot copy helps. |
| A client gives up while Origami is still reading | Use streaming and set the client's timeout to 10 minutes or more; streams send a keepalive every 10 s, and a non-streamed request whose client closes its sending side is treated as gone. At 256K a cold prompt can take an hour: set the timeout to match. A request queued behind it can hit the 3,600 s queue limit. |
| A coding agent's first request fails or is refused as too long | The default context is 8K. Claude Code and Codex need `origami config set context 32k` or more. |
| "A context longer than this Mac keeps in memory goes to a file, which needs … free" | The key/value file and saved prompts live in `context_dir` (default: the data folder). Free about 37 GB there (a long conversation uses more while the server runs, 52 GB in a full-context test), or move `context_dir` or `data_dir` to a bigger drive. |
| "macOS interrupted Origami's GPU work (the display or another app needed it); please retry your request" | macOS stopped Origami's GPU work three times in one request, usually while Screen Sharing, a video or a busy browser used the GPU. Origami already retried twice from its last saved state. Retry the request (Pi and the OpenAI SDKs do so on this 503). Fewer GPU-heavy apps during long prompts help. |
| Replies stop mid-sentence | `max_tokens` was reached. With thinking on, Origami keeps room for the answer by closing the think block. |
| After an update, the first follow-up is slow once | The update changed the engine's arithmetic (or was the first to a version that keeps saved prompts), so its saved prompts cannot be used and the first long prompt is read again. The request's `prompt_restore` reason says so (`saved_by_other_build`). An update with the same arithmetic keeps them: its first start says "saved prompts: this build was checked against the folder's canary in N s". |
| "saved prompts: … have no canary yet …, so this build cannot check them and saves its own" | The build that saved them had not yet been idle for two minutes after its first save, when it writes the canary. The new build starts its own saves; nothing is lost but those prompts' reuse. |
| "saved prompts: WARNING: this build computes different bits …" | The new build's probe gave other bits than the build that saved the prompts, though their sources match (another compiler or macOS can do this). Nothing is restored from that folder; this build saves in its own. Report it with both builds' versions. |
| "Conversation saves are paused: N GB free, and a save keeps 10.0 GB free plus its image" | The drive with `context_dir` is nearly full, so replies are not saved and the next long prompt may be read again from the start. Free disk space there, or lower `context` (smaller images). Saves resume by themselves; the server log says when. |
| "Copy on … not built: it needs … plus 10.0 GB kept free, and … has … free" | Not enough room on that drive for a further copy. The message says how much copies from an earlier Origami version hold there; delete them in the dashboard's **Models** tab, free other space, or choose another drive. The model is not reloaded. |
| The dashboard says the hot copy is not possible: "… has … free; the copy needs … and keeps 20 GB free" | Not enough room for the hot copy on that drive. Free space there or choose another drive on the card; it is built while the model serves. |
| Saved prompts of an earlier build of a model are gone | Saved prompts are kept by builds that compute the same bits. A folder of the same model this build cannot read, unused for 14 days, is removed, and one may be when a save is short of disk; never one holding a pinned or named conversation. Another model's saved prompts are never removed this way: `origami cache prune` removes them (Qwen, GLM and DeepSeek folders, and a GLM or DeepSeek context left on disk), listing each with its size first. |
| Old saved prompts take space (after upgrades, a model replaced) | Saved prompts no build can read any more stay on disk: older builds' folders, those of a model file no model uses, a GLM or DeepSeek context left on disk. The dashboard's **Models** view lists them under **Drives and files**, Saved prompts, per model with their size and age (the storage total says so above 1 GB); **Remove unusable…** lists what goes with sizes and removes them, also while a model is loaded. `origami cache` marks each folder "in use", "usable by this build" or "unusable: <reason>", and `origami cache prune` removes the unusable ones the same way. A folder a running server holds (its `.lock`) is never removed; one with pinned or named conversations goes only when ticked (`--yes`). `origami cache prune --all` also removes this build's, with the model stopped (`origami stop`). |
| The dashboard says "The model will reload after the current reply" | A reload (new settings or a new hot copy) waits for the reply being written; the model keeps serving until then. |
| "too little memory" | Close large apps, lower `context` to 8k, or set `memory_limit`. |
| "the model does not fit in the memory limit" | The limit leaves too little for the model to start (GLM needs about 21 GB). Raise it up to the maximum the message names for this Mac, or unset it and let `origami tune --memory` measure one. |
| "this model does not fit this Mac at this context" | Even the largest limit this Mac allows is less than the model needs at this context. Use a smaller `context`, a model that needs less memory (Qwen needs the least), or a Mac with more memory. Qwen says the same when even this Mac's largest limit cannot start it. |
| "Port 11439 is already in use" | Another program, often another `origami serve`, listens there. `origami status` shows whether it is Origami; quit that one, or start this one with `--port`. |
| The model folder is refused | `origami inspect <folder>` shows what it holds; see [models.md](models.md). |
| "the drive … is not connected" (`model list`, `setup`, `serve`) | The model's drive is unplugged or asleep. Connect it. A server whose model's drive was not mounted when it started (after a power cut, say) waits for it (its status says "waiting for …") and loads the model as soon as the drive mounts; Stop ends the wait. If it is connected under another name (renamed, or `/Volumes/<name> 1`), `origami doctor` names the new place and `origami relink` uses it. |
| The model's drive dropped off while serving (status `degraded`, requests fail with `drive_stalled` "The model drive is not connected") | Requests fail at once while it is gone; what is already in memory and the hot copy stay. Reconnect the drive: within about 5 s Origami opens the model's files again, checks they are the same files (size, modification time, inode, and 64 KiB of content), and reads continue with no reload (the log says "the model drive returned after N s"). If another drive or another model is now at that path, it stays `degraded` and says "A different file is at …": reconnect the right drive, or Reload to load what is there. The dashboard may still offer Reload after the drive is back; it is not needed once requests answer again. |
| "… is shorter than its header says: the download is incomplete" | A GGUF was cut short (an interrupted download). Download it again. |
| A move stopped part way (a crash, a sleep, a closed browser) | `origami move list` shows it; with the server quit, `origami move recover` finishes it (the copy was checked) or puts the original back, as the dashboard's **Recover…** does. A hot copy listed as "belongs to <model> before its move" waits for its model's files to be readable; `origami move recover` then carries it over. |
| A `.partial` file or folder in the models folder | An unfinished conversion, listed with its size by `origami store list`. Run the same `origami model add` again (a Qwen conversion resumes), or delete it. |
| `origami cache prune` keeps a build "it holds chat-1" | That build has pinned or named conversations; `origami cache prune --yes` removes it too (on the dashboard: tick it in **Remove unusable…**). "a running server holds it": a server has that folder open; it can be removed once that server has stopped. |
| An expert copy may be damaged | `origami verify`. A record that fails its checksum while serving is no longer read from that copy. GLM and DeepSeek check each hot-copy record on its first read and read a bad one from the model file instead; `/v1/status` counts them as `rejected_records`. The log has one line per rejected record and one when it is repaired. |
| Apps get a 503 with `"code": "stopped"` | Origami was stopped with `origami stop` or the dashboard. Run `origami start` or press **Start**. |
| A 503 "Request buffers are busy; retry in a moment" | Several large requests (images, long conversations) arrived at once. Retry; up to 64 requests can wait in the queue, but memory for request bodies is sized for 4 large ones (2 at 256K). An upload too slow to finish in time also gets it when another request needs its buffer. |
| A 408 "HTTP headers deadline exceeded", "HTTP request deadline exceeded" or "HTTP request body arrived too slowly" | The request did not arrive in time: headers within 10 s, the whole request within 30 s, a body at 16 KB per 5 s at least. On a slow network, send smaller bodies (images) or put a proxy that buffers uploads in front ([integration.md](integration.md#tls)). |
| A 429 "Too many wrong keys from this address" | That device (or proxied client) sent 10 wrong keys within a minute. Fix its key and wait for the minute to end; programs on this Mac are never locked out. |
| A 503 "Too many connections" for one device only | It has 8 connections whose request has not fully arrived (all other devices together: 32). Something on it is opening connections without sending; close it. |
| A streamed reply stops and the request shows as cancelled | The client took no byte of the reply for 30 s (a suspended app, a stuck network), so Origami dropped it, as if it had disconnected; the conversation is kept. |
| `origami doctor` warns "listen lan sends the key in clear" | Other devices send the key over plain HTTP. Use a TLS proxy ([integration.md](integration.md#tls)) or a VPN. |
| "… did not start within 60 minutes and was stopped" | The model process hung while loading. The server log says where; `origami doctor` checks the setup. |
| "prompt cache: off (another Origami server is using the data folder …)" | Two servers share one `context_dir`; the second saves no prompts and keeps no usage counts. Give each its own: `origami config set context_dir <folder>` (or `--context-dir`). |
| Something else | `origami doctor` says what to fix; `origami report` prints a block to paste into a GitHub issue. |

## The model drive got slow

Origami reads the same parts of the model files very often. Some SSDs (seen on a DRAM-less NVMe drive) then read those
parts at a fraction of their speed, for seconds at a time, while other files on the same drive read at full speed.
The dashboard says so on the drive's Health line (with the command to copy), `/v1/status` in that row's `slow_reads`, and `origami doctor`
with a `fix` line, once more than 20% of the drive's reading time (over at least a minute of reading) went below a
third of the speed the same drive reached shortly before.

Quit `origami serve` (or stop the service that runs it: `origami stop` alone keeps the server answering, and refresh
refuses while one does), then write the files' own bytes back where they are, so the drive stores them on fresh blocks:

```sh
origami refresh       # --model <name> for another model; --yes skips the question
```

`origami refresh` reads each part of every weight file and writes the same bytes back to the same place, flushes them,
reads the file again and compares the two SHA-256 digests; the file's dates are kept. It takes about as long as reading
the model three times (the doctor line gives a lower bound). It is safe to stop at any moment: every write repeats the
bytes just read, and running it again continues where it stopped (`refresh.json` in the data folder remembers which
files are done; a server does not start that model until the refresh is finished). If a file reads back different
bytes, it stops and names the file, and nothing else changes: the file's bytes may no longer be the model's, so check
the drive, download that file again and delete `refresh.json` (running the refresh again refuses until then).

Rewriting changes the files' change time, which is part of how Origami recognises a model's files, so afterwards it
moves the hot copy, further copies, usage counts, tuning and saved prompts to the files' new identity. That is done
only because the bytes were proven unchanged; a file replaced or edited meanwhile is treated as a new model (its copies
are listed as orphans, its saved prompts are not used). A model with low-bit experts (`model add --base`) depends on its base's files:
refreshing the base moves the variant's copies and counts too, and `origami refresh --model <variant>` also rewrites
its experts file. Copies on a drive that is not connected keep the old identity. The slowness
can come back after many terabytes of reads; run it again then.

### How often a refresh is needed

What brings the slowness back is how often the same blocks are read, not time. On the development Mac (an external
DRAM-less NVMe drive holding all three models) the files were rewritten on 2026-11-10 by hand, then with `origami
refresh` on 2026-11-11, and twice on 2026-11-15: under heavy testing (benchmarks reading terabytes a day, mostly the
same 68 GB of Qwen experts) the slow periods were back within hours to a day, after a few to about ten terabytes of
reads (25 to 60 passes over the most-read data); in normal use it took weeks. A drive that does not do this never needs it.

So Origami counts, per set of files, what it read since they were last written, in `drive-reads.json` in the data
folder. The dashboard's Drive line shows that number (for example "4.2 TB read") and a sentence under it: "4.2 TB
read since its files were last rewritten on 2026-11-15; slow 3% of the reading time. Last time the slowdown began
after about 6.1 TB." "Slow" is the share of reading time spent below a third of the drive's own recent speed, as
above. Once a slowdown has been seen and the files refreshed, the line turns to a warning with the command to copy
when 80% of that amount has been read again ("A slowdown is likely soon"), so you can refresh at a quiet moment
before it hurts. `origami doctor` prints the same line for every set of files counted, `/v1/status` has it as the
row's `since_rewrite`. `origami refresh` starts the count again.

### The hot copy's drive

The hot copy (on the Mac's internal drive by default) is read the same way, and the internal drive cannot be
swapped, so its reads are counted too: the dashboard's Hot copy line warns when its drive reads it slowly, or when
it nears the amount at which it slowed down before. The cure is the same as for the model's files. Quit
`origami serve`, then:

```sh
origami refresh --copy     # --model <name> for another model; --yes skips the question
```

It writes the copy's own bytes back where they are, flushes them, reads the copy again and compares the two SHA-256
digests. Nothing else changes: the copy keeps its experts, the server uses it as before, and its count starts again.
It takes about as long as reading the copy twice, needs no free space, and is safe to stop and run again.

## After upgrading from 2.x

Origami 3.0 identifies a model by its files on this disk, not only by their names and sizes, so the fast copies 2.x
made (a Qwen mirror in `store/<model>-<fingerprint>`, a GLM or DeepSeek hot copy in `family-<key>`) are not used.
The dashboard's **Models** tab lists them as "made by an earlier Origami version for <model>"; delete them there to get the
space back. The hot copy is built again while the model serves. Saved prompts from 2.x are not reused either.

## After upgrading from 3.0 (Qwen)

Qwen's hot copy is no longer built by `origami setup`: the server builds it while serving, after the first reply (until
then the dashboard says it is learning). The 3.0 copy (`mirror.json` and
`mirror.bin` in the hot copy's store) is not read any more; it is removed when the new one is first built, and until
then the dashboard's **Models** tab lists it as an offline mirror of an earlier version. Further copies an earlier version made on other
drives are still read as they are; new ones are written in the hot copy's format (`copy/` in the store's model folder).

## After upgrading from 3.1

Saved prompts are now kept across updates that do not change the engine's arithmetic. Those 3.1 saved were keyed on its
build and cannot be checked, so the first start of this version begins without them, once; the log says an unusable
older folder exists, and a request that would have used one says why in its `prompt_restore` (`saved_by_other_build`).
They go after 14 days unused, or sooner when a save is short of disk; pinned or named ones stay until `origami cache
prune`.

## What Origami is not

- A general model runner: it runs Qwen3.8-Flash-Next, GLM-5.3-Flash and DeepSeek-V4.1-Flash, in the formats listed
  in [models.md](models.md).
- A multi-user server: one request runs at a time and the others wait.
- For video or audio input. GLM and DeepSeek read text only.
- For Intel Macs or Linux.
