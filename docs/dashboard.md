# Dashboard

`origami dashboard` opens the running server's dashboard in your browser (normally
`http://127.0.0.1:11439/dashboard`). It works on the Mac that runs Origami and, with `listen lan`, on other devices
that have the key (see [integration.md](integration.md#other-devices-on-your-network)). Paths always refer to the Mac
that runs Origami.

The header always shows the state, by one name used on every tab: **Ready** (a green square), **Loading**,
**Stopping**, **Stopped**, **Failed** (red), **Reload needed** (saved settings the running model does not use yet),
**Job running**, and **Drive gone** when the model drive stops answering (also in `origami status`; `degraded` in `/health`; it clears
by itself once the drive is back); **Working** while a reply is written. Notices stand in one place, under the header, in one style: a
failed start or a stopped model, a waiting reload or paused saves, saved settings that differ from the running model
(with **Reload to apply** and **Discard the saved change**), and a job's progress. The header also has
**Stop and unload**, which gives the model's memory back while keeping the dashboard up; **Start** loads it again,
and after a failed start the same button is **Try again**.
Under it, the crane, the model's name and what it runs as: its context, and whether its replies are the model's own
(**Exact**) or **differ** (a fast quality level, or low-bit experts). A reload asked for while a reply is being written
waits for it, and a banner says so; the same banner says when conversation saves are paused for lack of disk. A job
(adding, converting, tuning, moving) shows its progress and **Cancel** on every tab, then its outcome until dismissed.

## Tabs

- **Now.** What is happening, and the one next action when something is needed.
  - **Slowed by**, at the top while something slows Origami down now, also while a request runs: one line per
    condition, what and then the fix, the costliest first and settings advice last. Memory 1 GB or more under what Origami may
    use says why (cut by macOS, started at the budget learned from this Mac's memory use, or started small), names the three
    apps that hold the most and that it grows back by itself as memory frees;
    just after Qwen loads, that it is warming up and that the first reply ends it; replies that waited on a drive for a
    fifth of their writing time or more at full memory, that the conversation uses more of the model than fits and
    whether this Mac can keep more;
    the others are a drive reading slowly or due for `origami refresh`, gone or not allowed, the hot copy building, without room or failed, heat, Low Power Mode,
    conversation saves paused, a prompt read again (and why), a job, waiting requests, a memory limit this Mac does not
    keep or one under the measured limit, and never tuned. `origami status` prints the same lines; apps read
    `slowed_by` ([integration.md](integration.md)).
  - On a first run: the models Origami found on this Mac (**Add**), the download it is made for, or your own file.
    A model added but not loaded: **Load**. Stopped: **Start**. A model that did not start: why (a memory limit of
    yours that is too small is named), **Try again** and **Change the setup**.
  - The speed (the reply being written; otherwise all the tokens of the last five replies over the time they took to
    write, one mark per reply), the first-word time, how fast prompts are read and
    how much of them was reused. **Measure** sends a short test reply.
  - Three bars: where the last request's time went (waiting, setting up, reading the prompt, writing, saving), where
    the experts were read from (memory, hot copy, the model drive), and this Mac's memory ("Origami uses 17.8 GB · free
    5.8 GB · other apps and macOS 10.8 GB"), with under it how far Origami may still grow within its budget and how much
    of its limit it holds back after macOS took memory back, and why. Under the first, the last request in two lines (its
    first word: prompt tokens read and reused from a saved copy, and why fewer were reused than could have been; its
    writing: GPU time and the wait on each drive) and one line, the same `origami status` gives, in the words the
    Slowed-by box uses: "Slower than usual: macOS took back about 1.8 GB of Origami's memory during this reply, so more
    of the model had to be read from the drive", tokens of the prompt read again (and why, in a sentence), a drive the
    reply waited on, or "Nothing unusual".
  - **Health**, one line each. **Memory** says how much Origami uses of what it may use, who decided that figure (you,
    a measurement Origami made on a given day, which finds the most memory this Mac keeps for Origami without macOS
    taking any back, Origami's own estimate, or a memory search running now), and only then what to know or do: macOS
    took memory back (when and why, and that it grows back by itself), Origami started lower and grows, your limit is
    more than this Mac keeps or less than a measurement found (with **Use 25.5 GB**). Its buttons: **Measure the best
    limit** (or **Measure again**), **Measure by itself when idle** / **Stop measuring by itself**, **Change the
    limit…**. Then each model drive, what was read from it since its files were last written, with a warning and the
    `origami refresh` command when it reads far below its own speed; the hot copy (size, share of expert uses it
    holds, build progress); tuning; who can reach Origami.
  - **Requests**: the last ten, each with its time bar. **Saved prompts**: open, rename, keep, edit or delete.
- **Setup.** Context, quality (only the levels this model offers), thinking and memory limit as dials,
  with this Mac's memory answering each move; a dial set to something that changes replies turns copper.
  **Tune for this Mac** (below), sampling, **Presets** and **Restore last working settings**. **Presets** keep the
  saved model, context, quality, thinking, memory limit and sampling under a name: **Save as preset**, then on each
  **Use** (saves its values, for the next reload), **Update** (keeps the settings saved now), **Rename…** and
  **Delete…**. Tuning results come back with a preset's model and context; hot copy sizes stay as they are. The bar at the bottom appears only
  when there is something to save: **Save** keeps a draft (the bar then offers **Reload** alone), **Save & reload**
  loads it. Settings that cannot run on this Mac (a memory limit the context does not fit in) say why and cannot be
  saved.
- **Models.** Every added model as a row: **Load** switches to it, **Remove…** unlists it (files stay). A low-bit
  variant is marked "replies differ from exact". **Add a model** takes a Hugging Face name, a folder or a `.gguf`; for
  a `.gguf` it also offers **Low-bit experts for** an exact model already added (the dashboard's
  `origami model add <experts.gguf> --base <model>`). Then where the running model's experts are read from (the hot
  copy's drive and size are changed here), and the drives and files.
- **Connect.** The address (the network one once Origami listens on the network) and model an app needs, who can reach Origami (this Mac, or this network with a key and a
  QR code), and the log.

Memory sizes on the page are the same decimal GB that `memory_limit` takes and `origami status` prints:
`memory_limit 22.5GB` shows as 22.5 GB. Only this Mac's own size is named as macOS names it (32 GB).

Background refreshes keep your edits. A draft made before the settings changed elsewhere (another dashboard,
`origami config`, or an edit of the settings file) is marked "based on older settings"; saving it is refused only
when a setting it changes was changed since, and the page then shows both values with **Keep mine** and **Take
theirs**. Values from a lost connection are marked stale. A button that cannot act now says why next to it (and in its
tooltip), with the step that frees it where there is one, such as **Reload to apply**. The folder browser offers the Mac's drives, pasted paths, search, pages
and new folders. A flag or environment variable that sets a setting wins and keeps its control disabled. It works at
phone width, in light and dark, and from the keyboard.

## The hot copy's size

**Change…** on the hot copy (Models, where the experts are read from) shows a bar graph of the experts' popularity:
one bar per group of experts in the order the copy takes them (most used at the left), its height the uses per expert
(a log scale when they span orders of magnitude). Bars inside the size you choose are solid, the rest faint; a dashed
mark shows what the copy holds now, and what the drive cannot hold (its free space less the 20 GB it keeps) is hatched.
The slider sets the size in GB or as a share of the experts (**GB** / **% of experts**); two lines follow it: the
share of expert uses the size covers, and how often the last expert in and the next one left out are used. **Use this**
saves it as `hot_copy` in GB; **Automatic** and **Off** stay in the menu beside it. Until a thousand tokens are counted
the second line says it is still learning.

Below it, on a drive that holds another model's copy, every model's copy there: its size and limit, what an idle one
keeps when the loaded model's copy grows (its floor, or all of it with the box ticked), what it gave or took, and a
**Keep when idle** box per model (`hot_copy_keep` of that model). The copy's own row says how many swaps wait for the
hourly allowance (records, GB and the uses they hold). Under **Drives and files**, each drive's disk budget: what it
keeps free, what copies gave back for saved prompts and the saves skipped for space.

## Models: drives and files

This view lists every file Origami knows about, with its real size on disk: full models, smaller-expert copies,
hot copies, saved prompts and leftovers it can identify, including files in folders Origami used before. A missing
disk shows as unavailable. Files Origami cannot prove it owns are never offered for deletion.

- **Move…** shows the destination and the space it needs. On the same drive a move is a rename. Across drives,
  Origami copies each file, checks its SHA-256 by reading it back, updates its records, then removes the original.
  The destination needs the item's size plus 10 GB free. A model using those files pauses and reloads afterwards.
  A moved model keeps its hot copies, usage counts, tuning and (GLM and DeepSeek) saved prompts; Qwen's saved prompts
  start again.
- **Delete…** previews the item, its size and what it affects; for a model download no model uses any more, it says the whole download would have to be fetched again. Deleting a hot copy also turns automatic hot copying
  off for that model. Smaller experts an earlier version made are listed as not used and can be deleted. Files another model
  uses, and folders holding Origami's own data, are protected. Deleting a model removes only its own files (its weights,
  config, tokenizer, template and licence, and what Origami wrote there); a folder that holds anything else stays, and
  the result names what is left.
- **Unlist…** removes a model from the library and keeps its files; a Qwen low-bit variant's small generated folder
  of links goes too, as `origami model remove` does, never the base's files or the experts file. **Locate…** checks
  and records a new location; the start-failure notice then says **Try again** loads it from there.
- A copy that Origami 2.x made shows as "made by an earlier Origami version for <model>; not used by this version;
  safe to delete". This version never reads it; **Remove** frees its space, and the hot copy is built again.
- An interrupted move stays visible. **Recover…** checks the remaining copies, then finishes the cleanup or puts the
  original back. Stop the loaded model first. If the copies differ or a disk is missing, recovery keeps every file
  and reports the problem.

- **Saved prompts** are listed per model, in the words `origami cache` uses: those the running build reads (**in use**
  or **usable by this build**) with their size, conversations and how many are pinned or named, and apart those no
  build can read any more (**unusable**: made by an earlier or another Origami build, of a model file no model uses, a
  GLM or DeepSeek context left on disk) with their size and when they were last used. Above 1 GB of those, the storage
  total says so. **Remove unusable…** lists every folder that goes with its size, and removes them, also while a model
  is loaded: a folder a running server holds is never removed. A folder with pinned or named conversations, or whose
  index cannot be read, is named with its own box and goes only when ticked. The result says what was kept and why.

Full and partial expert copies move independently, even when they share a folder. Turning a cache off keeps its
files for reuse.

## Tuning for this Mac

`origami tune` (or **Tune for this Mac** in the dashboard's Setup tab) measures the running model's
timing-only switches against its current settings on this Mac, in about 8 minutes, and keeps a switch only if it was
faster in every round with identical replies. After its first measurement it knows how long this model needs: the
default limit then grows to fit (up to 30 minutes, saying so); a limit you choose (`--minutes`, or the dashboard's
Time limit) is kept, and the run measures the candidates that fit and names the ones left out. The switches are the GPU keep-alive, and GLM's and DeepSeek's
next-layer expert guesses, their depth and spinning.

- Origami applies the result whenever it serves that model and context on this Mac; your own settings still win.
- `origami tune --recheck` checks the result still beats the defaults (offered after a macOS update); `--undo`
  removes it.
- It starts only while the loaded model is idle, runs its saved settings and no other job runs; otherwise `origami
  tune` (and `tune --memory`) stops with the reason the dashboard shows, such as "Reload to apply saved settings first".
- It stops, changing nothing, when an app sends a request, the Mac is too warm, memory runs short, Origami restarts
  or the conditions change too much. Ordinary background activity and a steady Low Power Mode are accepted.
- Nothing measured is lost: each finished comparison is kept (in `tuning.json`), and the next run within a day, for the
  same model, context, settings and Origami build, continues from it ("continuing: N of M measurements kept from …")
  while a fresh baseline still agrees with the kept one within 3%; otherwise it says so and measures again.
  At the time limit the comparison in progress finishes and the run decides from what it has.
- Every ending says what was measured, what was kept and what to do next (often nothing).
- "The defaults are fastest" is a result: it is recorded, and the dashboard shows "Tuned: defaults are fastest
  (date)". The defaults were tuned by hand on the development Mac, so expect it on an M2 Pro.

The dashboard's **Tune for this Mac** finds one of three things:

- **Faster settings, same replies** compares timing switches (GPU keep-alive; for GLM and DeepSeek next-layer
  guesses, their depth and spinning) and prompt grouping within the memory Origami has.
  Each candidate is measured next to the current settings and a control, with retries and a separate
  confirmation. A candidate is kept only if it gains at least 3% and twice the measured noise, keeps the replies,
  and does not slow down prompt reading. Noisy runs raise the bar; an inconclusive run changes nothing.
  Qwen's prompt grouping is **NUMERIC**: matching the test replies does not guarantee identical replies on other
  prompts. GLM's and DeepSeek's grouping keeps the same arithmetic (**BITWISE**).
- **The speed of each quality level** measures the quality levels the running model offers down to the limit you choose,
  with context, memory and thinking fixed, and reports token rate, first-token time and their
  variation. These are speed measurements, not scores of answer quality.

- **The fastest memory limit**: see below.

The first two use temperature 0 without changing your saved sampling settings or saving the test prompts. Review the
results before applying them. Other requests, memory pressure or an
unavailable model interrupt it. From the command line: `origami tune --review true` saves a proposal without
applying it, and `origami tune --tradeoffs true` compares exact with the model's mildest level after it (`dynamic`) without changing
settings (`--quality-floor <level>` compares down to that level).

### The fastest memory limit

`origami tune --memory` (or **Measure the best limit** on the dashboard's Memory line, or **The fastest memory limit** in
Setup) finds the `memory_limit` that writes fastest on this
Mac for the running model and context. Too low a limit starves the expert cache; too high a one makes macOS swap.

- It measures three or four limits: your current one, the level that held after cuts (if any), the automatic budget,
  a step between that and the most macOS lets Origami have, and a step below it. For each, the model reloads at that
  limit, reads a long prompt, then writes a first reply and a repeat.
- A limit that macOS cut for memory pressure, or that made the Mac swap, is out whatever its speed, and higher ones
  are not tried. Only cuts the Mac asks for count (pressure, fresh swap), the same on every model: the model holding
  itself to the limit under test is that limit working. A limit too small for the model to start at is out, and the
  next one up is still measured. The highest limit kept wins: more memory is faster unless the Mac starts cutting or
  swapping, and those are measured reliably while one trial's speed is not. Speed only guards against something
  being wrong: a kept limit loses to the next lower kept one only if both its first reply and its repeat are over 15%
  slower, and it is measured once more first (the better of its two tries counts). Where kept limits held the same
  expert cache (a Mac that keeps every expert in memory), more memory buys nothing: the lowest of them is chosen, and once two such limits are kept the higher ones are not tried. When fewer than four were
  measured, one more halfway to the next limit up is tried (from the highest kept, or with none kept, from the
  highest the model did not start at), and wins only if it is kept too.
- When even the lowest limit is cut or swapped (other apps hold the memory), it goes on downwards: next the level
  macOS's cuts left the budget at (else 1.5 GB lower), at most three more, never below the least the model runs with;
  the first kept is suggested. A cut there adds a line naming the three largest other processes and the swap in use
  ("Other apps hold memory: …; swap 8.1 of 9.2 GB used."). If none is kept, the result says so and names the lowest
  limit tried.
- It takes about 4 minutes for Qwen, 6 for GLM and 10 for DeepSeek on the tested Mac (`--minutes`, default 30, bounds it), refuses while requests run and stops when an
  app sends one. Your saved limit is reloaded at the end, also after a failure or Cancel; a limit under test is never
  saved and leaves the memory history alone.
- The result ("Chosen: 25.5GB — 10.1 tok/s on a repeat (highest limit this Mac kept); your current limit 30.0GB —
  cut 2 times; set it with `origami config set memory_limit 25.5GB`") shows in `origami status`, `/v1/status`
  (`memory_learned.limit_advice`) and the dashboard's Memory line, ahead of the unmeasured suggestion. Origami never
  changes the setting itself; the dashboard's **Use …** sets it and reloads, after asking. Without a `memory_limit`
  the automatic budget uses the result at the next start, until it goes stale (see
  [settings.md](settings.md#memory)), and the memory line says so.
- **Measure by itself when idle** on the Memory line (`memory_auto_measure`) lets the server run this search by itself after
  half an hour without requests, while no limit of yours is set and none measured is current; the line then says what
  it found or why it stopped ([settings.md](settings.md#memory)).
