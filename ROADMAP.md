# Origami 3.1 roadmap

The single work queue for Origami 3.1: everything to test or build next, most urgent first. Work top to bottom; inside a tier, entries are ordered by value per effort. When an entry ships, delete it here and mention it in CHANGELOG.md.
IDs: H = phase 0, S = foundations, F = faster with identical output, N = opt-in (changes output), R = reliability, P = release. Size is relative effort for agent-driven work, where most of the time goes into validation: S = hours, M = a day or two, L = several days. "Audit O29" and similar refer to entries in the 2.4.x audit queue.

## 0. Rules of the road

**Where time goes today** (M2 Pro, 32 GB: the only Mac that has ever run Origami)
- DeepSeek in real use: 0.55 tok/s at ~59K context, median 142 s to the first token. 89% of the time is generation, and nearly all of that is waiting for expert reads. The GPU is a small share.
- About 47% of loaded bytes come from the external drive, which sat at a hard 125 MB/s cap 38–74% of the time (the SSD slowing down on heavily read blocks; see H1).
- GLM: ~0.48 tok/s at 8K; 78% of loads come from the external drive and RAM cache hits are only ~2%. Qwen: ~7.5 tok/s, about 1.5x SlotStream.
- DeepSeek usage counters (12M expert uses): the RAM cache (~1,000 records of 18.8 MB) covers ~45% of uses; ~2,000 records ~62%; ~3,500 records 75%; ~6,350 records (RAM plus a ~100 GB fast copy) 90%. So a ~100 GB fast copy on the internal SSD would cut the external drive's share from ~40% to ~10%.

**Rules**
- The drive is fixed and the fast copies are grown; timings taken before 2026-11-10 are not trustworthy.
- Exact by default. "Output unchanged" means identical tokens, logits and state. Anything that changes output is opt-in and needs its quality measured on checkable tasks.
- Measure carefully: one benchmark at a time. Never compare single runs that differ by less than ~50% unless the read test shows the drive healthy. Prefer bytes read per device, cache misses and interleaved A/B/A/B repeats. Use production-like settings: automatic memory and a large fast copy.
- Stop spending effort on: cache eviction policy, draft/speculative decoding (until the drive is fixed), megakernels, expert substitution, delta loading, decode read coalescing, lossless compression of expert bytes. Reasons are in sections 7 and 8.
- Keep experiments small. No evidence folders in the repo.
- Every experiment switch either becomes the default or is deleted within one release. There are 60 environment switches today.
- Keep all three models advancing together. The shared streaming core (S1) is what makes that cheap.
- This file is the only queue.

## 0.5 Where the time goes (measured 2026-11-12 to 2026-11-17)

Kept as the record behind the entries below, not as a queue. The ideas in it that were not built (items 5, 9 and 10, and the leftovers of 11a and 11b) are parked with section 7.2 and reopen with a real-model round on a healthy drive.

A day and a half of 3.1 work went into the shared core, cleanup and reliability; the only speed gained in code is prompt reading on GLM and DeepSeek (17–28%). Decode is unchanged. On the healthy drive the old premise ("nearly all time is waiting for reads") no longer holds everywhere: decode on GLM and DeepSeek reads 3.6 and 2.7 GB per token at about 3 GB/s, one drive's ceiling; prompt reading runs at 1.7–3 GB/s, below what the drives can deliver, and one kernel change cut GLM's first token 78 → 63 s at the same bytes; Qwen decodes at 2.1 GB/s, below the ceiling. Work in this order, and start no speed item without a breakdown that shows what it can gain:

Measured 2026-11-12 with the standard benchmark (production-size copies, clean drive; times less slow seconds):

| | 2,500-token prompt | 200-token decode | 7,700-token prompt |
|---|---|---|---|
| GLM | 50 s at a 21.5 GB limit: GPU busy 40 s, read wait 3 s. 78 s at 30 GB: memory pressure turns prompt read-ahead off (read wait 39 s) | 1.15 tok/s: read wait 80%, GPU wait 19%; both drives busy all the time at 2.2 and 1.5 GB/s, under their ceilings; 3.4 GB read per token at a 38% cache hit rate | 176 s: GPU busy 121 s, read wait 57 s |
| DeepSeek | 104 s: read wait 54 s, GPU busy 42 s, host and unwaited GPU work 33 s | 1.52 tok/s: read wait 73%, GPU wait 25%; drives at 1.8 and 1.3 GB/s | 158 s: GPU busy 88 s, read wait 57 s |
| Qwen (20 GB copy) | 34 s: GPU busy about 28 s | 7.3 tok/s at about 73% GPU use | 62 s at 91% GPU use |

The copies buy +46% decode speed on GLM, +35% on DeepSeek, +15–24% on Qwen, and 18–21% on prompts. Raising GLM's limit to 30 GB bought nothing in decode and cost 57% on the prompt.

So: first-token time is mostly GPU work on all three models; decode on GLM and DeepSeek is reads on two drives that are both below their ceilings, then GPU wait. In this order:

1. (Shipped: prompt read-ahead survives memory pressure. A request that starts under memory pressure still decodes with a small cache until pressure clears — 0.93 against 1.17 tok/s on GLM in one run; the cache should regrow toward its budget once it is safe. S.)
2. **Prompt reading on GLM and DeepSeek is bound by reads and the GPU, not by host work.** Shipped: grouped expert runs, the faster attention-value kernel, honest timing notes (the "host time" was a full GPU queue and read waits that were not counted; the request thread computes 1–2% of the time). A 2,500-token DeepSeek prompt reads about 180 GB with the model drive busy over 90%. Long prompts now read each layer once (the prompt borrows working memory from the cache). Levers left, in order: (a) **check the decode after a long prompt**: memory pressure cut the budget during the 7,700-token prompt on both models (cache 14.2 → 12.3 GB on DeepSeek, 10.0 → 9.5 GB on GLM afterwards), and GLM's GPU time rose 14 s in the one run; measure a long prompt followed by a decode and lower the loan's cap (half of the cache today) if the cache does not come back. S. (b) more of each model on the internal drive (item 7). (c) price the kernels behind the few 26–43 ms command buffers so they split (layer 0 KDA on GLM, expert and dense buffers on DeepSeek). S. Opt-in candidate: route DeepSeek's runs of 8 rows or fewer through the larger kernel (17% of its prompt GPU time; changes sums). Decode guesses: ranks are judged by the reads they start (DeepSeek waste 40 → 24 GB per 200 tokens at the same speed) and GLM's guess is on by default (about 12% less decode time with slow-drive seconds left out — confirm on a freshly refreshed drive). Scored on real routing traces: the cross-layer guess names 61–69% of misses; the previous token's routing and usage counters name almost none (those experts are in memory); a learned correction was not estimated (the trace has no hidden states).
3. **Decode reads: move bytes to the copy drive.** Measured on the real files (burst replay): the engine already reads each layer's records at the drives' ceilings (internal 3.2–4.2 GB/s, external 2.9–3.0 GB/s), and splitting records into chunks or a contiguous layout changes nothing (0.90–1.04x). What bounds a layer is the external drive, which takes 15 ms while the copy drive finishes in 9 ms and waits. A split where both finish together would cut decode read wait by about a third: that needs a larger share of misses held by the copy (larger copies: F11) or another fast drive (F10).
4. **DeepSeek's 33 s of host time** in a 104 s prompt: find it.
5. **Qwen's GPU work** (prefill and decode are GPU-bound): profile, then Qwen's exact fusions (parked, section 7).
6. Decode GPU wait (175 ms per token on GLM and DeepSeek): dispatch count (F8; the keep-alive's rest is parked in section 7).
7. A second fast drive or larger copies (F10, F11); memory (N5); the opt-in modes (N4: on Qwen the fast modes read a third of the bytes at 10 and 9 of 14 tasks correct against 12).

From a survey of other SSD-streaming engines (2026-11-14; ds4/DwarfStar, Slotstream, InferSSD, oMLX, Strata, DeepInfer-Mac, TensorFold, flash-moe, llama.cpp, and the offloading papers): where numbers are comparable, none decodes faster than Origami at exact 4-bit output on a similar Mac (Qwen 7.3 tok/s here against 4.1–6.2 on 32 GB Macs elsewhere; DeepSeek 1.52 against 1.2–1.35 on a 24 GB Mac); faster figures come from 2–3-bit packs, fewer experts, or Macs with much faster drives. Their self-reported numbers were not reproduced. Queued from it:

8. **Run each expert once per prompt call**, not once per 2,048-row slice (ds4, Strata, oMLX use 4,096–8,192-row chunks; Strata measured 700 → 1,100 prompt tok/s). Exact if the matrix kernel's per-row sums do not depend on the row count: test that bitwise first. Part of item 2.
9. **A learned correction to the next-layer guess** (Slotstream: a per-layer linear correction lifted agreement 0.73 → 0.80 and cut decode reads 9%, identical output). Could make GLM's guess worth turning on. Needs captured hidden states and routing; limited by drives already at their ceilings. M.
10. **Read-disturb care for the internal hot copy**: it is read the same way that slowed the external drive, on a drive that cannot be replaced. Extend the slow-read signal and `origami refresh` to the copy (drive slowdown, section 7). S.
11. Opt-in: **published calibrated low-bit experts.** `faster` and `fastest` skip experts (10/14 and 9/14 tasks on Qwen); `expert_bits 2` is Origami's own uncalibrated requantisation. Measured on Qwen (2026-11-14, only the experts swapped, 40 prompts, exact = 12/14 tasks): ISTA GSQ-RCO **IQ3_XXS** 12/14, 94.9% next-token agreement, 17% fewer bytes per token; ISTA **Q2_0** 12/14, 91.2%, 65% fewer bytes, records half the size (1.38 MB against 2.76 MB); `expert_bits 2` 90.0%, larger and 1.5x slower prompt kernels than 4-bit. Every GGUF type runs 20–30% faster than 4-bit in prompt kernels and the same at decode. Both files match our checkpoint. Order:
    - a. Shipped for Qwen: queued decode for GGUF experts and `origami model add <experts.gguf> --base <model>` (Q2_0: 12/14 tasks, about 14 tok/s warm). The hot copy for this mode shipped in 3.1.2 (records of each layer's own size). On the real IQ3_S file (2026-11-20, one request, 2,517 prompt tokens and 220 written, saved prompts cleared): the same reply with and without the copy; a 6 GB copy (2,872 records) served 12% of expert reads and the model drive's share fell from 41% to 29%; a 40 GB copy holds 19,554 of the 24,576 records. Speed not measured: compare decode speed and bytes per drive with and without it, drive healthy. Left: per-layer slot sizes for mixed-size files (IQ3_XXS stalls 23 ms per token on reads); MTP for this mode; a variant added with `model add --base` does not take its base's `hot_copy` size; `model list` shows the variant's size as the base's; a base whose router is stored quantised is refused rather than compared; no model-free check covers the server's status JSON. One open point: reading a queued group's record count inside the kernel gave stale values in layers whose ten experts were all cached, cause not understood; the count now comes from the indirect grid like the 4-bit kernels.
    - b. GLM and DeepSeek: `model add <experts.gguf> --base <model>` shipped. **First real file, GLM with Strata's experts-only Q2_K/Q3_K file (2026-11-15):** pairs with `--yes` and loads; same checkpoint (same-expert correlation 0.96–0.99 against 0.00 for a different expert); 12 of 14 tasks against exact's 10 (noise at this size, no loss shown); 92.0% next-token agreement; with NO fast copy it decodes at 1.14–1.30 tok/s and reaches first token in 39–47 s, against 0.8–0.98 tok/s and 54–70 s for exact with its 70 GB copy (drive throttling during the runs); twice the cache slots. **With a 20 GB fast copy (clean drive):** 1.73–1.88 tok/s and 39 s to first token, against 1.19 tok/s and 56 s for exact with its 70 GB copy, and 1.03–1.19 tok/s for the variant with no copy; replies identical with and without the copy; the copy covered 52% of uses counted on the benchmark's own prompts, so real traffic will gain less. Shipped since: counts seeded from the base, refresh carries a variant and rewrites its experts file, status shows both files. Left: while a Qwen variant's refresh is unfinished nothing stops a server from starting its base; DeepSeek needs 115–132 GB (63 GB free on the model drive); drop Origami's own requantiser as a tier source.
    - c. **The mixed mode: design (2026-11-16; nothing built).** Each expert has two records: the exact one and a published low-bit one (Qwen: 2.76 MB and 1.38 MB). Rule per token and layer, experts ranked by gate weight: the top expert is always exact (wait for it); any other expert uses its exact record if that is in memory, else its low-bit record if that is in memory, else a READ of the low-bit record once the experts ranked above it carry 0.6 of the weight, otherwise a read of the exact record. Nothing is skipped at this level (skipping stays in `faster`/`fastest`). An expert served low-bit gets its exact record read in the background at read-ahead priority, so experts used often become exact and only the cold tail stays low-bit. What it needs: (1) the cache holding both formats — a second slot size class in the shared slot core (the Family cache already runs several classes), with the class sizes set from miss statistics; (2) one layer command running a 4-bit group and a GGUF-kernel group with the reduction in rank order (Qwen's queued path can already run GGUF kernels on records the GPU looks up); (3) the read path choosing the file per expert; (4) a dial level, named so nobody takes it for exact. Costs to state plainly: replies then depend on what happens to be in memory, so two runs of the same prompt can differ; and the gain on Qwen is bounded by what is already measured (exact about 10 tok/s, all-low-bit 14). **Offline estimate (2026-11-16, recorded traces replayed through a simulated cache; scripts in the session scratch folder `mixed-estimate`):** with the background read of the exact record the mode is SLOWER than exact on all three models (Qwen −6 to −14%, GLM −25 to −32%, DeepSeek −14 to −25%): those reads load drives that are already busy and push other records out. **Drop the background promotion.** Without it, at the 0.6 threshold and nine tenths of the cache for exact records: Qwen +17 to +27% at 97.2–98.5% agreement, GLM +25% at 96.3–97.8%, DeepSeek +25 to +44% at 96.9–98.4% — speed clears the bar, agreement sits on it. A low-bit record in memory is used at any rank but the top, so a third to a half of uses go low-bit, not only a cold tail. Weak points: agreement is interpolated linearly from the all-low-bit figure (if errors add up otherwise Qwen falls to 95–96%); the Qwen trace is one prompt repeated; the GLM speed model is 10–18% generous; DeepSeek's low-bit size and read share are assumed. With a cache twice the size the Qwen gain mostly disappears. **Next, before building the cache work:** one real measurement of agreement — a hook that runs only the experts below the 0.6 line at rank 1 or lower through the low-bit kernel, everything in memory, at thresholds 0.4 / 0.6 / 0.8, against exact output on the 40 prompts. Build (1)–(4) only if that shows 97% or better at a threshold that keeps the predicted 15%.
      - **Stage A measured on Qwen (2026-11-16; branch `mixed-stage-a`, not merged; switch `ORIGAMI_MIXED_EXPERTS`; 40 prompts forced along exact's replies, no hot copy, 24 GB).** Agreement with exact: 96.3% / 97.6% / 98.0% at the 0.4 / 0.6 / 0.8 weight line (0.8 on 12 prompts only: the drive dropped), all low-bit 91.3%: as the offline estimate said. Speed is half of exact (3.6 against 7.3 tok/s) with twice the bytes read (445 against 237 MB per token at 0.6). Two causes in the counters: the 1 GB low-bit cache holds about 740 records and hit 31% of 985K low-bit uses (680K reads of 1.4 MB), and exact misses did not fall (293K against 335K) because read-ahead still fetches the exact record of experts that end up low-bit (`prerouting_false` 180K against 67K) and the exact cache is 1 GB smaller. Open question before any slice B: an expert is needed exact at one token and low-bit at another, so the mode may widen the working set instead of shrinking it. With a 4 GB low-bit cache the 0.6 line still read 408 MB per token at 4.3 tok/s (exact 237 and 7.3). **Parked** (see section 7): the all-low-bit variant is the fast option.
Checked and closed again: lossless coding of the 4-bit records (order-0 entropy of sampled hot-copy records: 7.68 of 8 bits on DeepSeek, 7.79 on GLM, so at most 3–4% saved; the 33% claimed elsewhere is for another weight format).

12. **Qwen decode: half of each token is not GPU work** (measured on the real model 2026-11-14, split now in the request log). Per token, warm: kernels 53 ms, GPU stalled on late expert reads 44 ms, host-only gaps 13 ms; the first decode after a prompt adds 30 ms. Floor with no read waits: about 65 ms (15 tok/s). Qwen is the owner's daily driver, so in this order:
    - a+b. One cause: **a memory limit the Mac cannot hold** (30 GB on a 32 GB Mac swaps; 25.77 GB gives 92–105 ms per token against 174–177). Shipped: the limit is a ceiling, recovery returns to the level that held, the limit this Mac can keep is suggested. `origami tune --memory` measures the fastest limit the Mac keeps (Qwen here: 25.5 GB; 27 GB was cut and swapped). Measured after the 2026-11-15 drive refresh: Qwen 24.0 GB (10.0 tok/s repeat; 25.5 GB the same within 3%), GLM 22.5 GB (1.3 tok/s), DeepSeek 22.5 GB (1.9–2.1 tok/s); 24 GB swapped on GLM and DeepSeek, 27 GB on Qwen. Only cuts macOS asks for count as cuts. Left: each limit is measured once, so repeat close calls; the first decode of a session at a too-high limit still collapses until history has three cuts; a hard-killed tune leaves the server at the trial limit until reload; GLM refuses to start below about 21 GB (one layer group gets fewer than its 16 slots) and says so only in the log; a saved result with no suggestion hides the level-that-held advice. S. Production's limits were set to the measured values on 2026-11-15. Shipped: automatic mode uses the measured limit, holds and climbs back after pressure, and marks a measurement stale. Left: run `tune --memory` in production's data folder for each model, then remove the explicit limits there (the owner's call); `memory_auto_measure` shipped (opt-in; proven on Qwen: starts by itself, a mid-run request is answered in 10.6 s and stops it, backs off a day) and the calibration now picks the highest kept limit (Qwen: 25.5 GB); not yet tried on GLM, DeepSeek or on battery; Cancel on an automatic run leaves the line saying "measuring" until the next try; a real squeeze by another app has only been checked model-free.
    - c. **Starting reads earlier does not pay (2026-11-15, routing trace of 500 tokens and real-model A/B).** Reads are bound by latency (about 2.9 ms each), not by the drive's throughput, and a layer waits if any one of its reads is late: 69 of the late reads per token had already been guessed and started, 28 had not. A two-layers-ahead guess covers 64% of reads and started 17 per token earlier, but its extra router cost 6 ms of kernels per token and removed under one waiting layer: net loss. The same layer's previous-token routing covers 0% (those experts are in memory); co-occurrence two layers back 12%. What is left: lower latency per read (the internal copy: 1.5 ms against 2.3–2.9), and layers 0 and 1 (7–8 ms of GPU idle per token: layer 0 has no guess, layer 1's matches 47%): a token-to-layer-0 table covers 31% of layer-0 late reads (at most 1 ms); keeping both layers' experts in memory would cost 2.8 GB of cache worth more than it saves — not done.
    - d. Copy size: 40 GB serves 85% of misses from the internal drive, 60 GB 99% (10 GB 29%, 20 GB 52%). A copy read takes 1.5 ms against 2.3 ms: 60 GB is worth 2–4 ms over 40 GB.
    - e. Host-only gaps: shipped the greedy sampler shortcut (0.15 ms per token). The 3 ms readout is 2 ms of output projection that cannot be skipped. `decode_prerouting_margin` 0.1 is the default (12% fewer bytes per token, same speed).

Process: at most two agents at once; one full suite and one real-model round per batch of merges; sanitizer runs and repeat loops only for concurrency changes; roadmap and changelog once per batch.

## 0.7 3.1 is complete: nothing is left to build (2026-11-18)

Everything queued for 3.1 has shipped or been decided against; CHANGELOG.md says what shipped, sections 7 and 8 what was parked or closed and why. There is no queue left: what was not built is parked in section 7, each entry with what reopens it. The journey audit (J1–J91) and the outside review of 2026-11-17 are closed: every finding was fixed, decided or is listed below. A structured review of the whole 3.1 diff (pr-agent, twelve areas, then a second pass over its own fixes) ran on 2026-11-18: 47 findings, 38 real and fixed (most with a check that fails without the fix), 7 shown not real, 2 kept as decided (where the `tool_choice` instruction sits, section 7; GLM and DeepSeek closing the think block before `max_tokens`, the owner's decision of 2026-11-17).

The final build was deployed on the development Mac on 2026-11-18 (the owner's word and permission; where and how to roll back: `AGENTS.local.md`). What 3.1 still needs is the owner's hands, nothing else:

- **Release.** 3.1.1 is the release (section 0.8): the changelog is dated and the README has the clone address. Left for the owner: make the repository public and tag v3.1.1.
- **A replug test** (audit J37): `stat -f '%d %i'` on a model file, replug the drive with another drive plugged in first, `stat` again. If the device number changes, nothing breaks (the volume UUID is noted per path and `origami relink` rebinds), but say so here.
- **`origami relink` on a real renamed drive** (checked with stand-in folders only).
- **The dashboard, live** (owner's order of 2026-11-15): a hot copy build from the page and its health line; a real download from the Add form; the page's macOS permission notice on a fresh binary and its "why was this slow" line on a real slow request (both were seen on the deployed build on 2026-11-18 in `/health` and `origami status`, the same data and rules; the page itself was not looked at).
- **After a day of real use:** read the counters named in the limits review (sections 5 and 6 of the owner's copy) and redo the per-phase sum per model; no code is queued.

## 0.8 3.1.1: a second Mac (2026-11-20)

An M1 MacBook Air (16 GB, macOS 27, Command Line Tools 27 with Swift 6.4) with Qwen on an exFAT portable SSD over USB 3 (0.6 GB/s) installed 3.1.1 from the README and runs it: 0.8–1.3 tok/s, 15–23 s to the first word, 142–159 s for a 4,650-token prompt (the README's table). What that Mac showed is fixed in 3.1.1 (CHANGELOG.md): the crash of a Swift 6.4 build on its first request, the hot copy and prompt reuse on exFAT, the empty reply when the model process dies, the early `origami refresh` advice, the unbounded scan for other processes, two download notes and three checks. Left:

- **Report the compiler fault to Swift.** A Void closure that only copies bytes, passed through a `rethrows` helper and specialized, ends in a tail call to `memmove` with a temporary in x21 (Swift 6.3.3 and 6.4, arm64, `-O`). `Tests/error-register-check.py` finds it in a binary; a ten-line reproducer is still to write. S.
- **GLM and DeepSeek on a 16 GB Mac** were not run (200 and 500 GB of files). *Reopens when* someone has them on such a Mac.
- **`context_dir` set by hand to an exFAT folder** still means no reuse at all (the in-memory frontier keeps an unlinked file there, which exFAT does not report as unlinked); `doctor` now says so. Keeping that file on the Mac's own drive would end it. S.
- **`legacyOriginal`**, the expert mode that is the configuration default and that `ExpertExecution` refuses, is still in the planner's checks (25 places). Cleanup only. S.
- The first request after a start takes about a minute on that drive even for 20 tokens (every expert is read once at 0.6 GB/s); nothing is queued for it.

## 0.9 3.1.2: a hot copy for GGUF experts (2026-11-20)

Shipped (CHANGELOG.md; the record behind it is in 0.5, item 11a). Seen at its deployment:

- **A hard link made to a model file gives the model another identity.** The fingerprint binds each file's change time (`LocalModelIdentity.file`), and `ln` changes it: a low-bit variant whose experts file was hard-linked into another folder lost its counts, saved prompts, memory measurement and would have lost its hot copy (all keyed by the fingerprint; the files stay on disk under the old key). Removing the link changes it again. Either leave the change time out when size, modification time, birth time and inode are unchanged, or re-key what the old fingerprint owns at the next start, as `origami refresh` does. S.

## 0.10 3.1.3: the next turn does not wait for the saves (2026-11-21)

Shipped (CHANGELOG.md). Found by a comparison with Strata v0.2.17 on 2026-11-20, which answered a repeated 8K prompt in 1.2 s against Origami's 5.1 s: the time was the wait for the previous reply's prompt-cache saves. Left:

- **A save put off can be skipped.** It keeps its image only while the conversation does, so a state the next request's growth evicts (an older snapshot, such as the system prompt's) is not written, and a later new chat reads that part again. *Reopens when* a request's record shows such a loss (`restore_fallbacks`).
- **GLM and DeepSeek** save through `PromptCache.writeState` inside the request and are unchanged; whether their next turn waits is not measured. S.
- **A new chat right after a reply** still waits for the saves (it may need them from disk, and its growth may evict what they hold).

## 0.11 3.1.4: the first Mac from outside (2026-11-22)

A user ran 3.1.3 on a Mac Studio M4 Max with 128 GB and reported it (issue #1 on GitHub; the README has the numbers): the first M4, the first Mac that keeps every one of Qwen's experts in memory, the first build with Swift 6.4.2. Shipped from it (CHANGELOG.md): no automatic hot copy there, a memory search that neither chooses between limits holding the same experts nor goes on above them, a readable report. Left:

- **`swift build -c release` failed there** with "Could not initialize build system" (Command Line Tools 27.2, where SwiftPM's default build system is the new one); `--build-system native` built it. Not reproduced: the Command Line Tools 27.0 build with the default, and 26 has only the native one. The README names the fallback. *Reopens when* someone has 27.2 at hand: find what the new build system needs (Xcode?) and whether the manifest can ask for the native one.
- **`origami doctor` for GLM and DeepSeek** does not know whether every expert fits in memory without the loaded model, so on such a Mac it says a hot copy will be built where the server says none is needed (`FamilySetup.rows`, marked `shortcut:`). S.
- **Not measured:** the cold start on that Mac without a hot copy against with one (the rule above assumes the difference is minutes, once per start), and anything on 64 or 96 GB.

## 0.12 3.1.5: an upgrade no longer empties the saved prompts (2026-11-22)

Three upgrades in three days each emptied the saved prompts: the owner's next turn of a 50,000-token conversation was read again in nine minutes, on a model that had started with its smallest cache because other apps held memory, and nothing on the page said either. Shipped: saved prompts keyed on a digest of the arithmetic sources with a canary per folder (F2, in place of the per-entry replay first designed); the live "Slowed by" list; old saved prompts shown and removed from the dashboard; permission messages that name the program macOS asks about. What is left:

- **The canary's coverage.** Its probe (Qwen: 600 prompt tokens and 3 decode steps; GLM and DeepSeek: 16 and 2) does not reach long-context attention, a context kept in files, images, or the batched per-expert prompt kernel of GLM and DeepSeek; those are guarded by the digest alone. Widen it only if a digest miss is ever found.
- **A build replaced before its first two idle minutes** leaves saves no later build can check (they stay in its folder and go by the usual rules).
- **`origami cache` and the storage map** say "usable by this build" for a folder that still waits for its canary check; `PromptCanary.notCheckedYet` has the words.
- **"Slowed by":** helper programs show under their process names ("node", "python3.14"), and the memory line is long with three apps named.
- **`origami refresh` drops the model's saved prompts** ("they cannot be re-stamped"): the key holds the model files' identity, which a rewrite changes. Seen on 2026-11-22 (nothing of value lost that day). Key them on the weights' content so a refresh carries them over, as it does the hot copy and the usage counts.
- **Memory that grows past the learned start** (3.1.6): on the real model, watch whether the budget climbs when memory frees, whether macOS cuts it back, and whether the learned start keeps learning from those cuts.
- **The launcher** is described for this Mac only; `origami service install` still installs `origami` itself as the job's program, so an upgrade there asks for the permission again.

## 1. Phase 0: done

H1 and H2 were done on 2026-11-10, H3 on 2026-11-11. H3's screen on the real GLM and DeepSeek (one cold start per variant, replies identical to control everywhere) found no experiment switch that beats the default by 3%: early prefetch, prefetch into allocated slots, arriving-first, read lanes at any worker count, buffered reads, the protected cache, compact shrink, the admission variants, grouped prefill gate/up, Engram overlap and memo are neutral or worse; adaptive prefetch off, warm-after-restore and one prompt slice hurt. They are deleted (S4). Not screened: Qwen's switches, DeepSeek's restore and startup looks, the draft items (parked), and O18, which refuses exact mode. The screen's own findings were fixed and confirmed on the real models the same day (the hedge rule, the restart reply, the second long conversation); what is left of them is parked in section 7 (drive slowdown, GLM resume points). Baselines on 3.1 (short runs, 8 GB copy): first token on a 2,500-token prompt GLM 80 s, DeepSeek 149 s, Qwen 37 s; 200-token decode 0.72, 1.08 and 6.59 tok/s.

Sections 2 to 6 held the queue until 2026-11-18. They are gone: what they held and 3.1 did not build is parked in section 7.

## 7. Parked (and what reopens them)

Nothing here is queued. Each entry is parked until what its **Reopens when** line (or its *Reopens* clause) names: a second fast drive, another Mac, the owner's account or decision, a healthy model drive, or a measurement that shows the need. Work starts again only when an entry's condition is met; it then moves into a queue of its own. Entries keep their IDs (S, F, N, R, P), which the code and the docs cite.

### 7.1 Foundations

#### S1 · Shared expert-streaming core
- **Engines:** all · **Size:** L (several days in three stages; mostly validation) · **Expected:** Family gets hedged reads, multi-drive copies, demand-first scheduling and read timeouts for free; every later read-path change is written once.
- **Why:** The Family engine (GLM, DeepSeek) has a simpler second copy of everything Qwen's engine does for reads: cache, I/O scheduler, fast copy, read-ahead. That duplication is why keeping the engines level is expensive, and it is where the worst 2.x bug lived.
- **What:** Keep each model's forward pass. Extract one shape-driven core into `Sources/OrigamiEngine/Streaming/`, built on Qwen's design: slot state machine with tickets and pins, verified publication, a scheduler with a demand reserve, priorities, cancellation and hedged reads. Family's per-record-size pools and its prompt-sweep eviction (evict what the current prompt sweep will not reach again) become strategies. Stages: (1) GLM and DeepSeek reads go through Qwen's scheduler, with one drive-split estimator and one fast-copy store: about a day, low risk; (2) one cache core: two to three days, the risky part, because Qwen's cache assumes one record size and a fixed 48×512 table while GLM and DeepSeek need several record sizes, wave-based prompt reading and memory lending to the conversation state, and about 100 Family checks are written against today's internals and must move with it; (3) prefetch and admission policies: a day or two, which also settles prefetch into already-allocated slots under memory warning (audit O38, O37) and the per-model prefetch defaults. The switch deletions (S4) are done or parked (2026-11-18).
- **Check:** After every stage, identical tokens, logits and state on all three models (the bitwise checks are the referee). Bytes per device and read wait per token no worse than 3.0. Family hedged reads fire when a drive stalls.
- **Source:** review; audit PAR02, PAR03, PAR04, O13, O51, O12, O37, O38, PFS15, offload-prefetch-byte-credit.
- **Progress:** stage 1 is done and checked on the real models (2026-11-11: replies and reference logits identical to 3.0.1 on all three; external bytes no worse; GLM's first token 78 s → 63 s on the test prompt): the IO scheduler, the drive-split rule (`earliestSource`) and the hot copy's store and keeper are shared code in `Streaming/`; Qwen's hot copy is built while serving. Stage 2a is done: Qwen's cache runs on the shape-driven slot core (`Streaming/ExpertSlotCore`), with the same decisions as before. Stage 2b is done: GLM's and DeepSeek's pools run on the same core (a size class per pool, `Family/Streaming/FamilyEviction` as policy), with the same decisions as before on a seeded trace of every cache call; a record is found only once its read is verified, and returned slots are pinned. Stage 2c is done: GLM's and DeepSeek's demand reads of copy-held records are hedged by the shared `Streaming/ExpertReadReplica` race (model-free checks `family_hedged_read_*`; not yet measured on a real stall). Stage 2 is complete. Stage 3 is settled by H3's screen: no prefetch or admission variant wins, so the defaults stay and the variants are deleted; there is no policy code left to share. S1 is complete except for confirming the last merges on the real models. Multi-drive copies for GLM and DeepSeek are their own entry (F10).
- **Left for stage 2:** one real-model round with bytes per device for the several-copy reads (one rule picks among N sources, several hot-copy stores per model, further copies in the hot-copy format; merged 2026-11-16 and in production since build 3, whose round reported replies and speed only). Then F10's serving side.
- **Reopens when:** a Mac with two fast copy drives for the several-copy round (bytes per device), or the next read-path change, which is then written once in the shared core.

**Parity gaps** (most of them close with S1)
- Qwen has, Family lacks: multi-drive copies (F10); MTP serving (parked, section 7); BF16 state (parked); async copy-on-write checkpoint saves (F7); fused grouped prefill kernels (F8).
- Family has, Qwen lacks: protected cache split (deleted); phase timers and prefetch observer (audit O01, O12: S1 gives both engines one set).
- DeepSeek only, by model design: native drafter (deleted 2026-11-18; drafting is parked); mask-aware indexing (audit O05; GLM's indexer has no candidate mask); Engram memo and overlap (parked, section 7; no other model has Engram); default next-layer prefetch (off on GLM: low hit rate).

#### S2 · Run on other Macs
- **Engines:** all · **Size:** S · **Expected:** unknown; nothing but an M2 Pro 32 GB has ever run it.
- **Why:** The goal is wide Apple Silicon support, but memory budgets, kernel choices and drive speeds were all tuned on one machine.
- **What:** Rent hourly cloud Macs: at least one M1, one base 16–24 GB machine, one 64 GB+ Max and one M5. Run what fits on each (Qwen at minimum) and collect `origami report`.
- **Check:** Each machine loads, answers checkable tasks with reference-matching tokens, and reports tok/s, memory and drive speed. Every failure becomes an R entry.
- **Source:** review; audit O09, O39, PAR05.
- **Reopens when:** the owner rents the cloud Macs (it needs the owner's account); every failure found there becomes an R entry.

#### S3 · Real CI and an automatic parity test
- **Engines:** all · **Size:** M · **Expected:** regressions caught before they reach a model run.
- **Why:** CI has never actually run on GitHub, and parity between engines and releases is checked by hand on real models.
- **What:** Push and get `.github/workflows/ci.yml` green on the macos-26 runner (both jobs are checked locally only: builds, the suite with its skip rule, `--no-gpu`, CLI, dashboard control, fuzzing, Python checks and `Tests/tsan-checks.sh`). On GitHub, find out whether the runner's paravirtual GPU passes `ModelContract.validate(device:)`; if it does not, the Metal checks skip by design, and a check that crashes (instead of failing) on that device cannot be caught by the skip rule. A Qwen synthetic model is still missing (`synthetic_glm_end_to_end` and `synthetic_deepseek_end_to_end` cover the family engine): Qwen's kernels and `ModelContract.validate` fix every dimension at the real model's (48×512 experts, hidden 2,560, a 320M-row PLE table, a 2 GB phase workspace), so it first needs its geometry taken from the config, which is S1-sized work. Capture real-model reference tensors for the vision tower (`vision_encoder_reference` is skipped everywhere). The control endpoints' settings writers (`/v1/config`, `/v1/hot-copy`) live in the `origami` target and are fuzzed only up to the server's type gate.
- **Check:** CI is green on every push; changing one kernel's arithmetic makes the synthetic test fail; the fuzzers run without findings; no check is skipped for want of an input CI could provide.
- **Source:** review; audit P01.
- **First run (2026-11-20, macos-26, private repository):** the runner's paravirtual GPU does not pass `ModelContract.validate(device:)`. Green there: both builds, the error-register check, engine-checks (its GPU checks skipped by rule), fuzzing, the Thread Sanitizer job (63 and 66 minutes; a third run passed its 75-minute limit, now 120) and the scripts that need no GPU. The command-line, dashboard-control, memory-search and ingress scripts start servers and plan a model's memory, which the host check refuses there: `ci.yml` skips them as steps on such a runner (`GPU`). Left: run those four on a self-hosted Apple-silicon runner, or give the model-free scripts a test switch past the host check; the parity test.
- **Reopens when:** the repository is public (runner minutes are free then).

#### S5 · Finish per-machine calibration
- **Engines:** all · **Size:** M · **Expected:** unmeasured on other Macs; to be confirmed neutral or better on the development Mac.
- **Why:** 3.1 measures each drive per Mac (1, 4 and 8 readers, about 3 s a drive, saved by volume; at first start, `origami setup`, `origami tune`, after 4 weeks, or when serving disagrees 3x for 10 minutes) and replaced the development Mac's drive curves with a neutral one. Not yet run on a real model. Other numbers from that Mac remain: reader and lane counts, the GPU work budget's cost figures, spin budgets, memory reserves, the reuse calibration (`layerReadCost`, reuse probabilities), kernel row floors. Only one kernel choice is both proven bit-identical and worth tuning: the grouped MMA floor on M2–M4 (the M5 tensor-ops path is not bit-identical and must not be tuned).
- **What:** Confirm on the real models that bytes per drive and speed are no worse than with the old curves (Qwen's model drive is measured with single reads while its checkpoint reads are nine spans; a measurement taken in a slow period of the drive stays until the disagreement rule clears it). Tune the grouped MMA floor with an in-engine harness that checks bits during the benchmark. Fold the memory tuner (audit O09) and `origami tune`'s result into one calibration file with one staleness rule. Derive the remaining numbers from measurements where a wrong value costs speed.
- **Check:** On the S2 machines the calibrated values differ from the defaults and tok/s is equal or better. Bitwise checks pass for every chosen variant.
- **Source:** review; audit O09; 2026-11-12.
- **Reopens when:** the S2 machines exist, and a healthy model drive allows the real-model confirmation here.

### 7.2 Faster, output unchanged

Every entry must give identical tokens, logits and state. Family hedged reads, read timeouts and demand-first scheduling come with S1 and are not repeated here.

#### F4 · Qwen: faster lone misses and less host waiting
- **Engines:** Qwen · **Size:** M · **Expected:** Qwen's own calibration reads 1.57 GB/s at queue depth 1 vs 2.93 GB/s at depth 4; end-to-end gain unmeasured.
- **Why:** A single missing expert is read as one request; hedged reads only start when something polls; the router wait pins a CPU core.
- **What:** Split a lone-miss record read at projection boundaries; the down projection is needed last, so it may arrive later (audit DO08). Signal the late expert group from I/O completion instead of a host wake. Start hedges from a timer. Replace the pinned spin with an adaptive one (on Family, turning spin off cut CPU work 75% at about the same speed, audit O20).
- **Check:** Bitwise; latency per miss; CPU seconds per token; interleaved tok/s.
- **Source:** review; audit DO08, O20.
- **Reopens when:** a healthy drive shows lone misses are still a measurable share of Qwen's decode (splitting them was neutral on one fast drive).

#### F6 · Token-keyed read-ahead for the first layers
- **Engines:** all · **Size:** M · **Expected:** unknown.
- **Why:** Early layers route mostly on the token itself, yet their reads only start when routing runs. The current next-layer guess is fairly accurate but gives little lead time.
- **What:** Start the first layers' reads the moment a token is sampled, from a token-to-experts table built from usage counters. Add a same-layer previous-token predictor next to the cross-layer guess. Real routing stays in charge: every miss still loads the real expert.
- **Check:** First replay saved routing traces: share of early-layer experts predicted and lead time gained. Then bytes arriving before use and interleaved tok/s. Bitwise.
- **Source:** review; audit DO07, offload-cross-layer-predictor.
- **Reopens when:** a replay of saved routing traces (no code) shows the first layers' experts can be predicted from the token with lead time to spare.

#### F7 · Cheaper conversation saves
- **Engines:** all · **Size:** M · **Expected:** up to ~15 GB of writes avoided during one long Qwen prompt; time saved unmeasured.
- **Why:** Qwen writes conversation frontiers densely (up to ~15 GB during a long prompt), saves more than one image per reply, and the next request waits for saves. Family saves are synchronous (small states take 43–45 ms; long ones are unmeasured).
- **What:** Write frontiers at turn boundaries and geometric spacing; one saved image per reply; let the next request start while saves finish. Give Qwen idle-time work: mirror rebuild, cache warm (an idle unload is the `idle_unload_minutes` setting). Checkpoint a cancelled long prefill so a retry resumes. Give Family async copy-on-write saves like Qwen's.
- **Check:** Bitwise restore; bytes written per request; first-token time for the request after a long save; a cancelled prefill resumes where it stopped.
- **Source:** review; audit O07, PAR06.
- **Reopens when:** a long Qwen prompt is measured to write several GB of frontiers, or the request after it is seen waiting on saves.

#### F8 · Family: fewer GPU dispatches
- **Engines:** GLM / DeepSeek · **Size:** L · **Expected:** unmeasured; shows mostly in prompt reading and on big-memory Macs, where reads stop dominating.
- **Why:** Prompt reading issues 6 GPU dispatches per expert per 2,048-row slice, and the batched kernel covers only up to 8 rows. A DeepSeek token takes ~1,800 dispatches and 130–160 command buffers.
- **What:** A batched prefill expert kernel for any row count (as Qwen's grouped kernels do), cached encode plans, and fused small attention dispatches, all with the same summation order.
- **Check:** Bitwise state and logits; dispatches and command buffers per token; prompt-reading time.
- **Source:** review; audit PFS14, PFS16, O36.
- **Reopens when:** a big-memory Mac (S2) shows prompt reading bound by dispatches rather than reads.

#### F11 · Hot copies the owner steers: always the most used, visible, on every drive chosen
- **Engines:** all · **Size:** L · **Expected:** disk reads move to the fastest drives and stay there as usage shifts; the owner sees what a bigger or smaller copy buys.
- **Why:** owner's order of 2026-11-17. Built (2026-11-17): re-ranking in place, the popularity graph, copy places on every drive chosen (`hot_copy_more`) with a size and graph per drive on the page. Left: the check on the real models.
- **What:** Measure it on the three models with two copy drives, and compare per-drive bytes after a rebuild with the predicted coverage (from F1, whose sharing between models shipped as `copies-share-drive`).
- **Check:** Bitwise on all three models. A workload change moves the copy's contents without a rebuild and without a dip in the share of reads from the copy; with two copy drives, bytes per drive follow their speeds; unplugging a copy drive fails no request.
- **Source:** owner, 2026-11-17.
- **Split reads of one expert across two drives: opt-in** (`SplitReads`, `ORIGAMI_SPLIT_READS=1`). Measured 2026-11-17 on one fast drive (Qwen, 18 GB hot copy, 200 decode tokens): about 670 splits per 200 tokens on about 1 s of lone waits in a 21 s decode, 9.72 tok/s on against 9.75 off. It becomes the default only if a Mac with two or more fast copy drives shows a clear decode gain with replies identical, and `lone_demand_wait_seconds` (always counted) is a large share of decode time there; otherwise it is deleted in 3.2.
- **Reopens when:** a second fast drive is at hand: the two-drive round on the three real models.

#### F10 · Multi-drive copies for GLM and DeepSeek
- **Engines:** GLM / DeepSeek · **Size:** M · **Expected:** a second fast drive is used as Qwen uses one; no gain on a Mac with one fast drive.
- **Why:** All three engines build and read a copy on every drive chosen (`hot_copy_more`, `HotCopyPlaces`). Since `copies-dashboard` the copies start at the sum of their places' measured rates and each record goes to the place expected to finish it first (model-free check `copy_places_follow_speeds`). Left: new copies have no `verify --deep` against the checkpoint (R12), and the measurement.
- **What:** `verify --deep` for every place; measure on the real models with two copy drives.
- **Check:** Bitwise; with two copy drives, bytes per device follow the drives' speeds; a drive that disappears is skipped.
- **Source:** S1 stage 2 (2026-11-11).
- **Reopens when:** a second fast drive is at hand (the same round as F11).

### 7.3 Faster, opt-in (changes output)

Everything here stays off by default and ships only with quality measured on checkable tasks that have enough thinking room. New ideas (N1–N3) are tested offline first with the same output-error measure used in audit O29 and O52.

#### N4 · Harder quality tests for the fast modes
- **Engines:** all · **Size:** S · **Expected:** confirms or corrects which fast modes each model offers.
- **Why:** Since 3.0 each model offers only the fast modes that clearly helped it (Qwen: faster, fastest; GLM: faster; DeepSeek: dynamic-fastest), but the evidence is 10–14 easy single-turn tasks per model. DeepSeek's "faster" was only run once (4 tasks, no saving).
- **What:** Score long-form and coding tasks per model in each offered mode and in the hidden ones worth a second look (DeepSeek faster, GLM dynamic-fastest is known worse). Update the one table in `QualityDial.offered` from the results.
- **Check:** At least 10 checkable tasks per model including coding and long-form; score and time against exact, interleaved.
- **Source:** audit PAR08, O16, O54, R43.
- **Reopens when:** a fast mode is to be offered, withdrawn or made a default: this scoring decides it, and the other entries of 7.3 wait for its tasks.

#### N5 · Resident Q4 dense weights for GLM and DeepSeek
- **Engines:** GLM / DeepSeek · **Size:** S · **Expected:** +3.4 GB of expert cache, 9–13% less time.
- **Why:** Smaller resident matrices free memory for experts (183 extra slots on DeepSeek), and hard-task quality held in a first screen.
- **What:** Finish the existing opt-in (`ORIGAMI_FAMILY_RESIDENT_Q4_PACK`) and offer it as a setting or as part of the presets.
- **Check:** At least 10 scored tasks per engine at production memory, and fewer slow-drive bytes in interleaved runs. Drop it on any quality loss.
- **Source:** audit O40, O09.
- **Reopens when:** N4's tasks exist to score it.

#### N6 · Fewer experts with a thinking budget
- **Engines:** all · **Size:** S · **Expected:** two experts solved one task in 10.5 s vs 44–71 s with six; quality depends on the budget.
- **Why:** Fewer experts need more thinking room. With two experts a code-tracing task failed at 64 thinking tokens and passed at 96 (127 s).
- **What:** Expose expert count and a minimum thinking budget as one setting instead of two independent knobs.
- **Check:** Scored tasks per engine, including the known code-tracing case (audit R24).
- **Source:** audit O27, O26, R24.
- **Reopens when:** N4's tasks exist to score it.

#### N1 · Importance-ordered experts
- **Engines:** all · **Size:** M · **Expected:** unknown; could halve the bytes read for low-weight experts with no extra disk.
- **Why:** Today a low-weight missing expert is either dropped (cheap, lossy) or read in full (exact, expensive). There is nothing in between.
- **What:** An expert's hidden units can be reordered without changing its math: gate/up rows and down columns move together, in blocks of 32 to stay compatible with MXFP4 block scales (or the format's own scale groups). Write the fast copy with the most important blocks first, using activation statistics (the activation observer already exists). Then load low-weight experts as a prefix, for example half. Full records remain numerically equivalent.
- **Check:** Offline first: error of a half-prefix expert vs dropping it. Then confirm full reordered records give the same tokens; reordering changes the summation order, so if they are not bit-identical the reordered copy itself is opt-in. Then scored tasks and bytes per token.
- **Source:** review; audit DO16.
- **Reopens when:** N4's tasks exist, after the offline error measurement.

#### N3 · Quality dial for prompt reading (Family)
- **Engines:** GLM / DeepSeek · **Size:** M · **Expected:** unmeasured; aimed at the 142 s median first-token wait.
- **Why:** Prompts always run exact, and a 300-token tool result touches most experts of every layer, which is close to reading the whole model.
- **What:** Optionally read only the experts that carry most of the routing weight across the new prompt rows, and skip the rest.
- **Check:** First-token time and bytes read while reading the prompt; scored tasks that include long tool results.
- **Source:** review.
- **Reopens when:** N4's tasks include long tool results.

#### N2 · Two-layer storage of the 4-bit codes
- **Engines:** all · **Size:** L · **Expected:** unknown; gives low-bit copies without audit O29's ~70 GB of extra disk.
- **Why:** A low-bit copy of rarely used experts is gentler than dropping them (audit O29), but a separate copy costs disk space.
- **What:** Store each record as a coarse half and a refining half. The coarse half alone is a low-bit version of the same expert; reading the refining half later upgrades it to exact (bitwise once recombined). This allows "run coarse now, refine in the background" for late experts.
- **Check:** Recombined records are bitwise identical. Coarse-only error measured as in O29, which found 2-bit too lossy for Qwen and a last resort for GLM/DeepSeek, so the coarse half may only suit low-weight experts. Then scored tasks.
- **Source:** review; audit O29.
- **Reopens when:** N1's offline measurement shows a coarse half is usable.

#### N7 · JSON mode: show the schema to the model, and the keywords left out
- **Engines:** all · **Size:** S · **Expected:** better answers under a schema; fewer refused schemas.
- **Why:** 3.1 guarantees valid JSON when a request asks for it (`json_object` or a JSON Schema subset, on every wire and model, after any thinking, with tool calls still possible). The schema only constrains sampling: the model is not told it, so the answer's content depends on the prompt describing it (DeepSeek's template has a `response_format` slot that is not filled). Refused today: `pattern`, `minLength`, `minimum`/`maximum`, `oneOf`, tuple `items`.
- **What:** Put the schema into the prompt where the template has a place for it, or as a short system note; add the string and number bounds.
- **Check:** Scored tasks with a schema, with and without the schema in the prompt; the new keywords in the acceptor table.
- **Source:** review; 2026-11-12.
- **Reopens when:** a user reports weak answers under a schema, or a schema Origami refuses.

### 7.4 Reliability and compatibility

Only what 3.0 has not fixed yet. Check each against the 3.0 code before starting.

#### R22 · Tuning that does not fail: the real-model part
- **Engines:** all · **Size:** M · **Expected:** `origami tune` gives the same choice on a rerun, and the speed it shows is one a real first reply reaches.
- **Why:** owner's order of 2026-11-16. Shipped in 3.1 (branch `tuning-never-fails`): the time limit is sized from the first measurement (the default grows up to 30 minutes, an owner's `--minutes` is kept with fewer candidates), every finished round is kept in `tuning.json` (`partial@<key>`) and the next run within a day continues it, "the defaults are fastest" is a recorded result, every ending says what was measured, kept and next, and `origami start` retries a failed start. Each ending has a stub-server check (`Tests/tuning-flow-checks.py`). Still open: single runs on a drive that throttles decide the winner, and the measurement is the warm short-prompt speed (seen 2026-11-16 on Qwen: 13 tok/s measured, 9 tok/s on the first real request, a 7,403-token Hermes prompt with nothing restored). `tune --memory` now keeps its finished trials the same way (`memory-partial@<key>`, J54); `--tradeoffs` was not reworked.
- **Endings (inventory 2026-11-17, production history: `aborted` 4, `too_noisy` 2, `already_fastest` 2, `comparison` 2):**

  | Ending | Before | Now |
  |---|---|---|
  | Time limit before any candidate | "needs about N minutes; choose a longer limit", nothing kept | the default grows to fit; an owner's limit measures what fits, keeps it, the next run continues |
  | Time limit mid-run | "Stopped: Time limit reached", nothing kept | the round in progress finishes, decided from what it has, rounds kept, next run continues |
  | Time limit between candidates | "No reliable improvement" with "tested 1 of 2" | "Incomplete", the measured candidates kept, next run measures the rest |
  | App request, restart, counter reset, pressure, changed model or memory | "Stopped: <reason>. Nothing was changed." | the same reason, finished rounds kept, next run continues |
  | Server gone mid-run | raw URL error, exit 1 | "Stopped: Origami stopped answering …", rounds kept |
  | Ctrl-C / Cancel | "Stopped. Nothing was changed." | rounds kept (saved as each ends); Cancel says so |
  | Too noisy | "Inconclusive: <reason>" | the same, with measured / kept / next |
  | No candidate faster (dashboard) | not recorded; the page offered the same run | recorded; "Tuned: defaults are fastest (date)" |
  | Faster | "+N%", review or applied | the same, with measured / kept / next |
- **What:** Decide from interleaved repeats and bytes read per device, not one timing; show a production-like speed (a long prompt, first reply) next to the warm figure, saying which is which.
- **Check:** On each real model, three runs in a row on a warm and on a throttling drive give the same choice.
- **Source:** owner, 2026-11-16.
- **Reopens when:** a healthy model drive: three runs in a row per model on a warm drive, then on a throttling one.

#### R7 · Built-in TLS
- **Engines:** all · **Size:** M · **Expected:** the key never travels in clear on a LAN without a second program.
- **Why:** 3.1 made LAN mode safe against slow or stuck clients (header and body deadlines, a minimum upload rate, per-device caps, fair body buffers, a bounded writer that drops a client which stops reading, a wrong-key limit) and documents a TLS proxy in front (Caddy or nginx, with `trusted_proxy loopback`); the recipes have not been run against a real proxy here. Built-in TLS was not done: the front uses raw sockets with blocking reads and writes in every path, so it needs a transport rewrite or an in-process TLS bridge, about 600–900 lines.
- **What:** Run the two proxy recipes once and fix what they show. Then decide whether built-in TLS is worth its size: certificate and key in settings, a helper that makes a self-signed certificate and prints its fingerprint, plain HTTP refused on that port.
- **Check:** With TLS on, a request works, plain HTTP is refused, and the key is not accepted in clear.
- **Source:** review; audit PFS20; 2026-11-12.
- **Reopens when:** a user needs the LAN mode without a proxy in front, or the owner decides it is worth its size.

#### R8 · Filesystems and macOS versions
- **Engines:** all · **Size:** M · **Expected:** works beyond APFS and the latest macOS.
- **Done (2026-11-18):** exFAT and case-sensitive APFS disk images on macOS 26: model moves (exFAT refused the exclusive rename: fixed), stores, and the file features probed. exFAT: no hole punch (the `punchable` probe says so), no clones (saves write whole images), no sparse files (a save or context file takes its full size at once), xattrs as `._` files, F_FULLFSYNC works; a folder Origami makes shows as private (0700, the user's) although the drive ignores ownership (`noowners`, `MNT_IGNORE_OWNERSHIP`): saved prompts are now off there (2026-11-18; no setting turns them on). Case-sensitive APFS: no difference found. Minimum macOS 15 and the macOS 26 SDK are in README's Requirements.
- **Left (a model, another Mac):** serve a model and save prompts with the model, `context_dir` and the hot copy on exFAT (the engine checks' scratch folder ignores `TMPDIR`, so they ran on the internal drive); HFS+; the oldest supported macOS (S2).
- **Source:** review.
- **Reopens when:** another Mac (S2) or an HFS+ drive is at hand; a model served from exFAT needs a test window.

### 7.5 Release

#### P3 · README numbers: what is left
- **What:** The README and docs carry the 2026-11-15 numbers from the development Mac. Left: numbers from other Macs (S2); INTERNALS.md still holds 2.x measurements; two licences could not be confirmed and the README says so (mlx-community's Qwen 4-bit states none; the ISTA Qwen GSQ-RCO card says the base model's licence in its text and apache-2.0 in its metadata).
- **Reopens when:** numbers from other Macs exist (S2); the two licences when their publishers state them.

#### P5 · Signed, notarized binary and a Homebrew tap
- **Engines:** all · **Size:** M · **Expected:** one-command install.
- **Why:** Users should not need the macOS 26 SDK to try Origami, and unsigned binaries trigger Gatekeeper warnings.
- **What:** Sign and notarize release builds in CI; publish a Homebrew tap.
- **Check:** `brew install` on a clean Mac runs `origami report` without warnings.
- **Source:** review.
- **From the journey audit:** until then the README needs the real clone URL and a symlink instead of copying the binary and bundle (J45, section 0.7).
- **Reopens when:** the owner has an Apple developer certificate for signing, and the repository is public (S3).

### 7.6 Smaller items

- **Mixed exact and low-bit experts per token** (0.5 item 11c; branch `mixed-stage-a`, not merged). Quality held (97.6% agreement at the 0.6 line against 91.3% all low-bit), but an expert is needed exact at one token and low-bit at another, so both records compete for the same memory: twice the bytes of exact and about half its speed, also with a 4 GB low-bit cache. *Reopens when* the choice can be made per expert for a whole reply (one record per expert in memory), or on Macs where all exact experts stay in memory.
- **Draft and speculative decoding** (audit O21, O22, O43, O44, O45, O46, O47, O49, RD03, R40, PAR05). Drafting saved 14–29% of reads but showed no time gain: it mostly saved reads the fast copy had already made cheap (external-drive bytes fell only 2%). MTP on 32 GB costs ~2.5 plain steps per check; Qwen pair verification cost 2.02x a plain step. *Reopens when* the drive is fixed and H3's retest shows a time gain, or on 64 GB+ Macs where experts stay in memory.
- **CPU expert lane** (O23, offload-cpu-gpu-lane). GPU work was 2.5 s of 43–61 s replies, and a CPU lane reads the same cold bytes. *Reopens when* a machine is compute-bound (all experts resident) and spare CPU is measured.
- **Lower-precision Family state, BF16 like Qwen** (O24, PAR06). Saves ~419 MB at 128K but changes output. *Reopens for* contexts of 64K+ where memory or save time is measured to matter.
- **DeepSeek activation fidelity** (RD06). Origami skips some FP4/FP8 activation round trips the original model uses; the effect on quality is unmeasured. *Reopens when* N4's task scoring (7.3) exists; close it if 10+ scored tasks show no difference. GLM's original arithmetic is unchecked.
- **Index candidate reuse or a smaller retrieval budget** (DO18). Changes output. *Reopens when* long-context index time is measured to matter.
- **Finer prefill dependency pruning** (PFS12). Upper layers and non-final rows are already skipped. *Reopens when* a simulation on long prompts predicts a real cut.
- **Wider prefill grouping** (PFS11). Cannot help short suffixes. *Reopens with* a measured suffix above 4,096 rows.
- **Branch/prefix cache** (PFS08). The one recorded miss was a selection problem (R27). *Reopens when* traces show repeated branch misses.
- **Non-exact kernel fusion** (PFS16, the parts that change summation order). *Reopens when* compute, not reads, dominates.
- **Prompt summarizing** (O31). Changes the conversation. *Reopens on* user demand, as an opt-in.
- **Continuous batching** (O32). Serves throughput, not single-request speed. *Reopens if* several users share one server.
- **Tool-turn priority aging** (audit S10). User turns deliberately go first. *Reopens if* tool turns time out behind user turns.
- **Reply-text delta path** (PFS19). *Reopens when* profiling long outputs shows text rebuilding matters.
- **Waits: what is left after the bounds** (R2, 2026-11-18). With the bounded GPU wait, three buffer-retention checks failed about 1 run in 15 until given the 2 s the engine allows elsewhere for memory to come back; why memory returns later with the watchdog is not understood. Qwen's GPU bounds have no end-to-end check (there is no synthetic Qwen model). *Reopens when* a real-model run shows memory coming back late, or with a synthetic Qwen model.
- **Experiment switches waiting for a measurement** (S4, 2026-11-18). Four stay opt-in, each to become the default or be deleted by 3.2: `ORIGAMI_SPLIT_READS` (no gain with one fast copy drive, 9.72 against 9.75 tok/s; made for two or more), `ORIGAMI_FAMILY_SAME_DRIVE_HOT_COPY` (the single-read records below), `ORIGAMI_FAMILY_LAZY_STARTUP_WORKSPACE` (let DeepSeek start at 11 GB with 8K context; what it costs the first prompt step is unmeasured) and `ORIGAMI_FAMILY_REFRESH_IDLE_OVERHEAD` (O42: 490 MB of overhead allowance kept after prompt reading against 115 MB in a fresh process; what refitting between requests does to speed is unmeasured). *Reopens with* a Mac with two fast copy drives, an interleaved real-model run, and a 16–24 GB Mac (S2) respectively.
- **GLM resume points inside a multi-slice call** (R21, decided 2026-11-18). A prompt call reads its slices layer-major, so no slice has every layer before the call's last layer, and a cancel goes back to the call's start (40% into a 7,300-token turn re-read the whole turn). A point inside the call means keeping the call's four hidden streams at the layer reached until the resend: 16 × hidden bytes per row, about a third of a slice's working memory (4 (11h + e + 24) bytes per row), so up to ~0.13 GB per 2,048-token slice on GLM's 32 GB plan (~0.5 GB for that turn), taken from the expert pool until the next request, plus a mid-call resume path in the forward pass. Shorter calls instead read every expert once more per extra call. Hedged reads cannot cover the model drive's slow periods (a record outside the copy has no second source; only a larger copy or a refresh helps). *Reopens if* the log shows cancels inside long prompts are common in real use.
- **Drive slowdown: the rest** (R15). The tally per drive, the warn-ahead line and the healthy-rate floor shipped. Left: the 0.2 `slow_reads` threshold has not been compared with the read test on a slowed drive; a low-bit variant's experts file is counted with its model's folder; the tally is keyed by folder path (a moved model starts a new count; the relink, R25, owns re-keying). Bytes at server start and the keeper's reads while a copy grows are not measured. *Reopens with* the drive's next slow period: run the read test beside a server's tally and compare.
- **Qwen: the per-tile attention command** (R17). The expert layer command was cut into buffers of about 17 ms; the per-tile command (attention, router and shared expert of one 256-row tile) is still one buffer, estimated at 9 ms on a linear-attention layer and possibly over the 25 ms budget on a sparse-attention layer at 8K keys. *Reopens with* a Metal System Trace of the real model: if a tile's buffer nears 25 ms, split it after attention (`afterAttention`) with a submit that does not wait; bitwise on the real model.
- **Qwen's exact fusions and bookkeeping** (F9). Gate and up fused in the expert kernel; the readout in the last layer's command with GPU argmax at temperature 0; layer-invariant victim minima (48x less scanning); a small LRU for embedding and PLE rows; an online reuse-probability table; a 16 KiB slot stride. Each claims a small per-token saving; none is measured, and Qwen's prompt and decode are GPU-bound. *Reopens with* a per-token CPU and GPU profile of the real model that shows one of them above 1% of a token; each then needs a bitwise check.
- **Family: single-read records and Engram overlap** (F3; audit O14, O25). The contiguous same-drive copy (`ORIGAMI_FAMILY_SAME_DRIVE_HOT_COPY=1`) cut task time 5–22% in one test, before the drive was fixed and the internal hot copy existed; DeepSeek's Engram reads still wait instead of overlapping GPU work (unbuilt). *Reopens with* an interleaved run on the real GLM and DeepSeek (A, B, A, B) with read calls per token, CPU time per miss and bytes per device.
- **GPU keep-alive on GLM and DeepSeek: the rest** (F5, 2026-11-16). On by default (`gpu_keepalive` auto): GPU time per 200 tokens 37.7 → 25.9 s on GLM and 37.8 → 23.2 s on DeepSeek, replies identical. Unmeasured: power; the GLM low-bit variant; skipping it when every expert is resident (it costs GLM 1–2 ms of wall time per token there). *Reopens with* a power meter, a low-bit GLM run, or a Mac where every expert stays in memory.
- **Release references from the models' own files** (P2, 2026-11-12). The stored GLM and DeepSeek references come from other model files, and the thresholds (0.99; 0.999 decode against prompt) are tighter than llama.cpp agrees with itself (cosine 0.984 on a 316-token prompt; 0.92–0.97 between a position read as prompt and decoded, on GLM's shaky positions; Origami matches llama.cpp's tokens). New references need llama.cpp run on the real files (top-k logits kept); the half-precision prompt question needs scored tasks (section 4). *Reopens before* the next release check: store them, set each threshold from llama.cpp's own prompt-against-decode gap, and confirm a one-bit kernel change fails.
- **Small-memory Macs on real hardware** (R6). The plan is checked on simulated 8, 16 and 24 GB Macs (`plan_mac_sizes`, `plan_small_macs`): Qwen starts at 8K from 16 GB and refuses at 8 GB and at 32K on 16 GB with the line that names what would fit; GLM and DeepSeek refuse with "does not fit this Mac at this context" and a smaller context or a model that needs less memory. Unknown: whether macOS swaps or kills the process at the floor under real background load; pacing cache recovery by read cost (O04) waits for that. *Reopens with* a 16–24 GB Mac (S2): start, then a long reply under background load without swap growth or being killed.
- **Qwen decode against build 6: about 1% slower, inside the noise** (2026-11-18). Replies identical; prompts the same speed; decode 2–3% slower before the layer-0 pin was removed and split reads made opt-in, about 1% after, on a drive that was throttling in every round. Suspect: the cache sized at prompt time (commit 558c6a8). *Reopens when* a healthy drive allows four interleaved pairs, with and without that sizing.
- **Long sessions: replaying the empty-reply session** (audit J62). The two paths that gave empty content now answer with the reasoning and say so (`completion_tokens_details.empty_answer`); the last request of the recorded 33-request session was not replayed on the real model. *Reopens when* a session ends in an empty reply again, or with the next real-model test window.
- **The `tool_choice` instruction sits in the last user message** (structured review, 2026-11-18). It stays put through a tool loop, so a loop reuses its saved prompt; a client that sends `required` or a named tool on every request reads the previous turn's tool results again at each new user turn. The system-prompt placement had the opposite cost (a full re-read when the choice changes). *Reopens when* a client that always sends it is in real use.
- **GLM and DeepSeek: a session that moves on to a prompt whose save already exists may not save the conversation again** (noted by the sessions work of 2026-11-18; not reproduced). *Reopens when* a long session shows a prompt read again that the "why was this slow" line puts down to "it had not been saved".
- **Small gaps left by the structured review's fixes** (2026-11-18). A save's give-back can lose the hot copy's writer lock to a keeper swapping record after record and gives up after 2 s (the save then asks the next copy); a memory search from the terminal killed without its closing restore (`kill -9`) leaves the next terminal search refused until the server restarts; an in-place change is skipped while any other copy object of the folder is open in the same server (a status read), and the state then says "another process". *Reopens when* any of them is seen in use.
- **Qwen's guarded first split range runs inline before the hedge timer** (outside review; `SplitReads.swift`). Split reads are opt-in and gave no gain on one fast drive. *Reopens with* the decision on `ORIGAMI_SPLIT_READS` (experiment switches, above).
- **Family read overlap: the counters before any change** (study of 2026-11-17). The study's first fix shipped (read-ahead survives memory pressure); its other ideas need per-layer counters first (busy time per drive by what the serving thread waits on, demand misses per drive and layer step, the split planner's state). *Reopens when* a real-model round on a healthy drive is planned for GLM or DeepSeek decode.

## 8. Closed — do not repeat

- **Saved prompts on a drive mounted with ownership ignored** (owner, 2026-11-18): off on exFAT and FAT, which have no file ownership; on an external APFS or Mac OS Extended drive (macOS mounts those with ownership ignored by default) they stay on, with one line at the start saying every user of the Mac can read them.
- **Locate re-binding copies to a checkpoint with identical config, index and shard sizes** (outside review; owner, 2026-11-18): accepted as it is; the safe fix would rebuild the copy on every relocate.
- **Cache eviction/admission tuning** (O17, offload-protected-lrfu): a perfect cache would cut only 10–17% of decode time; frequency and protected split saved 12–14% of reads with no clear time gain. Only H3's single retest of O17 margin variants remains.
- **Megakernels**: the GPU is a small share of time; they do not touch the expert-read wait.
- **Expert substitution** (O52): on all three models every substitute caused more output error than dropping the expert.
- **Delta loading between experts** (O53): experts in a layer are unrelated, so differences are as large as the experts.
- **Parallel chunked record reads, contiguous record layout for decode** (2026-11-12, burst replay on the real drives): 0.90–1.04x of whole reads; a decode burst already runs both drives at their ceilings.
- **Decode read coalescing** (offload-coalesced-read-planning): 1.5% fewer read calls, same bytes. Contiguous records (parked, section 7) cover the rest.
- **Lossless compression of expert bytes** (O30): saves 1–10% of bytes (O30's scale trick 4.4%), under 1% of request time. The unbuilt branch was not carried into 3.0.
- **Reusing an expert's output from the previous token** (O28): none of 4,920 adjacent comparisons reached 0.9 similarity.
- **Separate prefetch byte budget** (offload-prefetch-byte-credit): speculative reads were 0.43% of decode reads and wasted 0.08%; S1's demand reserve handles interference.
- **DeepSeek prefetch on or off at its memory limit** (2026-11-11): A/B/A/B on the real model, 84.3 / 85.2 / 87.6 / 83.8 s and the same bytes: no difference, the default stays.
- **Coalescing usage-count writes** (O08): 17–19 ms per request.
- **First-touch page preparation** (O19): 0.16–0.24 s in 180–464 s runs.
- **More prompt-encoding caches** (PFS06): the whole phase is 0.0063% of service time.
- **Reordering prompts to get cache hits** (PFS07): it changes token IDs; fix the client that sends unstable prompts instead.
- **Treating DeepSeek "fastest" thinking-marker leaks as a parser bug** (R43): the model does it in that mode; count it against the mode (N4).
- **Family early guesses and the curve-based copy/model split** (2026-11-17, GLM, 24 GB copy, healthy drive, 200 decode tokens, replies identical): default 204.1 s (1.04 tok/s); guesses issued at the router result and yielding to demand 219.9 s (0.96 tok/s, 8% slower); the split by measured drive curves 202.5 s (no change). Both switches deleted; the decode overlap counters stay.
- **Keeping layer 0's 200 most-used Qwen experts resident across prompts** (2026-11-17, Qwen, 18 GB hot copy, 200 decode tokens, interleaved, replies identical): 9.75 tok/s with it, 9.81 without: no gain and about 0.6% slower decode, because the 0.55 GB pinned comes out of the cache every other layer shares. Reverted; `cache_layer0_misses` stays as the measurement that would reopen it.
- **Bounding model shutdown's read joins** (R2, 2026-11-18): not on a request's path; the process is ending.
- **The eviction cost of a rejected hot-copy record and a cancellable restore** (R21): both shipped before 2026-11-18 (a rejected record costs a model-drive read until repaired, `family_hot_copy_repair`; a cancel during a restore lets it finish, `synthetic_restore_cancel_keeps`).
- **Shrinking optional caches when a denied expert slab stays denied** (R5, 2026-11-18): the optional caches (tool anchor, observation and trace reserves) already come off the pool's target before any slab is sized (`poolTarget`), so a slab still denied after its wait is denied for fixed owners or a lowered budget, which no cache shrink lifts; no such failure has been seen since the wait-and-retry shipped.
- **Splitting the last long functions** (S6, 2026-11-18): `FamilyBackend.attemptBody`, `ServingProtocol.parse`, `RuntimeConfiguration.load` and Qwen's decode steps; a refactor with no user effect, done when someone changes them for a reason. The DeepSeek native drafter (about 600 lines with its checks) was deleted instead of given a seam; Qwen's verifier cost probe stays as drafting's measurement, and the two fault-injection self-checks stay beside the private file descriptors they need.
- **Adopting the fast copies 2.x made** (R12, 2026-11-18): 3.1 builds a hot copy while serving, about a minute per 40 GB, and the 2.x upgrade notes already tell users to delete the old copies in the Models view; adoption needed a full re-read of every record against two formats, more code and risk than one rebuild costs. No 2.x copies are left on the development Mac.
