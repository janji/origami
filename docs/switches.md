# Environment variables (`ORIGAMI_*`)

Experiment switches are temporary: each one either becomes the default or is deleted by the release it names, and the
open ones are parked in [ROADMAP.md](../ROADMAP.md) section 7 with the measurement that decides them. Settings are permanent; use `origami config` for them rather than
variables where you can.

Status: **setting** is a supported user setting; **experiment** is an opt-in trial that may change replies or speed
and is not yet proven; **debug** is for measuring or checking the engine and should stay at its default otherwise;
**internal** is set by Origami itself; **retired** does nothing any more.
A switch that is "on" with `1` is off for any other value unless the table says otherwise.

## User settings

| Name | Default | What it does | Status |
|---|---|---|---|
| `ORIGAMI_<SETTING>` | unset | Any setting that `origami config` lists, by its name in capitals: `ORIGAMI_CONTEXT=32k`, `ORIGAMI_PORT`, `ORIGAMI_MODEL`, `ORIGAMI_DATA_DIR`, `ORIGAMI_MEMORY_LIMIT`, `ORIGAMI_QUALITY`, `ORIGAMI_HOT_COPY_DIR`, `ORIGAMI_ALLOW_FAST`… A flag wins over it; it wins over a named model's own value and the settings file (`Sources/origami/Settings.swift`). See [settings.md](settings.md). | setting |
| `ORIGAMI_CONFIG` | `~/Library/Application Support/Origami/config.json` | Which settings file to read and write (same as `--config`), for example one per llama-swap entry. | setting |
| `ORIGAMI_STORES` | unset | Store folders, comma-separated; replaces the configured stores for that run (no size limits). | setting |
| `ORIGAMI_API_KEY` | unset | The API key every request must carry (the `api_key` setting); keeps it off the command line. | setting |
| `ORIGAMI_EXPERT_BITS` | unset | Retired (`expert_bits` is gone): the model's own experts run, and start, `status` and `doctor` say the variable is still set. Remove it. | retired |

`HF_TOKEN` (not an Origami variable) is used for gated Hugging Face downloads.

## Qwen and shared engine switches

| Name | Default | What it does | Status |
|---|---|---|---|
| `ORIGAMI_EXECUTION` | unset | A JSON object of Qwen execution-plan overrides, such as `{"prefill_double_buffer":false}`. The text becomes part of the saved-prompt identity, so overridden runs never share saved prompts with plain ones. Keys it sets (such as `gpu_keepalive`) take precedence over the matching setting and over `origami tune` results. | debug |
| `ORIGAMI_NO_BFLOAT_MATRICES` | unset | Any value: run the M1 kernels (no bfloat16 matrix units) on a newer Mac, to compare or to rule them out. Same replies on M1–M4; on M5 it also turns off the matrix-unit prompt kernel. | debug |
| `ORIGAMI_SCRATCH_COMPACTION` | on | `0` gives every temporary GPU buffer its own memory instead of reusing it after its last use (the reference layout, for comparison). Applies to all three models; arithmetic is the same either way. | debug |
| `ORIGAMI_MEMORY_CALIBRATION` | unset | Set by the server on a model it loads at a limit `origami tune --memory` measures (all three models): its memory history goes to a throwaway file, so the trial neither uses nor changes what this Mac learned. Not for users. | internal |
| `ORIGAMI_SPLIT_READS` | off | `1`: split a lone demand miss across drives that hold it (parts read from each at once). For Macs with two or more fast copy drives: on one fast drive it gained nothing (9.72 against 9.75 tok/s off). All three models; replies are the same either way; `split_reads.lone_demand_wait_seconds` shows what is at stake either way. Decided by 3.2 on a Mac with two fast copy drives: the default or deleted. | experiment |

## GLM and DeepSeek switches

These are read by the engine for GLM-5.3-Flash and DeepSeek-V4.1-Flash (`Sources/OrigamiEngine/Family/`).

Timing switches that `origami tune` also sets. A variable fixes the value, so tuning leaves it alone:

| Name | Default | What it does | Status |
|---|---|---|---|
| `ORIGAMI_FAMILY_PREFETCH` | on | `1` or `0`: during a reply, read the experts the next layer will probably need while this layer runs. | debug |
| `ORIGAMI_FAMILY_PREFETCH_DEPTH` | all | How many of those guessed reads may start, best guesses first (0 or more). | debug |
| `ORIGAMI_FAMILY_SPIN` | on | `0`: wait for expert reads by sleeping instead of spinning a CPU core (spinning keeps the GPU fed; Low Power Mode also turns it off). | debug |
| `ORIGAMI_FAMILY_PREFILL_SLICES` | fitted to memory | `1` to `4`: how many prompt slices to process per step; below the default it caps a long prompt's loan too. | debug |

Other experiments:

| Name | Default | What it does | Status |
|---|---|---|---|
| `ORIGAMI_FAMILY_SAME_DRIVE_HOT_COPY` | off | `1`: allow a hot copy on the same drive as the model (single-read records). Decided by 3.2 with an interleaved real-model run: the default or deleted. | experiment |
| `ORIGAMI_FAMILY_GROUPED_DECODE` | on | `0`: compute each expert projection separately instead of grouped (the exact reference path); its saved prompts are kept apart from the grouped path's. | debug |
| `ORIGAMI_FAMILY_GROUPED_GATE_UP` | on | `0`: compute the gate and up projections separately during a reply. | debug |
| `ORIGAMI_FAMILY_LAZY_STARTUP_WORKSPACE` | off | `1`: at load, reserve working memory for a one-token step instead of the largest prompt step; each step still reserves what it needs (starts in about 1.5 GB less). Decided by 3.2 on a 16–24 GB Mac: the default or deleted. | experiment |
| `ORIGAMI_FAMILY_REFRESH_IDLE_OVERHEAD` | off | `1`: re-measure memory overhead between requests and refit the expert cache. Decided by 3.2 with a real-model run: the default or deleted. | experiment |
| `ORIGAMI_FAMILY_RESIDENT_Q4_PACK` | unset | A folder holding a prepared pack of smaller always-resident weights (Q8_0 to Q4_K). **Changes replies**; answer quality is not validated. Decided by 3.2 with quality measured on checkable tasks (ROADMAP N5): the default or deleted. | experiment |
| `ORIGAMI_FAMILY_PHASE_TIMING` | off | `1`: time each host-side phase of a request and report it in the status; log the SHA-256 digests of the state and logits a reply starts from and of each decode step's logits and the state it ends at, to compare runs. | debug |
| `ORIGAMI_FAMILY_ROUTE_TRACE` | off | `1`: record which experts each token was routed to (bounded), with their weights, the next layer's guess and which experts were in memory, and report it in the status. | debug |
| `ORIGAMI_FAMILY_PREFETCH_OBSERVATIONS` | off | `1`: record how guessed reads turned out and report it in the status, with the last 256 hedge decisions (`hedge_trace`). | debug |

## Variables read only by the checks

These are read by `engine-checks` and the scripts in `Tests/`, never by the server.

| Name | What it does |
|---|---|
| `ORIGAMI_MODEL` | A model folder for the checks that can also run on a real model (tokenizer, quality dial, group-32 conversion, tool syntax, vision). In `Tests/cli-checks.sh` it is the `model` setting, like the other `ORIGAMI_<SETTING>` variables there. |
| `ORIGAMI_VISION_REFERENCE` | With `ORIGAMI_MODEL`: a folder with a reference capture of the vision part, to compare against. |
| `ORIGAMI_IMAGE_LINK` | An image URL, to check fetching a real image link. |
| `ORIGAMI_FAMILY_GGUF` | A GLM or DeepSeek GGUF for `family_reference_logits`. |
| `ORIGAMI_FAMILY_REFERENCE` | A folder of `<name>.tokens.json` and `<name>.logits.json` reference logits for `family_reference_logits`. |
| `ORIGAMI_FAMILY_PROMPTS` | Comma-separated prompt names to limit `family_reference_logits` to. |
| `ORIGAMI_FAMILY_EXPERT_GB` | Expert memory in GB for `family_reference_logits` (default: what is free). |
| `ORIGAMI_FAMILY_DECODE` | Tokens to decode after the first prompt in `family_reference_logits` (default 16). |
| `ORIGAMI_CONVERT_GLM` | A GLM-5.3-Flash safetensors folder, to check converting a few layers. |
| `ORIGAMI_CONVERT_LAYERS` | Which layers that check converts (default `0,3`). |
| `ORIGAMI_SYNTHETIC_WRITE_REFERENCE` | `1`: `synthetic_glm_end_to_end` and `synthetic_deepseek_end_to_end` write their reference (`Tests/Fixtures/synthetic/`) instead of comparing; only after a change meant to alter output. |
| `ORIGAMI_TUNING_HOST` | Set by `Tests/tuning-flow-checks.py` and read by `origami tune`: a JSON object of fixed host readings (`memory_bytes`, `swapout_bytes`, `memory_pressure`, `thermal_state`, `low_power`) in place of the Mac's, so the check does not depend on the swapping of the machine running it; `minute_seconds` shortens a minute of the tuning time limit, so the time-limit checks take seconds, and `reported_timing` makes it take each reply's first-token and writing seconds from the fixture (`usage.check_timing`) instead of the clock. |
| `ORIGAMI_AUTO_MEASURE_IDLE` | Seconds without a request before `memory_auto_measure` measures (default 1800). A test aid, to see it start without waiting half an hour; `Tests/auto-measure-checks.py` sets it to 1. |
| `ORIGAMI_TEST_WORKER` | A program `origami serve` starts in place of the model process, with the same arguments. `Tests/dashboard-control-checks.py` sets it to a stub that answers as a loaded model, to check what the dashboard says about a running model without one. |
| `ORIGAMI_TEST_MOVE_PAUSE` | A file name: while that file exists, a move that copies stops after its copy (before the originals are compared) and creates `<file>.copied`, for at most 30 s. `Tests/dashboard-control-checks.py` sets it to write a source file during a move. |
| `ORIGAMI_TEST_CANARY_AT_LOAD` | `1`: the saved prompts' canary probe runs at every load, also for a new or empty folder (whose canary is otherwise written while idle once something is saved), and a folder without a canary takes this build's at once. For the real-model check that two loads give the same canary; never for serving. |
| `ORIGAMI_TEST_VOLUMES` | Folders that stand in for the mounted drives, as `folder=uuid,…`, for `origami relink`, `doctor`, `model list` and the storage map. `Tests/cli-checks.sh` sets it to check a renamed drive without one. |
| `ORIGAMI_JOB` | Set by the server for the commands it runs as a dashboard job. `origami tune` started by hand asks the server whether tuning may start (the dashboard's rule and words); one the server started as a job has passed that check already. |
| `ORIGAMI_UNKNOWN_CREDENTIAL` | A dummy variable set by a check to prove unknown variables are never recorded. |

Checks that need one of these report "skipped" with the reason when it is not set.

## Names that look like variables but are not

`ORIGAMI_EXACT_IO_H`, `ORIGAMI_DEVICE_PROBE_H`, `ORIGAMI_KV_ARGS`, `ORIGAMI_MM_TABLE`, `ORIGAMI_MV_BATCH`,
`ORIGAMI_MV_GATHER`, `ORIGAMI_MV_ROWS`, `ORIGAMI_MV_TABLE`, `ORIGAMI_PAGED_KV`, `ORIGAMI_FRAG` and the `ORIGAMI_PV_*`
and `ORIGAMI_QK_*` names are C and Metal macros inside the sources. `ORIGAMI_TUNING_OUTCOME`
is a marker line that a tuning run prints for the dashboard to read.
