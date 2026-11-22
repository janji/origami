# Working on Origami (guide for coding agents)

## Goal

Origami aims to be the best SSD-streaming inference engine for Qwen3.8-Flash-Next, GLM-5.3-Flash and
DeepSeek-V4.1-Flash on Apple Silicon. It should work well on every Apple Silicon Mac, not just the development one,
and it is a public GitHub project.

## Layout

```text
Sources/origami/                 the command-line tool: settings, setup, the server front, dashboard state, tuning
Sources/OrigamiEngine/Core/      small shared basics: exact file reads, bfloat16, GPU retry
Sources/OrigamiEngine/Models/    model identity, compatibility, layout and tokenizer
Sources/OrigamiEngine/Formats/   MLX, safetensors and GGUF reading, conversion, expert packs
Sources/OrigamiEngine/Memory/    the memory ledger, budget and governor
Sources/OrigamiEngine/Streaming/ expert read and cache code shared by all models, the quality dial
Sources/OrigamiEngine/Qwen/      the Qwen3.8-Flash-Next engine
Sources/OrigamiEngine/Family/    the GLM-5.3-Flash and DeepSeek-V4.1-Flash engine
Sources/OrigamiEngine/{Conversation,Serving,Planning}/   saved prompts; HTTP, APIs, dashboard; the memory plan
Sources/OrigamiEngineChecks/     checks that need engine internals; only engine-checks links them
Sources/CExactIO/                C helpers for uncached file reads
Tests/                           engine-checks (Tests/EngineChecks/), CLI and dashboard check scripts
```

How it works: [INTERNALS.md](INTERNALS.md). User docs: [README.md](README.md) and `docs/`. Every `ORIGAMI_*`
environment variable: [docs/switches.md](docs/switches.md).

## Commands

```sh
swift build                                       # debug
swift build -c release                            # needs the macOS 26 SDK; no warnings in our sources
.build/release/engine-checks --only name,name     # the checks your change touches
.build/release/engine-checks > checks.json        # the full suite, once before a release build
Tests/cli-checks.sh .build/release/origami        # the command-line contract
python3 -B Tests/error-register-check.py .build/release/origami   # a release build: a compiler fault that crashed 3.1.0 (see the script)
.build/release/engine-checks --fuzz gguf --seconds 60 --seed 1   # fuzz a parser (gguf, safetensors, http, template, socket)
swift format format -i -r --configuration .swift-format Sources Tests
python3 -B scripts/arithmetic-digest.py           # after changing a file that can change bits (arithmetic_digest_current says so)
```

The checks are a plain executable printing JSON (no XCTest), so they build with the Command Line Tools alone.

## Rules of the road

- **The only work queue is [ROADMAP.md](ROADMAP.md).** Read its "Rules of the road" first. Do not create other
  queues, status files or evidence folders in the repository.
- **Exact by default.** Anything that changes output is opt-in, and needs its quality measured on checkable tasks
  (answers that can be scored right or wrong), not only its speed.
- **All three models advance together.** Shared read and cache code belongs in `Sources/OrigamiEngine/Streaming/`,
  not copied into `Qwen/` or `Family/`.
- **Smallest complete change.** Match the surrounding code. Add one focused check per non-trivial change. Run the
  relevant checks with `--only`, and the full suite once before a release build.
- **Experiment switches are temporary.** Every experiment switch either becomes the default or is deleted within
  one release. List new ones in [docs/switches.md](docs/switches.md).
- **No machine-specific code.** No constants, paths or measurements from one Mac in the sources.
- **Measuring speed:**
  - one benchmark at a time, and never compile during a timed run;
  - use production-like settings (the defaults, a real context size, a warm or cold cache stated explicitly);
  - do not trust single-run time differences on a drive that may throttle: compare bytes read per device, cache
    misses and interleaved repeats (A, B, A, B).
- **Never touch a running production server.** Do not deploy to, restart or reconfigure it, and never write to its
  data folder, without the owner's explicit say. A test server uses its own port and its own data folder
  (`--port`, `ORIGAMI_DATA_DIR`, `ORIGAMI_CONFIG`).
- **Reports to the owner:** short and in plain language. What worked, what did not, what is next.
- **Machine-specific notes** (model paths, drive quirks, production service details) live in an untracked
  `AGENTS.local.md`. Read it if it exists; never commit it.
