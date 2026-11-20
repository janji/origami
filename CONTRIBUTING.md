# Contributing to Origami

## Reporting a problem, or a Mac we have not tested

Run `origami report` while `origami serve` is running and paste its output into a GitHub issue. It shows the
version, the Mac, the memory plan, the stores and a short benchmark, with home-folder paths shortened to `~`.
Reports from Macs other than the one tested M2 Pro are especially welcome. Also say how you start Origami, which
app sends the requests, and what you expected. For a wrong answer, include the smallest request that shows it, at
temperature 0.

## Build

```sh
swift build                    # debug
swift build -c release         # Xcode 26 or Command Line Tools 26 (macOS 26 SDK); runs on macOS 15 or later
```

The build must have no warnings in Origami's own sources.

## Checks

```sh
.build/release/engine-checks > checks.json        # every check; model-free, needs an Apple Silicon GPU
.build/release/engine-checks --only name,name     # just the named checks, for a quick loop
Tests/cli-checks.sh .build/release/origami        # the command-line contract
```

`engine-checks` must report 0 failed. There is no XCTest: the checks are a plain executable that prints JSON, so
they build with the Command Line Tools alone. Checks that need a real model or other input report "skipped" with the
reason (see [docs/switches.md](docs/switches.md#variables-read-only-by-the-checks)); so do GPU checks on a machine
without an Apple Silicon GPU, such as CI's virtual machines.

## Formatting

```sh
swift format format -i -r --configuration .swift-format Sources Tests
```

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

[INTERNALS.md](INTERNALS.md) explains how it all works.

## Rules

- **Exact by default.** A change must not alter replies unless it is an explicit opt-in mode. The bit-exact checks
  are the judge: if one fails, the change is wrong, not the check.
- Every setting that can change replies is labelled **BITWISE** or **NUMERIC** in
  [docs/settings.md](docs/settings.md#which-settings-change-replies); keep the label honest.
- Every bug fix, and every limit on untrusted input (model files, HTTP requests, images), comes with a model-free
  check.
- No machine-specific constants, paths or measurements in the code.
- A change that claims speed needs before/after numbers from the same Mac and prompt, with the expert cache in the
  same state, in the pull request.
- New work comes from [ROADMAP.md](ROADMAP.md).
