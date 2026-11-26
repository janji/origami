# Benchmark helpers

Scripts for measuring an Origami build on your own machine. They never change your configuration, models or
running servers; each run writes into a new output folder you name. Inputs (request bodies, launch manifests,
encoded prompts) are yours to supply: nothing here ships measurements.

- `stream_request.py`: sends one or more OpenAI-style streaming chat requests to a running server and records every frame and the server's usage.
- `origami_compare.py`: runs a "before" and a "candidate" `origami` binary one after the other, each in a fresh isolated process, on the same request.
- `family_features.py`: screens optional engine features (variants of `origami_tune` settings) in one loaded candidate process per model family.
- `monitor_origami_resources.py`: watches memory, swap, pressure, thermal state and disk while a comparison runs, and interrupts it when a limit is crossed.
- `sample_device_counters.py`: samples the storage driver's read and write counters over a test window (read-only; counters include other processes).
- `quality_tasks.py`: 12 checkable tasks (code run against hidden tests, long tool results, math, format constraints, an agent tool call), each scored right or wrong by code; runs against a running server, `--compare` two result files (exact against a fast mode or a low-bit build), `--self-test` checks every scorer offline.
- `native_compare.py`: shared process ownership and host-snapshot helpers the scripts above import, plus a driver for other engines given a launch manifest.

Run `--help` on any script for its arguments. Unit tests (no server, model or GPU):
`python3 -B -m unittest discover -s Tools/bench`.
