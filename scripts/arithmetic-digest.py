#!/usr/bin/env python3
"""Writes Sources/OrigamiEngine/Core/ArithmeticDigest.swift: the SHA-256 over the sources that can change the bits of a
saved state or of the logits computed from it. Saved prompts are keyed on it instead of the executable's hash, so a
build that changes none of these files keeps them (each build still checks a folder's canary once: PromptCanary).

Run it from anywhere after changing a file it covers; `engine-checks --only arithmetic_digest_current` says when to.
The digest: for each covered file in byte order of its path from the repository's root, the path, a NUL, the hex
SHA-256 of its contents and a newline; the SHA-256 of all that. Covered: every .swift, .metal, .c and .h file under an
INCLUDED path that is not under an EXCLUDED one. When unsure whether a file can change bits, it is in.
"""
import hashlib
import os
import sys

INCLUDED = [
    "Sources/CExactIO/",  # C helpers under the exact reads the weights and states come through
    "Sources/OrigamiEngine/Core/",  # bfloat16, exact file reads, the state files' layout
    "Sources/OrigamiEngine/Formats/",  # weight decoding, GGUF and safetensors parsing, ggml matrix kernels, record layouts
    "Sources/OrigamiEngine/Models/",  # tensor locations, model layout and geometry
    "Sources/OrigamiEngine/Conversation/",  # the saved state's format, its capture and restore
    "Sources/OrigamiEngine/Qwen/",  # Qwen's kernels and forward pass
    "Sources/OrigamiEngine/Family/",  # GLM's and DeepSeek's kernels and forward passes
    "Sources/OrigamiEngine/Serving/GenerationDriver.swift",  # Qwen's prompt read schedule and decode loop
    "Sources/OrigamiEngine/Streaming/QualityDial.swift",  # which rows are exact
    "Sources/OrigamiEngine/Streaming/FastLateMissSkip.swift",  # the exact timeline, fast rows kept out of it
    "Sources/OrigamiEngine/Planning/RuntimeConfiguration.swift",  # the execution switches that choose kernels
    "Sources/OrigamiEngine/Planning/ReleaseExecution.swift",  # their release values
]
EXCLUDED = [
    "Sources/OrigamiEngine/Core/ArithmeticDigest.swift",  # the digest itself
    "Sources/OrigamiEngine/Core/CommandBudget.swift",  # how GPU work is cut into command buffers, not what it computes
    "Sources/OrigamiEngine/Core/GPURetry.swift",  # retries a reply after a lost GPU
    "Sources/OrigamiEngine/Core/GPUWait.swift",  # waits for the GPU
    "Sources/OrigamiEngine/Core/HostGuard.swift",  # refuses hosts
    "Sources/OrigamiEngine/Core/RequestGPUKeepAlive.swift",  # keeps the GPU's clock up
    "Sources/OrigamiEngine/Models/DownloadResume.swift",  # downloads
    "Sources/OrigamiEngine/Models/ModelCompatibility.swift",  # refuses models
    "Sources/OrigamiEngine/Models/ModelRefresh.swift",  # re-binds files to a changed model
    "Sources/OrigamiEngine/Models/VerifiedContentReceipt.swift",  # content receipts
    "Sources/OrigamiEngine/Models/LocalModelIdentity.swift",  # names; the identities they make enter the epoch themselves
    "Sources/OrigamiEngine/Models/TokenizerStack.swift",  # text to token ids: saves are keyed on the ids
    "Sources/OrigamiEngine/Conversation/PromptCache.swift",  # folders, index and slot choice, never a state's bits
    "Sources/OrigamiEngine/Conversation/PromptEncodingCache.swift",  # text to token ids
    "Sources/OrigamiEngine/Qwen/Execution/PrefillExpertReadAhead.swift",  # reads ahead
    "Sources/OrigamiEngine/Qwen/Execution/PrefillReadPolicy.swift",  # chooses reads
    "Sources/OrigamiEngine/Qwen/Streaming/ExpertCache.swift",  # caches the records read (same bytes)
    "Sources/OrigamiEngine/Qwen/Streaming/ExpertCacheEviction.swift",
    "Sources/OrigamiEngine/Qwen/Streaming/ExpertGPUResidency.swift",
    "Sources/OrigamiEngine/Qwen/Streaming/ExpertReuseCalibration.swift",
    "Sources/OrigamiEngine/Qwen/Streaming/ExpertReusePolicy.swift",
    "Sources/OrigamiEngine/Qwen/Streaming/ExpertSourceOnlineCalibration.swift",  # which drive a record is read from
    "Sources/OrigamiEngine/Qwen/Streaming/ExpertSourceSelection.swift",
    "Sources/OrigamiEngine/Qwen/Streaming/HostStagePrefetch.swift",
    "Sources/OrigamiEngine/Qwen/Streaming/HotExpertRanking.swift",
    "Sources/OrigamiEngine/Qwen/Streaming/KnownPLERows.swift",
    "Sources/OrigamiEngine/Qwen/Streaming/PLERowPrefetch.swift",
    "Sources/OrigamiEngine/Qwen/Streaming/PairResidencyBackoff.swift",
    "Sources/OrigamiEngine/Qwen/Streaming/RoutingProfile.swift",  # usage counts
    "Sources/OrigamiEngine/Qwen/Streaming/SourceExpertLoader.swift",
    "Sources/OrigamiEngine/Qwen/Streaming/SpanReadPool.swift",
    "Sources/OrigamiEngine/Family/FamilyGPUProfile.swift",  # profiling
    "Sources/OrigamiEngine/Family/FamilyHostPhaseTiming.swift",  # timing
    "Sources/OrigamiEngine/Family/FamilyRouteTrace.swift",  # route traces
    "Sources/OrigamiEngine/Family/FamilySupport.swift",  # tokenizer files and the key, which enters the epoch itself
    "Sources/OrigamiEngine/Family/Streaming/FamilyEviction.swift",  # the expert slots: same bytes, other places
    "Sources/OrigamiEngine/Family/Streaming/FamilyHedges.swift",
    "Sources/OrigamiEngine/Family/Streaming/FamilyHotCopySource.swift",
    "Sources/OrigamiEngine/Family/Streaming/FamilyPoolState.swift",
    "Sources/OrigamiEngine/Family/Streaming/FamilyPrefetchObservations.swift",
    "Sources/OrigamiEngine/Family/Streaming/FamilyPrefetchPolicy.swift",
    "Sources/OrigamiEngine/Family/Streaming/FamilySlabStorage.swift",
]
EXTENSIONS = (".swift", ".metal", ".c", ".h")
TARGET = "Sources/OrigamiEngine/Core/ArithmeticDigest.swift"


def covered(root):
    files = []
    for entry in INCLUDED:
        path = os.path.join(root, entry)
        if os.path.isfile(path):
            files.append(entry)
            continue
        for folder, _, names in os.walk(path):
            files += [os.path.relpath(os.path.join(folder, n), root) for n in names if n.endswith(EXTENSIONS)]
    files = {f.replace(os.sep, "/") for f in files}
    return sorted(f for f in files if not any(f == e or f.startswith(e) for e in EXCLUDED))


def digest(root):
    total = hashlib.sha256()
    for f in covered(root):
        with open(os.path.join(root, f), "rb") as handle:
            total.update(f.encode() + b"\0" + hashlib.sha256(handle.read()).hexdigest().encode() + b"\n")
    return total.hexdigest()


def swift_list(items):
    return "[\n" + "".join(f'        "{i}",\n' for i in items) + "    ]"


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    value = digest(root)
    text = f"""// Written by scripts/arithmetic-digest.py: do not edit by hand.

/// SHA-256 over the sources that can change the bits of a saved state or of the logits computed from it: every .swift,
/// .metal, .c and .h file under `included` and not under `excluded` (the script says why each is there). Saved prompts are
/// keyed on it (`PromptCanary`), not on the executable; `arithmetic_digest_current` checks it is up to date.
public enum ArithmeticDigest {{
    public static let value = "{value}"
    package static let included = {swift_list(INCLUDED)}
    package static let excluded = {swift_list(EXCLUDED)}
}}
"""
    with open(os.path.join(root, TARGET), "w") as handle:
        handle.write(text)
    print(value)


if __name__ == "__main__":
    sys.exit(main())
