#!/usr/bin/env python3
"""Writes Sources/OrigamiEngine/Core/ArithmeticDigest.swift: per engine (Qwen with MLX experts, Qwen with GGUF experts,
GLM and DeepSeek), the SHA-256 over the sources that can change the bits of that engine's saved state or of the logits
computed from it. Saved prompts are keyed on their engine's digest instead of the executable's hash, so a build that
changes none of an engine's files keeps that engine's saves (each build still checks a folder's canary once:
PromptCanary).

Run it from anywhere after changing a file it covers; `engine-checks --only arithmetic_digest_current` says when to.
A digest: for each of the engine's files in byte order of its path from the repository's root, the path, a NUL, the hex
SHA-256 of its contents and a newline; the SHA-256 of all that. Covered: every .swift, .metal, .c and .h file under an
INCLUDED path that is not under an EXCLUDED one. Each covered file is every engine's, except those under an ONLY path,
which are the named engines' alone (code only those engines run). When unsure whether a file can change bits, it is in;
when unsure whether an engine runs it, it is that engine's.
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
ENGINES = ["qwen", "qwen_gguf", "family"]  # Qwen with MLX experts, Qwen with GGUF experts, GLM and DeepSeek
QWEN, GGUF = ["qwen", "qwen_gguf"], ["qwen_gguf", "family"]
ONLY = [
    ("Sources/OrigamiEngine/Qwen/", QWEN),
    ("Sources/OrigamiEngine/Serving/GenerationDriver.swift", QWEN),  # the family engine reads one constant of it
    ("Sources/OrigamiEngine/Streaming/FastLateMissSkip.swift", QWEN),  # the family engine reads one message of it
    ("Sources/OrigamiEngine/Planning/", QWEN),
    ("Sources/OrigamiEngine/Models/ModelContract.swift", QWEN),
    ("Sources/OrigamiEngine/Models/ModelLayout.swift", QWEN),
    ("Sources/OrigamiEngine/Formats/PackStore.swift", QWEN),  # the family converter reads its tensor spans
    ("Sources/OrigamiEngine/Formats/StoreBuilder.swift", QWEN),
    ("Sources/OrigamiEngine/Formats/StoreFormat.swift", QWEN),
    ("Sources/OrigamiEngine/Formats/GGUFExperts.swift", ["qwen_gguf"]),  # Qwen's experts read from GGUF banks
    ("Sources/OrigamiEngine/Formats/GGUF.swift", GGUF),
    ("Sources/OrigamiEngine/Formats/GGUFConversion.swift", GGUF),
    ("Sources/OrigamiEngine/Formats/GGUFGrids.swift", GGUF),
    ("Sources/OrigamiEngine/Formats/GGUFQuantize.swift", GGUF),
    ("Sources/OrigamiEngine/Formats/GGMLMatrixKernels.swift", GGUF),
    ("Sources/OrigamiEngine/Family/", ["family"]),
    ("Sources/OrigamiEngine/Models/ModelFamily.swift", ["family"]),
]
# Each engine's saved prompts from before digests were per engine: 3.1.5 to 3.1.8.3 keyed every engine's on one digest,
# BEFORE in 3.1.8.3. SAME_SINCE: each engine's digest in the first build keyed per engine, whose arithmetic is 3.1.8.3's
# (it changed no covered file but PromptCanary.swift, and that only in which digest a marker names). While an engine's
# digest is still that, the folders 3.1.8.3 keyed are its own (`ArithmeticDigest.before`; each build still checks their
# canary); once it changes, they are another arithmetic's. Never set SAME_SINCE to a later digest (a later build cannot
# vouch that its arithmetic is 3.1.8.3's): once no engine keeps it, delete it with BEFORE.
BEFORE = "83f737875a3242733c5d4b1d5748e28d592a589506ffccb881ee9d4388355d90"
SAME_SINCE = {
    "qwen": "3f31c5aed024514da0cb6556a4a61fa1752eff423f1ec6fd0d450bcef9e4aa64",
    "qwen_gguf": "222e35c2eeadaf20306293f53135b8d988008da50ddb384fc937e83614980ac7",
    "family": "220daf7b984cc339f1f7a497b08021873fb9d37c84813e793a6478a9da2ceea0",
}
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


def engines_of(path):
    for prefix, engines in ONLY:
        if path == prefix or path.startswith(prefix):
            return engines
    return ENGINES


def digest(root, engine):
    total = hashlib.sha256()
    for f in covered(root):
        if engine not in engines_of(f):
            continue
        with open(os.path.join(root, f), "rb") as handle:
            total.update(f.encode() + b"\0" + hashlib.sha256(handle.read()).hexdigest().encode() + b"\n")
    return total.hexdigest()


def swift_list(items):
    return "[\n" + "".join(f"        {i},\n" for i in items) + "    ]"


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    values = {e: digest(root, e) for e in ENGINES}
    names = {"qwen": "qwen", "qwen_gguf": "qwenGGUF", "family": "family"}

    def engine(e):
        before = f'"{BEFORE}"' if SAME_SINCE.get(e) == values[e] else "nil"
        return (
            f"    public static let {names[e]} = ArithmeticDigest(\n"
            f'        name: "{e}", value: "{values[e]}",\n        before: {before})\n'
        )

    only = [f'("{p}", [{", ".join(chr(34) + e + chr(34) for e in es)}])' for p, es in ONLY]
    text = f"""// Written by scripts/arithmetic-digest.py: do not edit by hand.

/// Per engine, the SHA-256 over the sources that can change the bits of its saved state or of the logits computed from
/// it: every .swift, .metal, .c and .h file under `included` and not under `excluded`, less those `only` gives to other
/// engines (the script says why each is there). Saved prompts are keyed on their engine's (`PromptCanary`), not on the
/// executable; `arithmetic_digest_current` checks they are up to date.
public struct ArithmeticDigest: Sendable, Equatable {{
    /// "qwen", "qwen_gguf" or "family", as in `only`.
    public let name: String
    public let value: String
    /// 3.1.8.3's single digest while this engine's files are as they were then: the folders it keyed are this engine's
    /// (`PromptCanary.adopt`; their canary is still checked). nil once they changed.
    public let before: String?
{engine("qwen")}{engine("qwen_gguf")}{engine("family")}    /// The engine that runs a model: Qwen's two by where its experts are read from, or GLM's and DeepSeek's.
    public static func forQwen(ggufExperts: Bool) -> ArithmeticDigest {{ ggufExperts ? qwenGGUF : qwen }}
    public static let all = [qwen, qwenGGUF, family]
    package static let included = {swift_list(chr(34) + i + chr(34) for i in INCLUDED)}
    package static let excluded = {swift_list(chr(34) + e + chr(34) for e in EXCLUDED)}
    package static let only: [(String, [String])] = {swift_list(only)}
}}
"""
    with open(os.path.join(root, TARGET), "w") as handle:
        handle.write(text)
    for e in ENGINES:
        print(e, values[e], "(keeps 3.1.8.3's saves)" if SAME_SINCE.get(e) == values[e] else "")


if __name__ == "__main__":
    sys.exit(main())
