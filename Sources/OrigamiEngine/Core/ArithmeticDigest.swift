// Written by scripts/arithmetic-digest.py: do not edit by hand.

/// Per engine, the SHA-256 over the sources that can change the bits of its saved state or of the logits computed from
/// it: every .swift, .metal, .c and .h file under `included` and not under `excluded`, less those `only` gives to other
/// engines (the script says why each is there). Saved prompts are keyed on their engine's (`PromptCanary`), not on the
/// executable; `arithmetic_digest_current` checks they are up to date.
public struct ArithmeticDigest: Sendable, Equatable {
    /// "qwen", "qwen_gguf" or "family", as in `only`.
    public let name: String
    public let value: String
    /// 3.1.8.3's single digest while this engine's files are as they were then: the folders it keyed are this engine's
    /// (`PromptCanary.adopt`; their canary is still checked). nil once they changed.
    public let before: String?
    public static let qwen = ArithmeticDigest(
        name: "qwen", value: "3c7cf269a0cf2228a85642b931c59dd14522a00e9522f69d6d6e5ef7732a83e8",
        before: nil)
    public static let qwenGGUF = ArithmeticDigest(
        name: "qwen_gguf", value: "80c54dcc7e8cc69bb2bc4ce08ae3ea5ff23b448591b77f0867b0e4cde15c414d",
        before: nil)
    public static let family = ArithmeticDigest(
        name: "family", value: "220daf7b984cc339f1f7a497b08021873fb9d37c84813e793a6478a9da2ceea0",
        before: "83f737875a3242733c5d4b1d5748e28d592a589506ffccb881ee9d4388355d90")
    /// The engine that runs a model: Qwen's two by where its experts are read from, or GLM's and DeepSeek's.
    public static func forQwen(ggufExperts: Bool) -> ArithmeticDigest { ggufExperts ? qwenGGUF : qwen }
    public static let all = [qwen, qwenGGUF, family]
    package static let included = [
        "Sources/CExactIO/",
        "Sources/OrigamiEngine/Core/",
        "Sources/OrigamiEngine/Formats/",
        "Sources/OrigamiEngine/Models/",
        "Sources/OrigamiEngine/Conversation/",
        "Sources/OrigamiEngine/Qwen/",
        "Sources/OrigamiEngine/Family/",
        "Sources/OrigamiEngine/Serving/GenerationDriver.swift",
        "Sources/OrigamiEngine/Streaming/QualityDial.swift",
        "Sources/OrigamiEngine/Streaming/FastLateMissSkip.swift",
        "Sources/OrigamiEngine/Planning/RuntimeConfiguration.swift",
        "Sources/OrigamiEngine/Planning/ReleaseExecution.swift",
    ]
    package static let excluded = [
        "Sources/OrigamiEngine/Core/ArithmeticDigest.swift",
        "Sources/OrigamiEngine/Core/CommandBudget.swift",
        "Sources/OrigamiEngine/Core/GPURetry.swift",
        "Sources/OrigamiEngine/Core/GPUWait.swift",
        "Sources/OrigamiEngine/Core/HostGuard.swift",
        "Sources/OrigamiEngine/Core/RequestGPUKeepAlive.swift",
        "Sources/OrigamiEngine/Models/DownloadResume.swift",
        "Sources/OrigamiEngine/Models/ModelCompatibility.swift",
        "Sources/OrigamiEngine/Models/ModelRefresh.swift",
        "Sources/OrigamiEngine/Models/VerifiedContentReceipt.swift",
        "Sources/OrigamiEngine/Models/LocalModelIdentity.swift",
        "Sources/OrigamiEngine/Models/TokenizerStack.swift",
        "Sources/OrigamiEngine/Conversation/PromptCache.swift",
        "Sources/OrigamiEngine/Conversation/PromptEncodingCache.swift",
        "Sources/OrigamiEngine/Qwen/Execution/PrefillExpertReadAhead.swift",
        "Sources/OrigamiEngine/Qwen/Execution/PrefillReadPolicy.swift",
        "Sources/OrigamiEngine/Qwen/Streaming/ExpertCache.swift",
        "Sources/OrigamiEngine/Qwen/Streaming/ExpertCacheEviction.swift",
        "Sources/OrigamiEngine/Qwen/Streaming/ExpertGPUResidency.swift",
        "Sources/OrigamiEngine/Qwen/Streaming/ExpertReuseCalibration.swift",
        "Sources/OrigamiEngine/Qwen/Streaming/ExpertReusePolicy.swift",
        "Sources/OrigamiEngine/Qwen/Streaming/ExpertSourceOnlineCalibration.swift",
        "Sources/OrigamiEngine/Qwen/Streaming/ExpertSourceSelection.swift",
        "Sources/OrigamiEngine/Qwen/Streaming/HostStagePrefetch.swift",
        "Sources/OrigamiEngine/Qwen/Streaming/HotExpertRanking.swift",
        "Sources/OrigamiEngine/Qwen/Streaming/KnownPLERows.swift",
        "Sources/OrigamiEngine/Qwen/Streaming/PLERowPrefetch.swift",
        "Sources/OrigamiEngine/Qwen/Streaming/PairResidencyBackoff.swift",
        "Sources/OrigamiEngine/Qwen/Streaming/RoutingProfile.swift",
        "Sources/OrigamiEngine/Qwen/Streaming/SourceExpertLoader.swift",
        "Sources/OrigamiEngine/Qwen/Streaming/SpanReadPool.swift",
        "Sources/OrigamiEngine/Family/FamilyGPUProfile.swift",
        "Sources/OrigamiEngine/Family/FamilyHostPhaseTiming.swift",
        "Sources/OrigamiEngine/Family/FamilyRouteTrace.swift",
        "Sources/OrigamiEngine/Family/FamilySupport.swift",
        "Sources/OrigamiEngine/Family/Streaming/FamilyEviction.swift",
        "Sources/OrigamiEngine/Family/Streaming/FamilyHedges.swift",
        "Sources/OrigamiEngine/Family/Streaming/FamilyHotCopySource.swift",
        "Sources/OrigamiEngine/Family/Streaming/FamilyPoolState.swift",
        "Sources/OrigamiEngine/Family/Streaming/FamilyPrefetchObservations.swift",
        "Sources/OrigamiEngine/Family/Streaming/FamilyPrefetchPolicy.swift",
        "Sources/OrigamiEngine/Family/Streaming/FamilySlabStorage.swift",
    ]
    package static let only: [(String, [String])] = [
        ("Sources/OrigamiEngine/Qwen/", ["qwen", "qwen_gguf"]),
        ("Sources/OrigamiEngine/Serving/GenerationDriver.swift", ["qwen", "qwen_gguf"]),
        ("Sources/OrigamiEngine/Streaming/FastLateMissSkip.swift", ["qwen", "qwen_gguf"]),
        ("Sources/OrigamiEngine/Planning/", ["qwen", "qwen_gguf"]),
        ("Sources/OrigamiEngine/Models/ModelContract.swift", ["qwen", "qwen_gguf"]),
        ("Sources/OrigamiEngine/Models/ModelLayout.swift", ["qwen", "qwen_gguf"]),
        ("Sources/OrigamiEngine/Formats/PackStore.swift", ["qwen", "qwen_gguf"]),
        ("Sources/OrigamiEngine/Formats/StoreBuilder.swift", ["qwen", "qwen_gguf"]),
        ("Sources/OrigamiEngine/Formats/StoreFormat.swift", ["qwen", "qwen_gguf"]),
        ("Sources/OrigamiEngine/Formats/GGUFExperts.swift", ["qwen_gguf"]),
        ("Sources/OrigamiEngine/Formats/GGUF.swift", ["qwen_gguf", "family"]),
        ("Sources/OrigamiEngine/Formats/GGUFConversion.swift", ["qwen_gguf", "family"]),
        ("Sources/OrigamiEngine/Formats/GGUFGrids.swift", ["qwen_gguf", "family"]),
        ("Sources/OrigamiEngine/Formats/GGUFQuantize.swift", ["qwen_gguf", "family"]),
        ("Sources/OrigamiEngine/Formats/GGMLMatrixKernels.swift", ["qwen_gguf", "family"]),
        ("Sources/OrigamiEngine/Family/", ["family"]),
        ("Sources/OrigamiEngine/Models/ModelFamily.swift", ["family"]),
    ]
}
