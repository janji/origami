import OrigamiEngine

/// Pure simulated-host policy: no sysctl, authorization, model, or GPU access.
func wiredMemoryPolicyLimits() throws {
    func require(_ value: Bool, _ message: String) throws { if !value { throw EngineError(message) } }
    let gib = WiredMemoryPolicy.gibibyte
    let mib = WiredMemoryPolicy.mebibyte
    for (ram, maximum) in [(8, 5), (16, 13), (24, 21), (32, 28), (64, 56), (96, 84), (128, 112), (192, 168)] as [(UInt64, UInt64)] {
        let limit = WiredMemoryPolicy.maximumBudgetBytes(physicalMemoryBytes: ram * gib)
        try require(limit == maximum * gib, "\(ram) GiB host has an incorrect safe ceiling")
        let plan = try WiredMemoryPolicy.plan(requestedBudgetBytes: limit, physicalMemoryBytes: ram * gib, currentLimitMiB: 1)
        try require(
            plan.maximumBudgetBytes == limit && plan.requiredLimitMiB == maximum * 1024 && plan.needsRaise,
            "\(ram) GiB plan did not preserve headroom and MiB units")
    }
    for ram in [UInt64(0), 1, 3 * gib] {
        try require(WiredMemoryPolicy.maximumBudgetBytes(physicalMemoryBytes: ram) == 0, "Small host underflowed")
    }
    let oddRAM = 32 * gib + 12345
    let oddMax = WiredMemoryPolicy.maximumBudgetBytes(physicalMemoryBytes: oddRAM)
    try require(oddMax % mib == 0 && oddMax <= oddRAM - max(3 * gib, oddRAM / 8), "MiB rounding crossed OS headroom")
    let boundary = try WiredMemoryPolicy.plan(requestedBudgetBytes: 20 * gib + 1, physicalMemoryBytes: 32 * gib, currentLimitMiB: 20480)
    try require(boundary.requiredLimitMiB == 20481 && boundary.needsRaise, "A partial MiB was rounded down")
}

func wiredMemoryPolicyValidation() throws {
    func require(_ value: Bool, _ message: String) throws { if !value { throw EngineError(message) } }
    func rejects(_ budget: UInt64, _ ram: UInt64, _ ceiling: UInt64?, _ code: String) throws {
        do {
            _ = try WiredMemoryPolicy.plan(requestedBudgetBytes: budget, physicalMemoryBytes: ram, currentLimitMiB: ceiling)
        } catch let error as EngineError where error.code == code { return }
        throw EngineError("Invalid wired-memory plan was not refused with \(code)")
    }
    let gib = WiredMemoryPolicy.gibibyte
    // Actual production numbers: sufficient current system ceiling needs no elevation.
    let actual = try WiredMemoryPolicy.plan(requestedBudgetBytes: 25_770_000_000, physicalMemoryBytes: 32 * gib, currentLimitMiB: 28672)
    try require(!actual.needsRaise && actual.currentLimitMiB == 28672, "Existing ceiling should skip authorization")
    let exact = try WiredMemoryPolicy.plan(requestedBudgetBytes: 28 * gib, physicalMemoryBytes: 32 * gib, currentLimitMiB: 28672)
    try require(!exact.needsRaise && exact.requiredLimitMiB == 28672, "Equal ceiling requested elevation")
    let alreadyHigher = try WiredMemoryPolicy.plan(requestedBudgetBytes: 20 * gib, physicalMemoryBytes: 32 * gib, currentLimitMiB: 32768)
    try require(!alreadyHigher.needsRaise && alreadyHigher.currentLimitMiB == 32768, "Existing larger system ceiling would be lowered")
    let automaticEnough = try WiredMemoryPolicy.plan(
        requestedBudgetBytes: 20 * gib, physicalMemoryBytes: 32 * gib,
        currentLimitMiB: 0, defaultLimitBytes: 24 * gib)
    try require(
        !automaticEnough.needsRaise && automaticEnough.currentLimitMiB == 0 && automaticEnough.effectiveLimitMiB == 24576,
        "Automatic system ceiling was confused with zero bytes or needlessly changed")
    let automaticRaise = try WiredMemoryPolicy.plan(
        requestedBudgetBytes: 28 * gib, physicalMemoryBytes: 32 * gib,
        currentLimitMiB: 0, defaultLimitBytes: 24 * gib)
    try require(
        automaticRaise.needsRaise && automaticRaise.requiredLimitMiB == 28672 && automaticRaise.effectiveLimitMiB == 24576,
        "Reboot-default system ceiling cannot be raised to a validated budget")
    for invalidDefault in [UInt64(0), 1, 33 * gib, UInt64.max] {
        do {
            _ = try WiredMemoryPolicy.plan(
                requestedBudgetBytes: 20 * gib, physicalMemoryBytes: 32 * gib,
                currentLimitMiB: 0, defaultLimitBytes: invalidDefault)
            throw EngineError("Invalid automatic-ceiling recommendation was accepted")
        } catch let error as EngineError where error.code == "wired_memory_unavailable" {}
    }
    try rejects(0, 32 * gib, 28672, "invalid_configuration")
    try rejects(28 * gib + 1, 32 * gib, 32768, "invalid_configuration")
    try rejects(UInt64.max, 32 * gib, 32768, "invalid_configuration")
    try rejects(1, 0, 1, "invalid_configuration")
    for ceiling in [nil, UInt64(0), UInt64(Int32.max) + 1, UInt64.max] as [UInt64?] {
        try rejects(20 * gib, 32 * gib, ceiling, "wired_memory_unavailable")
    }
    try rejects(WiredMemoryPolicy.maximumBudgetBytes(physicalMemoryBytes: UInt64.max), UInt64.max, 1, "invalid_configuration")
}
