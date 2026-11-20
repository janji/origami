import OrigamiEngine

/// Repeated misses must not make every speculative pair replay, and a cache that becomes resident must be probed
/// again within a bounded number of pairs. A success restores the zero-host-round-trip path immediately.
func pairResidencyProbeBacksOffAndRecovers() throws {
    func require(_ value: Bool, _ message: String) throws {
        guard value else { throw EngineError("Pair residency probe: " + message) }
    }
    var policy = PairResidencyBackoff()
    try require(policy.shouldProbe(), "a new sequence did not try the resident path")
    for expected in [1, 3, 7, 15, 31, 31, 31] {
        policy.recordMiss()
        for _ in 0..<expected { try require(!policy.shouldProbe(), "retried before the bounded backoff expired") }
        try require(policy.shouldProbe(), "did not retry after \(expected) streamed pairs")
    }
    policy.recordHit()
    for _ in 0..<8 { try require(policy.shouldProbe(), "a resident hit kept a stale delay") }
    policy.recordMiss()
    try require(!policy.shouldProbe() && policy.shouldProbe(), "a success did not reset the miss history")
    for _ in 0..<10_000 { policy.recordMiss() }
    for _ in 0..<31 { try require(!policy.shouldProbe(), "the saturated delay changed") }
    try require(policy.shouldProbe(), "repeated misses made recovery unbounded")
    policy = .init()
    try require(policy.shouldProbe(), "a fresh request inherited the previous request's delay")
}

/// Custom whole-schedule plans may disable early demand independently of MTP or resident pre-encoding. They must
/// decline a pair before routing changes state unless both the queue and minimum phase cache fit its worst case.
func pairUnionAdmissionRequiresCompleteReserve() throws {
    for (window, prefillSlots, early, supported) in [
        (16, 32, true, true),  // Release reservation, including the streamed/non-preencoded path.
        (16, 32, false, false),  // Ordinary window cannot admit two disjoint top-ten rows.
        (20, 32, false, true),  // A custom explicit union-sized window needs no early-demand reservation.
        (10, 16, true, false),  // The queue fits, but a reduced phase cache cannot retain twenty records.
        (10, 32, true, true),  // Early demand reserves twenty independently of the ordinary window.
    ] {
        let configuration = try ExpertExecutionConfiguration(
            mode: .slabPackedPreferred,
            cacheBytes: UInt64(32 * PackStore.expertBytes), slotsPerSlab: 8,
            demandWindow: window, readConcurrency: 2, prefillCacheSlots: prefillSlots)
        var accepted = false
        do {
            try QwenPairSchedule.requireUnionCapacity(configuration, earlyDemand: early)
            accepted = true
        } catch let error as EngineError where error.code == "pair_unsupported" {}
        guard accepted == supported else {
            throw EngineError("Pair admission differs for window \(window), prefill slots \(prefillSlots), early demand \(early)")
        }
    }
}
