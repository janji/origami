import Foundation
import OrigamiEngine

func familyHostPhaseTimingChecks() throws {
    func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw EngineError("Family host timing: " + message) }
    }
    struct Failed: Error {}
    var clockCalls = 0
    let times: [Double] = [10, 10.125, 20, 20.5]
    let timing = FamilyHostPhaseTiming(enabled: true) {
        defer { clockCalls += 1 }
        return times[clockCalls]
    }
    let value = timing.measure(.sampleForceReplay) { 42 }
    do {
        try timing.measure(.decodeForward) { throw Failed() }
        throw EngineError("Timed failure was swallowed")
    } catch is Failed {}
    let counters = timing.counters
    try require(value == 42 && clockCalls == 4, "measurement changed result or added clock reads")
    try require(
        counters["host_wall.sample_force_replay.seconds"] == 0.125
            && counters["host_wall.sample_force_replay.calls"] == 1
            && counters["host_wall.decode_forward.seconds"] == 0.5
            && counters["host_wall.decode_forward.calls"] == 1, "failed/successful calls not retained separately")
    try require(
        counters.count == FamilyHostPhaseTiming.Phase.allCases.count * 2 + 4
            && counters["host_wall.prefill_feed.calls"] == 0, "schema is not bounded or zero observations are missing")
    try require(
        counters["host_wall.nested_in_stage_seconds"] == 1
            && counters["host_wall.model_calls_include_io_gpu_wait"] == 1
            && counters["host_wall.exclusive_kernel_timing"] == 0, "overlap semantics are ambiguous")
    let disabled = FamilyHostPhaseTiming(enabled: false) {
        clockCalls += 1
        return 0
    }
    try require(disabled.measure(.emitCallback) { 7 } == 7, "disabled measurement changed result")
    do {
        try disabled.measure(.prefillSave) { throw Failed() }
        throw EngineError("Disabled failure swallowed")
    } catch is Failed {}
    try require(clockCalls == 4 && disabled.counters.isEmpty, "disabled timing read the clock or published counters")
    try require(
        !FamilyHostPhaseTiming.enabled(environment: [:])
            && !FamilyHostPhaseTiming.enabled(environment: ["ORIGAMI_FAMILY_PHASE_TIMING": "0"])
            && FamilyHostPhaseTiming.enabled(environment: ["ORIGAMI_FAMILY_PHASE_TIMING": "1"]), "timing is not explicitly opt-in")
}
