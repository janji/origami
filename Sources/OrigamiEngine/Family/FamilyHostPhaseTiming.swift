import Foundation

/// Optional host wall observations on the model-owning thread. Fixed-size aggregates only: no per-token history,
/// GPU synchronization, or interpretation of command-buffer labels as exclusive kernel time.
package final class FamilyHostPhaseTiming {
    package enum Phase: Int, CaseIterable {
        case prefillFeed, prefillForward, prefillSave, decodeForward, sampleForceReplay
        case detokenizeTools, stopFilter, toolAnchorCopy, emitCallback

        package var name: String {
            switch self {
            case .prefillFeed: "prefill_feed"
            case .prefillForward: "prefill_forward"
            case .prefillSave: "prefill_checkpoint_save"
            case .decodeForward: "decode_forward"
            case .sampleForceReplay: "sample_force_replay"
            case .detokenizeTools: "detokenize_tool_detection"
            case .stopFilter: "stop_filter"
            case .toolAnchorCopy: "tool_anchor_copy"
            case .emitCallback: "emit_callback"
            }
        }
    }
    package static func enabled(environment: [String: String] = ProcessInfo.processInfo.environment) -> Bool {
        environment["ORIGAMI_FAMILY_PHASE_TIMING"] == "1"
    }
    private let enabled: Bool, clock: () -> Double
    private var seconds: [Double], calls: [UInt64]

    package init(enabled: Bool, clock: @escaping () -> Double = { ProcessInfo.processInfo.systemUptime }) {
        self.enabled = enabled
        self.clock = clock
        seconds = enabled ? Array(repeating: 0, count: Phase.allCases.count) : []
        calls = enabled ? Array(repeating: 0, count: Phase.allCases.count) : []
    }
    /// Counts every entered call, including one that throws. Disabled observations never read the clock.
    @inline(__always) package func measure<T>(_ phase: Phase, _ body: () throws -> T) rethrows -> T {
        guard enabled else { return try body() }
        let start = clock()
        defer {
            let elapsed = clock() - start
            if elapsed.isFinite && elapsed > 0 { seconds[phase.rawValue] += elapsed }
            calls[phase.rawValue] &+= 1
        }
        return try body()
    }
    /// These intervals nest within stage_seconds, not alongside it. Model-call intervals include host work,
    /// synchronous expert I/O and existing GPU waits; callback time can include downstream parser/network work.
    /// Their sum is not a complete partition of the attempt, and none is physical I/O or exclusive GPU time.
    package var counters: [String: Double] {
        guard enabled else { return [:] }
        var result: [String: Double] = [
            "host_wall.schema_version": 1, "host_wall.nested_in_stage_seconds": 1,
            "host_wall.model_calls_include_io_gpu_wait": 1, "host_wall.exclusive_kernel_timing": 0,
        ]
        for phase in Phase.allCases {
            result["host_wall.\(phase.name).seconds"] = seconds[phase.rawValue]
            result["host_wall.\(phase.name).calls"] = Double(calls[phase.rawValue])
        }
        return result
    }
}
