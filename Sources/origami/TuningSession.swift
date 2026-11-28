import Foundation
import OrigamiEngine

extension Tuning {
    typealias TimedReply = (ttft: Double, total: Double, prompt: Int, cached: Int, completion: Int, text: String, gaps: [Double])

    final class Disrupted: Error, CustomStringConvertible {
        let description: String
        init(_ reason: String) { description = reason }
    }

    /// One sequential tuning session owns its completed requests, including discarded measurements.
    /// Keeping this across samples also catches an app request that finishes while the tuner is settling.
    final class RequestOwnership {
        private var completed: UInt64?
        private var recent: [Data: Int]

        init(_ state: [String: Any]) {
            completed = completedRequests(state)
            recent = completed == nil ? recentCounts(state) : [:]
        }

        func accept(_ state: [String: Any], ownedCompletion: Bool = false) throws {
            let next = completedRequests(state)
            if let completed {
                guard let next,
                    ownedCompletion ? next > completed && next - completed == 1 : next == completed
                else {
                    throw Aborted("Another request completed or the model restarted during tuning")
                }
            } else {
                guard next == nil else { throw Aborted("Request accounting changed during tuning") }
                let rows = recentCounts(state)
                if ownedCompletion {
                    let added = rows.reduce(0) { $0 + max(0, $1.value - (recent[$1.key] ?? 0)) }
                    let removed = recent.reduce(0) { $0 + max(0, $1.value - (rows[$1.key] ?? 0)) }
                    // A full legacy ring can replace an identical row, so zero visible additions is valid.
                    // One completion can otherwise append one row and evict at most one old row.
                    guard added <= 1, removed <= added else {
                        throw Aborted("A request from an app arrived or the request history changed")
                    }
                } else {
                    guard rows == recent else { throw Aborted("A request from an app arrived or the request history changed") }
                }
                recent = rows
            }
            // Advance before sample eligibility checks: a disrupted owned request still completed.
            completed = next
        }
    }

    /// A machine-readable terminal result for the dashboard job, separate from human progress lines.
    static func outcome(_ value: String, reason: String? = nil) {
        var fields = ["outcome": value]
        if let reason { fields["reason"] = reason }
        if let data = try? JSONSerialization.data(withJSONObject: fields, options: [.sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        {
            print("ORIGAMI_TUNING_OUTCOME \(json)")
        }
    }

    /// Seconds in a minute of a tuning limit: 60, or the tuning-flow check's `minute_seconds` (`ORIGAMI_TUNING_HOST`), so
    /// its time limits pass in seconds.
    static let minute: Double = checkHost?["minute_seconds"] as? Double ?? 60
    /// The tuning-flow check's `ORIGAMI_TUNING_HOST` object, or nil (a real Mac).
    static var checkHost: [String: Any]? {
        ProcessInfo.processInfo.environment["ORIGAMI_TUNING_HOST"].flatMap {
            (try? JSONSerialization.jsonObject(with: Data($0.utf8))) as? [String: Any]
        }
    }

    /// The tuning-flow check's Mac (`ORIGAMI_TUNING_HOST`, a JSON object of `memory_bytes`, `swapout_bytes`,
    /// `memory_pressure`, `thermal_state`, `low_power`): fixed readings, so the flow it checks does not depend on the
    /// swapping or memory pressure of the machine running it (a small CI runner swaps during the comparison).
    static func conditions(_ state: [String: Any]) -> TuningConditions.Snapshot {
        func fixed(_ key: String) -> NSNumber? { checkHost?[key] as? NSNumber }
        return TuningConditions.Snapshot(
            monotonic: ProcessInfo.processInfo.systemUptime,
            physicalMemoryBytes: fixed("memory_bytes")?.uint64Value ?? ProcessInfo.processInfo.physicalMemory,
            swapoutBytes: fixed("swapout_bytes")?.uint64Value ?? swapoutBytes(),
            memoryPressure: fixed("memory_pressure")?.intValue ?? Int(pressureLevel()),
            thermalState: fixed("thermal_state")?.intValue ?? ProcessInfo.processInfo.thermalState.rawValue,
            lowPower: fixed("low_power")?.boolValue ?? ProcessInfo.processInfo.isLowPowerModeEnabled,
            budgetBytes: ((state["memory"] as? [String: Any])?["budget_bytes"] as? NSNumber)?.uint64Value)
    }

    static func idleState(
        _ client: Client, key: String, ownership: RequestOwnership,
        finishing: Bool = false
    ) async throws -> [String: Any] {
        var state = try await client.call("v1/status")
        // Finishing our own HTTP reply can briefly precede queue retirement. Never wait on a queued app request.
        if finishing {
            for _ in 0..<4 {
                let queue = state["queue"] as? [String: Any] ?? [:]
                guard queue["busy"] as? Bool == true, (queue["queued"] as? Int ?? 0) == 0 else { break }
                try await Task.sleep(for: .milliseconds(200))
                state = try await client.call("v1/status")
            }
        }
        guard state["status"] as? String == "ready",
            (state["tuning"] as? [String: Any])?["key"] as? String == key
        else { throw Aborted("The loaded model or context changed") }
        let queue = state["queue"] as? [String: Any] ?? [:]
        guard queue["busy"] as? Bool != true, (queue["queued"] as? Int ?? 0) == 0 else { throw Aborted("A request from an app arrived") }
        try ownership.accept(state, ownedCompletion: finishing)
        return state
    }

    /// Start as soon as the model is idle. Warning pressure and stable Low Power Mode are ordinary user conditions.
    /// Only critical conditions wait, bounded by both a short grace period and the user's tuning deadline.
    static func readyToMeasure(
        _ client: Client, key: String, deadline: Date,
        ownership: RequestOwnership
    ) async throws -> [String: Any] {
        let waitUntil = min(deadline, Date().addingTimeInterval(20))
        var announced = false
        while true {
            guard Date() < deadline else { throw Aborted("Time limit reached") }
            let state = try await idleState(client, key: key, ownership: ownership)
            let assessment = TuningConditions.assess(conditions(state))
            guard assessment.disposition == .critical else { return state }
            let reason = assessment.reason ?? "Critical system pressure"
            guard Date() < waitUntil else { throw Aborted(reason) }
            if !announced {
                print("Waiting briefly: \(reason)")
                announced = true
            }
            try await Task.sleep(for: .seconds(min(2, max(0.01, waitUntil.timeIntervalSinceNow))))
        }
    }

    static func completedRequests(_ state: [String: Any]) -> UInt64? {
        ((state["requests"] as? [String: Any])?["completed"] as? NSNumber)?.uint64Value
    }

    /// Older servers have no monotonic completion count. Compare complete rows as a multiset rather than rounding
    /// request times to seconds, which mistakes fast adjacent tuning replies for another application's traffic.
    static func recentCounts(_ state: [String: Any]) -> [Data: Int] {
        var result: [Data: Int] = [:]
        for row in (state["requests"] as? [String: Any])?["recent"] as? [[String: Any]] ?? [] {
            if let data = try? JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]) { result[data, default: 0] += 1 }
        }
        return result
    }

    static func measuredRequest(
        _ client: Client, key: String, prompt: String, tokens: Int,
        extra: [String: Any], deadline: Date, warmup: Bool,
        ownership: RequestOwnership,
        anchor: TuningConditions.Snapshot? = nil,
        minimumCompletionTokens: Int = 2
    ) async throws -> (run: TimedReply, after: TuningConditions.Snapshot) {
        let before = try await readyToMeasure(client, key: key, deadline: deadline, ownership: ownership)
        let initial = conditions(before)
        if !warmup, let anchor {
            let movement = TuningConditions.assess(initial, after: anchor)
            if movement.disposition != .usable { throw Disrupted(movement.reason ?? "Conditions changed during this round") }
        }
        let run: TimedReply
        do {
            run = try await Diagnostics.timed(
                client.base, prompt: prompt, maximumTokens: tokens,
                apiKey: client.key, extra: extra, deadline: deadline)
        } catch { throw Aborted(Date() >= deadline ? "Time limit reached" : "A tuning request failed: \(error)") }
        let after = try await idleState(client, key: key, ownership: ownership, finishing: true)
        let oldMemory = before["memory"] as? [String: Any] ?? [:]
        let newMemory = after["memory"] as? [String: Any] ?? [:]
        for field in ["cap_bytes", "target_bytes", "budget_mode"] {
            guard String(describing: oldMemory[field]) == String(describing: newMemory[field]) else {
                throw Aborted("The selected memory settings changed")
            }
        }
        let observed = conditions(after)
        let assessment = TuningConditions.assess(observed, after: initial)
        if assessment.disposition == .critical { throw Aborted(assessment.reason ?? "Critical system pressure") }
        if !warmup {
            guard run.completion >= minimumCompletionTokens else { throw Disrupted("The reply was too short for this measurement") }
            if assessment.disposition == .disrupted { throw Disrupted(assessment.reason ?? "This measurement was disrupted") }
            if let anchor {
                let movement = TuningConditions.assess(observed, after: anchor)
                if movement.disposition != .usable { throw Disrupted(movement.reason ?? "Conditions changed during this round") }
            }
        } else if assessment.disposition == .disrupted {
            print("Warm-up discarded: \(assessment.reason ?? "the machine was settling")")
        }
        return (run, observed)
    }
}
