import Foundation
import OrigamiEngine

extension Tuning {
    /// Practical, request-scoped comparisons of the levels the model offers (`offered`, exact first) down to `floor`: a
    /// level it does not offer is its mildest fast one, as is no floor. The governor remains active and no settings are
    /// applied automatically.
    static func explore(
        _ settings: Settings, client original: Client, key: String, minutes: Double, offered: [String], floor: String?,
        ownership: RequestOwnership
    ) async throws {
        guard settings.allowFast else { throw CLIError("Quality tradeoffs are disabled by allow_fast") }
        let floor = floor.map { QualityDial.aliases[$0] ?? $0 }
        guard (3...30).contains(minutes) else { throw CLIError.usage("--minutes must be 3 to 30 for --tradeoffs") }
        guard offered.count > 1 else { throw CLIError("This model has no fast level to compare: it runs exact only") }
        if let floor, QualityDial.level(floor) == nil {
            throw CLIError.usage(
                "--quality-floor \(floor) is not a quality level: this model offers \(offered.dropFirst().joined(separator: ", "))")
        }
        let last = floor.flatMap { offered.firstIndex(of: $0) } ?? 1
        let deadline = Date().addingTimeInterval(minutes * 60)
        let revision = DashboardState.revision(settings)
        var client = original
        client.deadline = deadline
        try? FileManager.default.removeItem(at: proposalFile(settings))
        let levels = Array(offered.prefix(last + 1))
        let iso = ISO8601DateFormatter()
        var rates: [String: [Double]] = [:]
        var first: [String: [Double]] = [:]
        var notes: [String] = []
        print(
            "Comparing quality modes with your current settings; normal background activity is allowed. Settings change only if you apply a result."
        )
        do {
            // Discard cold work without treating ordinary cache filling or governor recovery as failed measurements.
            // Short replies and two rotated rounds fit slow SSD-streaming models into the user's time limit.
            var completed = 0
            for level in levels {
                _ = try await measuredRequest(
                    client, key: key, prompt: prompts[0], tokens: 16,
                    extra: [
                        "origami_mode": level, "origami_tune": [String: Any](),
                        "reasoning_effort": settings.reasoningEffort ?? "none",
                    ], deadline: deadline, warmup: true, ownership: ownership)
                completed += 1
                print("Warming up · \(level)  \(completed * 100 / (levels.count * 3))%")
            }
            for pass in 0..<2 {
                let order = Array(levels.dropFirst(pass % levels.count)) + levels.prefix(pass % levels.count)
                var accepted = false
                for attempt in 0..<2 {
                    let anchor = conditions(try await readyToMeasure(client, key: key, deadline: deadline, ownership: ownership))
                    var round: [(name: String, rate: Double, ttft: Double)] = []
                    do {
                        for level in order {
                            let sample = try await measuredRequest(
                                client, key: key, prompt: prompts[pass % prompts.count], tokens: 16,
                                extra: [
                                    "origami_mode": level, "origami_tune": [String: Any](),
                                    "reasoning_effort": settings.reasoningEffort ?? "none",
                                ], deadline: deadline, warmup: false,
                                ownership: ownership, anchor: anchor)
                            let run = sample.run
                            round.append((level, Double(run.completion - 1) / max(0.001, run.total - run.ttft), run.ttft))
                        }
                        // Publish an entire balanced round only. Never keep the faster part of a disrupted round.
                        for sample in round {
                            rates[sample.name, default: []].append(sample.rate)
                            first[sample.name, default: []].append(sample.ttft)
                        }
                        completed += levels.count
                        print("Measured round \(pass + 1) of 2  \(completed * 100 / (levels.count * 3))%")
                        accepted = true
                        break
                    } catch let disruption as Disrupted {
                        notes.append(disruption.description)
                        print(
                            "Discarded round: \(disruption.description). \(attempt == 0 ? "Retrying once within your time limit." : "No comparable round obtained.")"
                        )
                    }
                }
                guard accepted else { throw Disrupted("Conditions changed substantially in both attempts; no comparison was applied") }
            }
            // Validate after the final sample too; a last-arm model/settings change must never produce a proposal.
            _ = try await idleState(client, key: key, ownership: ownership)
            guard revision == DashboardState.revision(settings) else { throw Aborted("Settings changed during the comparison") }
            let results: [[String: Any]] = levels.compactMap { level in
                guard let samples = rates[level], samples.count >= 2 else { return nil }
                return [
                    "quality": level, "decode_tps": SelfTuning.median(samples),
                    "first_token_seconds": SelfTuning.median(first[level] ?? []),
                    "samples": samples.count, "spread": (samples.max()! - samples.min()!) / max(0.001, SelfTuning.median(samples)),
                ]
            }
            guard results.count == levels.count else { throw Disrupted("Some modes have too few completed measurements") }
            try Settings.write(
                [
                    "kind": "tradeoffs", "key": key, "revision": revision, "results": results,
                    "notes": notes, "date": iso.string(from: Date()),
                ], to: proposalFile(settings))
            record(settings, key: key, ["outcome": "comparison", "results": results, "notes": notes])
            print("Comparison ready. Review the speeds and variation; answer quality is not scored. Settings unchanged.")
            outcome("comparison")
        } catch {
            let reason = Date() >= deadline ? "Time limit reached" : "\(error)"
            let result = error is Disrupted ? "incomplete" : "aborted"
            record(settings, key: key, ["outcome": result, "reason": reason, "notes": notes])
            print("Comparison \(result == "aborted" ? "interrupted" : "incomplete"): \(reason). Settings unchanged.")
            outcome(result, reason: reason)
        }
    }
}
