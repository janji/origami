import Foundation

/// Causal next-cycle reuse probability, calibrated on early tuning routes only.
/// This metadata never changes target routing or arithmetic. The cache owns its
/// synchronization and the ledger charge, including all fixed history arrays.
package final class ExpertReusePolicy {
    package static let admittedBytes: UInt64 = 2 * 1024 * 1024
    package static let transientSlots = 24
    private var last = [Int](repeating: -1_000_000, count: 48 * 512)
    private var gap = [Int](repeating: 1_000_000, count: 48 * 512)
    private var mass = [Double](repeating: 0, count: 48 * 512)
    private var probability = [Double](repeating: 0, count: 48 * 512)
    private var protected = [Bool](repeating: false, count: 48 * 512)
    private var protectedKeys: [Int] = []
    private var layerWeight = [Double](repeating: 1, count: 48)
    // Immutable between execution phases, installed by the cache under its lock. This array is
    // covered by the existing 2 MiB owner; no source/queue callback runs during victim selection.
    private var storageCost = [Double](repeating: 1, count: 48 * 512)
    private var token = -1, previousLayer = 47
    package private(set) var rows: UInt64 = 0
    package private(set) var decisionNanoseconds: UInt64 = 0
    package private(set) var replayResets: UInt64 = 0
    private let ledger: MemoryLedger
    private let reservation: MemoryLedger.Reservation

    package init(ledger: MemoryLedger) throws {
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: Self.admittedBytes, category: "expert-reuse-metadata")
        reset()
    }
    deinit { try? ledger.release(reservation) }

    package func reset() {
        last = Array(repeating: -1_000_000, count: 48 * 512)
        gap = Array(repeating: 1_000_000, count: 48 * 512)
        mass = Array(repeating: 0, count: 48 * 512)
        probability = Array(repeating: ExpertReuseCalibration.probabilities[9 * 8 * 4 + 3], count: 48 * 512)
        protected = Array(repeating: false, count: 48 * 512)
        protectedKeys.removeAll(keepingCapacity: true)
        token = -1
        previousLayer = 47
    }

    private func bucket<T: Comparable>(_ value: T, _ bounds: [T]) -> Int {
        var result = 0
        for bound in bounds {
            if value < bound { break }
            result += 1
        }
        return result
    }

    package func beginRow(layer: Int, experts: [Int]) throws -> [Int] {
        // Up to the routed ten: a quality dial that drops experts records only those the token uses.
        guard (0..<48).contains(layer), (1...10).contains(experts.count), Set(experts).count == experts.count,
            experts.allSatisfy({ (0..<512).contains($0) })
        else {
            throw EngineError("Reuse history requires one authoritative layer cycle of 1 to 10 experts")
        }
        let began = DispatchTime.now().uptimeNanoseconds
        defer { decisionNanoseconds += DispatchTime.now().uptimeNanoseconds - began }
        if layer != (previousLayer + 1) % 48 {
            // Exact exception repair may discard an already staged next router
            // and replay an earlier layer. Discard uncertain history rather
            // than refusing or retaining observations from discarded work.
            reset()
            token = layer == 0 ? -1 : 0
            replayResets += 1
        }
        if layer == 0 { token += 1 }
        previousLayer = layer
        rows += 1
        for key in protectedKeys { protected[key] = false }
        protectedKeys = experts.map { layer * 512 + $0 }
        for key in protectedKeys {
            protected[key] = true
            let age = token - last[key]
            mass[key] = mass[key] * exp(-Double(age) / 32) + (1 - exp(-1.0 / 32))
            gap[key] = age
            last[key] = token
        }
        // Each layer's next opportunity advances only at its own authoritative
        // row. Other layers retain the probability for their upcoming visit.
        for key in (layer * 512)..<((layer + 1) * 512) {
            let age = token + 1 - last[key]
            let frequency = mass[key] * exp(-Double(age) / 32)
            let a = bucket(age, [1, 2, 3, 5, 9, 17, 33, 65, 129])
            let f = bucket(frequency, [0.01, 0.03, 0.06, 0.12, 0.25, 0.5, 0.8])
            let g = bucket(gap[key], [1, 4, 16])
            probability[key] = ExpertReuseCalibration.probabilities[(a * 8 + f) * 4 + g]
        }
        for other in 0..<48 {
            let distance = Double((other - layer + 47) % 48 + 1) / 48
            layerWeight[other] = ExpertReuseCalibration.layerReadCost[other] / (0.5 + distance)
        }
        return experts.sorted {
            let left = score(layer * 512 + $0)
            let right = score(layer * 512 + $1)
            return left == right ? $0 < $1 : left > right
        }
    }

    @discardableResult package func setStorageCosts(_ costs: [Double]) -> Bool {
        guard costs.count == storageCost.count, costs.allSatisfy({ $0.isFinite && (0.125...8).contains($0) }),
            costs != storageCost
        else { return false }
        storageCost = costs
        return true
    }
    package func score(_ key: Int) -> Double { probability[key] * layerWeight[key / 512] * storageCost[key] }
    package func isProtected(_ key: Int) -> Bool { protected[key] }
    package func recordDecisionTime(_ value: UInt64) { decisionNanoseconds += value }
}
