import Darwin
import Foundation
import Metal
import ObjectiveC

/// Experimental decode-only semantics. An omitted source FFN contribution is applied once at the
/// following attention boundary, using the original input and gate. This is intentionally approximate.
/// Version 1 replaces tail/late skipping, keeps the largest routed expert and permits one pending expert.
public struct DelayedExpertPolicy: Codable, Sendable, Equatable {
    public var version: Int
    public var maximumMass: Double
    public var maximumExpertMass: Double
    public var maximumExperts: Int
    public init(version: Int = 1, maximumMass: Double = 0.05, maximumExpertMass: Double = 0.05, maximumExperts: Int = 1) {
        self.version = version
        self.maximumMass = maximumMass
        self.maximumExpertMass = maximumExpertMass
        self.maximumExperts = maximumExperts
    }
    public var signature: String { "defer-v\(version)-m\(maximumMass)-e\(maximumExpertMass)-n\(maximumExperts)" }
    public func validated() throws -> Self {
        guard version == 1, maximumExperts == 1, maximumMass.isFinite, maximumExpertMass.isFinite,
            maximumMass > 0, maximumMass <= 0.1, maximumExpertMass > 0, maximumExpertMass <= maximumMass
        else {
            throw EngineError(
                "Delayed experts require version 1, one expert, and finite mass limits in (0, 0.1]", code: "invalid_configuration")
        }
        return self
    }
    /// Pure selection, with original (not renormalised immediate) routed gates. The caller supplies
    /// actual incomplete/nonresident candidates and refuses the policy at final/token/exact boundaries.
    package func rank(weights: [Float], late: Set<Int>, resident: Set<Int> = [], capacity: Int) -> Int? {
        guard capacity > 0, weights.count > 1, weights.allSatisfy({ $0.isFinite && $0 >= 0 }),
            let largest = weights.indices.max(by: { weights[$0] < weights[$1] })
        else { return nil }
        let mass = weights.reduce(0, +)
        guard mass.isFinite, mass > 0 else { return nil }
        return weights.indices.filter {
            $0 != largest && late.contains($0) && !resident.contains($0) && weights[$0] > 0
                && Double(weights[$0] / mass) <= min(maximumMass, maximumExpertMass)
        }.min { weights[$0] == weights[$1] ? $0 < $1 : weights[$0] < weights[$1] }
    }
}

/// Versioned tiny mathematical reference. Streams are [stream][hidden], never a flattened hidden
/// vector. GLM and DeepSeek use comb[destination + 4 * source], exactly as hc_expand does.
/// Next attention/HC coefficients and recurrent/KV updates are evaluated on the immediate streams.
/// DeepSeek's lagged pre coefficients remain those already computed; injection does not replay them.
package enum DelayedExpertReference {
    package enum Family: Equatable { case qwen, glm, deepseek }
    package static func inject(
        _ streams: [Float], expert: [Float], gate: Float, sourcePost: [Float],
        family: Family, nextCombine: [Float] = []
    ) throws -> [Float] {
        guard !expert.isEmpty, streams.count == expert.count * 4, sourcePost.count == 4,
            family == .qwen || nextCombine.count == 16
        else { throw EngineError("Delayed HC reference geometry differs") }
        let h = expert.count
        var result = streams
        for d in 0..<4 {
            var scale = sourcePost[d]
            if family != .qwen {
                scale =
                    nextCombine[d] * sourcePost[0] + nextCombine[d + 4] * sourcePost[1]
                    + nextCombine[d + 8] * sourcePost[2] + nextCombine[d + 12] * sourcePost[3]
            }
            for j in 0..<h {
                if family == .qwen {
                    let delta = LayerArithmetic.round(expert[j] * gate)
                    result[d * h + j] = LayerArithmetic.round(result[d * h + j] + LayerArithmetic.round(delta * scale))
                } else {
                    result[d * h + j] += (expert[j] * gate) * scale
                }
            }
        }
        return result
    }
}

/// One separately owned request slot. It never borrows phase scratch. The associated reservation
/// follows the last Metal buffer owner; request cleanup joins work before dropping this slot.
package final class DelayedExpertStorage {
    private final class Owner {
        let ledger: MemoryLedger, reservation: MemoryLedger.Reservation
        init(_ ledger: MemoryLedger, bytes: UInt64, footprint: UInt64?) throws {
            self.ledger = ledger
            reservation = try ledger.reservePhysical(bytes: bytes, category: "delayed-expert-slot", footprint: footprint)
        }
        deinit { try? ledger.release(reservation) }
    }
    private enum Association { nonisolated(unsafe) static var key: UInt8 = 0 }
    package let hidden: Int
    /// FP32 input for families (Qwen stores BF16 in its leading half), one expert output, four source post gates.
    package let input, output, post: MTLBuffer
    package let allocatedBytes: UInt64
    /// `footprint` replaces the process-footprint sample for physical admission (checks only).
    package init(device: MTLDevice, ledger: MemoryLedger, hidden: Int, footprint: UInt64? = nil) throws {
        guard hidden > 0 else { throw EngineError("Invalid delayed expert width") }
        func allocate(_ count: Int) throws -> MTLBuffer {
            let bytes = count * 4
            let page = Int(getpagesize())
            let charge = (bytes + page - 1) / page * page
            let owner = try Owner(ledger, bytes: UInt64(charge), footprint: footprint)
            // Request the admitted physical extent, not the tiny logical payload. Metal may
            // suballocate shorter requests at a granularity below the host VM page size;
            // the owned extent and final-owner ledger charge must describe the same bytes.
            guard let b = device.makeBuffer(length: charge, options: .storageModeShared), b.allocatedSize == charge else {
                throw EngineError("Cannot admit delayed expert storage", code: "insufficient_memory")
            }
            objc_setAssociatedObject(b, &Association.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
            return b
        }
        self.hidden = hidden
        input = try allocate(hidden)
        output = try allocate(hidden)
        post = try allocate(4)
        allocatedBytes = UInt64(input.allocatedSize + output.allocatedSize + post.allocatedSize)
    }
}

public struct DelayedExpertCounters: Sendable {
    public var deferredExperts: UInt64 = 0
    public var deferredMass: Double = 0
    public var deadlineJoins: UInt64 = 0
    public var capacityRefusals: UInt64 = 0
    public var appliedContributions: UInt64 = 0
    public var earlyReadSubmissions: UInt64 = 0
    /// Demand bytes were still outstanding at submission; this is not an exclusive I/O overlap timer.
    public var earlySubmissionsWithDemand: UInt64 = 0
    public init() {}
}
