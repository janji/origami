import Foundation

/// Placement only: every selected source contains the same complete expert.
/// Default policy preserves the existing packed-first path and joined fallback.
public enum ExpertSourcePolicy: String, Codable, Sendable {
    case packedPreferred, earliestCompletion
}

/// Frozen whole-record host-read curves. The internal curve already includes
/// nine original tensor-span reads; external is one packed record read. Rates
/// are aggregate per drive at the listed concurrency, never per worker.
public struct ExpertSourceCalibration: Codable, Sendable {
    public let queueDepths: [Int]
    public let internalBytesPerSecond, externalBytesPerSecond: [Double]
    public let internalLatencySeconds, externalLatencySeconds: Double
    public let label: String
    /// Replica hedge delay. Absent: four times this calibration's single-record external estimate
    /// (the original rule). Set it to keep the hedge timing when the routing rates change.
    public let replicaDelaySeconds: Double?

    public init(
        queueDepths: [Int], internalBytesPerSecond: [Double], externalBytesPerSecond: [Double],
        internalLatencySeconds: Double = 0, externalLatencySeconds: Double = 0, label: String,
        replicaDelaySeconds: Double? = nil
    ) {
        self.queueDepths = queueDepths
        self.internalBytesPerSecond = internalBytesPerSecond
        self.externalBytesPerSecond = externalBytesPerSecond
        self.internalLatencySeconds = internalLatencySeconds
        self.externalLatencySeconds = externalLatencySeconds
        self.label = label
        self.replicaDelaySeconds = replicaDelaySeconds
    }
    enum CodingKeys: String, CodingKey {
        case queueDepths = "queue_depths"
        case internalBytesPerSecond = "internal_bytes_per_second"
        case externalBytesPerSecond = "external_bytes_per_second"
        case internalLatencySeconds = "internal_latency_seconds"
        case externalLatencySeconds = "external_latency_seconds"
        case label
        case replicaDelaySeconds = "replica_delay_seconds"
    }

    /// The curve when this Mac's drives are not measured yet: both drives alike, 1 GB/s at every depth. Equal curves
    /// split reads by outstanding bytes alone (ties to the pack), so no drive is favoured on another Mac's numbers; the
    /// rate only sets the hedge delay (four single-record reads), on the late side, and the online estimator can only
    /// shorten it. `DriveSpeed` measurements replace it (`init(model:pack:recordBytes:)`).
    public static let neutral = ExpertSourceCalibration(
        queueDepths: DriveSpeed.queueDepths, internalBytesPerSecond: DriveSpeed.queueDepths.map { _ in 1e9 },
        externalBytesPerSecond: DriveSpeed.queueDepths.map { _ in 1e9 }, label: "neutral")

    /// Curves from this Mac's measured drives (`DriveSpeed`): the model's drive and the pack's, at the depths both
    /// measured. The replica hedge waits 1.33x one record's single-reader pack read (the setup rule since 1.0).
    public init(model: DriveSpeed, pack: DriveSpeed, recordBytes: Int) {
        let depths = model.queueDepths.filter(pack.queueDepths.contains)
        self.init(
            queueDepths: depths, internalBytesPerSecond: depths.map { model.rate(depth: $0) },
            externalBytesPerSecond: depths.map { pack.rate(depth: $0) },
            label: "measured-\(Int(min(model.measuredAt, pack.measuredAt)))",
            replicaDelaySeconds: min(1, max(0.0001, 1.33 * Double(recordBytes) / pack.rate(depth: 1))))
    }

    public func validate() throws {
        guard (1...7).contains(queueDepths.count), queueDepths.first == 1,
            queueDepths == queueDepths.sorted(), Set(queueDepths).count == queueDepths.count,
            queueDepths.allSatisfy({ (1...64).contains($0) }),
            internalBytesPerSecond.count == queueDepths.count, externalBytesPerSecond.count == queueDepths.count,
            (internalBytesPerSecond + externalBytesPerSecond).allSatisfy({ $0.isFinite && (1_000_000...64_000_000_000).contains($0) }),
            [internalLatencySeconds, externalLatencySeconds].allSatisfy({ $0.isFinite && (0...60).contains($0) }),
            !label.isEmpty, label.utf8.count <= 256,
            replicaDelaySeconds.map({ $0.isFinite && (0.0001...1).contains($0) }) ?? true
        else {
            throw EngineError("Invalid bounded expert-source calibration", code: "invalid_configuration")
        }
    }

    /// Copies of the pack on `drives` drives read at once: d reads in flight keep min(d, drives) drives busy, each at
    /// depth d / drives. Model-drive rates, latencies and the hedge delay stay.
    public func spread(over drives: Int) -> ExpertSourceCalibration {
        let external = queueDepths.map { depth in
            Double(min(depth, drives)) * externalBytesPerSecond[queueDepths.lastIndex { $0 <= max(1, depth / drives) }!]
        }
        return ExpertSourceCalibration(
            queueDepths: queueDepths, internalBytesPerSecond: internalBytesPerSecond, externalBytesPerSecond: external,
            internalLatencySeconds: internalLatencySeconds, externalLatencySeconds: externalLatencySeconds,
            label: label + "-\(drives)-drives", replicaDelaySeconds: replicaDelaySeconds)
    }

    /// Conservative outstanding work includes running, queued, cancellation and
    /// callback-retirement charges. There is no guessed pread progress or online
    /// training. Caller validates and holds the actual queue admission lock.
    public func estimatedSeconds(
        source: IOSource, bytesIncludingNew: UInt64,
        jobsIncludingNew: Int, maximumConcurrent: Int
    ) -> Double {
        let depth = max(1, min(maximumConcurrent, jobsIncludingNew))
        let index = queueDepths.lastIndex(where: { $0 <= depth })!
        let rate = source == .internalSSD ? internalBytesPerSecond[index] : externalBytesPerSecond[index]
        let latency = source == .internalSSD ? internalLatencySeconds : externalLatencySeconds
        return latency + Double(bytesIncludingNew) / rate
    }
}
