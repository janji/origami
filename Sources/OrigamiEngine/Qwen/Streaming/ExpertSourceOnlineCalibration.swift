import Dispatch
import Foundation

/// Bounds of the online source estimator. Every estimate is a multiple of the
/// static calibration's value within [floor, ceiling] of the frozen curves. The
/// estimates drive source placement and the replica hedge delay, which may move within
/// [floor, delayCeiling] x the static delay; delayCeiling is at most 1, so the
/// hedge can fire earlier than the static rule when the external drive is slow,
/// never later (a measured 11.7 ms delay missed the decode handoff bound; the static 3.88 ms met it).
public struct ExpertSourceOnlineCalibration: Codable, Equatable, Sendable {
    public let floorMultiple, ceilingMultiple, smoothing: Double
    public let warmupObservations: Int
    public let delayCeilingMultiple: Double

    public init(
        floorMultiple: Double = 0.25, ceilingMultiple: Double = 4, smoothing: Double = 0.02,
        warmupObservations: Int = 32, delayCeilingMultiple: Double = 1
    ) {
        self.floorMultiple = floorMultiple
        self.ceilingMultiple = ceilingMultiple
        self.smoothing = smoothing
        self.warmupObservations = warmupObservations
        self.delayCeilingMultiple = delayCeilingMultiple
    }
    enum CodingKeys: String, CodingKey {
        case floorMultiple = "floor_multiple"
        case ceilingMultiple = "ceiling_multiple"
        case smoothing
        case warmupObservations = "warmup_observations"
        case delayCeilingMultiple = "delay_ceiling_multiple"
    }
    /// An absent delay ceiling is the forced maximum, 1; a present one may only lower it.
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            floorMultiple: try values.decode(Double.self, forKey: .floorMultiple),
            ceilingMultiple: try values.decode(Double.self, forKey: .ceilingMultiple),
            smoothing: try values.decode(Double.self, forKey: .smoothing),
            warmupObservations: try values.decode(Int.self, forKey: .warmupObservations),
            delayCeilingMultiple: try values.decodeIfPresent(Double.self, forKey: .delayCeilingMultiple) ?? 1)
    }
    public func validate() throws {
        guard floorMultiple.isFinite, ceilingMultiple.isFinite, smoothing.isFinite, delayCeilingMultiple.isFinite,
            floorMultiple > 0, floorMultiple <= 1, ceilingMultiple >= 1, ceilingMultiple <= 64,
            smoothing > 0, smoothing <= 1, (1...1_000_000).contains(warmupObservations),
            delayCeilingMultiple >= floorMultiple, delayCeilingMultiple <= 1
        else {
            throw EngineError(
                "expert_source_online_bounds: floor_multiple must be in (0, 1], ceiling_multiple in [1, 64], smoothing in (0, 1], warmup_observations in 1...1000000 and delay_ceiling_multiple in [floor_multiple, 1] (the hedge never fires later than the static delay)",
                code: "invalid_configuration")
        }
    }
}

/// Path counters of the online estimator, serialized in the experts snapshot.
public struct ExpertSourceOnlineSnapshot: Codable, Sendable {
    public struct Source: Codable, Sendable {
        /// Completed record reads that trained this source; `rejected` had a non-positive or non-finite time.
        public let observations, rejected: UInt64
        public let warm: Bool
        /// Current multiple of the static estimate consumed by decisions (1 until warm), the raw
        /// smoothed multiple, and the clamped observation range seen so far.
        public let multiple, smoothedMultiple, minimumObservedMultiple, maximumObservedMultiple: Double
        /// Online single-record estimate at queue depth one, in seconds, and the static one.
        public let singleRecordSeconds, staticSingleRecordSeconds: Double
        public let observedSeconds: Double
        public let observedBytes: UInt64
    }
    public let settings: ExpertSourceOnlineCalibration
    public let internalSSD, externalSSD: Source
    /// Kept for older status consumers. Mirror observations now have their own curve and are not skipped.
    public let hotInternalReadsSkipped: UInt64
    public let hotMirror: Source
    /// Earliest-completion selections supplied a curve; `Differing` counts adjusted curves,
    /// not proof that the winning source differs from a counterfactual static placement.
    public let placementDecisions, placementDecisionsDiffering: UInt64
    /// Replica hedge delays derived online, and how many were shorter than the static delay.
    public let hedgeDelays, hedgeDelaysDiffering: UInt64
    public let lastHedgeDelaySeconds, minimumHedgeDelaySeconds, maximumHedgeDelaySeconds: Double
    public var observations: UInt64 { internalSSD.observations &+ externalSSD.observations &+ hotMirror.observations }
    public var decisionsDiffering: UInt64 { hedgeDelaysDiffering &+ placementDecisionsDiffering }
}

/// Per-source running multiple of the static calibration, trained from actual
/// record reads on the reader threads. The multiple is the exponentially
/// weighted ratio of observed service time to the static prediction at the
/// observed drive concurrency. Placement takes one immutable snapshot before queue admission;
/// mirror and original-span reads train separate curves. Bytes never change.
public final class ExpertSourceOnlineEstimator: @unchecked Sendable {
    public let calibration: ExpertSourceCalibration
    public let settings: ExpertSourceOnlineCalibration
    public let maximumConcurrent: Int
    private struct State {
        var observations: UInt64 = 0, rejected: UInt64 = 0
        var smoothed = 1.0, minimumObserved = Double.infinity, maximumObserved = 0.0
        var seconds = 0.0, bytes: UInt64 = 0
        /// `clock` seconds of the last observation.
        var at = 0.0
    }
    private let lock = NSLock()
    private var internalState = State(), externalState = State(), hotState = State()
    private var hotSkipped: UInt64 = 0
    private var placements: UInt64 = 0, differingPlacements: UInt64 = 0
    private var demandPlacements: UInt64 = 0
    private var hedgeDelays: UInt64 = 0, hedgeDelaysDiffering: UInt64 = 0
    private var lastHedgeDelay = 0.0, minimumHedgeDelay = Double.infinity, maximumHedgeDelay = 0.0
    /// Seconds, for settling an idle source back toward the static curve (`LiveDriveEstimate`); checks pass their own.
    private let clock: @Sendable () -> Double

    public init(
        calibration: ExpertSourceCalibration, settings: ExpertSourceOnlineCalibration,
        maximumConcurrent: Int, clock: @escaping @Sendable () -> Double = { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }
    ) throws {
        try calibration.validate()
        try settings.validate()
        guard (1...64).contains(maximumConcurrent) else {
            throw EngineError("Online source calibration requires a bounded read concurrency", code: "invalid_configuration")
        }
        self.calibration = calibration
        self.settings = settings
        self.maximumConcurrent = maximumConcurrent
        self.clock = clock
    }

    private func clamp(_ value: Double) -> Double {
        min(max(value, settings.floorMultiple), settings.ceilingMultiple)
    }

    /// Static service time of one read among `concurrency` running reads: the
    /// aggregate curve at that depth shared by the running reads.
    public func predictedSeconds(source: IOSource, bytes: Int, concurrency: Int) -> Double {
        let depth = max(1, min(maximumConcurrent, concurrency))
        return calibration.estimatedSeconds(
            source: source, bytesIncludingNew: UInt64(depth) * UInt64(max(0, bytes)),
            jobsIncludingNew: depth, maximumConcurrent: maximumConcurrent)
    }

    /// One completed read. Called from the reader thread after the record was
    /// read; the observation is clamped before it enters the running estimate.
    public func observe(source: IOSource, bytes: Int, seconds: Double, concurrency: Int, hotInternal: Bool = false) {
        let predicted = predictedSeconds(source: source, bytes: bytes, concurrency: concurrency)
        let valid = seconds.isFinite && seconds > 0 && bytes > 0 && predicted.isFinite && predicted > 0
        let ratio = valid ? clamp(seconds / predicted) : 0
        lock.lock()
        defer { lock.unlock() }
        // Under the lock, so concurrent readers never move `state.at` backwards.
        let now = clock()
        func update(_ state: inout State) {
            guard valid else {
                state.rejected &+= 1
                return
            }
            state.observations &+= 1
            if state.observations == 1 {
                state.smoothed = ratio
            } else {
                state.smoothed += LiveDriveEstimate.weight(settings.smoothing, gapSeconds: now - state.at) * (ratio - state.smoothed)
            }
            state.smoothed = clamp(state.smoothed)
            state.minimumObserved = min(state.minimumObserved, ratio)
            state.maximumObserved = max(state.maximumObserved, ratio)
            state.seconds += seconds
            state.bytes &+= UInt64(bytes)
            state.at = now
        }
        if source == .internalSSD {
            if hotInternal { update(&hotState) } else { update(&internalState) }
        } else {
            update(&externalState)
        }
    }

    /// Legacy counter for callers that cannot identify a mirror observation.
    public func skipHotInternalRead() {
        lock.lock()
        hotSkipped &+= 1
        lock.unlock()
    }

    private func multipleLocked(_ source: IOSource, hotInternal: Bool = false) -> Double {
        let state = source == .internalSSD ? (hotInternal ? hotState : internalState) : externalState
        guard state.observations >= UInt64(settings.warmupObservations) else { return 1 }
        return LiveDriveEstimate.settled(clamp(state.smoothed), stored: 1, idleSeconds: clock() - state.at)
    }
    /// Current multiple of the static estimate; exactly 1 until the source is warm.
    public func multiple(for source: IOSource) -> Double {
        lock.lock()
        defer { lock.unlock() }
        return multipleLocked(source)
    }

    /// Small frozen curves: queue selection holds only the scheduler lock, never the estimator's.
    /// Cold curves retain the measured setup calibration until enough actual reads arrive.
    package func placementCalibration(hotInternal: Bool, recordDecision: Bool = true) -> ExpertSourceCalibration {
        lock.lock()
        let internalMultiple = multipleLocked(.internalSSD, hotInternal: hotInternal)
        let externalMultiple = multipleLocked(.externalSSD)
        if recordDecision {
            placements &+= 1
            if internalMultiple != 1 || externalMultiple != 1 { differingPlacements &+= 1 }
        }
        lock.unlock()
        func rates(_ values: [Double], _ multiple: Double) -> [Double] {
            values.map { min(64_000_000_000, max(1_000_000, $0 / multiple)) }
        }
        return ExpertSourceCalibration(
            queueDepths: calibration.queueDepths,
            internalBytesPerSecond: rates(calibration.internalBytesPerSecond, internalMultiple),
            externalBytesPerSecond: rates(calibration.externalBytesPerSecond, externalMultiple),
            internalLatencySeconds: min(60, calibration.internalLatencySeconds * internalMultiple),
            externalLatencySeconds: min(60, calibration.externalLatencySeconds * externalMultiple),
            label: calibration.label, replicaDelaySeconds: calibration.replicaDelaySeconds)
    }

    /// Occasionally place an ordinary demand read on an idle alternative so a stale/cold
    /// estimate cannot permanently starve it. The scheduler refuses this hint if it is busy.
    package func idleProbeSource(hotInternal: Bool) -> IOSource? {
        lock.lock()
        defer { lock.unlock() }
        demandPlacements &+= 1
        guard demandPlacements % 64 == 1 else { return nil }
        let internalCount = hotInternal ? hotState.observations : internalState.observations
        if internalCount < UInt64(settings.warmupObservations) { return .internalSSD }
        if externalState.observations < UInt64(settings.warmupObservations) { return .externalSSD }
        return demandPlacements / 64 % 2 == 0 ? .internalSSD : .externalSSD
    }

    /// Replica hedge delay: the static base scaled by the online change in the
    /// internal-to-external time ratio, within [floor, delayCeiling] x base with
    /// delayCeiling at most 1. A slower external primary hedges sooner; nothing
    /// makes the hedge fire later than the static delay.
    public func hedgeDelaySeconds(base: Double, primary: IOSource = .externalSSD, hotInternal: Bool = false) -> Double {
        lock.lock()
        defer { lock.unlock() }
        let internalMultiple = multipleLocked(.internalSSD, hotInternal: hotInternal)
        let externalMultiple = multipleLocked(.externalSSD)
        let scale = primary == .externalSSD ? internalMultiple / externalMultiple : externalMultiple / internalMultiple
        let delay = min(max(base * scale, base * settings.floorMultiple), base * settings.delayCeilingMultiple)
        hedgeDelays &+= 1
        if delay != base { hedgeDelaysDiffering &+= 1 }
        lastHedgeDelay = delay
        minimumHedgeDelay = min(minimumHedgeDelay, delay)
        maximumHedgeDelay = max(maximumHedgeDelay, delay)
        return delay
    }

    public func snapshot() -> ExpertSourceOnlineSnapshot {
        lock.lock()
        defer { lock.unlock() }
        func describe(_ source: IOSource, _ state: State, hotInternal: Bool = false) -> ExpertSourceOnlineSnapshot.Source {
            let staticSingle = calibration.estimatedSeconds(
                source: source, bytesIncludingNew: UInt64(PackStore.expertBytes),
                jobsIncludingNew: 1, maximumConcurrent: 1)
            let multiple = multipleLocked(source, hotInternal: hotInternal)
            return .init(
                observations: state.observations, rejected: state.rejected,
                warm: state.observations >= UInt64(settings.warmupObservations),
                multiple: multiple, smoothedMultiple: state.smoothed,
                minimumObservedMultiple: state.observations == 0 ? 0 : state.minimumObserved,
                maximumObservedMultiple: state.maximumObserved,
                singleRecordSeconds: staticSingle * multiple, staticSingleRecordSeconds: staticSingle,
                observedSeconds: state.seconds, observedBytes: state.bytes)
        }
        return .init(
            settings: settings, internalSSD: describe(.internalSSD, internalState),
            externalSSD: describe(.externalSSD, externalState), hotInternalReadsSkipped: hotSkipped,
            hotMirror: describe(.internalSSD, hotState, hotInternal: true),
            placementDecisions: placements, placementDecisionsDiffering: differingPlacements,
            hedgeDelays: hedgeDelays, hedgeDelaysDiffering: hedgeDelaysDiffering,
            lastHedgeDelaySeconds: lastHedgeDelay,
            minimumHedgeDelaySeconds: hedgeDelays == 0 ? 0 : minimumHedgeDelay,
            maximumHedgeDelaySeconds: maximumHedgeDelay)
    }
}
