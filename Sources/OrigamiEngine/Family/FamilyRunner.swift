import CExactIO
import Darwin
import Foundation
import Metal
import Synchronization

/// What a family engine's forward pass is built from (GLM-5.3-Flash, DeepSeek-V4.1-Flash): the GPU queue, one command
/// buffer and encoder filled at a time and flushed where the host needs a result (the router), reusable working
/// buffers, ggml matrices, and the routed-expert step (top-k on the host, records read into the slot pool, each used
/// expert run over its rows). Decode is one row, a prompt many; the code is the same.
package class FamilyRunner: @unchecked Sendable {
    package struct FFN {
        let gate, up, down: FamilyWeights.Matrix
        package init(gate: FamilyWeights.Matrix, up: FamilyWeights.Matrix, down: FamilyWeights.Matrix) {
            self.gate = gate
            self.up = up
            self.down = down
        }
    }
    package typealias Ref = FamilyKernels.Ref

    let weights: FamilyWeights
    package var experts: FamilyExperts!
    let kernels: FamilyKernels
    let matrices: GGMLMatrixKernels
    let queue: MTLCommandQueue
    let g: ModelGeometry
    let hidden: Int
    package let contextLimit: Int
    /// Rows a prompt is cut into inside a layer (attention working memory scales with it).
    let chunk = 256
    /// Rows of a prompt each expert call takes (a prompt is cut into slices of this many rows, as it always was).
    package static let promptSlice = 2048
    /// Rows from which a prompt call is expected to use nearly every expert of the next MoE layer
    /// (`PromptReadRows.minimum`: GLM 246, DeepSeek 439), so `moe` reads that layer's first wave ahead. It changes only
    /// which reads are issued and when, never the output.
    package var readAheadRows: Int { PromptReadRows.minimum(expertsPerToken: g.expertsPerToken, experts: g.experts) }
    /// Performance cores (`hw.perflevel0`; all cores where it is missing): a prompt's top-k is split over them, as an
    /// efficiency core's share would finish last.
    static let performanceCores: Int = {
        var count: Int32 = 0
        var size = MemoryLayout<Int32>.size
        return sysctlbyname("hw.perflevel0.logicalcpu", &count, &size, nil, 0) == 0 && count > 0
            ? Int(count) : ProcessInfo.processInfo.activeProcessorCount
    }()
    /// Slices one forward pass reads (`promptRows`): each layer's experts are read once for all of them, not once a
    /// slice, which nearly every slice uses whole. Set by `slicePrompts` before the expert pool is sized.
    package private(set) var promptSlices = 1
    private var selectedPrefillSlices: Int?
    /// Slices a call of the prompt being read takes when more than `promptSlices` (`planPrompt`).
    private var plannedSlices: Int?
    /// The request's grouping, bounded by the working memory reserved at load, or by the prompt's loan (`planPrompt`)
    /// unless a smaller grouping is chosen. Changing it allocates nothing.
    package var prefillSlices: Int {
        get { selectedPrefillSlices.flatMap { $0 < promptSlices ? $0 : nil } ?? plannedSlices ?? promptSlices }
        set { selectedPrefillSlices = max(1, min(newValue, promptSlices)) }
    }
    package var promptRows: Int { Self.promptSlice * prefillSlices }
    package private(set) var position = 0
    /// The quality dial every MoE layer routes with; the serving backend sets it to exact for prompts.
    package var routing = QualityDial.exact
    /// Opt-in scheduling of an already selected delayed contribution. The policy remains approximate;
    /// its original input, gate and fixed next-attention injection boundary stay unchanged.
    package var earlyDelayedReads = false
    /// Benches only: a pause before each decode layer's experts, as a read leaves the GPU idle (`--bench-family-prefill`).
    package var benchReadGap: TimeInterval = 0
    package var benchGapSpins = false
    package var routeTrace: FamilyRouteTrace?
    package let wantsRouteTrace = ProcessInfo.processInfo.environment["ORIGAMI_FAMILY_ROUTE_TRACE"] == "1"
    package let refreshesIdleOverhead = ProcessInfo.processInfo.environment["ORIGAMI_FAMILY_REFRESH_IDLE_OVERHEAD"] == "1"
    package private(set) var idleOverheadRefreshes = 0
    package var shouldContinue: () -> Bool = { true }
    package private(set) var delayedExpertCounters = DelayedExpertCounters()
    private var delayedStorage: DelayedExpertStorage?
    private final class DelayedRead: @unchecked Sendable {
        var source = 0, target = 0
        var gate: Float = 0, limit: Float = 0
        let done = DispatchSemaphore(value: 0)
        var slots: [FamilyExperts.Slot] = []
        var error: Error?
        func prepare(source: Int, gate: Float, limit: Float) {
            self.source = source
            target = source + 1
            self.gate = gate
            self.limit = limit
            slots.removeAll(keepingCapacity: true)
            error = nil
        }
    }
    private var delayedReadSlot: DelayedRead?
    private var delayedRead: DelayedRead?
    /// Once per request, outside a forward; no buffers are allocated for a disabled policy.
    package func prepareDelayedExpertStorage(for quality: QualityDial) throws {
        guard command == nil, delayedRead == nil else { throw EngineError("Delayed policy changes require an idle family runner") }
        delayedExpertCounters = .init()
        delayedStorage = nil
        delayedReadSlot = nil
        guard let policy = quality.delayedExperts else { return }
        _ = try policy.validated()
        guard let ledger, ledger.optionalGrowthAllowed else {
            delayedExpertCounters.capacityRefusals += 1
            return
        }
        do {
            delayedStorage = try DelayedExpertStorage(device: weights.device, ledger: ledger, hidden: hidden)
            delayedReadSlot = DelayedRead()
        } catch let error as EngineError where error.code == "insufficient_memory" { delayedExpertCounters.capacityRefusals += 1 }
    }
    package func releaseDelayedExpertStorage() {
        // Teardown never recycles a cache slot while its reader or an encoded command still owns it.
        if let read = delayedRead {
            read.done.wait()
            delayedRead = nil
        }
        delayedStorage = nil
        delayedReadSlot = nil
    }
    /// Fixed same-token deadline, after next attention has written its streams and before its FFN HC.
    /// No cache mutation can occur between starting the delayed fetch and this joined boundary.
    package func injectDelayedExperts(at layer: Int, streams: MTLBuffer, combine: MTLBuffer) throws {
        guard let read = delayedRead else { return }
        guard layer == read.target, let slot = delayedStorage else { throw EngineError("Delayed family contribution crossed its deadline") }
        _ = try flush(wait: false)  // independent attention runs while the already-authorized source read finishes
        onReads { read.done.wait() }
        delayedRead = nil
        delayedExpertCounters.deadlineJoins += 1
        // The semaphore is consumed exactly once, including when encoding or the GPU join fails.
        // `read` and the reusable request slot keep its record owners alive through forwardPass.end().
        if let error = read.error { throw error }
        memset(slot.output.contents(), 0, hidden * 4)
        try runExperts(encoder, slots: read.slots, input: slot.input, output: slot.output, weights: [read.gate], limit: read.limit)
        let (grid, group) = grid2(hidden, 1)
        try kernels.dispatch(
            encoder, "hc_delayed_add", [Ref(slot.output), Ref(slot.post), Ref(combine), Ref(streams)],
            ints: [Int32(hidden)], grid: grid, group: group)
        // The fetch's stamps protect records until this join. A following MoE may then reuse its slots.
        try flush(wait: true)
        delayedRead = nil
        delayedExpertCounters.appliedContributions += 1
    }

    /// Where forward time goes (seconds, summed over calls): waiting for the GPU, the GPU's own execution time, waiting
    /// for expert reads, and in all; `commits` counts the command buffers committed, `longest` is the longest one's GPU
    /// time and `longestStep` its label, `largestWork` the most multiply-adds one held. `gpuByCommandLabels` groups whole
    /// buffers by their open-time label; a buffer can contain later phases too, so these are diagnostic labels, not
    /// exclusive phase timings.
    package struct Timing: Sendable {
        package var gpuWait = 0.0, reads = 0.0, total = 0.0, calls = 0, gpu = 0.0, commits = 0, longest = 0.0, largestWork = 0.0
        /// Completed buffers that ran longer than `CommandBudget.seconds`.
        package var overBound = 0
        package var longestStep = ""
        package var gpuByCommandLabels: [String: Double] = [:]
        package init() {}
    }
    package var timing: Timing {
        timingLock.lock()
        defer { timingLock.unlock() }
        return timingState
    }
    private var timingState = Timing()
    /// Benches: the longest buffer counted from now.
    package func resetLongestBuffer() {
        timingLock.lock()
        timingState.longest = 0
        timingLock.unlock()
    }
    private let timingLock = NSLock()
    private var command: MTLCommandBuffer?
    package var encoder: MTLComputeCommandEncoder!

    package init(weights: FamilyWeights, context: Int) throws {
        self.weights = weights
        g = weights.geometry
        guard let queue = weights.device.makeCommandQueue() else { throw EngineError("No Metal command queue") }
        self.queue = queue
        kernels = try FamilyKernels(device: weights.device)
        matrices = try GGMLMatrixKernels(device: weights.device)
        hidden = g.hidden
        contextLimit = min(context, g.context)
        stateCapacity = contextLimit
    }

    /// The expert slot pool, its slots counted for the most it may hold (`poolMaximum`): `budget` bytes
    /// (`expert_memory`), else what `memoryLimit`, else 70% of the Mac's memory (the rest stays with other apps, as
    /// Origami's planner leaves it below 80 GB), leaves once the dense weights, working buffers and the whole context's
    /// state are counted (`expertBudget`), with the per-position state past `stateCapacity` (not allocated) lent on top;
    /// `reserveState` takes it back before the state needs it. With `ledger` (the memory governor's), the weights and
    /// working buffers, the state and every slab are reserved in it. Automatic mode fits free memory; explicit mode
    /// targets the chosen allowance unless macOS already reports pressure. Returns free memory before the pool.
    func makeExperts(budget: UInt64?, memoryLimit: UInt64?, ledger: MemoryLedger?) throws -> UInt64 {
        if lazyStartupWorkspace {
            // Prove that at least one real forward plan and the mandatory expert floor fit at startup.
            // Larger requests still admit their actual geometry before any scratch allocation.
            let counts =
                try scratchCompactionEnabled
                ? ScratchLifetimeCompaction.familyPlan(scratchPlan(rows: 1), compact: true).counts
                : scratchPlan(rows: 1)
            let minimum = try counts.values.reduce(UInt64(0)) { total, count in
                let bytes = UInt64(try Self.scratchLength(count))
                let (sum, overflow) = total.addingReportingOverflow(bytes)
                guard !overflow else { throw EngineError("Startup scratch capacity overflow", code: "insufficient_memory") }
                return sum
            }
            currentWorkingBytes.store(minimum, ordering: .relaxed)
        }
        let free = availableMemoryBytes()
        let share = ProcessInfo.processInfo.physicalMemory / 10 * 7
        let lent = lendsState ? positionStateBytes(at: contextLimit) - positionStateBytes(at: stateCapacity) : 0
        poolMaximum = budget ?? expertBudget(within: memoryLimit ?? share) + lent
        lendsPromptWorking = budget == nil
        // A limit without room for two layers' experts per token is refused (insufficient_memory); a busy Mac only starts small.
        let start = try FamilyExperts.startBytes(weights)
        if budget == nil { try requireLimit(memoryLimit ?? share, start: start, floor: 0, margin: ledger?.transientMargin ?? 0) }
        // Capacity is metadata only. A measured working plan can lend unused scratch without rebuilding pools;
        // the active startup limit still charges the full allowance until the first completed-boundary plan.
        experts = try FamilyExperts(weights: weights, budget: poolMaximum + (lendsPromptWorking ? workingBytes : 0))
        guard poolMaximum >= experts.leastBytes else {
            throw EngineError(
                "\(OrigamiPlan.gigabytes(poolMaximum)) is left for experts and they need at least \(OrigamiPlan.gigabytes(experts.leastBytes))",
                code: "insufficient_memory")
        }
        if let ledger {
            // Fixed owners and the rounded physical two-wave cache floor must fit before any slab can be allocated.
            let fixed = UInt64(weights.residentBytes) + workingBytes + heldState
            try requireLimit(ledger.ceiling, start: 0, floor: experts.leastAllocatedBytes, margin: ledger.transientMargin)
            // Startup allocations were pre-admitted individually. Only final owner bookkeeping follows this handoff;
            // no new buffer is allocated before the complete fixed/state reservation set succeeds.
            weights.endStartup()
            var admitted: [MemoryLedger.Reservation] = []
            do {
                let dense = try ledger.reserve(bytes: UInt64(weights.residentBytes), category: "family-dense")
                admitted.append(dense)
                let working = try ledger.reserveAllowance(bytes: workingBytes, category: "family-working")
                admitted.append(working)
                let state = try ledger.reserve(bytes: fixedStateBytes, category: "family-state")
                admitted.append(state)
                let positions = try ledger.reserve(bytes: positionStateBytes(at: stateCapacity), category: "family-state")
                admitted.append(positions)
                denseReservation = dense
                workingReservations = [(working, workingBytes)]
                stateReservation = state
                positionReservation = positions
            } catch {
                for reservation in admitted { try? ledger.release(reservation) }
                throw error
            }
            if ledger.usesAvailableMemoryFloor {
                // Working buffers are not allocated yet: free memory reaches the floor once they are.
                let held = UInt64(weights.residentBytes) + heldState
                ledger.setUsable(held + free > ledger.availableFloorBytes ? held + free - ledger.availableFloorBytes : 0)
            } else if MemoryPressureLevel.current()?.isElevated == true {
                ledger.setUsable(fixed + experts.leastAllocatedBytes)
            } else {
                ledger.setUsable(ledger.maximumUsableBytes)
            }
            self.ledger = ledger
            experts.ledger = ledger
            reconcilePhysicalAdmission()
        }
        experts.admitPrefetchObservations()
        if wantsRouteTrace, let ledger {
            routeTrace = FamilyRouteTrace(
                layers: g.layers, experts: g.experts, ranks: g.expertsPerToken,
                ledger: ledger, retainingBytes: experts.leastAllocatedBytes)
        }
        experts.setLimit(bytes: poolTarget())
        return free
    }

    /// The least memory limit `makeExperts` starts with: the dense weights, working buffers and held state, then for the
    /// experts the larger of their slab-rounded two-wave floor (`floor`, with the ledger's `margin`) and the budget every
    /// pool's share needs (`start`, `FamilyExperts.startBytes`) less the working buffers lent to it.
    func leastLimit(start: UInt64, floor: UInt64, margin: UInt64) -> UInt64 {
        let lent = lendsPromptWorking ? workingBytes : 0
        return UInt64(weights.residentBytes) + workingBytes + heldState + max(floor + margin, start > lent ? start - lent : 0)
    }
    /// The least memory limit this model starts with at its context (`leastLimit`), as `/v1/status` reports it.
    package var minimumLimit: UInt64 {
        leastLimit(start: experts.startBytes, floor: experts.leastAllocatedBytes, margin: ledger?.transientMargin ?? 0)
    }
    /// The refusal when the smallest limit that works (`least`) is more than this Mac allows (`maximum`), or nil.
    package static func beyondThisMac(least: UInt64, maximum: UInt64) -> EngineError? {
        guard least > maximum else { return nil }
        return EngineError(
            "this model does not fit this Mac at this context: the smallest memory limit that works is "
                + "\(OrigamiPlan.gigabytes(least)) and this Mac allows at most \(OrigamiPlan.gigabytes(maximum)); use a smaller context "
                + "or a model that needs less memory",
            code: "insufficient_memory")
    }

    /// Refuses a `limit` under `leastLimit`, saying what is short.
    private func requireLimit(_ limit: UInt64, start: UInt64, floor: UInt64, margin: UInt64) throws {
        let least = leastLimit(start: start, floor: floor, margin: margin)
        guard limit < least else { return }
        if let refusal = Self.beyondThisMac(
            least: least, maximum: WiredMemoryPolicy.maximumBudgetBytes(physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory))
        {
            throw refusal
        }
        // Tenths of a GB, or MB under 1 GB; what is short rounded up, what is left down, so the two never read as equal.
        func size(_ bytes: UInt64, up: Bool = false) -> String {
            let (unit, name) = bytes >= 1_000_000_000 ? (UInt64(100_000_000), "GB") : (1_000_000, "MB")
            let units = (bytes + (up ? unit - 1 : 0)) / unit
            return name == "GB" ? "\(units / 10).\(units % 10) GB" : "\(units) MB"
        }
        let fixed = UInt64(weights.residentBytes) + workingBytes + heldState
        throw EngineError(
            "the model does not fit in a \(size(limit)) limit: after \(size(UInt64(weights.residentBytes))) of dense weights, "
                + "\(size(workingBytes)) of working buffers and \(size(heldState)) of context state, "
                + "\(size(limit > fixed ? limit - fixed : 0)) is left for experts and they need at least \(size(least - fixed, up: true)); "
                + "the smallest limit that works is \(size(least, up: true)), or use a smaller context",
            code: "insufficient_memory")
    }

    /// The memory governor's ledger (`makeExperts`), and the state's reservation in it (`reserveState` moves it).
    package var ledger: MemoryLedger?
    private var denseReservation: MemoryLedger.Reservation?
    private var stateReservation: MemoryLedger.Reservation?
    private var positionReservation: MemoryLedger.Reservation?
    deinit {
        for reservation in [denseReservation, stateReservation, positionReservation].compactMap({ $0 }) {
            try? ledger?.release(reservation)
        }
        for (reservation, _) in workingReservations { try? ledger?.release(reservation) }
    }
    /// The most the expert slots may hold: `expert_memory`, else what the memory limit leaves with the state not held lent
    /// on top (`reserveState` moves it with the state).
    package var poolMaximum: UInt64 = 0
    /// Optional tool-boundary state, charged against the expert pool while it is retained.
    package var toolAnchorBytes: UInt64 = 0
    /// Optional per-model CPU caches remain charged when sizing expert capacity. Subclasses own and retire them.
    package var optionalModelCacheBytes: UInt64 {
        (experts?.observationReservedBytes ?? 0) + (routeTrace == nil ? 0 : FamilyRouteTrace.reservedBytes)
    }
    /// The conversation state held in memory now.
    package var heldState: UInt64 { fixedStateBytes + positionStateBytes(at: stateCapacity) }
    private var fixedStateBytes: UInt64 { stateBytes - positionStateBytes(at: contextLimit) }
    /// What the expert slots may hold: `maximum`, within what the governor's budget leaves once the dense weights,
    /// working buffers and `state` are counted.
    package func poolTarget(maximum: UInt64? = nil, state: UInt64? = nil, working: UInt64? = nil) -> UInt64 {
        let maximum = maximum ?? poolMaximum
        let optional = toolAnchorBytes + optionalModelCacheBytes
        let limit = maximum > optional ? maximum - optional : 0
        guard let ledger else { return limit }
        let others = UInt64(weights.residentBytes) + max(working ?? workingBytes, workingRetainedBytes) + (state ?? heldState) + optional
        let overhead = ledger.physicalOverheadBytes
        let physicalOwners = overhead < ledger.ceiling ? ledger.ceiling - overhead : 0
        let usable = min(ledger.usableBytes, physicalOwners)
        return min(limit, usable > others ? usable - others : 0)
    }
    /// The memory governor's trim (`FamilyServingBackend.trimExperts`, on the governor's thread): the limit fitted to
    /// `poolTarget` now when no request runs (`idle`); else only `trimWanted` is set, as the target reads the state the
    /// request may be reallocating, and the next MoE layer's start or the request's end fits it (`fitIfTrimmed`). The
    /// Cache-owned bytes retired now. A deferred request reports zero until the safe boundary actually retires it.
    package func trimExperts(idle: Bool) -> UInt64 {
        guard idle else {
            trimLock.lock()
            trimWanted = true
            trimLock.unlock()
            return 0
        }
        let resident = experts.residentBytes
        fitExperts()
        return UInt64(max(0, resident - experts.residentBytes))
    }
    private var trimWanted = false
    private var hasPendingTrim: Bool {
        trimLock.lock()
        defer { trimLock.unlock() }
        return trimWanted
    }
    /// Decode reads the next MoE layer's guessed experts while this one runs (`moe`'s `next`), as `FamilyPrefetchPolicy`
    /// admits them. Measured on an M2 Pro with one external SSD and a hot copy: DeepSeek-V4.1 decodes 12-29% faster with
    /// it; GLM-5.3 about 10% faster (a third fewer demand misses) once the policy judged ranks by the reads they start. A
    /// model may set its own default; ORIGAMI_FAMILY_PREFETCH=1 or 0 overrides it, to measure another Mac.
    package class var guessesByDefault: Bool { true }
    /// Each MoE layer commits its router and then its shared expert before the host waits for the router, and decode
    /// commits its routed experts as soon as they are encoded, so the GPU works while the host picks, reads and encodes
    /// (the same commands in the same order: BITWISE). Off: the router's waited flush first, the experts committed with
    /// the next layer's router (checks compare the two).
    nonisolated(unsafe) package static var overlapsRouting = true
    /// Decode encodes and commits its resident experts' gate, up and down while the missing ones are read
    /// (`FamilyExperts.fetch`'s `ready`), the rest once they are in, and the weighted add over all in expert order: the
    /// same bits (each expert's output its own row; `add_weighted_rows_at` adds as `add_weighted_rows` does). Off: all
    /// once read.
    nonisolated(unsafe) package static var runsResidentFirst = true
    /// Qualification switches retain the existing independent projection/expert path as an exact reference.
    nonisolated(unsafe) package static var groupsDecodeExperts = ProcessInfo.processInfo.environment["ORIGAMI_FAMILY_GROUPED_DECODE"] != "0"
    nonisolated(unsafe) package static var groupsGateUp = ProcessInfo.processInfo.environment["ORIGAMI_FAMILY_GROUPED_GATE_UP"] != "0"
    package lazy var guessesNext = ProcessInfo.processInfo.environment["ORIGAMI_FAMILY_PREFETCH"].map { $0 != "0" } ?? Self.guessesByDefault
    private let trimLock = NSLock()
    /// The limit fitted to `poolTarget`: at a request's start, for a trim that arrived while one ran, and once the
    /// governor raised its budget again.
    /// What bounds the expert slots now, in plain words (status says it when the cache shrank).
    private var limitReason: String {
        guard let ledger else { return "the conversation state and working buffers took more of the memory limit" }
        let overhead = ledger.physicalOverheadBytes
        if (overhead < ledger.ceiling ? ledger.ceiling - overhead : 0) < ledger.usableBytes {
            return "memory the process holds outside its own buffers (\(OrigamiPlan.gigabytes(overhead))) leaves less room"
        }
        if ledger.usableBytes < ledger.maximumUsableBytes {
            return "macOS asked for memory back, and the memory governor lowered the budget to \(OrigamiPlan.gigabytes(ledger.usableBytes))"
        }
        return "the conversation state and working buffers took more of the memory limit"
    }
    package func fitExperts() {
        trimLock.lock()
        trimWanted = false
        trimLock.unlock()
        let target = poolTarget()
        if target != experts.budget { experts.setLimit(bytes: target, why: limitReason) } else { experts.rebalance() }
    }
    /// `fitExperts` if a trim arrived while the request ran, or the budget grew back (automatic: by 256 MB or more; a long reply
    /// squeezed early would otherwise keep its few slots to the end): called where no GPU work uses an expert slot and
    /// no fetch runs (a MoE layer after its router's wait, where only its shared expert may still run, before its wave
    /// size is read; the request's end).
    package func fitIfTrimmed() {
        trimLock.lock()
        let wanted = trimWanted
        trimLock.unlock()
        let hysteresis: UInt64 = ledger?.budgetMode == .explicit ? 0 : 256_000_000
        if wanted || poolTarget() > experts.budget + hysteresis { fitExperts() }
    }

    /// Startup/maximum first-slice allowance. Once a forward plan is available only its actual page-rounded scratch
    /// remains reserved; the rest can be lent to the expert cache.
    package static let baseWorkingBytes: UInt64 = 3 << 29
    /// Opt-in startup admission reserves a one-row plan instead of maximum prompt scratch. Each forward still
    /// reserves its complete measured working set before allocation, trimming experts first; phase limits remain.
    package static var lazyStartupWorkspaceRequested: Bool {
        ProcessInfo.processInfo.environment["ORIGAMI_FAMILY_LAZY_STARTUP_WORKSPACE"] == "1"
    }
    package let lazyStartupWorkspace = FamilyRunner.lazyStartupWorkspaceRequested
    private let currentWorkingBytes = Atomic<UInt64>(FamilyRunner.baseWorkingBytes)
    /// The working set's current allowance, also read by the status thread.
    package var workingBytes: UInt64 { currentWorkingBytes.load(ordering: .relaxed) }
    private var prefillWorkingBytes: UInt64 = FamilyRunner.baseWorkingBytes
    /// The working bytes of one prompt slice past the first (`slicePrompts`).
    private var sliceWorkingBytes: UInt64 = 0
    private var workingReservations: [(MemoryLedger.Reservation, UInt64)] = []
    private var lendsPromptWorking = false
    private var decodeWorkingSet = false
    /// Scoped to an explicitly requested generation block, never inferred from its row count.
    private var generationBlock = false
    /// Whether prompt-only workspace has been released for generation (one row or an explicit small block).
    package var isDecodePhase: Bool { decodeWorkingSet }
    /// Checks only: the reserve `slicePrompts` fits slices in, in place of this Mac's.
    nonisolated(unsafe) package static var promptReserve: Double?
    /// Sets `promptSlices` from the bytes a prompt row's working buffers take (`rowBytes`): up to four, no more than
    /// the context holds, and as many past the first as 1/32 of the Mac's memory (at most 2.5 GB) holds, which the
    /// working buffers then count. Before `makeExperts`, so the pool is sized without it.
    func slicePrompts(rowBytes: Int) {
        let reserve = Self.promptReserve ?? min(2.5e9, Double(ProcessInfo.processInfo.physicalMemory) / 32)
        let slice = Self.promptSlice
        promptSlices = max(1, min(4, (contextLimit + slice - 1) / slice, 1 + Int(reserve / Double(slice * rowBytes))))
        sliceWorkingBytes = UInt64(slice * rowBytes)
        prefillWorkingBytes = Self.baseWorkingBytes + UInt64(promptSlices - 1) * sliceWorkingBytes
        currentWorkingBytes.store(prefillWorkingBytes, ordering: .relaxed)
    }

    /// Before a prompt of `rows` new rows: when it has more slices than `promptSlices`, its calls may take as many as the
    /// expert pool can lend the working memory for (each layer's experts are read once a call, and a slice nearly uses
    /// them all): at most half of what the pool could hold with no working memory, never below its two-wave floor. The
    /// ledger admits each call's actual scratch as before (`prepareScratch` trims the pool first), and `beginDecode`
    /// returns it. Only with an automatic pool (`expert_memory` is a fixed cap) and no smaller grouping chosen.
    package func planPrompt(rows: Int) {
        plannedSlices = nil
        prefillWorkingBytes = Self.baseWorkingBytes + UInt64(promptSlices - 1) * sliceWorkingBytes
        let slices = (rows + Self.promptSlice - 1) / Self.promptSlice
        guard slices > prefillSlices, prefillSlices == promptSlices, lendsPromptWorking, sliceWorkingBytes > 0, let experts else { return }
        let room = poolTarget(maximum: poolMaximum + workingBytes, working: 0)
        let floor = experts.leastAllocatedBytes
        let lendable = min(room / 2, room > floor ? room - floor : 0)
        let fits = lendable > Self.baseWorkingBytes ? 1 + Int((lendable - Self.baseWorkingBytes) / sliceWorkingBytes) : 1
        let planned = min(slices, fits)
        guard planned > promptSlices else { return }
        plannedSlices = planned
        prefillWorkingBytes = Self.baseWorkingBytes + UInt64(planned - 1) * sliceWorkingBytes
    }

    /// The next forward plan reclaims its exact scratch requirement before allocation. A complete prompt-cache
    /// hit need not reserve the maximum prompt grouping. Neither phase transition opens an encoder.
    package func beginPrefill() throws {
        experts?.abandonPrefetchObservation()
        decodeWorkingSet = false
    }

    /// Release prompt references only after the preceding GPU work and autorelease pool finish. The first decode
    /// forward admits its exact plan; no fixed 1.5 GiB scratch reservation remains idle throughout a long reply.
    package func beginDecode() {
        guard !decodeWorkingSet else { return }
        autoreleasepool {
            releasePromptReferences()
            scratch.removeAll(keepingCapacity: true)
        }
        decodeWorkingSet = true
        planPrompt(rows: 0)
        // A prompt read through: its buffers may grow back (`commitLimit`).
        commandBudget.recover()
        // Shrinking cannot exceed the ledger's cap or require a new allocation.
        try? adjustWorkingAllowance(to: 0)
        reconcilePhysicalAdmission()
        if experts != nil { fitExperts() }
    }

    /// Phase boundaries have joined their GPU users. Wired dense/cache buffers prove a conservative
    /// resident subset; unwired state, escaped slabs and prospective scratch never justify lowering
    /// an earlier overhead observation. This samples phases, not each layer/token.
    private func reconcilePhysicalAdmission() {
        guard let ledger, ledger.budgetMode == .explicit else { return }
        let resident = UInt64(weights.wiredBytes) + UInt64(experts?.wiredBytes ?? 0)
        ledger.reconcilePhysicalOverhead(residentOwnedBytes: resident)
    }

    /// Optional request-boundary calibration after temporary prompt owners have retired. Uses the same
    /// conservative wired-owner proof as phase transitions; it never assumes all reserved bytes are resident.
    package func refreshIdleOverhead() {
        guard refreshesIdleOverhead, command == nil, encoder == nil, delayedRead == nil else { return }
        reconcilePhysicalAdmission()
        if experts != nil { fitExperts() }
        idleOverheadRefreshes += 1
    }

    /// Called only with no encoder. Planning reclaims space from experts before individual allocation admission;
    /// retained old buffers keep their own ledger charge until the last Metal owner releases them.
    private func adjustWorkingAllowance(to requested: UInt64) throws {
        let wanted = max(requested, workingRetainedBytes)
        let previous = workingBytes
        // Startup reserves the maximum before the first plan exists. Once planned, each actual buffer carries
        // its own reservation (including escaped old buffers); no unallocated allowance is double charged.
        for (reservation, _) in workingReservations { try ledger?.release(reservation) }
        let removedAllowance = !workingReservations.isEmpty
        workingReservations.removeAll()
        if removedAllowance { reconcilePhysicalAdmission() }
        guard wanted != previous else { return }
        let maximum =
            lendsPromptWorking
            ? (wanted > previous
                ? (poolMaximum > wanted - previous ? poolMaximum - (wanted - previous) : 0)
                : poolMaximum + previous - wanted) : poolMaximum
        if wanted > previous {
            experts?.setLimit(
                bytes: poolTarget(maximum: maximum, working: wanted), why: "reading the prompt borrowed working memory from it")
        }
        poolMaximum = maximum
        currentWorkingBytes.store(wanted, ordering: .relaxed)
    }

    /// Model fields which also hold scratch buffers must let them go before the working-set loan is made.
    func releasePromptReferences() {}
    /// What a memory limit leaves the expert slots once the dense weights, the working buffers and the conversation
    /// state (`stateBytes`, the whole context's) are counted.
    func expertBudget(within limit: UInt64) -> UInt64 {
        let held = UInt64(weights.residentBytes) + workingBytes + stateBytes
        return limit > held ? limit - held : 0
    }
    /// The conversation state's allocation for the whole context (without a per-position part on disk), set once the
    /// model has allocated it.
    var stateBytes: UInt64 = 0

    // MARK: the per-position state, lent to the expert slots

    /// The model's conversation state at `position` (`FamilyModel.stateParts`; each model's own).
    package func stateParts(at position: Int) -> [(buffer: MTLBuffer, bytes: Int)] { [] }
    package func stateBytes(at tokens: Int) -> UInt64 { UInt64(stateParts(at: tokens).reduce(0) { $0 + $1.bytes }) }
    /// Resident per-position allocation bytes, including padding rows; file-backed buffers contribute no resident loan.
    package func positionStateBytes(at capacity: Int) -> UInt64 { 0 }
    /// Publish replacement buffers only after every allocation and prefix copy succeeds.
    func allocatePositionState(_ capacity: Int) throws { try allocatePositionState(capacity, grow: regrown) }
    package func allocatePositionState(_ capacity: Int, grow: (MTLBuffer?, Int) throws -> MTLBuffer) throws {}
    /// Held while `reserveState` replaces the per-position buffers and while `stateParts` reads them: `/v1/status`
    /// reads the state's size on its own thread while a request runs.
    let stateLock = NSLock()
    /// A per-position buffer of `count` floats in place of `old`, holding its bytes up to the shorter one's length: every
    /// part `stateParts` names is a prefix of its buffer, so the conversation's rows are copied byte for byte.
    package func regrown(_ old: MTLBuffer?, _ count: Int) throws -> MTLBuffer {
        let buffer = try stateBuffer(count)
        if let old { buffer.contents().copyMemory(from: old.contents(), byteCount: min(old.length, buffer.length)) }
        return buffer
    }
    /// Positions an empty conversation's per-position state holds, and the step it grows in.
    package static let stateBase = 4096
    /// Positions the per-position state buffers hold (the context's unless `lendsState`).
    package private(set) var stateCapacity: Int
    /// Whether the per-position state past `stateCapacity` is lent to the expert slots: kept in memory, with no set
    /// `expert_memory`. Set before the state is first allocated.
    var lendsState = false { didSet { stateCapacity = lendsState ? min(contextLimit, Self.stateBase) : contextLimit } }
    /// The positions a state holding `capacity` takes for `positions` at `position`: an empty conversation's is fitted
    /// (whole `stateBase` steps), a longer one grows to the larger of that and twice its own, so a reply growing past it
    /// copies its rows a few times, not once a token; never past `limit`.
    package static func grownCapacity(for positions: Int, position: Int, holding capacity: Int, limit: Int) -> Int {
        let fit = min(limit, max(1, (positions + stateBase - 1) / stateBase) * stateBase)
        return position == 0 ? fit : positions > capacity ? min(limit, max(fit, 2 * capacity)) : capacity
    }
    /// Makes the per-position state hold `positions` (at most the context), between forward passes. The expert slots
    /// give up the bytes it adds before the buffers take them, and take back what a smaller one frees; the buffers are
    /// reallocated (GPU buffers may be committed whole, so lazy pages would not lend anything) with their rows copied.
    /// At position 0 no row is live: the old buffers go first, so a new conversation never holds both sizes.
    package func reserveState(_ positions: Int) throws { try reserveState(positions, grow: regrown) }
    /// The supplied allocator also lets the fixture exercise a partial replacement failure through real admission;
    /// `footprint` stands in for the measured process footprint in checks.
    package func reserveState(
        _ positions: Int, grow: (MTLBuffer?, Int) throws -> MTLBuffer, footprint: (() -> UInt64)? = nil
    ) throws {
        guard lendsState else { return }
        let capacity = Self.grownCapacity(for: positions, position: position, holding: stateCapacity, limit: contextLimit)
        guard capacity != stateCapacity else { return }
        if position == 0 && stateCapacity > 0 {
            // Kernels read rows below the position only, so nothing here is worth copying or keeping alongside the
            // new buffers. A later denial leaves capacity 0, consistently charged; the next request allocates.
            let held = positionStateBytes(at: stateCapacity)
            let empty = positionStateBytes(at: 0)
            stateLock.lock()
            do { try autoreleasepool { try allocatePositionState(0) { _, count in try stateBuffer(count) } } } catch {
                stateLock.unlock()
                throw error
            }
            stateCapacity = 0
            stateLock.unlock()
            poolMaximum += held - empty
            if let ledger, let old = positionReservation { try? ledger.shrink(old, to: empty) }
        }
        let held = positionStateBytes(at: stateCapacity)
        let wanted = positionStateBytes(at: capacity)
        let maximum = poolMaximum + held > wanted ? poolMaximum + held - wanted : 0
        // With live rows, old and new position buffers coexist until publication: the expert cache lends the complete
        // new allocation, and a denied reservation or failed allocation leaves the old state intact.
        let target = poolTarget(maximum: maximum, state: heldState + wanted)
        let physicalShortfall = ledger?.physicalAllocationShortfall(bytes: wanted, footprint: footprint?()) ?? 0
        let physicalTarget = UInt64(experts.residentBytes) > physicalShortfall ? UInt64(experts.residentBytes) - physicalShortfall : 0
        experts.setLimit(
            bytes: physicalShortfall > 0 ? min(target, physicalTarget) : target, why: "the conversation state grew to \(capacity) positions"
        )
        let replacement: MemoryLedger.Reservation?
        do {
            replacement = try admitState(bytes: wanted, footprint: footprint)
        } catch {
            fitExperts()
            throw error
        }
        stateLock.lock()
        do { try autoreleasepool { try allocatePositionState(capacity, grow: grow) } } catch {
            stateLock.unlock()
            if let replacement { try? ledger?.release(replacement) }
            fitExperts()
            throw error
        }
        stateCapacity = capacity
        stateLock.unlock()
        poolMaximum = maximum
        if let ledger {
            if let old = positionReservation { try? ledger.release(old) }
            positionReservation = replacement
        }
        experts.setLimit(bytes: poolTarget())
    }
    /// The state's physical admission. Pages of slabs or buffers just released can take hundreds of ms to leave the
    /// footprint, so while the charged owners fit, it waits for them (`ReclamationWait`), then once more with the
    /// expert slots at their floor, before it is denied. It probes without reserving until the probe fits: a failed
    /// reservation would count those pages in `physicalOverhead` for good. With live rows the wait ignores
    /// cancellation (at most about 2 s) and a denial leaves the old state whole, so the conversation survives both.
    private func admitState(bytes: UInt64, footprint: (() -> UInt64)?) throws -> MemoryLedger.Reservation? {
        guard let ledger else { return nil }
        func short() -> UInt64 { ledger.physicalAllocationShortfall(bytes: bytes, footprint: footprint?()) }
        let live = position > 0
        for floor in [false, true] {
            if floor { autoreleasepool { experts.setLimit(bytes: 0, why: "the conversation state needed the memory") } }
            if short() > 0, ledger.physicalAllocationShortfall(bytes: bytes, footprint: 0) == 0 {
                try ReclamationWait.wait(shouldContinue: live ? { true } : shouldContinue) { short() == 0 }
            }
            guard short() == 0 else { continue }
            // A logical denial gets the floor's second chance too; the last attempt's error is the request's.
            do {
                if let reservation = try ledger.reservePhysicalIfAvailable(bytes: bytes, category: "family-state", footprint: footprint?())
                {
                    return reservation
                }
            } catch  where !floor {}
        }
        throw EngineError("Physical memory admission denied: \(bytes) bytes for family-state", code: "insufficient_memory")
    }

    // MARK: command buffers

    /// Multiply-adds encoded into the open buffer since it opened (`mm` and the attention kernels; `encoded(_:)`).
    private var pendingWork = 0.0
    /// A buffer is committed once it holds `commitLimit` multiply-adds (`CommandBudget`: about 17 ms of `mm` on an M2 Pro
    /// at the start, fitted to measured GPU time, halved after an abort). At the floor GLM's attention runs a row a
    /// dispatch: a third slower.
    package static let commitWork = CommandBudget.work
    package static let commitSeconds = CommandBudget.seconds
    package static let commitFloor = CommandBudget.floor
    private let commandBudget = CommandBudget()
    /// Settable for checks.
    package var commitLimit: Double {
        get { commandBudget.limit }
        set { commandBudget.limit = newValue }
    }
    /// Counts multiply-adds encoded into the open buffer.
    package func encoded(_ work: Double) { pendingWork += work }
    /// Commits the open buffer without waiting once it holds `commitLimit`. Called only where the code reads `encoder`
    /// afresh (between row chunks and between experts), never inside a step that holds the encoder it was given.
    package func commitIfLong() throws { if pendingWork >= commitLimit { try flush(wait: false) } }
    /// One token row's indexer scoring over `entries` entries and its selection (`blocks` candidate blocks) in `mm`
    /// multiply-adds by time: index_scores runs about 30x slower per multiply-add than `mm`; index_select and
    /// candidate_blocks cost per entry and per block (measured on an M2 Pro: 0.54 s, 11 ms and 2.5 ms for 256 rows at
    /// 21K entries of 32 heads x 128, where 3e10 `mm` multiply-adds take 25 ms). index_scores has since run about 6x faster
    /// (eight heads at a time), so slices are shorter than they need be.
    package static func indexWork(entries: Int, heads: Int, dim: Int, blocks: Int = 0) -> Double {
        32 * Double(entries) * Double(heads * dim) + 2560 * Double(entries) + 4608 * Double(blocks)
    }
    /// One token row's latent attention (`attend_scores`, `softmax_rows`, `attend_values` over `selected` rows of `dim`
    /// for `heads` heads) in `mm` multiply-adds by time: its 2 x heads x selected x dim multiply-adds take about 6x an
    /// `mm` one in the eight-heads kernels.
    package static func attendWork(heads: Int, selected: Int, dim: Int) -> Double { 12 * Double(heads * selected) * Double(dim) }
    /// `body(first, count)` over `rows` rows in slices of at most `commitLimit` (`perRow` each, at least one row),
    /// committing first when a slice would take the open buffer past it, so work that grows with the context (the
    /// indexer's) never lengthens a buffer. `body` reads `encoder` afresh; a row's arithmetic is the same in any slice.
    package func sliced(_ rows: Int, perRow: Double, _ body: (Int, Int) throws -> Void) throws {
        let limit = commitLimit
        let slice = max(1, Int(limit / perRow))
        var done = 0
        while done < rows {
            let n = min(slice, rows - done)
            let work = Double(n) * perRow
            if pendingWork > 0, pendingWork + work > limit { try flush(wait: false) }
            try body(done, n)
            encoded(work)
            done += n
        }
    }

    /// What the engine is encoding ("layer 12 attention"), the label of the buffers opened while it holds, named by a
    /// GPU failure.
    var step = "start" { didSet { if step != oldValue { activityChanged?(step) } } }
    /// Latest activity only, reported on the model thread. This describes encoding/work in progress, not completed
    /// tokens or exclusive GPU phase timing; no barriers, history or extra allocations beyond the existing label.
    package var activityChanged: ((String) -> Void)?
    private func open() {
        // The properties retain the live objects. Their autoreleased references must not retain every completed
        // command (and all its bound buffers) until the entire forward pass has finished.
        // With 64 buffers not completed (Metal's queue limit) this blocks until the GPU ends one: a GPU wait, not host work.
        let started = Date()
        let activity = experts?.driveActivity
        let previous = activity?.enter(.gpu)
        defer { activity?.restore(previous) }
        autoreleasepool {
            command = queue.makeCommandBuffer()!
            command!.label = step
            encoder = command!.makeComputeCommandEncoder()!
        }
        timingLock.lock()
        timingState.gpuWait += Date().timeIntervalSince(started)
        timingLock.unlock()
        profile?.open = nil
        pendingWork = 0
    }
    /// The prefill benchmark's GPU profile (`FamilyGPUProfile`), set between forward passes.
    package var profile: FamilyGPUProfile? {
        didSet {
            let hook: ((MTLComputeCommandEncoder, String) throws -> MTLComputeCommandEncoder)? =
                profile == nil ? nil : { [unowned self] _, kernel in try profiled(kernel) }
            kernels.willDispatch = hook
            matrices.willDispatch = hook
        }
    }
    private var profiledMatrix: (name: String, rows: Int, bytes: Int)?
    private func profiled(_ kernel: String) throws -> MTLComputeCommandEncoder {
        let m = profiledMatrix
        profiledMatrix = nil
        encoder = try profile!.dispatch(
            step: step, kernel: kernel, matrix: m?.name, rows: m?.rows, bytes: m?.bytes ?? 0, current: encoder, command: command!)
        return encoder
    }
    /// Opens the first buffer of a forward pass; `end` commits whatever is left.
    /// The buffers of an earlier pass are not this one's: a pass checks its own before it returns (`forwardPass`).
    package func begin() {
        unchecked = []
        uncheckedWork = 0
        open()
    }
    /// Waits for it too: after a throw mid-layer the buffer still holds real work, and the caller's `reset` writes the
    /// same state on the CPU.
    package func end() {
        encoder.endEncoding()
        command!.commit()
        // Not thrown: a pass that failed has thrown already, and a lost GPU fails the next wait (`GPUWait`).
        try? command!.waitBounded(work: uncheckedWork + pendingWork, budget: commandBudget)
        command = nil
        encoder = nil
        unchecked.removeAll(keepingCapacity: true)
        uncheckedWork = 0
        timingLock.lock()
        timingState.commits += 1
        timingLock.unlock()
    }

    /// Commits what is encoded (waiting for it when asked) and opens the next buffer. A waited flush throws when this
    /// or an earlier unwaited buffer failed on the GPU, so stale router or logit values are never read as results.
    @discardableResult package func flush(wait: Bool) throws -> MTLCommandBuffer {
        encoder.endEncoding()
        let work = pendingWork
        command!.addCompletedHandler { [timingLock] done in
            timingLock.lock()
            self.timingState.largestWork = max(self.timingState.largestWork, work)
            if done.status == .completed { self.commandBudget.fit(work: work, seconds: done.gpuEndTime - done.gpuStartTime) }
            self.timingState.gpu += done.gpuEndTime - done.gpuStartTime
            if done.gpuEndTime - done.gpuStartTime > CommandBudget.seconds { self.timingState.overBound += 1 }
            let kind = (done.label ?? "").split(separator: " ").filter { Int($0) == nil && !$0.contains("+") && $0 != "rows" }.joined(
                separator: " ")
            self.timingState.gpuByCommandLabels[kind, default: 0] += done.gpuEndTime - done.gpuStartTime
            if done.gpuEndTime - done.gpuStartTime > self.timingState.longest {
                self.timingState.longest = done.gpuEndTime - done.gpuStartTime
                self.timingState.longestStep = done.label ?? ""
            }
            timingLock.unlock()
        }
        let committed = command!
        unchecked.append(committed)
        uncheckedWork += work
        committed.commit()
        open()
        timingLock.lock()
        timingState.commits += 1
        timingLock.unlock()
        if wait { try finish(committed) }
        return committed
    }
    /// Waits for a committed buffer (and so for every buffer committed before it); throws as a waited `flush` does.
    func finish(_ committed: MTLCommandBuffer) throws {
        let started = Date()
        let activity = experts?.driveActivity
        let previous = activity?.enter(.gpu)
        defer { activity?.restore(previous) }
        do { try committed.waitBounded(work: uncheckedWork, budget: commandBudget) } catch {
            // Every buffer before it has ended too (stalled), or none will be read again (lost).
            unchecked.removeAll()
            uncheckedWork = 0
            throw error
        }
        timingLock.lock()
        timingState.gpuWait += Date().timeIntervalSince(started)
        // Every buffer committed up to this one has finished (one queue, in order), so their own statuses are final; a
        // completion handler may not have run yet.
        let done = unchecked.firstIndex { $0 === committed }.map { $0 + 1 } ?? 0
        var failure = (unchecked.prefix(done) + [committed]).first { $0.status == .error }.map {
            (step: $0.label ?? "a step", error: $0.error)
        }
        unchecked.removeFirst(done)
        if unchecked.isEmpty { uncheckedWork = 0 }  // else an overcount: a later wait's bound is only more generous
        if let left = interruptWait {
            interruptWait = left > 0 ? left - 1 : nil
            if left == 0 { failure = (committed.label ?? "a step", EngineError.fakeInterruption) }
        }
        if let failure, EngineError.interrupted(failure.error) { commandBudget.interrupted() }
        timingLock.unlock()
        if let failure { throw EngineError.gpu("The GPU failed a family-engine step: \(failure.step)", failure.error) }
    }
    /// Buffers committed since the last waited flush, oldest first: a waited flush fails when any of them failed, so
    /// stale router or logit values (or state written after a stopped buffer) are never taken as results.
    private var unchecked: [MTLCommandBuffer] = []
    /// Multiply-adds committed since the last waited flush, which a wait's bound covers.
    private var uncheckedWork = 0.0
    /// Checks: the waited buffer after this many more fails as macOS's Impacting Interactivity abort would.
    package var interruptWait: Int?

    /// Counter A: the serving thread is in an expert fetch or waits on one while `body` runs (`DriveActivity.Host.reads`).
    private func onReads<T>(_ body: () throws -> T) rethrows -> T {
        let activity = experts?.driveActivity
        let previous = activity?.enter(.reads)
        defer { activity?.restore(previous) }
        return try body()
    }

    /// Runs `body` as one forward pass of `rows` new tokens: times it and advances the position when it succeeds.
    /// Inside its own autorelease pool: the command buffers and encoders it makes hold every bound buffer, and the
    /// serving thread's pool drains only when the request ends.
    func forwardPass<T>(rows: Int, generation: Bool = false, _ body: () throws -> T) throws -> T {
        // Direct model callers also have to reclaim the prompt allowance; serving calls this once before prefill.
        if generation { beginDecode() } else if rows > 1 { try beginPrefill() }
        let previousPurpose = generationBlock
        generationBlock = generation
        defer { generationBlock = previousPurpose }
        if position + rows > stateCapacity { try reserveState(position + rows) }
        return try autoreleasepool {
            let started = Date()
            // Counter A counts decode only: from here to the pass's end.
            if generation { experts?.driveActivity.window(true) }
            defer { if generation { experts?.driveActivity.window(false) } }
            try prepareScratch(scratchPlan(rows: rows))
            begin()
            defer {
                if let read = delayedRead {
                    read.done.wait()
                    delayedRead = nil
                }
                end()
                timingLock.lock()
                timingState.total += Date().timeIntervalSince(started)
                timingState.calls += 1
                timingLock.unlock()
            }
            let result = try body()
            guard delayedRead == nil else { throw EngineError("Delayed family contribution escaped its token", code: "delayed_deadline") }
            position += rows
            return result
        }
    }
    func resetPosition() {
        experts?.abandonPrefetchObservation()
        position = 0
    }
    /// A saved state was written into the buffers: the conversation is `position` tokens long.
    func restorePosition(_ position: Int) { self.position = position }

    // MARK: working buffers and common steps

    private var scratch: [String: MTLBuffer] = [:]
    private let scratchOwnership = FamilyScratchOwnership()
    package private(set) var scratchCompactionEnabled = ScratchLifetimeCompaction.enabledByDefault
    /// Planned saving at the last successful preparation (actual retained owners remain in workingRetainedBytes).
    package private(set) var scratchCompactionSavedBytes: UInt64 = 0
    private var scratchLogicalLengths: [String: Int] = [:]
    /// Fixtures/configuration may select before first allocation. Never invalidate aliases held by a live plan.
    package func configureScratchCompaction(_ enabled: Bool) throws {
        guard command == nil, encoder == nil, scratch.isEmpty, scratchOwnership.bytes == 0 else {
            throw EngineError("Scratch compaction must be selected before allocating working buffers", code: "invalid_configuration")
        }
        scratchCompactionEnabled = enabled
    }
    package var workingRetainedBytes: UInt64 { scratchOwnership.bytes }
    /// For checks/diagnostics at a completed forward boundary; no bookkeeping is added to the decode path.
    package var workingAllocatedBytes: UInt64 { scratch.values.reduce(0) { $0 + UInt64($1.length) } }
    /// Actual forward geometry, not the maximum prompt grouping or the whole context. Models add only the paths
    /// they contain. Index scores grow in small position quanta so ordinary one-row reuse needs no physical probe.
    package func scratchPlan(rows: Int) -> [String: Int] { [:] }
    func scratchPositionLimit(rows: Int) -> Int { min(contextLimit, (position + rows + chunk - 1) / chunk * chunk) }

    /// Common MoE capacities, also used by direct kernel fixtures before they open an encoder. A routed expert's
    /// prompt input is at most one slice; only row IDs/weights span all slices. Decode's batched path uses k rows.
    package func moeScratchPlan(rows: Int, sharedWidth: Int, slice: Int = FamilyRunner.promptSlice) -> [String: Int] {
        let c = min(rows, chunk)
        let r = min(rows, slice)
        let k = g.expertsPerToken
        var plan = [
            "router": rows * g.experts, "ffn-gate": c * sharedWidth, "ffn-up": c * sharedWidth,
            "exp-ids": rows * k, "exp-ws": rows * k, "exp-add": 3 * rows * k + 4 * g.experts * (rows + slice - 1) / slice,
            "exp-x": r * hidden,
            "exp-gate": r * g.expertWidth, "exp-up": r * g.expertWidth,
            "exp-out": max(r, rows == 1 ? k : 0) * hidden,
        ]
        if rows == 1 {
            plan["exp-gu"] = 2 * k * g.expertWidth
            if guessesNext { plan["router-next"] = g.experts }
        }
        return plan
    }

    /// Page-rounded bytes, including tiny allocations. Reject arithmetic overflow before allocating anything.
    package static func scratchLength(_ count: Int) throws -> Int {
        let page = Int(getpagesize())
        guard count >= 0, count <= (Int.max - page + 1) / 4 else {
            throw EngineError("Invalid family scratch capacity \(count)", code: "insufficient_memory")
        }
        return (max(16, count * 4) + page - 1) / page * page
    }

    /// All growth happens at a completed forward boundary. Old and new allocations both count while they coexist.
    /// When coexistence alone would exceed the working owner, dead scratch is retired before allocating its
    /// replacement. Conversation state is separate; a failed allocation can retry preparation without restoring it.
    /// No extra reservation counts the already charged working owner twice.
    /// `allowance` may only tighten that bound; the supplied footprint makes allocation checks deterministic.
    package func prepareScratch(_ counts: [String: Int], allowance: UInt64? = nil, footprint: (() -> UInt64)? = nil) throws {
        guard command == nil, encoder == nil else {
            throw EngineError("Family scratch must be prepared before opening a GPU encoder", code: "scratch_plan_mismatch")
        }
        let plan =
            try scratchCompactionEnabled
            ? ScratchLifetimeCompaction.familyPlan(counts, compact: true)
            : ScratchLifetimeCompaction.FamilyPlan(counts: counts, logicalLengths: [:], savedBytes: 0)
        var growth: [(name: String, length: Int)] = []
        var pending: UInt64 = 0
        var superseded: UInt64 = 0
        for (name, count) in plan.counts {
            let length = try Self.scratchLength(count)
            if let current = scratch[name], current.length >= length { continue }
            let (next, overflow) = pending.addingReportingOverflow(UInt64(length))
            guard !overflow else { throw EngineError("Family scratch capacity overflow", code: "insufficient_memory") }
            pending = next
            growth.append((name, length))
            superseded += UInt64(scratch[name]?.length ?? 0)
        }
        let phaseLimit = decodeWorkingSet ? Self.baseWorkingBytes : prefillWorkingBytes
        let limit = min(phaseLimit, allowance ?? phaseLimit)
        let live = workingAllocatedBytes
        let kept = live - superseded
        guard kept <= limit, pending <= limit - kept else {
            throw EngineError(
                "Family scratch needs \(live) live + \(pending) replacement bytes; working allowance is \(limit)",
                code: "insufficient_memory")
        }
        guard !growth.isEmpty else {
            try adjustWorkingAllowance(to: live)
            scratchLogicalLengths = plan.logicalLengths
            scratchCompactionSavedBytes = plan.savedBytes
            return
        }
        var completed = false
        defer {
            if !completed {
                // A denied/partial allocation never owns its prospective plan.
                // The inner autorelease pool has drained before this rollback;
                // escaped old buffers still count through workingRetainedBytes.
                try? adjustWorkingAllowance(to: workingAllocatedBytes)
                if experts != nil { fitExperts() }
            }
        }
        if live > limit || pending > limit - live {
            // The preceding forward joined its GPU work before this boundary. Nothing reads old scratch now;
            // retire only replaced capacities, so increasing rows/context never requires a double-sized owner.
            autoreleasepool {
                releasePromptReferences()
                for (name, _) in growth { scratch.removeValue(forKey: name) }
            }
        }
        try autoreleasepool {
            // Include escaped old buffers in the coexistence peak. They retain their allocation reservations.
            let retained = workingRetainedBytes
            guard retained <= limit, pending <= limit - retained else {
                throw EngineError("Retained family scratch exceeds the phase working allowance", code: "insufficient_memory")
            }
            let cacheBeforeTrim = ledger?.cacheReliefSnapshot.cacheOwnedBytes ?? 0
            try autoreleasepool {
                try adjustWorkingAllowance(to: retained + pending)
                if let experts {
                    experts.setLimit(bytes: min(experts.budget, poolTarget()), why: "reading the prompt needed working buffers")
                }
            }
            if let ledger {
                // The temporary governor target sizes optional cache, not mandatory request scratch. Count ALL
                // charged owners (history, delayed storage and escaped buffers too), plus the complete replacement
                // allocation. Model geometry alone can leave a small overrun after a pressure/recovery transition.
                let reserved = ledger.reservedBytes
                let usable = ledger.usableBytes
                let (projected, overflow) = reserved.addingReportingOverflow(pending)
                let temporaryShortfall = overflow ? UInt64.max : projected > usable ? projected - usable : 0
                func physicalProbe() -> (shortfall: UInt64, footprint: UInt64) {
                    let measured = footprint?() ?? origami_physical_footprint()
                    return (ledger.physicalAllocationShortfall(bytes: pending, footprint: measured), measured)
                }
                let missing = max(temporaryShortfall, physicalProbe().shortfall)
                if missing > 0, let experts {
                    // setLimit joins reads and preserves the two-wave streaming floor. Drain temporary Metal
                    // references before inspecting actual final-owner retirement; a real escaped view stays charged.
                    autoreleasepool {
                        let resident = UInt64(experts.residentBytes)
                        experts.setLimit(
                            bytes: min(experts.budget, resident > missing ? resident - missing : 0),
                            why: "reading the prompt needed working buffers")
                    }
                }
                let maximum = ledger.maximumUsableBytes
                var after = ledger.reservedBytes
                var physical = physicalProbe()
                let cacheAfterTrim = ledger.cacheReliefSnapshot.cacheOwnedBytes
                let retired = cacheBeforeTrim > cacheAfterTrim ? cacheBeforeTrim - cacheAfterTrim : 0
                if physical.shortfall > 0, retired > 0, after <= maximum, pending <= maximum - after,
                    ledger.physicalAllocationShortfall(bytes: pending, footprint: 0) == 0
                {
                    // Metal object retirement can precede the GPU driver's VM reclamation by hundreds of ms.
                    // Only a proved cache-owner release with a fitting charged envelope may wait. Do not discount
                    // the reported footprint or allocate against predicted reclamation. Ordinary admission never
                    // sleeps. A persistent excess still fails after at most 1.023 s of backoff at this boundary;
                    // cancellation is checked on both sides of each wait, before any allocation, unless the conversation
                    // has rows: then the wait finishes (as in `admitState`), so a cancel never resets it mid-token.
                    try ReclamationWait.wait(shouldContinue: position > 0 ? { true } : shouldContinue) {
                        physical = physicalProbe()
                        return physical.shortfall == 0
                    }
                    after = ledger.reservedBytes
                }
                guard after <= maximum, pending <= maximum - after,
                    physical.shortfall == 0
                else {
                    throw EngineError(
                        "Family scratch replacements would exceed memory_limit (owned \(after), pending \(pending), maximum \(maximum), footprint \(physical.footprint), physical shortfall \(physical.shortfall), expert bytes retired \(retired))",
                        code: "insufficient_memory")
                }
                // A temporary target below fixed owners/the streaming floor cannot be met by more cache eviction.
                // Essential work may proceed inside the immutable and physical limits; each allocation is still
                // admitted atomically below. Never enlarge the saved target or remove a retained owner's charge.
            }
            var replacements: [String: MTLBuffer] = [:]
            for (name, length) in growth {
                let reservation = try ledger?.reservePhysical(bytes: UInt64(length), category: "family-working", footprint: footprint?())
                guard let buffer = weights.device.makeBuffer(length: length, options: .storageModeShared) else {
                    if let reservation { try? ledger?.release(reservation) }
                    throw EngineError("Out of memory for working buffer \(name)", code: "insufficient_memory")
                }
                scratchOwnership.attach(buffer, ledger: ledger, reservation: reservation)
                replacements[name] = buffer
            }
            // Retire any old buffers still coexisting only after every allocation succeeded. These fields contain
            // scratch references, never conversation state; the next forward assigns them before using them.
            releasePromptReferences()
            for (name, buffer) in replacements { scratch[name] = buffer }
        }
        try adjustWorkingAllowance(to: workingAllocatedBytes)
        scratchLogicalLengths = plan.logicalLengths
        scratchCompactionSavedBytes = plan.savedBytes
        completed = true
    }

    /// A pre-admitted f32 buffer. Never flush or trim here: callers may hold this encoder or live expert slots.
    package func work(_ name: String, _ count: Int) throws -> MTLBuffer {
        guard count >= 0, count <= Int.max / 4 else {
            throw EngineError("Invalid family scratch request \(name): \(count)", code: "scratch_plan_mismatch")
        }
        let storage = scratchCompactionEnabled ? ScratchLifetimeCompaction.familyStorageName(name) : name
        // An alias's larger sibling cannot silently authorize an unplanned role or logical capacity.
        let logicalFits = !scratchCompactionEnabled || (scratchLogicalLengths[name].map { $0 >= count * 4 } ?? false)
        if logicalFits, let b = scratch[storage], b.length >= count * 4 { return b }
        throw EngineError(
            "Unplanned family scratch growth for \(name) (\(count) floats); prepare it before GPU encoding", code: "scratch_plan_mismatch")
    }
    /// Where the per-position conversation state lives when kept on disk (a context too large for memory here): a file
    /// per buffer, mapped and unlinked at once, which macOS writes out and reads back like any file. nil: in memory.
    package internal(set) var contextFolder: URL?
    /// A zeroed f32 buffer of per-position state: in memory, or in `contextFolder`. The same bytes either way, so replies
    /// are unchanged (BITWISE); only where the pages live differs.
    func stateBuffer(_ count: Int) throws -> MTLBuffer {
        guard let folder = contextFolder else { return try zeros(count) }
        return try Self.mappedBuffer(weights.device, count: count, folder: folder)
    }
    /// `count` zeroed floats in a file in `folder`, mapped shared and unlinked at once (so nothing is left when the
    /// process ends), as a GPU buffer.
    package static func mappedBuffer(_ device: MTLDevice, count: Int, folder: URL) throws -> MTLBuffer {
        let page = Int(getpagesize())
        let length = (max(16, count * 4) + page - 1) / page * page
        let url = folder.appendingPathComponent("state-\(getpid())-\(UUID().uuidString)")
        let fd = Darwin.open(url.path, O_RDWR | O_CREAT | O_EXCL, 0o600)
        guard fd >= 0 else { throw EngineError("Cannot create the context file in \(folder.path)", code: "storage_unavailable") }
        defer {
            close(fd)
            unlink(url.path)
        }
        guard ftruncate(fd, off_t(length)) == 0, let base = mmap(nil, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0), base != MAP_FAILED
        else {
            throw EngineError("No room for the context file in \(folder.path)", code: "storage_unavailable")
        }
        guard
            let buffer = device.makeBuffer(bytesNoCopy: base, length: length, options: .storageModeShared, deallocator: { munmap($0, $1) })
        else {
            munmap(base, length)
            throw EngineError("The GPU cannot use the context file in \(folder.path)", code: "storage_unavailable")
        }
        return buffer
    }

    /// A zeroed f32 buffer that lives with the conversation (not reused).
    func zeros(_ count: Int) throws -> MTLBuffer {
        try weights.startupAdmission?.reserve(bytes: UInt64(max(16, count * 4)))
        guard let buffer = weights.device.makeBuffer(length: max(16, count * 4), options: .storageModeShared) else {
            throw EngineError("Out of memory for the conversation state", code: "insufficient_memory")
        }
        memset(buffer.contents(), 0, buffer.length)
        return buffer
    }

    package func mm(_ e: MTLComputeCommandEncoder, _ m: FamilyWeights.Matrix, _ input: Ref, _ out: Ref, rows: Int) throws {
        encoded(Double(rows) * Double(m.width) * Double(m.outputs))
        if profile != nil {
            profiledMatrix = (m.name, rows, m.outputs * (m.width / GGUFQuant.sizes[m.type]!.0 * GGUFQuant.sizes[m.type]!.1))
        }
        try matrices.encode(
            e, type: m.type, weights: m.buffer, weightOffset: m.offset, width: m.width, outputs: m.outputs,
            input: input.buffer, inputOffset: input.offset, rows: rows, output: out.buffer, outputOffset: out.offset)
    }
    /// `mm` of each matrix z (one type and shape) from `input + z * inputStride` into `output + z * outputStride`: one
    /// dispatch for up to `GGMLMatrixKernels.batchCapacity` of them where the batch kernels apply (up to 8 rows, a
    /// quantized type), else one each; the same bits either way.
    package func mmBatch(
        _ e: MTLComputeCommandEncoder, _ ms: [FamilyWeights.Matrix], _ input: Ref, inputStride: Int, _ out: Ref, outputStride: Int,
        rows: Int
    ) throws {
        guard let m = ms.first else { return }
        guard rows <= 8, matrices.batches(type: m.type),
            ms.allSatisfy({ $0.type == m.type && $0.width == m.width && $0.outputs == m.outputs })
        else {
            for (z, matrix) in ms.enumerated() {
                try mm(
                    e, matrix, Ref(input.buffer, input.offset + z * inputStride), Ref(out.buffer, out.offset + z * outputStride), rows: rows
                )
            }
            return
        }
        var first = 0
        while first < ms.count {
            let part = ms[first..<min(ms.count, first + GGMLMatrixKernels.batchCapacity)]
            encoded(Double(part.count) * Double(rows) * Double(m.width) * Double(m.outputs))
            if profile != nil {
                profiledMatrix = (
                    m.name, rows, part.count * m.outputs * (m.width / GGUFQuant.sizes[m.type]!.0 * GGUFQuant.sizes[m.type]!.1)
                )
            }
            try matrices.encodeBatch(
                e, type: m.type, matrices: part.map { ($0.buffer, $0.offset) }, width: m.width, outputs: m.outputs,
                input: input.buffer, inputOffset: input.offset + first * inputStride, inputStride: inputStride, rows: rows,
                output: out.buffer, outputOffset: out.offset + first * outputStride, outputStride: outputStride)
            first += part.count
        }
    }
    /// Rows `first ..< first + count` of a matrix as a matrix of their own (one head's slice of a stacked tensor).
    func rowsOf(_ m: FamilyWeights.Matrix, first: Int, count: Int, width: Int? = nil) -> FamilyWeights.Matrix {
        let w = width ?? m.width
        let (block, bytes) = GGUFQuant.sizes[m.type]!
        return FamilyWeights.Matrix(
            buffer: m.buffer, offset: m.offset + first * (w / block * bytes), type: m.type, width: w, outputs: count, name: m.name)
    }
    func grid2(_ n: Int, _ rows: Int) -> (MTLSize, MTLSize) {
        (MTLSize(width: (n + 255) / 256, height: rows, depth: 1), MTLSize(width: 256, height: 1, depth: 1))
    }
    func rms(_ e: MTLComputeCommandEncoder, _ x: Ref, _ w: MTLBuffer?, _ y: Ref, n: Int, rows: Int, eps: Float) throws {
        try kernels.dispatch(
            e, "rms_norm", [x, Ref(w ?? x.buffer), y], ints: [Int32(n), w == nil ? 0 : 1], floats: [eps],
            grid: MTLSize(width: rows, height: 1, depth: 1),
            group: MTLSize(width: min(1024, max(32, (n + 31) / 32 * 32)), height: 1, depth: 1))
    }
    func permute(_ e: MTLComputeCommandEncoder, _ x: Ref, _ y: Ref, a: Int, b: Int, n: Int) throws {
        try kernels.dispatch(
            e, "permute3", [x, y], ints: [Int32(a), Int32(b), Int32(n)],
            grid: MTLSize(width: (n + 63) / 64, height: b, depth: a), group: MTLSize(width: 64, height: 1, depth: 1))
    }
    /// On `encoder`, committed between its matrices when long (a chunk of a wide dense FFN is more than a buffer should hold).
    func swigluFFN(_ f: FFN, _ input: Ref, _ out: Ref, rows: Int, limit: Float) throws {
        let width = f.gate.outputs
        let gate = try work("ffn-gate", rows * width)
        let up = try work("ffn-up", rows * width)
        try mm(encoder, f.gate, input, Ref(gate), rows: rows)
        try commitIfLong()
        try mm(encoder, f.up, input, Ref(up), rows: rows)
        let (grid, group) = grid2(width, rows)
        try kernels.dispatch(
            encoder, "swiglu", [Ref(gate), Ref(up), Ref(gate)],
            ints: [Int32(width), Int32(rows), Int32(width), Int32(width), Int32(width)], floats: [limit], grid: grid, group: group)
        try commitIfLong()
        try mm(encoder, f.down, Ref(gate), out, rows: rows)
    }
    /// `body` over `n` rows in chunks of `chunk`; a chunk's closure reads `encoder` afresh, so a long buffer is
    /// committed between chunks.
    func chunked(_ n: Int, _ body: (Int, Int) throws -> Void) throws {
        var start = 0
        while start < n {
            let rows = min(chunk, n - start)
            try body(start, rows)
            start += rows
            try commitIfLong()
        }
    }

    /// The routed experts of layer `l` for all rows: the router on the GPU, then the shared expert into `output` (on the
    /// GPU while the host picks the routed experts and reads the missing ones), top-k on the host (`probability` maps a
    /// router logit to its weight before the bias), then each used expert over its rows, records read as needed in waves
    /// of half the slot pool, each wave read while the one before runs. Decode with `next` (the next MoE layer's router)
    /// also runs that router on `next.input` (this layer's input when nil), at the same wait, and starts reading the
    /// experts it picks once this layer's are in (`FamilyExperts.prefetch`): the models pass the next layer's own mix and
    /// norm of these streams, close to the input it will route, and those reads then overlap this layer's GPU work. A prompt call of `readAheadRows` or more with `next` reads the next layer's first
    /// wave (experts 0 ..< wave, nearly always all used; an unused one is only a read) once only its own last wave can still run (`FamilyExperts.readAhead`),
    /// so the drive works through that wave, the next attention and shared expert. A prompt of several slices reads each expert once for all of them, but runs it
    /// once per `slice` rows its rows fall in, so every call keeps the rows (and so the kernels and bits) one slice's
    /// pass gave it. `keepRows` (a prompt's last layer, which only the head's last row reads on): only the experts these
    /// rows pick are read and run, each over all its rows as before, so the kept rows get the same bits and the others
    /// an unspecified sum.
    package func moe(
        layer l: Int, router: FamilyWeights.Matrix, bias: [Float], shared: FFN, input: MTLBuffer, output: MTLBuffer, rows n: Int,
        probability: @escaping @Sendable (Float) -> Float, normalize: Bool, scale: Float, sum: @Sendable (Float) -> Float, limit: Float,
        next: (layer: Int, router: FamilyWeights.Matrix, bias: [Float], input: MTLBuffer?)? = nil, slice: Int = FamilyRunner.promptSlice,
        keepRows: Range<Int>? = nil, delayedPost: MTLBuffer? = nil, mayDefer: Bool = false
    ) throws {
        let e = g.experts
        let (logits, guess) = try routeAndShared(
            router: router, shared: shared, input: input, output: output, rows: n, limit: limit, next: next)
        if n == 1 { experts.routerResultKnown() }
        let p = logits.contents().bindMemory(to: Float.self, capacity: n * e)
        var rowsOf = [[Int32]](repeating: [], count: e)
        var weightsOf = [[Float]](repeating: [], count: e)
        var probs = [Float](repeating: 0, count: e)
        let dial = routing
        let resident = FamilyRouting.needsResidency(dial) ? experts.resident(layer: l) : nil
        let picks = pickExperts(
            logits: p, rows: n, bias: bias, probability: probability, normalize: normalize, scale: scale, sum: sum, dial: dial,
            resident: resident)
        for (r, row) in picks.enumerated() {
            for (expert, weight) in row {
                rowsOf[expert].append(Int32(r))
                weightsOf[expert].append(weight)
            }
        }
        let used = (0..<e).filter { x in keepRows.map { k in rowsOf[x].contains { k.contains(Int($0)) } } ?? !rowsOf[x].isEmpty }
        if n == 1 { routeTrace?.record(position: position, layer: l, needed: used) }
        observeRouting(
            layer: l, picks: picks, logits: p, used: used, bias: bias, probability: probability, input: input, rows: n,
            guess: next.flatMap { next in guess.map { ($0, next.bias) } })
        // Verification still computes the same row/expert order. Its unique record demands use generation
        // admission rather than the prompt scan policy, even when several token rows share a record.
        let decodeAccess = n == 1 || generationBlock
        if decodeAccess { experts.observeRoutedDemand(layer: l, needed: used) }
        var predictionSent = false
        func prefetchNext() {
            guard !predictionSent, let next, let guess else { return }
            predictionSent = true
            let q = guess.contents().bindMemory(to: Float.self, capacity: e)
            for i in 0..<e { probs[i] = probability(q[i]) }
            experts.prefetch(
                layer: next.layer,
                experts: FamilyRouting.select(
                    probs: probs, bias: next.bias, k: g.expertsPerToken, dial: dial, resident: nil,
                    normalize: false, scale: 1, sum: { $0 }
                ).map(\.expert))
        }
        guard !used.isEmpty else { return }
        if n == 1, benchReadGap > 0 {
            // Woken by another thread as a read's completion is (a timed sleep can overshoot many times over here), or
            // with `benchGapSpins` spun through as `FamilyExperts.spins` spins for an awaited read.
            let until = DispatchTime.now() + benchReadGap
            if benchGapSpins {
                while DispatchTime.now() < until {}
            } else {
                let done = DispatchSemaphore(value: 0)
                DispatchQueue.global(qos: .userInitiated).async {
                    while DispatchTime.now() < until {}
                    done.signal()
                }
                done.wait()
            }
        }
        // The logical cache target can exceed its allocated slabs while growth is inhibited.
        // Admit both overlapping waves before any slot consumer starts; a later pressure signal
        // cannot invalidate already-owned capacity, and a joined trim replans below.
        let wave = try experts.prepareWave(layer: l, maximumExperts: used.count)
        // Decode: every used expert in one wave, then four dispatches for all of them (seven when the resident ones run first).
        if Self.groupsDecodeExperts, n == 1, used.count <= wave, 2 * used.count <= GGMLMatrixKernels.batchCapacity,
            let types = experts.bank(l)?.types,
            types.allSatisfy({ matrices.batches(type: $0) })
        {
            try runGroupedDecodeExperts(
                layer: l, used: used, weightsOf: weightsOf, input: input, output: output, limit: limit, dial: dial,
                delayedPost: delayedPost, mayDefer: mayDefer, prefetchNext: prefetchNext)
            return
        }
        try runExpertWaves(
            layer: l, used: used, wave: wave, rowsOf: rowsOf, weightsOf: weightsOf, rows: n, input: input, output: output,
            limit: limit, next: next, slice: slice, decodeAccess: decodeAccess, prefetchNext: prefetchNext)
    }

    /// `moe`'s router (and the next layer's guess) and its shared expert, committed in the order `moe` describes.
    private func routeAndShared(
        router: FamilyWeights.Matrix, shared: FFN, input: MTLBuffer, output: MTLBuffer, rows n: Int, limit: Float,
        next: (layer: Int, router: FamilyWeights.Matrix, bias: [Float], input: MTLBuffer?)?
    ) throws -> (logits: MTLBuffer, guess: MTLBuffer?) {
        let h = hidden
        let e = g.experts
        let logits = try work("router", n * e)
        let guess = n == 1 && next != nil && guessesNext ? try work("router-next", e) : nil
        try chunked(n) { first, rows in try mm(encoder, router, Ref(input, first * h * 4), Ref(logits, first * e * 4), rows: rows) }
        if let next, let guess { try mm(encoder, next.router, Ref(next.input ?? input), Ref(guess), rows: 1) }
        if Self.overlapsRouting {
            // The shared expert is committed behind the router before the host waits: the GPU runs it while the host wakes,
            // picks the routed experts and reads the missing ones. The same commands in the same order.
            let routed = try flush(wait: false)
            try chunked(n) { first, rows in
                try swigluFFN(shared, Ref(input, first * h * 4), Ref(output, first * h * 4), rows: rows, limit: limit)
            }
            try flush(wait: false)
            try finish(routed)
            // Only the shared expert may still run (no expert slot) and no fetch runs: a trim the governor asked for applies
            // here, before the wave size.
            fitIfTrimmed()
        } else {
            try flush(wait: true)
            fitIfTrimmed()
            try chunked(n) { first, rows in
                try swigluFFN(shared, Ref(input, first * h * 4), Ref(output, first * h * 4), rows: rows, limit: limit)
            }
            try flush(wait: false)
        }
        return (logits, guess)
    }

    /// `moe`'s routed experts of each row from the router's `logits`.
    private func pickExperts(
        logits p: UnsafeMutablePointer<Float>, rows n: Int, bias: [Float], probability: @escaping @Sendable (Float) -> Float,
        normalize: Bool, scale: Float, sum: @Sendable (Float) -> Float, dial: QualityDial, resident: Set<Int>?
    ) -> [[(expert: Int, weight: Float)]] {
        let e = g.experts
        // A prompt's rows split over the performance cores, each row's choice its own; gathered below in row order.
        var picks = [[(expert: Int, weight: Float)]](repeating: [], count: n)
        let lanes = max(1, min(Self.performanceCores, n / 64))
        let k = g.expertsPerToken
        nonisolated(unsafe) let logit = p
        picks.withUnsafeMutableBufferPointer { buffer in
            nonisolated(unsafe) let rows = buffer  // each lane writes only its own rows
            DispatchQueue.concurrentPerform(iterations: lanes) { lane in
                var probs = [Float](repeating: 0, count: e)
                for r in lane * n / lanes..<(lane + 1) * n / lanes {
                    for i in 0..<e { probs[i] = probability(logit[r * e + i]) }
                    rows[r] = FamilyRouting.select(
                        probs: probs, bias: bias, k: k, dial: dial, resident: resident, normalize: normalize, scale: scale, sum: sum)
                }
            }
        }
        return picks
    }

    /// `moe`'s opt-in decode observations (route trace, activations).
    private func observeRouting(
        layer l: Int, picks: [[(expert: Int, weight: Float)]], logits p: UnsafeMutablePointer<Float>, used: [Int], bias: [Float],
        probability: @escaping @Sendable (Float) -> Float, input: MTLBuffer, rows n: Int, guess: (MTLBuffer, bias: [Float])?
    ) {
        let e = g.experts
        if n == 1, let routeTrace, routeTrace.capturesDecode {
            // Opt-in O52 diagnostic: how close the best resident unchosen expert scores to each chosen one not in memory.
            let loaded = experts.observedReadyExperts(layer: l)
            let chosen = picks[0]
            let taken = Set(chosen.map(\.expert))
            let score = (0..<e).map { probability(p[$0]) + (bias.isEmpty ? 0 : bias[$0]) }
            let alternative = (0..<e).filter { loaded.contains($0) && !taken.contains($0) }.map { score[$0] }.max()
            let mass = chosen.reduce(Float(0)) { $0 + $1.weight }
            for (rank, pick) in chosen.enumerated() where !loaded.contains(pick.expert) {
                routeTrace.recordMissing(
                    score: score[pick.expert], alternative: alternative, share: mass > 0 ? pick.weight / mass : 0, rank: rank)
            }
            // The next layer's guess ranked as `FamilyRouting.select` ranks (its first k are the experts it guesses).
            var ranked: [(expert: Int, score: Float)] = []
            if let (buffer, nextBias) = guess {
                let q = buffer.contents().bindMemory(to: Float.self, capacity: e)
                let s = (0..<e).map { probability(q[$0]) + (nextBias.isEmpty ? 0 : nextBias[$0]) }
                ranked = (0..<e).sorted { s[$0] != s[$1] ? s[$0] > s[$1] : $0 < $1 }.prefix(3 * g.expertsPerToken).map { ($0, s[$0]) }
            }
            routeTrace.recordDetail(position: position, layer: l, chosen: chosen, guess: ranked, resident: loaded)
        }
    }

    /// `moe`'s decode path: every used expert in one wave (or one deferred, `mayDefer`).
    private func runGroupedDecodeExperts(
        layer l: Int, used: [Int], weightsOf: [[Float]], input: MTLBuffer, output: MTLBuffer, limit: Float, dial: QualityDial,
        delayedPost: MTLBuffer?, mayDefer: Bool, prefetchNext: () -> Void
    ) throws {
        let h = hidden
        step = "layer \(l) experts"
        let readStarted = Date()
        let w = used.map { weightsOf[$0][0] }
        if mayDefer, let policy = dial.delayedExperts, let storage = delayedStorage, let delayedPost, let read = delayedReadSlot,
            delayedRead == nil, ledger?.optionalGrowthAllowed != false
        {
            let resident = experts.resident(layer: l)
            let late = Set(used.indices.filter { !resident.contains(used[$0]) })
            if let rank = policy.rank(weights: w, late: late, capacity: experts.allocatedSlots(layer: l) - used.count) {
                let immediate = used.indices.filter { $0 != rank }
                let outstandingBefore = experts.counters.earlyOutstandingBatches
                var issuedEarly = false
                let submit: (() -> Void)? =
                    earlyDelayedReads
                    ? { [self] in
                        // fetch has already assigned and protected every immediate slot. Its submission
                        // callback permits only existing storage, and skips settling unrelated readers.
                        // No further cache activity occurs before the delayed fetch below joins this record.
                        let before = experts.counters.readAheads
                        experts.readAhead(layer: l, experts: [used[rank]])
                        issuedEarly = experts.counters.readAheads > before
                    } : nil
                let slots = try onReads {
                    try experts.fetch(layer: l, experts: immediate.map { used[$0] }, decode: true, submitted: submit)
                }
                if issuedEarly {
                    delayedExpertCounters.earlyReadSubmissions += 1
                    if experts.counters.earlyOutstandingBatches > outstandingBefore {
                        delayedExpertCounters.earlySubmissionsWithDemand += 1
                    }
                }
                try runExperts(encoder, slots: slots, input: input, output: output, weights: immediate.map { w[$0] }, limit: limit)
                // Router completion already joined the source activation/post. Copy before phase scratch aliases it.
                storage.input.contents().copyMemory(from: input.contents(), byteCount: h * 4)
                storage.post.contents().copyMemory(from: delayedPost.contents(), byteCount: 16)
                _ = try flush(wait: false)
                read.prepare(source: l, gate: w[rank], limit: limit)
                let pool = experts!
                let expert = used[rank]
                delayedRead = read
                DispatchQueue.global(qos: .userInitiated).async {
                    do { read.slots = try pool.fetch(layer: l, experts: [expert], decode: true, keepingLast: true) } catch {
                        read.error = error
                    }
                    read.done.signal()
                }
                // Once selected, the policy always uses its declared next-attention boundary.
                // A quickly completed read must not silently choose different source-layer arithmetic.
                delayedExpertCounters.deferredExperts += 1
                delayedExpertCounters.deferredMass += Double(w[rank] / w.reduce(0, +))
                // No guessing/next cache activity until injectDelayedExperts joins this read and GPU consumer.
                return
            }
        }
        // The resident experts run while the others are read (`runsResidentFirst`), in rows of their own.
        var first: [Int] = []
        let slots = try onReads {
            try experts.fetch(
                layer: l, experts: used, decode: true,
                ready: Self.runsResidentFirst
                    ? { [self] ready, slots in
                        try expertOutputs(encoder, slots: slots, input: input, first: first.count, of: used.count, limit: limit)
                        first += ready
                        try flush(wait: false)
                    } : nil)
        }
        timingLock.lock()
        timingState.reads += Date().timeIntervalSince(readStarted)
        timingLock.unlock()
        prefetchNext()
        if first.isEmpty {
            try runExperts(encoder, slots: slots, input: input, output: output, weights: w, limit: limit)
        } else {
            let rest = used.indices.filter { !first.contains($0) }
            let order = first + rest
            if !rest.isEmpty {
                try expertOutputs(
                    encoder, slots: rest.map { slots[$0] }, input: input, first: first.count, of: used.count, limit: limit)
            }
            var row = [Int32](repeating: 0, count: 64)
            for (r, z) in order.enumerated() { row[z] = Int32(r) }
            let (grid, group) = grid2(hidden, 1)
            try kernels.dispatch(
                encoder, "add_weighted_rows_at", [Ref(try work("exp-out", used.count * hidden)), Ref(output)],
                ints: [Int32(hidden), Int32(used.count)] + row, floats: w + Array(repeating: 0, count: 64 - used.count), grid: grid,
                group: group)
        }
        // Committed at once: the GPU runs the experts while the host encodes the next layer's attention and router.
        if Self.overlapsRouting { try flush(wait: false) } else { try commitIfLong() }
        return
    }

    /// `moe`'s waves: each read on another thread while the one before it runs.
    private func runExpertWaves(
        layer l: Int, used: [Int], wave: Int, rowsOf: [[Int32]], weightsOf: [[Float]], rows n: Int, input: MTLBuffer,
        output: MTLBuffer, limit: Float, next: (layer: Int, router: FamilyWeights.Matrix, bias: [Float], input: MTLBuffer?)?,
        slice: Int, decodeAccess: Bool, prefetchNext: () -> Void
    ) throws {
        let e = g.experts
        step = "layer \(l) experts"
        var offset = 0
        var batch = Array(used.prefix(wave))
        final class Read: @unchecked Sendable {
            var slots: [FamilyExperts.Slot] = []
            var error: Error?
            let done = DispatchSemaphore(value: 0)
        }
        func join(_ read: Read) throws -> [FamilyExperts.Slot] {
            let started = Date()
            onReads { read.done.wait() }
            timingLock.lock()
            timingState.reads += Date().timeIntervalSince(started)
            timingLock.unlock()
            if let error = read.error { throw error }
            return read.slots
        }
        let readStarted = Date()
        var slots = try onReads { try experts.fetch(layer: l, experts: batch, decode: decodeAccess) }
        timingLock.lock()
        timingState.reads += Date().timeIntervalSince(readStarted)
        timingLock.unlock()
        if n == 1 { prefetchNext() }  // decode is one wave (pools hold two): nothing else reads while the guesses are placed
        // The last buffer of the wave before this one: its slots, and the rows and weights below, are free once it finishes.
        var before: MTLCommandBuffer?
        let ahead = !generationBlock && n >= readAheadRows ? next?.layer : nil
        while offset < used.count {
            // A client that left stops a long prompt between waves, not only between layers; decode's one row finishes.
            guard n == 1 || shouldContinue() else { throw EngineError("Cancelled", code: "cancelled") }
            if let completed = before {
                try finish(completed)
                before = nil
            }
            if hasPendingTrim {
                // The preceding wave is joined and the prefetched wave has no GPU consumer yet. Drop its views,
                // apply the requested trim, and repartition only the unexecuted experts under the smaller limit.
                // Existing retained records are hits; records retired under pressure are read again on demand.
                slots.removeAll(keepingCapacity: true)
                autoreleasepool { fitExperts() }
                let reducedWave = try experts.prepareWave(layer: l, maximumExperts: used.count - offset)
                batch = Array(used[offset..<min(used.count, offset + reducedWave)])
                let started = Date()
                slots = try experts.fetch(layer: l, experts: batch, decode: decodeAccess)
                timingLock.lock()
                timingState.reads += Date().timeIntervalSince(started)
                timingLock.unlock()
            }
            let after = offset + batch.count
            let nextWave = min(max(1, experts.slots(layer: l) / 2), experts.allocatedSlots(layer: l) - batch.count)
            guard after == used.count || nextWave > 0 else {
                throw EngineError("Prepared expert waves lost their physical capacity", code: "insufficient_memory")
            }
            let nextBatch = Array(used[after..<min(used.count, after + nextWave)])
            // Every earlier wave has finished and the last is the last fetch's; nothing else uses the pool until the next layer.
            if nextBatch.isEmpty, let ahead {
                experts.readAhead(layer: ahead, experts: Array(0..<min(e, max(1, experts.slots(layer: ahead) / 2))))
            }
            // The next wave is read on another thread while this one is encoded and runs: into slots neither wave holds
            // (`keepingLast`), the wave before having finished. Pools hold at least two waves; encode order is unchanged.
            var next: Read?
            if !nextBatch.isEmpty {
                let read = Read()
                let pool = experts!
                let asked = nextBatch
                let decode = decodeAccess
                DispatchQueue.global(qos: .userInitiated).async {
                    do { read.slots = try pool.fetch(layer: l, experts: asked, decode: decode, keepingLast: true) } catch {
                        read.error = error
                    }
                    read.done.signal()
                }
                next = read
            }
            do {
                // Each expert's rows and weights at its own offset: the GPU reads them only after the whole wave is written
                // (the wave before has finished, and so has every earlier layer's, behind the router's wait).
                try runWave(
                    batch, slots, rowsOf: rowsOf, weightsOf: weightsOf, input: input, output: output, rows: n, limit: limit, slice: slice)
            } catch {
                next?.done.wait()
                throw error
            }  // the read must not run on while the caller resets
            offset = after
            if let next {
                before = try flush(wait: false)
                slots = try join(next)
                batch = nextBatch
            }
        }
    }

    /// Decode's routed experts (one row): gate and up of every expert in one dispatch (two when their types differ),
    /// SwiGLU over all, down in one, then `output += w_z * out_z` in expert order. The same bits as `runExpert` for
    /// each expert in turn: the same matrix kernels per expert, and the adds in the same order.
    func runExperts(
        _ e: MTLComputeCommandEncoder, slots: [FamilyExperts.Slot], input: MTLBuffer, output: MTLBuffer, weights w: [Float], limit: Float
    ) throws {
        let k = slots.count
        let (grid, group) = grid2(hidden, 1)
        try expertOutputs(e, slots: slots, input: input, first: 0, of: k, limit: limit)
        try kernels.dispatch(
            e, "add_weighted_rows", [Ref(try work("exp-out", k * hidden)), Ref(output)], ints: [Int32(hidden), Int32(k)],
            floats: w + Array(repeating: 0, count: 64 - k), grid: grid, group: group)
    }
    /// `runExperts`'s experts before the weighted add, `slots` into rows `first ..< first + slots.count` of the `total`
    /// a layer runs (exp-out; gate and up at their own offsets of exp-gu): decode runs its resident experts while the
    /// others are read, in rows of their own, and each expert's output has the same bits wherever its row is.
    func expertOutputs(
        _ e: MTLComputeCommandEncoder, slots: [FamilyExperts.Slot], input: MTLBuffer, first: Int, of total: Int, limit: Float
    ) throws {
        let h = hidden
        let width = g.expertWidth
        let k = slots.count
        let gu = Ref(try work("exp-gu", 2 * total * width), 2 * first * width * 4)
        let out = Ref(try work("exp-out", total * h), first * h * 4)
        func matrices(_ part: Int, width: Int, outputs: Int) -> [FamilyWeights.Matrix] {
            slots.map {
                FamilyWeights.Matrix(
                    buffer: $0.buffer, offset: $0.offsets[part], type: $0.types[part], width: width, outputs: outputs, name: "expert")
            }
        }
        let gates = matrices(0, width: h, outputs: width)
        let ups = matrices(1, width: h, outputs: width)
        let up = Ref(gu.buffer, gu.offset + k * width * 4)
        if Self.groupsGateUp, gates[0].type == ups[0].type {
            try mmBatch(e, gates + ups, Ref(input), inputStride: 0, gu, outputStride: width * 4, rows: 1)
        } else {
            try mmBatch(e, gates, Ref(input), inputStride: 0, gu, outputStride: width * 4, rows: 1)
            try mmBatch(e, ups, Ref(input), inputStride: 0, up, outputStride: width * 4, rows: 1)
        }
        let (grid, group) = grid2(width, k)
        try kernels.dispatch(
            e, "swiglu", [gu, up, gu], ints: [Int32(width), Int32(k), Int32(width), Int32(width), Int32(width)],
            floats: [limit], grid: grid, group: group)
        try mmBatch(e, matrices(2, width: width, outputs: h), gu, inputStride: width * 4, out, outputStride: h * 4, rows: 1)
    }

    /// One wave's experts over their rows (`moe`'s prompt path): each expert runs once per slice its rows fall in (a run:
    /// its rows in `ids`, Int32, ascending, their weights in `ws`), in expert order. Consecutive runs of one type go
    /// together, as many as the working buffers' rows and the open buffer's work allow: one gather of their rows, gate,
    /// up and down each in one dispatch (`encodeRowsBatch`; a run of 8 rows or fewer on its own `mm`, as `mm` runs it), one
    /// SwiGLU, then `scatter_add_ordered` adds every row's runs into it in run order. Each row gets the kernels and the adds
    /// one run at a time gave it, in the same order (`family_expert_runs_match_single`).
    private func runWave(
        _ batch: [Int], _ slots: [FamilyExperts.Slot], rowsOf: [[Int32]], weightsOf: [[Float]], input: MTLBuffer, output: MTLBuffer,
        rows n: Int, limit: Float, slice: Int
    ) throws {
        let total = batch.reduce(0) { $0 + rowsOf[$1].count }
        let ids = try work("exp-ids", total)
        let ws = try work("exp-ws", total)
        var runs: [(slot: FamilyExperts.Slot, at: Int, count: Int)] = []
        var at = 0
        for (expert, slot) in zip(batch, slots) {
            let rows = rowsOf[expert]
            let count = rows.count
            rows.withUnsafeBytes { (ids.contents() + at * 4).copyMemory(from: $0.baseAddress!, byteCount: count * 4) }
            weightsOf[expert].withUnsafeBytes { (ws.contents() + at * 4).copyMemory(from: $0.baseAddress!, byteCount: count * 4) }
            // One run per slice, in row order (the rows ascend).
            var start = 0
            while start < count {
                let part = rows[start...].prefix { Int($0) / slice == Int(rows[start]) / slice }.count
                runs.append((slot, at + start, part))
                start += part
            }
            at += count
        }
        // Every group's row order and starts, at their own offsets: the GPU reads them after the whole wave is written.
        let order = try work("exp-add", 3 * total + 4 * runs.count)
        var orderAt = 0
        let capacity = min(n, slice)  // the rows of exp-x
        let perRow = 3 * Double(hidden) * Double(g.expertWidth)
        var first = 0
        while first < runs.count {
            let types = runs[first].slot.types
            let most = min(capacity, Int(commitLimit / perRow))
            var end = first + 1
            var rows = runs[first].count
            while end < runs.count, end - first < GGMLMatrixKernels.rowsBatchCapacity, runs[end].slot.types == types,
                rows + runs[end].count <= most
            {
                rows += runs[end].count
                end += 1
            }
            if pendingWork > 0, pendingWork + Double(rows) * perRow > commitLimit { try flush(wait: false) }
            orderAt = try runGroup(
                runs[first..<end], rows: rows, ids: ids, ws: ws, order: order, orderAt: orderAt, input: input, output: output, limit: limit)
            first = end
            try commitIfLong()
        }
    }

    /// `runWave`'s runs `group` (`rows` in all, consecutive in `ids`), their row order for the add at `orderAt` of
    /// `order`; returns where the next group's goes.
    private func runGroup(
        _ group: ArraySlice<(slot: FamilyExperts.Slot, at: Int, count: Int)>, rows total: Int, ids: MTLBuffer, ws: MTLBuffer,
        order: MTLBuffer, orderAt: Int, input: MTLBuffer, output: MTLBuffer, limit: Float
    ) throws -> Int {
        let h = hidden
        let width = g.expertWidth
        let first = group.first!.at
        let x = try work("exp-x", total * h)
        let gate = try work("exp-gate", total * width)
        let up = try work("exp-up", total * width)
        let out = try work("exp-out", total * h)
        var (grid, group2) = grid2(h, total)
        try kernels.dispatch(
            encoder, "gather_rows", [Ref(input), Ref(ids, first * 4), Ref(x)], ints: [Int32(h), Int32(total)], grid: grid, group: group2)
        // Each part's runs of more than 8 rows in one dispatch (mul_mm), those of each smaller count in one (mul_mv).
        func part(_ p: Int, width: Int, outputs: Int, from source: MTLBuffer, into target: MTLBuffer) throws {
            var batch: [(slot: FamilyExperts.Slot, at: Int, count: Int)] = []
            var few = [[(slot: FamilyExperts.Slot, at: Int, count: Int)]](repeating: [], count: 9)
            // One run on its own, as `mm` runs it (no batch kernel for its type, or mul_mv for all rows under 64 wide).
            func single(_ run: (slot: FamilyExperts.Slot, at: Int, count: Int)) throws {
                try mm(
                    encoder,
                    FamilyWeights.Matrix(
                        buffer: run.slot.buffer, offset: run.slot.offsets[p], type: run.slot.types[p], width: width, outputs: outputs,
                        name: ["gate", "up", "down"][p]),
                    Ref(source, (run.at - first) * width * 4), Ref(target, (run.at - first) * outputs * 4), rows: run.count)
            }
            for run in group {
                if run.count <= 8 { few[run.count].append(run) } else if width >= 64 { batch.append(run) } else { try single(run) }
            }
            for (rows, runs) in few.enumerated() {
                guard let type = runs.first?.slot.types[p] else { continue }
                guard matrices.batches(type: type) else {
                    for run in runs { try single(run) }
                    continue
                }
                encoded(Double(runs.count * rows) * Double(width) * Double(outputs))
                if profile != nil {
                    let (block, bytes) = GGUFQuant.sizes[type]!
                    profiledMatrix = (["gate", "up", "down"][p], rows, runs.count * outputs * width / block * bytes)
                }
                try matrices.encodeFewRowsBatch(
                    encoder, type: type, matrices: runs.map { ($0.slot.buffer, $0.slot.offsets[p]) }, width: width, outputs: outputs,
                    input: source, inputOffsets: runs.map { ($0.at - first) * width * 4 }, rows: rows, output: target,
                    outputOffsets: runs.map { ($0.at - first) * outputs * 4 })
            }
            guard let type = batch.first?.slot.types[p] else { return }
            let rows = batch.reduce(0) { $0 + $1.count }
            encoded(Double(rows) * Double(width) * Double(outputs))
            if profile != nil {
                let (block, bytes) = GGUFQuant.sizes[type]!
                profiledMatrix = (["gate", "up", "down"][p], rows, batch.count * outputs * width / block * bytes)
            }
            try matrices.encodeRowsBatch(
                encoder, type: type, matrices: batch.map { ($0.slot.buffer, $0.slot.offsets[p]) }, width: width, outputs: outputs,
                input: source, inputOffsets: batch.map { ($0.at - first) * width * 4 }, rows: batch.map(\.count), output: target,
                outputOffsets: batch.map { ($0.at - first) * outputs * 4 })
        }
        for run in group {
            routeTrace?.recordPrefill(
                rows: run.count,
                eligible: run.count <= 8 && run.slot.types[0] == run.slot.types[1] && matrices.batches(type: run.slot.types[0]))
        }
        try part(0, width: h, outputs: width, from: x, into: gate)
        try part(1, width: h, outputs: width, from: x, into: up)
        (grid, group2) = grid2(width, total)
        try kernels.dispatch(
            encoder, "swiglu", [Ref(gate), Ref(up), Ref(gate)],
            ints: [Int32(width), Int32(total), Int32(width), Int32(width), Int32(width)],
            floats: [limit], grid: grid, group: group2)
        try part(2, width: width, outputs: h, from: gate, into: out)
        // Each output row's runs in run order: the group's rows sorted by output row, then by place.
        let rowIds = ids.contents().bindMemory(to: Int32.self, capacity: first + total) + first
        let sorted = (0..<total).sorted { (rowIds[$0], $0) < (rowIds[$1], $1) }
        let base = order.contents().bindMemory(to: Int32.self, capacity: orderAt + 3 * total + 4) + orderAt
        var targets = 0
        for (j, q) in sorted.enumerated() {
            base[j] = Int32(q)
            if j == 0 || rowIds[q] != rowIds[sorted[j - 1]] {
                base[total + targets] = rowIds[q]
                base[2 * total + targets] = Int32(j)
                targets += 1
            }
        }
        base[2 * total + targets] = Int32(total)
        (grid, group2) = grid2(h, targets)
        try kernels.dispatch(
            encoder, "scatter_add_ordered",
            [
                Ref(out), Ref(order, orderAt * 4), Ref(ws, first * 4), Ref(output), Ref(order, (orderAt + total) * 4),
                Ref(order, (orderAt + 2 * total) * 4),
            ], ints: [Int32(h), Int32(targets)], grid: grid, group: group2)
        return orderAt + 3 * total + 1
    }

    /// The head: the logits of the last row of `last` (a hidden row of `hidden` values).
    func head(_ normed: MTLBuffer, output: FamilyWeights.Matrix) throws -> [Float] {
        let logits = try work("logits", output.outputs)
        try mm(encoder, output, Ref(normed), Ref(logits), rows: 1)
        try flush(wait: true)
        let p = logits.contents().bindMemory(to: Float.self, capacity: output.outputs)
        return Array(UnsafeBufferPointer(start: p, count: output.outputs))
    }
}
