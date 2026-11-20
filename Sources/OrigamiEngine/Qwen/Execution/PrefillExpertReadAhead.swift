import Dispatch
import Foundation

/// Expert read-ahead for one whole-prefill scope. A single producer thread requests every expert record of every
/// layer in order (index = layer * 512 + expert), keeping at most `held` speculative requests (the cache's live slot
/// count minus the demand window, never below one layer). Routed demand may use that reserved window and takes the
/// next free queue slot ahead of the scan. There remain at most `pipeline` unjoined reads, retired in completion order.
/// The demand window borrows these requests instead of reading (`request`); `release` drops consumed records and
/// `finishLayer` the layer's unused ones. Reads only: no tensor arithmetic, record layout or consumption order changes
/// (BITWISE-neutral). Only complete handoff-pinned records reach the synchronous demand window; a failed read stays in
/// its request and the demand performs the joined internal-source fallback.
package final class PrefillExpertReadAhead: @unchecked Sendable {
    package static let metadataBytes: UInt64 = 4 * 1024 * 1024
    /// One read-ahead scope only. ExpertExecution retains the last closed
    /// scope's snapshot; these counters are not cumulative request totals.
    package struct Snapshot: Codable, Sendable {
        package let submitted, joined, consumed, skipped: Int
        /// Producer joins of a ready entry behind the oldest pending.
        package let outOfOrderRetirements: Int
        package let retained, maximumRetained, maximumPending: Int
        package let requestWaitSeconds: Double
        package let closed: Bool
        package var demandBypasses: Int? = nil
        package var pendingDemands: Int? = nil
    }
    private let condition = NSCondition()
    private let group = DispatchGroup()
    private let loader: SourceExpertLoader
    private let generation: IOScheduler.Generation
    private let configuration: ExpertExecutionConfiguration
    /// The layers it reads, `0..<layers`: a scope whose final layer skips its unread rows ends one layer early.
    package let layers: Int
    private let count, limit, pipeline: Int
    /// Experts per layer the model has (`ModelLayout.routedExperts`): a pruned GGUF's others are never read.
    private let routed = ModelLayout.routedExperts
    /// The cache's slots now: the memory governor may shrink the cache while this scope runs.
    private let cacheSlots: (@Sendable () -> Int)?
    private var requests: [Int: ExpertLoadRequest] = [:]
    /// Actual routed demand takes the producer's next free queue slot before
    /// its sequential scan, without exceeding the existing I/O queue depth.
    private var demanded: Set<Int> = []
    private var issued: Set<Int> = []
    // At most pipeline integer entries, inside the existing metadata owner.
    // Only the producer removes these; completion callbacks just signal ready.
    private var completed: Set<Int> = []
    private var next = 0, floor = 0
    private var stopped = false, closed = false
    private var failure: Error?
    private var submitted = 0, joined = 0, consumed = 0, skipped = 0
    private var outOfOrderRetirements = 0
    private var maximumRetained = 0, maximumPending = 0
    private var requestWaitSeconds = 0.0
    private var demandBypasses = 0

    package init(
        loader: SourceExpertLoader, generation: IOScheduler.Generation,
        configuration: ExpertExecutionConfiguration, layers: Int, cacheSlots: (@Sendable () -> Int)? = nil
    ) throws {
        // The cache may run below its configured size when the memory budget is lowered.
        let slots = min(configuration.effectivePrefillCacheSlots, cacheSlots?() ?? .max)
        guard (1...48).contains(layers), slots >= 512 + configuration.demandWindow, configuration.demandWindow >= 2 else {
            throw EngineError(
                "Prefill read-ahead needs \(512 + configuration.demandWindow) cache slots and a demand window of at least 2; "
                    + "the cache has \(slots) slots, the window is \(configuration.demandWindow) (layers \(layers))")
        }
        self.loader = loader
        self.generation = generation
        self.configuration = configuration
        self.cacheSlots = cacheSlots
        self.layers = layers
        count = layers * 512
        // Removed dictionary handles can still be held by the producer's joins.
        // Reserve the entire old demand window for those transient references.
        limit = slots - configuration.demandWindow
        pipeline = configuration.demandWindow - 1
        group.enter()
        DispatchQueue(label: "origami.prefill-expert-read-ahead").async { [self] in
            defer { group.leave() }
            produce()
        }
    }

    /// Records the producer may hold: the scope's limit, lowered while the governor keeps the cache smaller (never
    /// below one layer, which the governor's floor of 528 slots leaves room for).
    private var held: Int { min(limit, max(512, (cacheSlots?() ?? .max) - configuration.demandWindow)) }

    private func produce() {
        var pending: [(index: Int, request: ExpertLoadRequest)] = []
        defer {
            for item in pending { _ = loader.cancel(item.request) }
            for item in pending { loader.joinDiscarding(item.request) }
        }
        do {
            while true {
                condition.lock()
                while !stopped && pending.isEmpty && demanded.isEmpty && (next >= count || requests.count >= held) {
                    // Only close() can end a finished scope; a held one polls, as the cache may grow back meanwhile.
                    if next >= count { condition.wait() } else { _ = condition.wait(until: Date(timeIntervalSinceNow: 0.05)) }
                }
                let stop = stopped
                condition.unlock()
                if stop { return }
                while pending.count < pipeline {
                    condition.lock()
                    while next < count && next % 512 >= routed { next += 1 }
                    let urgent = demanded.min()
                    let index = urgent ?? next
                    let canSubmit = !stopped && index < count && requests.count < held + (urgent == nil ? 0 : configuration.demandWindow)
                    if canSubmit {
                        if let urgent { demanded.remove(urgent) } else { next += 1 }
                    }
                    let alreadyRequested = canSubmit ? !issued.insert(index).inserted : false
                    if canSubmit && urgent != nil && !alreadyRequested { demandBypasses += 1 }
                    condition.unlock()
                    if !canSubmit { break }
                    if alreadyRequested { continue }
                    let key = try ExpertKey(layer: index / 512, expert: index % 512)
                    // This scope is the sole read producer. The demand queues keep
                    // their depth of 4 and 16 admissions per physical drive;
                    // consumers borrow these requests instead of submitting.
                    let deadline = Date(timeIntervalSinceNow: 30)
                    var made: ExpertLoadRequest?
                    while made == nil {
                        do {
                            made = try loader.request(
                                key, generation: generation,
                                policy: configuration.prefillSourcePolicy, calibration: configuration.sourceCalibration)
                        } catch let error as EngineError where error.code == "expert_slots_busy" && Date() < deadline {
                            // The governor shrank the cache under this scope: wait (up to 30 s, measured, as every
                            // release wakes this wait) for the consumer to free a record.
                            condition.lock()
                            let stop = stopped
                            if !stop { _ = condition.wait(until: min(deadline, Date(timeIntervalSinceNow: 0.05))) }
                            condition.unlock()
                            if stop { return }
                        }
                    }
                    let request = made!
                    pending.append((index, request))
                    condition.lock()
                    submitted += 1
                    if !stopped && index >= floor && requests[index] == nil { requests[index] = request }
                    maximumRetained = max(maximumRetained, requests.count)
                    maximumPending = max(maximumPending, pending.count)
                    let cancel = stopped
                    condition.broadcast()
                    condition.unlock()
                    if cancel {
                        _ = loader.cancel(request)
                        return
                    }
                    try loader.notifyPrefillCompletion(request) { [weak self] in
                        guard let self else { return }
                        self.condition.lock()
                        defer { self.condition.unlock() }
                        guard !self.stopped else { return }
                        self.completed.insert(index)
                        self.condition.broadcast()
                    }
                }
                if !pending.isEmpty {
                    condition.lock()
                    // Do not let one slow oldest read hold up completed reads
                    // on either drive. This is a completion wait, not a poll.
                    var ready = pending.firstIndex { completed.contains($0.index) }
                    while !stopped && ready == nil {
                        condition.wait()
                        ready = pending.firstIndex { completed.contains($0.index) }
                    }
                    if stopped {
                        condition.unlock()
                        return
                    }
                    let selected = ready!
                    let item = pending.remove(at: selected)
                    completed.remove(item.index)
                    condition.unlock()
                    // Failed speculative reads remain explicit in the shared
                    // request; actual demand performs the existing joined
                    // internal-source fallback. Unused records are never used.
                    loader.joinDiscarding(item.request)
                    condition.lock()
                    joined += 1
                    if selected > 0 { outOfOrderRetirements += 1 }
                    condition.unlock()
                }
            }
        } catch {
            condition.lock()
            failure = error
            stopped = true
            condition.broadcast()
            condition.unlock()
        }
    }

    package func request(_ key: ExpertKey, shouldContinue: () -> Bool) throws -> ExpertLoadRequest {
        let start = DispatchTime.now().uptimeNanoseconds
        condition.lock()
        defer {
            requestWaitSeconds += Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            condition.unlock()
        }
        let index = key.layer * 512 + key.expert
        guard index >= floor && index < count else {
            throw EngineError("Read-ahead demand for layer \(key.layer) is outside the live layers \(floor / 512)..<\(count / 512)")
        }
        while requests[index] == nil {
            if let failure { throw failure }
            guard !stopped, shouldContinue() else { throw IOOperationCancelled() }
            demanded.insert(index)
            condition.broadcast()
            _ = condition.wait(until: Date(timeIntervalSinceNow: 0.025))
        }
        guard !stopped, shouldContinue() else { throw IOOperationCancelled() }
        demanded.remove(index)
        return requests[index]!
    }

    /// Caller already retired every synchronous CPU/GPU consumer pin.
    package func release(_ keys: ArraySlice<ExpertKey>) {
        condition.lock()
        defer {
            condition.broadcast()
            condition.unlock()
        }
        for key in keys {
            if requests.removeValue(forKey: key.layer * 512 + key.expert) != nil { consumed += 1 }
        }
    }
    package func finishLayer(_ layer: Int) throws {
        condition.lock()
        defer {
            condition.broadcast()
            condition.unlock()
        }
        guard layer * 512 == floor, layer < count / 512 else { throw EngineError("Read-ahead layers must retire in order") }
        floor = (layer + 1) * 512
        demanded = demanded.filter { $0 >= floor }
        let unused = requests.keys.filter { $0 < floor }
        for key in unused { requests.removeValue(forKey: key) }
        skipped += unused.count + (next < floor ? (next..<floor).filter { !issued.contains($0) }.count : 0)
        next = max(next, floor)
    }
    /// Idempotent; all reads join before the final producer handoff pins drop.
    /// Caller must have joined synchronous GPU consumers before entering here.
    package func close() {
        condition.lock()
        if closed {
            condition.unlock()
            return
        }
        stopped = true
        let current = Array(requests.values)
        condition.broadcast()
        condition.unlock()
        for request in current { _ = loader.cancel(request) }
        group.wait()
        for request in current { loader.joinDiscarding(request) }
        condition.lock()
        requests.removeAll()
        demanded.removeAll()
        issued.removeAll()
        completed.removeAll()
        closed = true
        condition.unlock()
    }
    package func snapshot() -> Snapshot {
        condition.lock()
        defer { condition.unlock() }
        var result = Snapshot(
            submitted: submitted, joined: joined, consumed: consumed, skipped: skipped,
            outOfOrderRetirements: outOfOrderRetirements,
            retained: requests.count, maximumRetained: maximumRetained, maximumPending: maximumPending,
            requestWaitSeconds: requestWaitSeconds, closed: closed)
        result.demandBypasses = demandBypasses
        result.pendingDemands = demanded.count
        return result
    }
}
