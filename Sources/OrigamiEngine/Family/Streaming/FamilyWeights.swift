import CExactIO
import Foundation
@preconcurrency import Metal
import Synchronization

/// Startup-only admission for actual requested buffers, not an estimate over all GGUF tensors (the large lookup
/// table and routed experts stay streamed). Charges retire before the final dense/state owners are installed.
package final class FamilyStartupAdmission {
    private let ledger: MemoryLedger
    private let footprint: () -> UInt64
    private var reservations: [MemoryLedger.Reservation] = []

    package init(
        ledger: MemoryLedger, fixedBytes: UInt64,
        footprint: @escaping () -> UInt64 = processFootprintBytes
    ) throws {
        self.ledger = ledger
        self.footprint = footprint
        // Existing owners are adopted, not allocated again.
        reservations.append(try ledger.reserve(bytes: fixedBytes, category: "family-startup"))
    }
    deinit { close() }
    /// Physical allocations are page rounded even when their logical Metal buffer length is smaller.
    @discardableResult package func reserve(bytes: UInt64) throws -> MemoryLedger.Reservation {
        let page = UInt64(getpagesize())
        let (padded, overflow) = bytes.addingReportingOverflow(page - 1)
        guard !overflow else {
            throw EngineError("Mandatory family allocation would exceed memory_limit", code: "insufficient_memory")
        }
        let reservation = try ledger.reservePhysical(
            bytes: padded / page * page, category: "family-startup",
            footprint: ledger.budgetMode == .explicit ? footprint() : nil)
        reservations.append(reservation)
        return reservation
    }
    package func release(_ reservation: MemoryLedger.Reservation) {
        guard let index = reservations.firstIndex(of: reservation) else { return }
        reservations.remove(at: index)
        try? ledger.release(reservation)
    }
    package func close() {
        for reservation in reservations { try? ledger.release(reservation) }
        reservations.removeAll()
    }
}

/// Allocates a shared buffer in whole pages and wires it (mlock) so the VM cannot compress or page it out from under
/// the GPU: a compressed weight page costs a decompression before every command that reads it (measured: 5 s of GPU
/// waits a token with 19 GB unwired on a 32 GB Mac). A failed wire leaves the buffer usable, only unwired.
func wiredBuffer(_ device: MTLDevice, length: Int, wired: inout Int) -> MTLBuffer? {
    let page = Int(getpagesize())
    let rounded = (max(16, length) + page - 1) / page * page
    guard let buffer = device.makeBuffer(length: rounded, options: .storageModeShared) else { return nil }
    var pages: UInt64 = 0
    if origami_wire_pages(buffer.contents(), rounded, page, &pages) == 0 { wired += rounded }
    return buffer
}

/// A GGUF model's weights for a family engine (GLM, DeepSeek): every tensor except the routed experts is read once
/// into shared Metal buffers in its stored ggml type (matrices) or as f32 (vectors); the routed experts stay in the
/// file and `FamilyExperts` reads the ones a token needs. Names resolve through aliases, since llama.cpp and
/// DwarfStar name a few tensors differently.
package final class FamilyWeights: @unchecked Sendable {
    /// A row-major ggml matrix: `outputs` rows of `width` values (GGUF dims [width, outputs]) at `offset` in `buffer`.
    package struct Matrix {
        package let buffer: MTLBuffer
        package let offset: Int
        package let type: Int
        package let width: Int
        package let outputs: Int
        package let name: String
        package init(buffer: MTLBuffer, offset: Int, type: Int, width: Int, outputs: Int, name: String) {
            self.buffer = buffer
            self.offset = offset
            self.type = type
            self.width = width
            self.outputs = outputs
            self.name = name
        }
    }

    package let gguf: GGUFSource
    package let geometry: ModelGeometry
    package let device: MTLDevice
    package let residentPack: FamilyResidentPack?
    private var files: [String: ExactFile] = [:]
    /// The GGUF's drive: its files reopen in place when it leaves and returns (`ModelDrive`).
    package let drive = ModelDrive()
    private let lock = NSLock()
    package private(set) var residentBytes = 0
    /// Bytes wired (mlock) of the resident ones.
    package private(set) var wiredBytes = 0
    /// Non-nil only inside a family model constructor; ordinary allocation paths add no host probes afterwards.
    package private(set) var startupAdmission: FamilyStartupAdmission?

    package func beginStartup(ledger: MemoryLedger?) throws {
        guard startupAdmission == nil else { throw EngineError("A family model is already loading") }
        guard let ledger else { return }
        // Startup admits every real allocation here. The one-row scratch minimum is known only after
        // geometry/layers are constructed, and makeExperts reserves it before publishing the model.
        startupAdmission = try FamilyStartupAdmission(ledger: ledger, fixedBytes: UInt64(residentBytes))
    }
    package func endStartup() {
        startupAdmission?.close()
        startupAdmission = nil
    }

    package init(gguf: GGUFSource, device: MTLDevice, residentPack: FamilyResidentPack? = nil) throws {
        self.gguf = gguf
        geometry = try ModelGeometry(gguf: gguf)
        self.device = device
        guard residentPack == nil || (residentPack?.sourceIdentity == FamilySupport.key(gguf) && residentPack?.conversion == .targetQ4)
        else {
            throw EngineError("Resident pack belongs to a different source model")
        }
        self.residentPack = residentPack
    }

    package func file(_ name: String) throws -> ExactFile {
        lock.lock()
        defer { lock.unlock() }
        if let open = files[name] { return open }
        let opened = try ExactFile(url: gguf.url(name), drive: drive)
        files[name] = opened
        return opened
    }

    /// The first tensor present among `names` (with `{L}` replaced by the layer).
    package func tensor(_ names: [String], layer: Int? = nil) -> GGUFSource.Tensor? {
        for name in names {
            let resolved = layer.map { name.replacingOccurrences(of: "{L}", with: "\($0)") } ?? name
            if let tensor = gguf.tensors[resolved] { return tensor }
        }
        return nil
    }

    func require(_ names: [String], layer: Int?) throws -> GGUFSource.Tensor {
        guard let tensor = tensor(names, layer: layer) else {
            let shown = names.map { n in layer.map { n.replacingOccurrences(of: "{L}", with: "\($0)") } ?? n }
            throw EngineError(
                "This \(geometry.family.displayName) GGUF lacks \(shown.joined(separator: " or "))", code: "model_incompatible")
        }
        guard tensor.length >= 0 else {
            throw EngineError("\(tensor.name) has the unknown ggml type \(tensor.type)", code: "model_incompatible")
        }
        return tensor
    }

    /// The raw bytes of a tensor in a new shared buffer.
    func load(_ tensor: GGUFSource.Tensor) throws -> MTLBuffer {
        let page = Int(getpagesize())
        let length = (max(16, Int(tensor.length)) + page - 1) / page * page
        try startupAdmission?.reserve(bytes: UInt64(length))
        var wired = 0
        guard let buffer = wiredBuffer(device, length: Int(tensor.length), wired: &wired) else {
            throw EngineError("Out of memory loading \(tensor.name) (\(tensor.length) bytes)", code: "insufficient_memory")
        }
        try file(tensor.file).read(
            offset: tensor.offset, into: UnsafeMutableRawBufferPointer(start: buffer.contents(), count: Int(tensor.length)))
        lock.lock()
        residentBytes += buffer.length
        wiredBytes += wired
        lock.unlock()
        return buffer
    }

    /// A matrix kept in its stored type (GGUF shape [width, outputs, …]: further dims fold into the rows).
    package func matrix(_ names: String..., layer: Int? = nil) throws -> Matrix { try matrix(names, layer: layer) }

    func matrix(_ names: [String], layer: Int?) throws -> Matrix {
        let tensor = try require(names, layer: layer)
        guard let width = tensor.shape.last, width > 0 else { throw EngineError("\(tensor.name) has no shape", code: "model_incompatible") }
        let outputs = tensor.shape.dropLast().reduce(1, *)
        guard ModelCompatibility.matrixTypes.contains(tensor.type) else {
            throw EngineError("\(tensor.name) is \(GGUFQuant.name(tensor.type)), which no Origami kernel reads", code: "model_incompatible")
        }
        if let residentPack, let packed = residentPack.layout(tensor.name) {
            let length = Int(packed.bytes)
            try startupAdmission?.reserve(bytes: UInt64(length))
            var wired = 0
            guard let buffer = wiredBuffer(device, length: length, wired: &wired) else {
                throw EngineError("Out of memory loading resident matrix \(tensor.name)", code: "insufficient_memory")
            }
            try residentPack.read(tensor.name, into: .init(start: buffer.contents(), count: length))
            lock.lock()
            residentBytes += buffer.length
            wiredBytes += wired
            lock.unlock()
            return Matrix(
                buffer: buffer, offset: 0, type: residentPack.conversion.targetType, width: width, outputs: outputs, name: tensor.name)
        }
        return Matrix(buffer: try load(tensor), offset: 0, type: tensor.type, width: width, outputs: outputs, name: tensor.name)
    }

    package func optionalMatrix(_ names: String..., layer: Int? = nil) throws -> Matrix? {
        guard tensor(names, layer: layer) != nil else { return nil }
        return try matrix(names, layer: layer)
    }

    /// Any tensor decoded to f32 on the CPU (norms, biases, small tables).
    package func floats(_ names: String..., layer: Int? = nil) throws -> [Float] {
        try floats(require(names, layer: layer))
    }

    package func optionalFloats(_ names: String..., layer: Int? = nil) throws -> [Float]? {
        guard let tensor = tensor(names, layer: layer) else { return nil }
        return try floats(tensor)
    }

    func floats(_ tensor: GGUFSource.Tensor) throws -> [Float] {
        let count = tensor.shape.reduce(1, *)
        guard GGUFQuant.readable.contains(tensor.type) else {
            throw EngineError("\(tensor.name) is \(GGUFQuant.name(tensor.type)), which Origami cannot decode", code: "model_incompatible")
        }
        // Both the encoded staging bytes and decoded CPU array coexist until dequantization finishes.
        let staging = try startupAdmission?.reserve(bytes: UInt64(tensor.length) + UInt64(count) * 4)
        defer { if let staging { startupAdmission?.release(staging) } }
        let raw = try file(tensor.file).read(offset: tensor.offset, length: Int(tensor.length))
        var out = [Float](repeating: 0, count: count)
        raw.withUnsafeBytes { source in
            out.withUnsafeMutableBufferPointer {
                GGUFQuant.dequantize(source.baseAddress!, type: tensor.type, count: count, into: $0.baseAddress!)
            }
        }
        return out
    }

    /// f32 values in a new shared buffer.
    package func buffer(_ values: [Float]) throws -> MTLBuffer {
        try startupAdmission?.reserve(bytes: UInt64(max(16, values.count * 4)))
        guard let buffer = device.makeBuffer(bytes: values, length: max(16, values.count * 4), options: .storageModeShared) else {
            throw EngineError("Out of memory for a \(values.count)-value buffer", code: "insufficient_memory")
        }
        lock.lock()
        residentBytes += buffer.length
        lock.unlock()
        return buffer
    }

    /// `count`: the values the kernels read from it; a shorter tensor is refused (the GPU does not check bounds).
    package func floatBuffer(_ names: String..., layer: Int? = nil, count: Int) throws -> MTLBuffer {
        let tensor = try require(names, layer: layer)
        guard tensor.shape.reduce(1, *) >= count else {
            throw EngineError(
                "\(tensor.name) holds \(tensor.shape.reduce(1, *)) values; this geometry reads \(count)", code: "model_incompatible")
        }
        return try buffer(try floats(tensor))
    }

    /// Dequantizes one row of a resident matrix on the CPU (token embeddings): row `row` of `width` values.
    package static func row(_ matrix: Matrix, _ row: Int, into out: UnsafeMutablePointer<Float>) {
        guard let (block, bytes) = GGUFQuant.sizes[matrix.type] else { return }
        let rowBytes = matrix.width / block * bytes
        GGUFQuant.dequantize(matrix.buffer.contents() + matrix.offset + row * rowBytes, type: matrix.type, count: matrix.width, into: out)
    }
}

/// The routed experts of a GGUF model, read from the file on demand into fixed pools of slots in shared Metal
/// buffers. One record is an expert's gate, up and down matrices back to back, each in the type its tensor has
/// (types may differ per layer and per projection). Layers whose records have the same (16 KiB-aligned) size share a
/// pool. Pools start with equal slots per layer; at request boundaries their shares can follow observed decode reuse
/// and saved read time per byte. Within a pool the coolest slot is reused first: each slot's heat grows by one per use and halves every
/// `halfLife` accesses, so experts a reply keeps coming back to stay over ones a prompt touched once. A prompt's reads
/// first reuse records a prompt placed and has passed (`Passed`). Slot placement changes no arithmetic.
///
/// The slots are an `ExpertSlotCore` (one size class per pool) with `FamilyEviction` as its policy: a record is found
/// only once its read is verified, a slot can never be reused under a reader (generation tickets), and the records a
/// fetch returns are pinned until the GPU work on them has finished by this type's caller protocol (the next fetch, or
/// the one after with `keepingLast`, a guess, or a limit change).
///
/// Threading: one owner at a time calls into it, never two at once: the serving thread during a request, the wave-read
/// thread that thread has handed it to and joins before using it again, or the warm-up while no request runs. Slot
/// state changes only under the core's lock. Its read jobs touch only their own slots and `Guess` (under `guessLock`).
/// Status readers use published snapshots, never the live state.
package final class FamilyExperts: @unchecked Sendable {
    /// Where one expert's three matrices are in the file.
    struct Bank {
        let tensors: [GGUFSource.Tensor]  // gate, up, down
        let types: [Int]  // per projection
        let bytes: [Int]  // per expert, per projection
        var record: Int { bytes.reduce(0, +) }
    }
    /// A resident record: buffer, offset of each projection, and their types and shapes.
    package struct Slot {
        package let buffer: MTLBuffer
        package let offsets: [Int]
        package let types: [Int]
        package init(buffer: MTLBuffer, offsets: [Int], types: [Int]) {
            self.buffer = buffer
            self.offsets = offsets
            self.types = types
        }
    }
    /// `prefetch…`: guessed records read (`prefetch`), asked for while resident or still read (used), asked for while
    /// still read (late), cancelled before their read began, and bytes read and evicted unasked (wasted); `predicted`:
    /// experts a decode fetch asked for after a guess at its layer, `predictedRight` the ones the guess held;
    /// `guessIssued`/`guessUsed` by rank in the guess (0 its best): reads begun, and those the next decode fetch of the
    /// layer asked for;
    /// `readAhead…`: a prompt's records read ahead (`readAhead`, their bytes also in `readBytes`), used and late.
    package struct Counters: Sendable {
        package var hits = 0, misses = 0, readBytes = 0, readSeconds = 0.0, hotReads = 0, hotBytes = 0
        /// Successful synchronous demand records only, excluding prefetch and warm-up. Record intervals overlap
        /// each other; source batch spans overlap the other drive and GPU work. Neither is exclusive wall time.
        package var demandHotServiceSeconds = 0.0, demandMainServiceSeconds = 0.0
        package var demandHotSpanSeconds = 0.0, demandMainSpanSeconds = 0.0
        package var demandHotRecords = 0, demandMainRecords = 0
        /// Sum of per-job time before its read closure starts. Jobs and source spans overlap;
        /// these are scheduler observations, not exclusive request wall time.
        package var demandHotQueueSeconds = 0.0, demandMainQueueSeconds = 0.0
        package var demandHotJobs = 0, demandMainJobs = 0
        /// Demand records the hot copy holds but the bandwidth split sent to the GGUF (what a hedged read could rescue).
        package var demandHeldMainRecords = 0
        package var demandTiming: [String: Double] {
            [
                "demand_hot_record_seconds": demandHotServiceSeconds, "demand_main_record_seconds": demandMainServiceSeconds,
                "demand_hot_batch_span_seconds": demandHotSpanSeconds, "demand_main_batch_span_seconds": demandMainSpanSeconds,
                "demand_hot_records": Double(demandHotRecords), "demand_main_records": Double(demandMainRecords),
                "demand_hot_queue_seconds": demandHotQueueSeconds, "demand_main_queue_seconds": demandMainQueueSeconds,
                "demand_hot_jobs": Double(demandHotJobs), "demand_main_jobs": Double(demandMainJobs),
                "demand_held_main_records": Double(demandHeldMainRecords),
            ]
        }
        package var allocationSeconds = 0.0
        package var prefetches = 0, prefetchUsed = 0, prefetchLate = 0, prefetchCancelled = 0, prefetchBytes = 0, prefetchWastedBytes = 0
        package var predicted = 0, predictedRight = 0, guessIssued: [Int] = [], guessUsed: [Int] = []
        package var readAheads = 0, readAheadUsed = 0, readAheadLate = 0
        package var guessRejectedConfidence = 0, guessRejectedQueue = 0, guessRejectedDeadline = 0, guessRejectedPressure = 0
        package var earlyBatches = 0, earlyOutstandingBatches = 0
        package var earlyDemandBytes = 0, earlyDemandHotBytes = 0
        /// Hedged demand reads (`fetch`): replicas fired (a record both drives hold whose read failed or outlasted its
        /// delay, read again from the other drive), records the replica won, records the primary won, replicas that
        /// failed, and the bytes replicas read (a queued loser is cancelled unread: at most this much was read twice).
        package var hedges = 0, hedgeReplicaWins = 0, hedgePrimaryWins = 0, hedgeReplicaFailures = 0, hedgeBytes = 0
        /// Record bytes read from the hot copy and from the model's own files (the GGUF) on every path: demand,
        /// read-ahead, decode guesses and warm-up (a hedge's second read stays in `hedgeBytes`). `hits`, `misses` and
        /// `hotReads` leave out records a guess read before they were asked for, so they cannot say where reads came from.
        package var copyBytes = 0, modelBytes = 0
        /// Decode guesses by the drive they were planned on (0 the copy, 1 the model's drive): reads begun, asked for,
        /// bytes evicted unasked, refusals by reason, and the seconds from their router result to their reads' submission
        /// and to their last part's end (sums and counts).
        package struct GuessDrive: Sendable {
            package var issued = 0, used = 0, wastedBytes = 0
            package var refusedConfidence = 0, refusedQueue = 0, refusedDeadline = 0, refusedPressure = 0
            package var issueSeconds = 0.0, issues = 0, completeSeconds = 0.0, completions = 0
        }
        package var guessDrives = [GuessDrive(), GuessDrive()]
        /// Per decode layer with a demand read: seconds from its router result to its first demand read submitted (sum,
        /// layers, and the longest since `resetRequestMaximum`).
        package var routerToDemandSeconds = 0.0, routerToDemandLayers = 0, routerToDemandMaximum = 0.0
        /// Demand records the copy held that went to the model's drive (`demandHeldMainRecords`): their read seconds, and
        /// the seconds by which the copy would have ended them sooner, after its own reads of the batch at its live rate.
        package var heldMainSeconds = 0.0, heldMainCopySoonerSeconds = 0.0
        /// The sums above, for per-request deltas (`decode.` in the phase counters).
        package var overlapTiming: [String: Double] {
            var notes: [String: Double] = [
                "router_to_demand_seconds": routerToDemandSeconds, "router_to_demand_layers": Double(routerToDemandLayers),
                "held_main_seconds": heldMainSeconds, "held_main_copy_sooner_seconds": heldMainCopySoonerSeconds,
            ]
            for (d, name) in ["copy", "model_drive"].enumerated() {
                let g = guessDrives[d]
                let prefix = "guess." + name + "."
                notes[prefix + "issued"] = Double(g.issued)
                notes[prefix + "used"] = Double(g.used)
                notes[prefix + "wasted_bytes"] = Double(g.wastedBytes)
                notes[prefix + "refused_confidence"] = Double(g.refusedConfidence)
                notes[prefix + "refused_queue"] = Double(g.refusedQueue)
                notes[prefix + "refused_deadline"] = Double(g.refusedDeadline)
                notes[prefix + "refused_pressure"] = Double(g.refusedPressure)
                notes[prefix + "router_to_issue_seconds"] = g.issueSeconds
                notes[prefix + "router_to_issue_count"] = Double(g.issues)
                notes[prefix + "router_to_complete_seconds"] = g.completeSeconds
                notes[prefix + "router_to_complete_count"] = Double(g.completions)
            }
            return notes
        }
        /// A phase's lookups, hits and bytes per source since `before` (`prefill_...`, `decode_...` phase counters).
        package func reads(since before: Counters, phase: String) -> [String: Double] {
            [
                phase + "_expert_lookups": Double(lookups - before.lookups), phase + "_expert_hits": Double(hits - before.hits),
                phase + "_expert_copy_bytes": Double(copyBytes - before.copyBytes),
                phase + "_expert_model_bytes": Double(modelBytes - before.modelBytes),
            ]
        }
        /// Asked records: from memory (`hits`), read on demand (`misses`) or read ahead by a guess.
        package var lookups: Int { hits + misses + readAheadUsed + prefetchUsed }
        /// Where the asked records came from: memory, and the rest in the shares of bytes read from the copy and the
        /// model's files (ROADMAP R14).
        package var split: (lookups: Int, memory: Double, hot: Double, drive: Double) {
            guard lookups > 0 else { return (0, 0, 0, 0) }
            let memory = Double(hits) / Double(lookups)
            let read = Double(copyBytes + modelBytes)
            let hot = read > 0 ? (1 - memory) * Double(copyBytes) / read : 0
            return (lookups, memory, hot, read > 0 ? 1 - memory - hot : 0)
        }
    }
    /// A guessed record being read (`prefetch`, or `readAhead` when `ahead`) in `parts` reads:
    /// `done` is left once they end, or at once when it was cancelled before one began (`started`, `cancelled` and
    /// `error` change under `guessLock`).
    /// A fetch's primary read that lost to its hedge while still reading (`hedge`) is kept as a demanded guess, begun,
    /// until it lands: verified if it read the whole record, else forgotten.
    final class Guess: @unchecked Sendable {
        /// Its slot's reservation (reading once its parts are submitted).
        let ticket: ExpertSlotTicket, ahead: Bool
        let done: DispatchGroup
        var observation: FamilyPrefetchObservations.Ticket?
        var started = false, cancelled = false, demanded = false, deliveredFromHot = false
        var error: Error?
        /// Its layer's router result and its last part's end (uptime ns; 0: none), for `GuessDrive`'s times.
        var routerAt: UInt64 = 0, endedAt: UInt64 = 0
        /// A wait for it outlived its bound (`ExpertReadStall`): later waits look without waiting (the owner's thread).
        var stalled = false
        let hedge: ExpertReadReplica?
        /// Why its record is not whole (`error`, or a lost primary's failure), once `done`.
        var failure: Error? {
            error ?? (hedge.map { $0.primarySucceeded } == false ? EngineError("The expert read its hedge replaced failed") : nil)
        }
        init(ticket: ExpertSlotTicket, ahead: Bool, parts: Int) {
            self.ticket = ticket
            self.ahead = ahead
            done = DispatchGroup()
            hedge = nil
            for _ in 0..<parts { done.enter() }
        }
        init(ticket: ExpertSlotTicket, losing hedge: ExpertReadReplica, done: DispatchGroup) {
            self.ticket = ticket
            ahead = false
            self.done = done
            self.hedge = hedge
            (started, demanded) = (true, true)
        }
    }

    /// Slots `first ..< first + count` of `slotBytes` each for `layers` layers, in slabs of up to `slabBytes`.
    struct Pool {
        let slotBytes: Int, first: Int, count: Int, initial: Int, perSlab: Int, layers: Int
        var initialSlabs: Int { (initial + perSlab - 1) / perSlab }
        // Preserve the initial last partial slab when reserving metadata for growth beyond the initial share.
        func slabRange(_ index: Int) -> Range<Int> {
            let start = index < initialSlabs ? index * perSlab : initial + (index - initialSlabs) * perSlab
            return start..<min(index < initialSlabs ? initial : count, start + perSlab)
        }
        /// The slab holding local slot `slot`.
        func slab(of slot: Int) -> Int { slot < initial ? slot / perSlab : initialSlabs + (slot - initial) / perSlab }
        func allocatedBytes(_ slots: Int) -> UInt64 {
            slots == 0 ? 0 : UInt64(slabRange(slab(of: slots - 1)).upperBound * slotBytes)
        }
    }
    /// Small enough that a lowered limit gives its bytes back (`setLimit` releases whole slabs).
    nonisolated(unsafe) package static var slabBytes = 64 << 20

    let weights: FamilyWeights
    let banks: [Int: Bank]
    /// Slots in all pools up to their limits, and their bytes.
    package var slotCount: Int { limits.reduce(0, +) }
    package var poolBytes: Int { zip(pools, limits).reduce(0) { $0 + $1.0.slotBytes * $1.1 } }
    private let slabWiring = FamilySlabWiring()
    package var wiring: FamilySlabWiring.Snapshot { slabWiring.snapshot }
    package var wiredBytes: Int { wiring.wiredBytes }
    private let pools: [Pool]
    /// The slots each pool may use (`setLimit`; all of them at first): slots past it hold nothing.
    private var limits: [Int]
    /// The bytes the limits were split from (`init`, `setLimit`).
    package private(set) var budget: UInt64
    private let maximumBudget: UInt64
    private let poolOf: [Int: Int]
    /// The slots, one size class per pool (key layer * experts + expert), and their victim choice. Slot `s` of pool `p`
    /// is the core's slot `s - pools[p].first` of class `p`.
    private let core: ExpertSlotCore<FamilyEviction>
    private var policy: FamilyEviction { core.policy }
    /// The tickets of the records each of the last fetches returned, oldest first, pinned until `releasePins`, and the
    /// replica leases each returned in place of a slot still read (`fetch`'s hedges), held as long.
    private var pins: [[ExpertSlotTicket]] = []
    private var replicaPins: [[ExpertReplicaLease]] = []
    /// A guess whose key leaves the lookup without its record being evicted (`forget`).
    private var forgetting: Int?
    /// The memory governor's ledger (`FamilyRunner.makeExperts`): each slab holds a reservation in it while allocated.
    package var ledger: MemoryLedger?
    private let wantsPrefetchObservations = ProcessInfo.processInfo.environment["ORIGAMI_FAMILY_PREFETCH_OBSERVATIONS"] == "1"
    private var prefetchObservations: FamilyPrefetchObservations?
    private var observationAdmissionRefused = false
    package var observationReservedBytes: UInt64 { prefetchObservations?.reservedBytes ?? 0 }
    package var observationNotes: [String: Any] {
        var notes =
            prefetchObservations?.snapshot ?? [
                "experimental": true, "enabled": false, "requested": wantsPrefetchObservations,
                "admission_refused": observationAdmissionRefused, "reserved_bytes": 0,
            ]
        notes["early_submission"] = [
            "batches": counters.earlyBatches, "with_outstanding_demand": counters.earlyOutstandingBatches,
            "demand_bytes_at_submission": counters.earlyDemandBytes,
            "hot_demand_bytes_at_submission": counters.earlyDemandHotBytes,
        ]
        if wantsPrefetchObservations { notes["hedge_trace"] = hedgeDecisions }
        notes["early_submission_scope"] =
            "cumulative; pending logical bytes by planned source at callback; fetch competitor bins remain entrance snapshots"
        return notes
    }
    /// Called once after fixed startup owners, while the mandatory expert floor is still lazy. The bounded
    /// diagnostic owner persists until model unload, including pressure, so previously collected counts survive.
    package func admitPrefetchObservations(enabled: Bool? = nil) {
        guard enabled ?? wantsPrefetchObservations, prefetchObservations == nil else { return }
        guard let ledger else {
            observationAdmissionRefused = true
            return
        }
        let g = weights.geometry
        prefetchObservations = FamilyPrefetchObservations(
            layers: g.layers, experts: g.experts, ranks: g.expertsPerToken,
            ledger: ledger, retainingBytes: leastAllocatedBytes)
        observationAdmissionRefused = prefetchObservations == nil
    }
    package func abandonPrefetchObservation() { prefetchObservations?.abandonPrediction() }
    /// No settle, promotion or wait is introduced. A just-completed valid guess counts as ready even before land;
    /// an incomplete read counts in I0. Failed/cancelled completed guesses count in neither set.
    private func observationResidency(layer: Int) -> (ready: [Int], inFlight: [Int]) {
        let e = weights.geometry.experts
        var ready: [Int] = []
        var reading: [Int] = []
        ready.reserveCapacity(e)
        reading.reserveCapacity(e)
        for expert in 0..<e {
            let key = layer * e + expert
            guard entry(key) != nil else { continue }
            if let guess = inFlight[key] {
                let complete = guess.done.wait(timeout: .now()) == .success
                guessLock.lock()
                let valid = guess.failure == nil && !guess.cancelled
                guessLock.unlock()
                guard valid else { continue }
                if complete { ready.append(expert) } else { reading.append(expert) }
            } else {
                ready.append(expert)
            }
        }
        return (ready, reading)
    }
    /// Observation only: do not land/cancel guesses or count already scheduled reads as avoidable traffic.
    /// Experts of `layer` whose records are in memory now, without settling or changing any read state.
    package func observedReadyExperts(layer: Int) -> Set<Int> { Set(observationResidency(layer: layer).ready) }
    package func observedAvailableOrPendingExperts(layer: Int) -> Set<Int> {
        let observed = observationResidency(layer: layer)
        return Set(observed.ready + observed.inFlight)
    }
    package func observeRoutedDemand(layer: Int, needed: [Int]) {
        guard let prefetchObservations else { return }
        let current = observationResidency(layer: layer)
        prefetchObservations.routed(layer: layer, needed: needed, ready: current.ready, inFlight: current.inFlight)
    }
    private func observationCompetitors(layer: Int, needed: [Int]) -> (queued: Int, running: Int) {
        let e = weights.geometry.experts
        var queued = 0
        var running = 0
        guessLock.lock()
        defer { guessLock.unlock() }
        for (key, guess) in inFlight
        where !guess.ahead && !(key / e == layer && needed.contains(key % e)) && !guess.demanded && !guess.cancelled {
            guard guess.done.wait(timeout: .now()) != .success else { continue }
            if guess.started { running += 1 } else { queued += 1 }
        }
        return (queued, running)
    }
    /// `key`'s slot (numbered across pools) and phase, if the lookup has it (reserved and being read included).
    private func entry(_ key: Int) -> (slot: Int, phase: ExpertSlotPhase)? {
        core.lock.lock()
        defer { core.lock.unlock() }
        guard let i = core.lookup[key] else { return nil }
        let p = poolOf[key / weights.geometry.experts]!
        return (pools[p].first + i, core.classes[p].slots[i]!.phase)
    }
    /// Generation fetch uses per key, including one demand per unique expert in an explicit verification block.
    /// These are record demands, not a count of committed tokens; hot-copy selection uses this profile.
    package private(set) var uses: [Int: Int] = [:]
    /// Reuse profiles are collected only at safe boundaries, from the existing decode counters. The hot path adds
    /// one count per fetch and reuses its existing read timer; no ranking or repartitioning runs during a fetch.
    private var decodeAccesses = 0, profiledAccesses = 0
    private var profiledUses: [Int: Int] = [:], demand: [Int: Double] = [:]
    private var allocationValues: [[Double]]?
    private var readCost: [Double]
    /// `body` under the core's lock.
    private func locked<T>(_ body: () throws -> T) rethrows -> T {
        core.lock.lock()
        defer { core.lock.unlock() }
        return try body()
    }
    /// Misses per key in quarters (4 per decode miss, 1 per prompt miss: a prompt slice reads each expert once for up
    /// to 2,048 rows, a decode miss costs a token's wait), what a hot copy is chosen from first: a record that is
    /// always resident gains nothing from a copy.
    package private(set) var missed: [Int: Int] = [:]
    /// Records also held on another drive, read from either (`copyShare`).
    private var hot: HotCopySet?
    /// Where hedged reads land (`fetch`): a token's experts of the largest record (at least two), allocated and charged
    /// to `ledger` at the first hedge, not before.
    package private(set) lazy var replicaArena: ExpertReplicaArena? = try? ExpertReplicaArena(
        ledger: ledger ?? MemoryLedger(), device: weights.device, records: max(2, weights.geometry.expertsPerToken),
        recordBytes: banks.values.map(\.record).max()!, lazily: true)
    /// A hedge never fires sooner than this (seconds): several Mach scheduler quanta (10 ms each), so a small record's
    /// read on a busy machine (a thread not scheduled yet, another process's I/O) is not taken for a stall, while the
    /// stalls a hedge is for (a throttling, sleeping or failing drive) last hundreds of milliseconds or more.
    package static let hedgeFloorSeconds = 0.05
    /// Each drive's current and healthy rate beside the moving averages (`FamilyDriveRates`).
    private let driveRates = FamilyDriveRates()
    /// The GGUF's drive: how much of its demand reading went far below its own healthy rate (`origami refresh`).
    package let modelDriveReads = SlowDriveReads()
    /// The last 256 hedge decisions, in the notes with ORIGAMI_FAMILY_PREFETCH_OBSERVATIONS=1.
    private let hedgeTrace = FamilyHedgeTrace()
    package var hedgeDecisions: [[String: Any]] { hedgeTrace.snapshot }
    /// Checks only: runs at the start of every demand read and hedge replica of `fetch`, with its drive and whether it is
    /// the replica; what it throws is that read's error.
    package var readProbe: (@Sendable (IOSource, Bool) throws -> Void)?
    /// Checks: runs on the reader thread before each part of a guessed or warm-up record (its key) is read (a hung
    /// drive, `ExpertReadStall`).
    package var backgroundProbe: (@Sendable (Int) -> Void)?
    /// Each drive's read rate (bytes per second), smoothed over the calls that read from it: from `seed` (this Mac's
    /// drive measurements), else equal, until reads are observed.
    private var bandwidth = (hot: 2e9, gguf: 2e9)
    private var seeded = (hot: 2e9, gguf: 2e9)
    /// Where the copy's moving average starts and an idle copy settles back to (`startCopyRate`): the attached copies'
    /// places summed when on several drives, else `seeded.hot`.
    private var copyStart = 2e9
    /// Each copy place's measured rate, by its copy's folder (`seed`).
    private var placeSeeds: [String: Double] = [:]
    /// Where the moving averages start (and `attach` restarts the copy's): this Mac's measured rates (`DriveSpeed`) of the
    /// copy's drive and the model's (the GGUF's), nil keeps the equal default; `places`, each copy place's rate by its
    /// copy's folder: copies on several drives start at the sum of theirs, since the split reads from all of them at once
    /// (each record then from the place expected to finish it first, `HotCopySet.pick`). Before serving starts.
    package func seed(copy: Double?, model: Double?, places: [URL: Double] = [:]) {
        placeSeeds = Dictionary(places.map { ($0.key.standardizedFileURL.path, $0.value) }) { a, _ in a }
        seeded = (copy ?? seeded.hot, model ?? seeded.gguf)
        bandwidth = seeded
        startCopyRate()
        refreshStorageCosts()
    }
    /// Restarts the copy's moving average at `copyStart`, from the attached copies and the seeds.
    private func startCopyRate() {
        let stores = hot?.stores ?? []
        let rates = stores.compactMap { placeSeeds[$0.folder.standardizedFileURL.path] }
        copyStart = stores.count > 1 && rates.count == stores.count ? rates.reduce(0, +) : seeded.hot
        bandwidth.hot = copyStart
    }
    /// The moving averages now, for comparing serving's view with the measurements (`DriveSpeedReview`).
    package var driveBandwidth: (copy: Double, model: Double) { (bandwidth.hot, bandwidth.gguf) }
    /// The copies reads come from now, with each drive's lane (`HotCopySet.laneRates`).
    package var hotCopySet: HotCopySet? { hot }
    private var readLatency = (hot: 0.0, gguf: 0.0)
    private var unservedHotBatches = 0, unservedGGUFBatches = 0
    private var inDemandSubmission = false
    package private(set) var counters = Counters()
    /// Guessed records still being read, by key; their slots are reserved or being read, never a victim. `unread`: guessed records no
    /// fetch has asked for yet (true: read ahead). `guessed`: the last guess, scored by the next decode fetch of its layer
    /// (`issued`: the rank of each expert whose read it began).
    private var inFlight: [Int: Guess] = [:]
    // Keep tickets beside guesses, not inside them: a queued job retains its Guess until its completion leaves the
    // group, and a ticket never retains its job, so a cache destroyed with queued work leaves no cycle.
    private var guessTickets: [Int: [IOTicket]] = [:]
    private var unread: [Int: Bool] = [:]
    /// The drive each unasked decode guess was planned on (`Counters.GuessDrive`).
    private var guessDrive: [Int: Int] = [:]
    /// The last decode router result (uptime ns) and whether its layer's first demand read is still to be timed.
    private var routerAt: UInt64 = 0, routerAwaitsDemand = false
    /// The serving thread has a decode layer's router result (counters D and E time from it).
    package func routerResultKnown() {
        routerAt = DispatchTime.now().uptimeNanoseconds
        routerAwaitsDemand = true
    }
    /// The next request's longest router-to-demand time starts from zero.
    package func resetRequestMaximum() { counters.routerToDemandMaximum = 0 }
    /// Counter A: every drive's busy and idle time and bytes by what the serving thread does (`DriveActivity`); lane 0
    /// the model's drive (its scheduler queue), then each copy place.
    package let driveActivity = DriveActivity(lanes: 1)
    /// Counter C: the split rule's estimates now (point values, not sums).
    package var splitPlannerNotes: [String: Double] {
        let planned = plannedBandwidth
        return [
            "split.copy_rate_bytes_per_second": bandwidth.hot, "split.model_rate_bytes_per_second": bandwidth.gguf,
            "split.copy_latency_seconds": readLatency.hot, "split.model_latency_seconds": readLatency.gguf,
            "split.copy_planned_rate_bytes_per_second": planned.hot, "split.model_planned_rate_bytes_per_second": planned.gguf,
        ]
    }
    private var guessed: (layer: Int, experts: Set<Int>, ranked: [Int], absent: Set<Int>, issued: [Int: Int], began: UInt64)?
    private var prefetchPolicy = FamilyPrefetchPolicy()
    /// Checks only: false issues every guessed read (`FamilyPrefetchPolicy` skips none), for fixtures that count them.
    package var adaptiveGuesses = true
    /// Reads a decode guess may begin, its best non-resident experts first (nil: all; ORIGAMI_FAMILY_PREFETCH_DEPTH).
    package var guessDepth = ProcessInfo.processInfo.environment["ORIGAMI_FAMILY_PREFETCH_DEPTH"].flatMap { Int($0) }.map { max(0, $0) }
    private let guessLock = NSLock()
    /// Checks only: runs in `settle` after it kept a guess and before it looks whether its read ended, the window in
    /// which a queued read may still cancel itself.
    package var settleProbe: (() -> Void)?
    /// Where every expert read runs (`makeReadScheduler`), with room for a fetch of every expert of a layer as three GGUF
    /// reads: demand first, guesses (a GGUF record as its three parts) at prefetch priority, as many at once per drive
    /// as a token has experts, the rest queued (and cancellable). Its job metadata is charged to a ledger of its own,
    /// not `ledger`, whose owners the budget checks count.
    package let reads: IOScheduler
    /// A guessed record's heat until a fetch asks for it: the first to go if none does.
    private static let guessHeat = 0x1p-20

    /// `budget`: bytes for expert slots (at least one layer's worth of experts per token).
    package init(weights: FamilyWeights, budget: UInt64) throws {
        self.weights = weights
        let g = weights.geometry
        let banks = try Self.banks(weights)
        self.banks = banks
        // 16 KiB alignment keeps every record page-aligned for the reads. A pool's share of the budget is its layers'
        // share of the slot bytes one slot per layer takes, so one record size gives budget / size slots, as before.
        let sizeOf = banks.mapValues(Self.slotBytes)
        let total = UInt64(sizeOf.values.reduce(0, +))
        let sizes = Set(sizeOf.values).sorted()
        let least = 2 * g.expertsPerToken
        let minimumBytes = sizes.reduce(UInt64(0)) { $0 + UInt64(least * $1) }
        startBytes = Self.startBytes(sizeOf, perToken: g.expertsPerToken)
        guard budget >= startBytes else {
            throw EngineError(
                "\(OrigamiPlan.gigabytes(budget)) is left for experts and they need at least \(OrigamiPlan.gigabytes(startBytes))",
                code: "insufficient_memory")
        }
        var pools: [Pool] = []
        var poolOf: [Int: Int] = [:]
        var first = 0
        var initial: [Int] = []
        for size in sizes {
            let layers = sizeOf.filter { $0.value == size }.keys
            let count = Self.share(budget, layers: layers.count, total: total)
            // Only metadata is sized for a larger share. Buffers are still allocated lazily, within `limits`.
            // Keep the other pools' two waves and never reserve extra metadata beyond this pool's distinct experts.
            let room = (budget - (minimumBytes - UInt64(least * size))) / UInt64(size)
            let capacity = max(count, min(layers.count * g.experts, Int(clamping: room)))
            for layer in layers { poolOf[layer] = pools.count }
            pools.append(
                Pool(
                    slotBytes: size, first: first, count: capacity, initial: count,
                    perSlab: max(1, min(count, Self.slabBytes / size)), layers: layers.count))
            initial.append(count)
            first += capacity
        }
        self.pools = pools
        self.poolOf = poolOf
        self.budget = budget
        maximumBudget = budget
        limits = initial
        readCost = Array(repeating: 0, count: pools.count)
        // Accesses over which a slot's heat halves: about four passes over the routed experts of every layer.
        let policy = FamilyEviction(
            first: pools.map(\.first), slotBytes: pools.map(\.slotBytes), slots: first, experts: g.experts,
            perToken: g.expertsPerToken, halfLife: Double(4 * g.expertsPerToken * banks.count))
        // The pools allocate, charge and wire their own slabs (`allocateLocked`); the core's ledger stays empty.
        let layers = banks.keys.max()! + 1
        core = try ExpertSlotCore(
            device: weights.device, ledger: MemoryLedger(),
            shape: ExpertSlotShape(
                layers: layers, expertsPerLayer: g.experts,
                classes: pools.map {
                    .init(
                        recordBytes: $0.slotBytes, slotsPerSlab: $0.perSlab, byteLimit: .max, maximumByteLimit: .max,
                        category: "family-experts", ownerSized: true)
                },
                layerClasses: (0..<layers).map { poolOf[$0] ?? 0 }),
            policy: policy, observer: policy)
        reads = try Self.makeReadScheduler(ledger: MemoryLedger(), jobs: 3 * g.experts, guesses: g.expertsPerToken)
        let activity = driveActivity
        reads.observeRuns { source, bytes in
            guard source == .externalSSD else { return }  // copy reads count per place, in `HotCopySet`
            if let bytes { activity.ended(0, bytes: bytes) } else { activity.began(0) }
        }
        policy.forgetting = { [unowned self] key in forgot(key) }
        policy.allocate = { [unowned self] p, slot in try allocateLocked(pool: p, through: slot - self.pools[p].first) }
    }

    /// The routed-expert layers' banks: each layer's gate, up and down tensors, checked against the geometry.
    static func banks(_ weights: FamilyWeights) throws -> [Int: Bank] {
        let g = weights.geometry
        var banks: [Int: Bank] = [:]
        for layer in g.moeLayers {
            let tensors = try ["gate", "up", "down"].map { try weights.require(["blk.{L}.ffn_\($0)_exps.weight"], layer: layer) }
            // The kernels read each expert as gate/up (expertWidth × hidden) and down (hidden × expertWidth).
            let shapes = [[g.expertWidth, g.hidden], [g.expertWidth, g.hidden], [g.hidden, g.expertWidth]].map { [g.experts] + $0 }
            let bytes = try zip(tensors, shapes).map { tensor, shape -> Int in
                guard tensor.shape == shape else {
                    throw EngineError(
                        "\(tensor.name) has shape \(tensor.shape); the model's geometry needs \(shape)", code: "model_incompatible")
                }
                guard tensor.length % Int64(g.experts) == 0 else {
                    throw EngineError("\(tensor.name) does not hold \(g.experts) experts", code: "model_incompatible")
                }
                return Int(tensor.length) / g.experts
            }
            banks[layer] = Bank(tensors: tensors, types: tensors.map(\.type), bytes: bytes)
        }
        guard !banks.isEmpty else {
            throw EngineError("This \(g.family.displayName) GGUF has no routed-expert layers", code: "model_incompatible")
        }
        return banks
    }
    private static func slotBytes(_ bank: Bank) -> Int { (bank.record + 16383) / 16384 * 16384 }
    /// The least budget the pools start with (`init`): every pool's share (`share`) holds two layers' experts per token,
    /// so the pool with the fewest layers sets it: two waves × the slot bytes of one slot per layer ÷ its layer count.
    /// More than two waves of every pool's records whenever the pools have unequal layer counts.
    static func startBytes(_ sizeOf: [Int: Int], perToken: Int) -> UInt64 {
        let total = UInt64(sizeOf.values.reduce(0, +))
        let fewest = UInt64(Dictionary(grouping: sizeOf.values) { $0 }.values.map(\.count).min() ?? 1)
        return (UInt64(2 * perToken) * total + fewest - 1) / fewest
    }
    /// `startBytes` for `weights`' experts, before a pool is made (`FamilyRunner.makeExperts` checks the limit with it).
    package static func startBytes(_ weights: FamilyWeights) throws -> UInt64 {
        startBytes(try banks(weights).mapValues(slotBytes), perToken: weights.geometry.expertsPerToken)
    }
    /// The least budget these pools started with (`startBytes`).
    package let startBytes: UInt64

    /// A pool's share of `budget`: its layers' share of the bytes one slot per layer takes (`total`).
    private static func share(_ budget: UInt64, layers: Int, total: UInt64) -> Int {
        Int(budget / total * UInt64(layers) + budget % total * UInt64(layers) / total)
    }

    /// Gives each pool its learned share of `bytes` (equal slots per layer until profiled), between forward passes:
    /// at least two layers' experts per token, at most its capacity. The state lent to the slots taken back (`FamilyRunner.reserveState`)
    /// or returned. Lowering cancels guesses not begun and waits for the others, forgets the records in slots past a
    /// limit (an unasked guess counts as wasted) and releases the slabs wholly past it; raising allocates nothing (slabs
    /// fill as slots are taken).
    package func setLimit(bytes: UInt64, why: String? = nil) {
        let before = UInt64(residentBytes)
        budget = bytes
        applyLimits(allocation(bytes: bytes))
        let after = UInt64(residentBytes)
        if let why, after < before {
            shrank = "The expert cache went from \(OrigamiPlan.gigabytes(before)) to \(OrigamiPlan.gigabytes(after)): \(why)."
        }
    }
    /// Why the cache last gave records back (`setLimit`'s `why`), one plain line for status; nil until it has.
    package private(set) var shrank: String?

    /// Reconsider the split at a request boundary, with no GPU use of slots or concurrent fetch. Eight decode passes
    /// (at least 256 expert uses) amortize the profile scan; a 5% gain above the mandatory floor avoids tiny changes.
    /// Profiles decay once per collection, so a new workload can supersede old demand without per-token maintenance.
    package func rebalance() {
        refreshStorageCosts()
        guard pools.count > 1, decodeAccesses - profiledAccesses >= max(256, banks.count * weights.geometry.expertsPerToken * 8) else {
            return
        }
        profiledAccesses = decodeAccesses
        var ranked = Array(repeating: [Double](), count: pools.count)
        var total = 0.0
        for (key, count) in uses {
            let value = (demand[key] ?? 0) * 0.5 + Double(count - (profiledUses[key] ?? 0))
            demand[key] = value
            profiledUses[key] = count
            if let p = poolOf[key / weights.geometry.experts] {
                ranked[p].append(value * locked { policy.storageMultiplier(key) })
                total += value
            }
        }
        // A small prior leaves unseen experts eligible, including a pool not yet touched in the profile window.
        let prior = total / Double(banks.count * weights.geometry.experts) * 0.05
        allocationValues = pools.indices.map { p in
            let pool = pools[p]
            let cost = readCost[p] > 0 ? readCost[p] : Double(pool.slotBytes) / bandwidth.gguf
            let values = ranked[p].map { max(prior, $0) }.sorted(by: >)
            return (0..<pool.count).map { i in
                let uses = i < values.count ? values[i] : (i < pool.layers * weights.geometry.experts ? prior : 0)
                return uses * cost / Double(pool.slotBytes)
            }
        }
        let next = allocation(bytes: budget)
        guard next != limits, let values = allocationValues else { return }
        func utility(_ counts: [Int]) -> Double {
            pools.indices.reduce(0) { sum, p in
                sum + values[p].prefix(counts[p]).dropFirst(2 * weights.geometry.expertsPerToken).reduce(0, +) * Double(pools[p].slotBytes)
            }
        }
        guard utility(next) > utility(limits) * 1.05 else { return }
        applyLimits(next)
    }

    private func allocation(bytes: UInt64) -> [Int] {
        Self.poolLimits(
            bytes: min(bytes, maximumBudget), sizes: pools.map(\.slotBytes), layers: pools.map(\.layers),
            capacities: pools.map(\.count), minimum: 2 * weights.geometry.expertsPerToken, values: allocationValues,
            allocatedBytes: { self.pools[$0].allocatedBytes($1) })
    }

    /// Marginal saved read time per byte, highest-demand expert first in each pool. Greedy whole-slot allocation
    /// charges the slabs a slot needs, respects each pool's two-wave floor, and retains that floor below its budget.
    /// With no profile, preserve equal slots per layer, trimming any rounding/floor excess from the fullest pool.
    package static func poolLimits(
        bytes: UInt64, sizes: [Int], layers: [Int], capacities: [Int], minimum: Int,
        values: [[Double]]? = nil, allocatedBytes: ((Int, Int) -> UInt64)? = nil
    ) -> [Int] {
        func charge(_ pool: Int, _ slots: Int) -> UInt64 { allocatedBytes?(pool, slots) ?? UInt64(sizes[pool] * slots) }
        var result = capacities.map { min(minimum, $0) }
        let floor = sizes.indices.reduce(UInt64(0)) { $0 + charge($1, result[$1]) }
        let budget = max(bytes, floor)
        let slotFloor = sizes.indices.reduce(UInt64(0)) { $0 + UInt64(sizes[$1] * result[$1]) }
        let slotBudget = max(bytes, slotFloor)
        if let values {
            var remaining = budget - floor
            var slotRemaining = slotBudget - slotFloor
            while true {
                var best: Int?
                var score = -Double.infinity
                var addedBytes: UInt64 = 0
                for p in sizes.indices where result[p] < capacities[p] && UInt64(sizes[p]) <= slotRemaining {
                    let added = charge(p, result[p] + 1) - charge(p, result[p])
                    guard added <= remaining else { continue }
                    let candidate = values[p][result[p]]
                    if candidate > score {
                        best = p
                        score = candidate
                        addedBytes = added
                    }
                }
                guard let p = best else { break }
                result[p] += 1
                remaining -= addedBytes
                slotRemaining -= UInt64(sizes[p])
            }
        } else {
            let total = sizes.indices.reduce(UInt64(0)) { $0 + UInt64(sizes[$1] * layers[$1]) }
            guard total > 0 else { return result }
            for p in sizes.indices { result[p] = min(capacities[p], max(result[p], share(slotBudget, layers: layers[p], total: total))) }
            var held = sizes.indices.reduce(UInt64(0)) { $0 + charge($1, result[$1]) }
            var slotHeld = sizes.indices.reduce(UInt64(0)) { $0 + UInt64(sizes[$1] * result[$1]) }
            while held > budget || slotHeld > slotBudget {
                guard
                    let p = sizes.indices.filter({ result[$0] > minimum }).max(by: {
                        Double(result[$0]) / Double(layers[$0]) < Double(result[$1]) / Double(layers[$1])
                    })
                else { break }
                let before = charge(p, result[p])
                result[p] -= 1
                held -= before - charge(p, result[p])
                slotHeld -= UInt64(sizes[p])
            }
        }
        return result
    }

    private func applyLimits(_ next: [Int]) {
        guard next != limits else { return }
        settle(keeping: [])
        for (key, guess) in inFlight {
            // A read that does not return keeps its slot: the pool keeps its size until it lands (a later call).
            guard landed(key, guess) else { return }
            _ = land(key)
        }
        // Callers have joined every GPU consumer of the slots.
        releasePins(keeping: 0)
        var retired: [ExpertSlabStorage] = []
        for (p, pool) in pools.enumerated() {
            let previous = limits[p]
            limits[p] = next[p]
            if limits[p] < previous {
                locked {
                    for s in (pool.first + limits[p])..<(pool.first + previous) where policy.owner(s, pool: p, core) != nil {
                        core.freeLocked(s - pool.first, sizeClass: p)
                        policy.scan[s] = false
                    }
                }
            }
            locked {
                // The slabs wholly past the limit, the last first.
                policy.retireFrom[p] = limits[p]
                core.classes[p].byteLimit = pool.allocatedBytes(limits[p])
                core.retireSlabsLocked(sizeClass: p, maximumSlabs: .max, retired: &retired)
            }
        }
        // Storage unwires only successful wires. The reservation follows the Metal buffer's final owner.
        retired.removeAll()
    }

    /// The bytes of the fewest slots a limit leaves: two layers' experts per token in every pool.
    package var leastBytes: UInt64 { UInt64(pools.reduce(0) { $0 + 2 * weights.geometry.expertsPerToken * $1.slotBytes }) }
    /// The physical slabs required by the two-wave minimum, including each pool's slab rounding.
    package var leastAllocatedBytes: UInt64 {
        pools.reduce(0) { $0 + $1.allocatedBytes(min($1.count, 2 * weights.geometry.expertsPerToken)) }
    }
    /// The slots of pool `p` up to its limit.
    private func usable(_ p: Int) -> Range<Int> { pools[p].first..<(pools[p].first + limits[p]) }

    package var residentBytes: Int { locked { core.classes.reduce(0) { $0 + Int($1.allocated) } } }
    /// The record sizes of the pools, for logs ("13 MiB", "11–18 MiB").
    package var slotSizes: String {
        let mib = pools.map { $0.slotBytes / 1_048_576 }
        return mib.min() == mib.max() ? "\(mib[0]) MiB" : "\(mib.min()!)–\(mib.max()!) MiB"
    }
    /// The slots `layer`'s experts can use.
    package func slots(layer: Int) -> Int { poolOf[layer].map { limits[$0] } ?? 0 }

    /// Physical slots already owned, capped by the current logical limit. Pressure may inhibit
    /// additional slabs without changing that limit; a prompt must not promise those absent slots.
    package func allocatedSlots(layer: Int) -> Int {
        guard let p = poolOf[layer] else { return 0 }
        return min(locked { policy.allocatedCount(p, core) }, limits[p])
    }

    /// Admit the current and next prompt wave before either consumes slots. Allocation can stop
    /// partway if pressure changes; only the resulting physical capacity determines the wave size.
    /// No expert is evicted/stamped or read here. Callers join prior fetches and apply trims first.
    /// Never request an unused second wave when every remaining expert fits in the first one.
    package func prepareWave(layer: Int, maximumExperts: Int) throws -> Int {
        guard maximumExperts > 0, let p = poolOf[layer] else {
            throw EngineError("Cannot prepare an empty or unknown expert wave")
        }
        let logical = min(maximumExperts, max(1, limits[p] / 2))
        let needed = min(maximumExperts, 2 * logical)
        var denied: Error?
        func allocate() throws {
            do {
                try locked { try allocateLocked(pool: p, through: needed - 1) }
                denied = nil
            } catch let error as EngineError where error.code == "insufficient_memory" {
                denied = error
            }
        }
        if allocatedSlots(layer: layer) < needed { try allocate() }
        // A denial that left no slot at all waits once for the slab to fit (`slabMayFit`) and tries again.
        if denied != nil, allocatedSlots(layer: layer) == 0, slabMayFit(pool: p) { try allocate() }
        let available = allocatedSlots(layer: layer)
        let wave = min(logical, available >= maximumExperts ? available : available / 2)
        guard wave > 0 else {
            throw denied ?? EngineError("Memory cannot hold the required expert wave", code: "insufficient_memory")
        }
        return wave
    }

    /// A denied slab's backpressure (ROADMAP R5): while the charged owners leave room for pool `p`'s next slab but the
    /// footprint does not (pages just released and not yet reclaimed), waits for them (`ReclamationWait`, about 1 s).
    /// Whether the slab fits now; a denial that waiting cannot lift (owners past the limit) returns at once.
    private func slabMayFit(pool p: Int) -> Bool {
        guard let ledger else { return false }
        let pool = pools[p]
        let bytes = UInt64(locked { pool.slabRange(core.classes[p].slabs.count).count } * pool.slotBytes)
        guard bytes > 0, ledger.physicalAllocationShortfall(bytes: bytes, footprint: 0) == 0 else { return false }
        func fits() -> Bool { ledger.physicalAllocationShortfall(bytes: bytes) == 0 }
        if !fits() { try? ReclamationWait.wait(shouldContinue: { true }) { fits() } }
        return fits()
    }

    /// Allocates pool `p`'s slabs, in order, through its slot `local` (under the core's lock).
    private func allocateLocked(pool p: Int, through local: Int) throws {
        let pool = pools[p]
        let last = pool.slab(of: local)
        while core.classes[p].slabs.count <= last {
            let range = pool.slabRange(core.classes[p].slabs.count)
            let slots = range.count
            guard range.lowerBound < 2 * weights.geometry.expertsPerToken || ledger?.optionalGrowthAllowed != false else {
                throw EngineError("Expert cache growth is temporarily inhibited by memory pressure", code: "insufficient_memory")
            }
            // Admission precedes allocation. A hard cap can never be bypassed by holding an uncharged slab.
            let reservation = try ledger?.reservePhysical(bytes: UInt64(slots * pool.slotBytes), category: "family-experts")
            let allocationStarted = ProcessInfo.processInfo.systemUptime
            let allocatedBuffer = autoreleasepool(invoking: {
                weights.device.makeBuffer(length: slots * pool.slotBytes, options: .storageModeShared)
            })
            counters.allocationSeconds += ProcessInfo.processInfo.systemUptime - allocationStarted
            guard let buffer = allocatedBuffer else {
                if let reservation { try? ledger?.release(reservation) }
                throw EngineError("Out of memory for expert slots", code: "insufficient_memory")
            }
            let storage = FamilySlabStorage(buffer: buffer, ledger: ledger, reservation: reservation, wiring: slabWiring)
            let first = core.insertSlabLocked(ExpertSlabStorage(buffer: buffer, retaining: storage), sizeClass: p, slots: slots)
            // Slabs come and go at the end, so the core numbers a slab's slots as the pool does.
            assert(first == range.lowerBound, "Expert slab out of order")
        }
    }

    private func describe(_ storage: ExpertSlabStorage, _ base: Int, layer: Int) -> Slot {
        let bank = banks[layer]!
        return Slot(
            buffer: storage.buffer, offsets: [base, base + bank.bytes[0], base + bank.bytes[0] + bank.bytes[1]], types: bank.types)
    }
    /// The ticket of `key`'s record, if the lookup has it.
    private func ticket(_ key: Int) -> ExpertSlotTicket? {
        locked {
            guard let i = core.lookup[key] else { return nil }
            let c = core.sizeClass(ofKey: key)
            return ExpertSlotTicket(sizeClass: c, slot: i, generation: core.classes[c].slots[i]!.generation, key: key)
        }
    }
    /// Pins `key`'s verified record for the current fetch (released by `releasePins`): where it is.
    private func pin(_ key: Int) throws -> (storage: ExpertSlabStorage, offset: Int) {
        guard let ticket = ticket(key) else { throw EngineError("Expert record left the cache") }
        return try pin(ticket)
    }
    private func pin(_ ticket: ExpertSlotTicket) throws -> (storage: ExpertSlabStorage, offset: Int) {
        let at = try core.pin(ticket)
        pins[pins.count - 1].append(ticket)
        return at
    }
    /// Unpins the records all but the last `keeping` fetches returned: their GPU work has finished.
    private func releasePins(keeping: Int) {
        guard pins.count > keeping else { return }
        locked {
            while pins.count > keeping {
                for ticket in pins.removeFirst() { try? core.unpinLocked(ticket) }
            }
        }
        replicaPins.removeFirst(max(0, replicaPins.count - keeping))
    }

    package func bank(_ layer: Int) -> (types: [Int], bytes: [Int])? { banks[layer].map { ($0.types, $0.bytes) } }
    /// Reads from `copy` the records it holds from now on (nil: none). Only while no fetch, prefetch or warm-up runs
    /// (`FamilyServingBackend.attachHotCopy`); reads in flight keep the copy they started with.
    package func attach(_ copy: HotCopy?) { attach(copies: copy.map { [$0] } ?? []) }
    /// Reads from `copies` (one store per drive, `HotCopySet`) the records they hold from now on; their drives share the
    /// copy's queue and rate. Only while no fetch, prefetch or warm-up runs.
    package func attach(copies: [HotCopy]) {
        driveActivity.setLanes(1 + copies.count)
        hot = copies.isEmpty ? nil : HotCopySet(copies, activity: (driveActivity, 1))
        startCopyRate()
        readLatency.hot = 0
        driveRates.reset(.internalSSD)
        unservedHotBatches = 0
        unservedGGUFBatches = 0
        refreshStorageCosts()
    }

    /// Per layer, log2 of a copied record's read time over the original's (frozen at idle/request boundaries), and with
    /// it every slot's cost (`FamilyEviction.cost`). Membership belongs to the attached copy, less the records it rejected
    /// (read from the model's files until a repaired copy is attached); no I/O or reader-thread lock is taken by victim
    /// selection.
    private func refreshStorageCosts() {
        let costs: [Int: Double] =
            hot == nil
            ? [:]
            : banks.mapValues { bank in
                let original = readLatency.gguf + Double(bank.record) / bandwidth.gguf
                let copied = readLatency.hot + Double(bank.record) / bandwidth.hot
                return log2(min(1, max(0.125, copied / max(1e-9, original))))
            }
        let offsets = hot?.heldOffsets
        locked { policy.setStorage(hot: offsets, logCost: costs, core) }
    }
    /// What evicting `key`'s record costs relative to one read from the model's files (checks only).
    package func storageMultiplier(_ key: Int) -> Double { locked { policy.storageMultiplier(key) } }

    /// Each drive's rate to plan with: its moving average, or its current rate while it is stalled (`FamilyDriveRates`).
    private var plannedBandwidth: (hot: Double, gguf: Double) {
        (
            driveRates.rate(.internalSSD, average: bandwidth.hot, stored: copyStart),
            driveRates.rate(.externalSSD, average: bandwidth.gguf, stored: seeded.gguf)
        )
    }

    /// `copyShare` on `reads` with this cache's drive estimates; `probe` (demand fetches): the starvation probe.
    private func copyShare(held: Int, other: Int, record: Int, priority: IOPriority = .demand, probe: Bool = false) -> Int {
        var share = Self.copyShare(
            held: held, other: other, record: record, partJobs: 3, on: reads, priority: priority,
            bandwidth: plannedBandwidth, latency: readLatency)
        if probe && held > 0 {
            // One existing demand read, at most once in 64 eligible batches, refreshes a starved
            // path's observations. No duplicate bytes or background probe is added.
            unservedHotBatches = share == 0 ? unservedHotBatches + 1 : 0
            unservedGGUFBatches = other + held - share == 0 ? unservedGGUFBatches + 1 : 0
            if unservedHotBatches >= 64 {
                share = 1
                unservedHotBatches = 0
            } else if unservedGGUFBatches >= 64 {
                share = max(0, share - 1)
                unservedGGUFBatches = 0
            }
        }
        return share
    }

    /// How many of `held` missing records the hot copy holds to read from it when `other` missing records only the GGUF
    /// holds, `partJobs` reads each, go to the GGUF's drive first: each in turn goes to the drive `reads` picks
    /// (`IOScheduler.earliestSource`, its outstanding reads plus this batch's earlier ones), the one whose queue would
    /// finish first, latency + (queued + record) / bandwidth, ties to the copy. A record no drive admits goes to the GGUF.
    package static func copyShare(
        held: Int, other: Int, record: Int, partJobs: Int, on reads: IOScheduler, priority: IOPriority,
        bandwidth: (hot: Double, gguf: Double), latency: (hot: Double, gguf: Double)
    ) -> Int {
        let estimate: IOScheduler.Estimate = { source, bytes, _, _ in
            source == .internalSSD ? latency.hot + Double(bytes) / bandwidth.hot : latency.gguf + Double(bytes) / bandwidth.gguf
        }
        var pendingBytes: [IOSource: UInt64] = [.externalSSD: UInt64(other * record)]
        var pendingJobs: [IOSource: Int] = [.externalSSD: other * partJobs]
        var share = 0
        for _ in 0..<held {
            let source = try? reads.earliestSource(
                availableSources: [.internalSSD, .externalSSD], estimate: estimate, ties: .internalSSD, priority: priority,
                byteCount: record, pendingBytes: pendingBytes, pendingJobs: pendingJobs)
            let fromCopy = source == .internalSSD
            pendingBytes[fromCopy ? .internalSSD : .externalSSD, default: 0] += UInt64(record)
            pendingJobs[fromCopy ? .internalSSD : .externalSSD, default: 0] += fromCopy ? 1 : partJobs
            if fromCopy { share += 1 }
        }
        return share
    }

    /// One expert's record (gate, up and down back to back, as a slot holds it) read from the GGUF into `into`.
    package func readRecord(layer: Int, expert: Int, into: UnsafeMutableRawPointer) throws {
        guard let bank = banks[layer], (0..<weights.geometry.experts).contains(expert) else {
            throw EngineError("No routed expert \(expert) in layer \(layer)")
        }
        var at = 0
        for part in 0..<3 {
            try weights.file(bank.tensors[part].file).read(
                offset: bank.tensors[part].offset + Int64(expert * bank.bytes[part]),
                into: UnsafeMutableRawBufferPointer(start: into + at, count: bank.bytes[part]))
            at += bank.bytes[part]
        }
    }

    /// Decode waits for its reads by spinning on the serving thread (up to twice their expected time, then blocking):
    /// measured on the model-free bench with uncached reads from an external SSD (M2 Pro), a serving thread asleep
    /// while every read sleeps in the drive left the GPU about 1.9x slower on the next layers' work, and spinning kept
    /// it at full speed and woke sooner. Timing only: the same bytes land in the same slots. ORIGAMI_FAMILY_SPIN=0
    /// blocks, to measure another Mac; so does Low Power Mode (a core spinning through every read costs battery).
    nonisolated(unsafe) package static var spins = ProcessInfo.processInfo.environment["ORIGAMI_FAMILY_SPIN"].map { $0 != "0" } ?? true
    /// Checks only: reads at once on the GGUF's drive and on the hot copy's, in place of 64 (a check stalls a drive).
    nonisolated(unsafe) package static var checkReadLimits: (main: Int, hot: Int)?

    /// A scheduler for expert reads (the hot copy's on `internalSSD`, the GGUF's on `externalSSD`): 64
    /// reads at once per drive, of which guesses
    /// (prefetch priority) may take `guesses` (at least one is kept for demand), room for `jobs` demand jobs per drive
    /// (the most one fetch submits) and as many guess jobs, and no byte bound a demand job could meet. Read hygiene is
    /// on for its prefetch holds and queued-only cancellation. Its job metadata (2 KiB per admitted job) is reserved
    /// in `ledger`.
    package static func makeReadScheduler(ledger: MemoryLedger, jobs: Int, guesses: Int = 0) throws -> IOScheduler {
        func limits(_ concurrent: Int) -> IOScheduler.Limits {
            let demand = min(32_768, max(64, jobs))
            return IOScheduler.Limits(
                maximumConcurrent: concurrent, reservedDemandConcurrent: max(1, concurrent - max(0, guesses)),
                maximumOutstanding: 2 * demand, reservedDemandOutstanding: demand,
                maximumOutstandingBytes: 1 << 62, reservedDemandBytes: 1 << 61)
        }
        return try IOScheduler(
            ledger: ledger, internalLimits: limits(checkReadLimits?.hot ?? 64), externalLimits: limits(checkReadLimits?.main ?? 64),
            readHygiene: true)
    }

    /// A batch's hedged reads (`fetch`, `FamilyHedges`): `perform` gives each job's ticket to `submitted` and its
    /// outcome to `ended` (in its completion), waits on `wake` (signalled by every job and every replica) and at each
    /// wake asks `poll` to fire the hedges due: it says whether every record has its bytes and when the next hedge is
    /// due. A job `left` says belongs to a primary that lost to its replica is not joined (its owner joins it before its
    /// slot is reused).
    package struct Hedging {
        let wake: DispatchSemaphore
        let submitted: (Int, IOTicket) -> Void
        let ended: @Sendable (Int, IOOutcome) -> Void
        let poll: () -> (settled: Bool, next: UInt64?)
        let left: (Int) -> Bool
    }

    /// `body` for 0 ..< count as demand jobs on `scheduler` (job i on `source(i)`, `bytes(i)` long), then `submitted` and
    /// `meanwhile` on this thread; then waits, spinning up to `spin` seconds before it blocks. Every submitted job is
    /// joined before it returns or throws (but those
    /// `hedging` leaves): a refused submission, or a job that did not run, throws once the others ended. With `hedging`
    /// the wait also ends once every record has its bytes, and a blocked wait wakes when the next hedge is due. Nothing
    /// runs for an empty batch but `meanwhile`. With `stall` (seconds, `ExpertReadStall`) a batch not done by then
    /// throws `drive_stalled` without joining: a hung pread cannot be cancelled, so its jobs are joined on a thread of
    /// their own, which then runs `stalled` (the caller frees the slots there, never before).
    package static func perform(
        _ count: Int, on scheduler: IOScheduler, source: (Int) -> IOSource = { _ in .externalSSD },
        bytes: (Int) -> Int = { _ in 1 }, spin: Double?, submitted: (() -> Void)? = nil,
        meanwhile: () -> Void = {}, hedging: Hedging? = nil, stall: Double? = nil, stalled: (@Sendable () -> Void)? = nil,
        _ body: @escaping @Sendable (Int) -> Void
    ) throws {
        guard count > 0 else {
            meanwhile()
            return
        }
        final class Completion: Sendable {
            let count = Atomic<Int>(0)
            let changed: DispatchSemaphore
            init(_ changed: DispatchSemaphore) { self.changed = changed }
        }
        let completion = Completion(hedging?.wake ?? DispatchSemaphore(value: 0))
        let ended = hedging?.ended
        let generation = scheduler.currentGeneration
        var tickets: [IOTicket] = []
        var refused: Error?
        for i in 0..<count {
            do {
                tickets.append(
                    try scheduler.submit(
                        source: source(i), priority: .demand, generation: generation, byteCount: bytes(i),
                        operation: { _ in body(i) },
                        completion: { _, outcome in
                            ended?(i, outcome)
                            completion.count.add(1, ordering: .releasing)
                            completion.changed.signal()
                        }))
                hedging?.submitted(i, tickets[i])
            } catch {
                refused = error
                break
            }
        }
        if refused == nil {
            submitted?()
            meanwhile()
        }
        let deadline = DispatchTime.now().uptimeNanoseconds + UInt64(max(0, spin ?? 0) * 1e9)
        let stallAt = stall.map { DispatchTime.now() + $0 } ?? .distantFuture
        var observed = 0
        // Hedged: asked at the first pass, after every wake and change, once the next hedge is due, and once every job
        // ended (a replica may still be read): the wait ends when it says every record has its bytes.
        let polled = refused == nil ? hedging : nil
        var poll = true
        var next: UInt64?
        while true {
            let done = completion.count.load(ordering: .acquiring)
            if let polled {
                if poll || done != observed || done == tickets.count || next.map({ DispatchTime.now().uptimeNanoseconds >= $0 }) == true {
                    let state = polled.poll()
                    if state.settled { break }
                    next = state.next
                }
            } else if done == tickets.count {
                break  // The caller runs the final batch without another intermediate commit.
            }
            poll = false
            if done != observed, done < tickets.count {
                observed = done
            } else if DispatchTime.now().uptimeNanoseconds >= deadline {
                if DispatchTime.now() >= stallAt {
                    let total = tickets.indices.reduce(0) { $0 + bytes($1) }
                    let left = tickets.enumerated().filter { hedging?.left($0.offset) != true }.map(\.element)
                    Thread.detachNewThread {
                        for ticket in left { _ = ticket.join() }
                        stalled?()
                    }
                    throw ExpertReadStall.error(bytes: total, seconds: stall ?? 0)
                }
                _ = completion.changed.wait(timeout: min(next.map { DispatchTime(uptimeNanoseconds: $0) } ?? .distantFuture, stallAt))
                poll = true
            } else if polled != nil, completion.changed.wait(timeout: .now()) == .success {
                poll = true
            }
        }
        var lost = false
        for (i, ticket) in tickets.enumerated() where hedging?.left(i) != true {
            if case .succeeded = ticket.join() {} else { lost = true }
        }
        if let refused { throw refused }
        if lost { throw EngineError("An expert read was cancelled before it ran") }
    }

    /// Spins until `ready` or for `seconds`, whichever comes first.
    package static func spin(_ seconds: Double, until ready: () -> Bool) {
        let end = DispatchTime.now().uptimeNanoseconds + UInt64(max(0, seconds) * 1e9)
        while !ready() && DispatchTime.now().uptimeNanoseconds < end {}
    }

    /// The races of a fetch batch's hedged records: each missing record (`keys`) the copy holds now. Its primary is its
    /// jobs in `jobs` (record, part); its replica reads the other drive (the copy, else the GGUF's three
    /// parts) once the primary failed, or once both hold:
    /// - the primary is older than its minimum delay, twice the record's read time at its drive's healthy rate, and
    ///   never sooner than `hedgeFloorSeconds`;
    /// - the replica would end first: the other drive's latency + (its outstanding bytes + the record) / its planned rate
    ///   (`plannedBandwidth`) is below the primary's remaining time, the bytes still queued on its drive ahead of it and
    ///   its own / its drive's current rate, a rate no higher than those bytes over the time the primary has waited
    ///   (a read that has not ended in t seconds is read no faster than that).
    /// Nil when no record qualifies.
    private func hedge(
        _ keys: [Int], jobs: [(miss: Int, part: Int)], bank: Bank, files: [ExactFile],
        probe: (@Sendable (IOSource, Bool) throws -> Void)?
    ) -> FamilyHedges? {
        guard let hot, keys.contains(where: hot.holds), let arena = replicaArena else { return nil }
        let start = DispatchTime.now().uptimeNanoseconds
        var queued: [IOSource: Double] = [:]
        for drive in reads.snapshot().drives { queued[drive.source] = Double(drive.outstandingBytes) }
        // Per record: the bytes on its primary's drive ahead of it and its own, this batch's earlier jobs included.
        // shortcut: counts reads already outstanding before the batch as ahead even once they ended (capped by what the
        // drive still has outstanding at each decision); exact positions would need the scheduler's queue order.
        var ahead = Array(repeating: 0.0, count: keys.count)
        for job in jobs {
            let source: IOSource = job.part == -1 ? .internalSSD : .externalSSD
            queued[source, default: 0] += Double(job.part < 0 ? bank.record : bank.bytes[job.part])
            ahead[job.miss] = queued[source]!
        }
        let generation = reads.currentGeneration
        let wake = DispatchSemaphore(value: 0)
        let e = weights.geometry.experts
        let trace = hedgeTrace
        var marks: [FamilyHedgeTrace.Mark?] = []
        let races = keys.enumerated().map { n, key -> ExpertReadReplica? in
            guard hot.holds(key) else {
                marks.append(nil)
                return nil
            }
            let parts = jobs.filter { $0.miss == n }
            let primary: IOSource = parts[0].part == -1 ? .internalSSD : .externalSSD
            let mark = FamilyHedgeTrace.Mark(key: key, primary: primary, start: start)
            marks.append(mark)
            let race = ExpertReadReplica(
                scheduler: reads, arena: arena, generation: generation, recordBytes: bank.record,
                read: { source, into in
                    try probe?(source, true)
                    if source == .internalSSD {
                        try hot.read(key, into: into)
                    } else {
                        try Self.read(
                            -2, layer: key / e, expert: key % e, of: e, bank: bank, files: files, hot: nil,
                            into: into.baseAddress!)
                    }
                },
                wake: {
                    let now = DispatchTime.now().uptimeNanoseconds
                    trace.lock.withLock { mark.replicaEnd = now }
                    wake.signal()
                })
            let healthy = driveRates.healthy(primary) ?? (primary == .internalSSD ? bandwidth.hot : bandwidth.gguf)
            race.expect(
                parts: parts.count, source: primary, alternate: primary == .internalSSD ? .externalSSD : .internalSSD,
                delay: max(Self.hedgeFloorSeconds, 2 * Double(bank.record) / healthy))
            return race
        }
        let primaries = marks.map { $0?.primary ?? .externalSSD }
        let (rates, average, stored, latency, record, scheduler) = (
            driveRates, bandwidth, (hot: copyStart, gguf: seeded.gguf), readLatency, Double(bank.record), reads
        )
        let decide: FamilyHedges.Decide = { n, elapsed in
            let primary = primaries[n]
            let other: IOSource = primary == .internalSSD ? .externalSSD : .internalSSD
            var outstanding: [IOSource: Double] = [:]
            for drive in scheduler.snapshot().drives { outstanding[drive.source] = Double(drive.outstandingBytes) }
            let left = max(1, min(outstanding[primary] ?? 0, ahead[n]))
            let current = rates.current(primary) ?? (primary == .internalSSD ? average.hot : average.gguf)
            let primaryLeft = left / min(current, ahead[n] / max(1e-6, elapsed))
            let otherRate = rates.rate(
                other, average: other == .internalSSD ? average.hot : average.gguf, stored: other == .internalSSD ? stored.hot : stored.gguf
            )
            let replica = (other == .internalSSD ? latency.hot : latency.gguf) + ((outstanding[other] ?? 0) + record) / otherRate
            // Not yet: the waited-time bound alone reaches the replica's time once elapsed = replica x ahead / left.
            return (replica < primaryLeft, primaryLeft, replica, replica * ahead[n] / left)
        }
        return FamilyHedges(
            wake: wake, record: jobs.map(\.miss), races: races, marks: marks, trace: trace, rates: driveRates,
            recordBytes: bank.record, start: start, decide: decide)
    }

    /// The records of `experts` of `layer`, reading the missing ones in parallel (a record the hot copy holds from
    /// whichever drive `copyShare` gives it, the others from the GGUF). Slots of the experts asked for are not reused by this call; earlier ones may be (callers
    /// finish their GPU work first). The records returned stay pinned until the next call begins, or the one after when
    /// that one is `keepingLast`. `decode`: generation cache policy; each unique record demand counts as a use,
    /// including a bounded verification block. `keepingLast`:
    /// the slots the previous call returned are not reused either (their GPU work may still run). A guessed record
    /// (`prefetch`) asked for is a hit that counts as the miss it saved, awaited if still read; one whose read failed or
    /// was cancelled is read here as a miss. Guesses not asked for and not begun are cancelled first, so their reads do
    /// not queue before these. `ready`: called once on this thread while the others are still read, with the indices
    /// (ascending) and slots of the resident records. Its error is thrown once the reads end.
    /// `submitted` runs once on this owner after all fresh demand jobs have been submitted, including all-cold
    /// batches. It may start a delayed expert's read ahead; it must not recursively fetch or change the cache budget.
    package func fetch(
        layer: Int, experts: [Int], decode: Bool = false, keepingLast: Bool = false, submitted: (() -> Void)? = nil,
        ready: (([Int], [Slot]) throws -> Void)? = nil
    ) throws -> [Slot] {
        guard let bank = banks[layer], let p = poolOf[layer] else { throw EngineError("Layer \(layer) has no routed experts") }
        let observations = decode ? prefetchObservations : nil
        let observationStarted = observations == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        var observationBytes = 0
        let competitors = observations == nil ? (queued: 0, running: 0) : observationCompetitors(layer: layer, needed: experts)
        var observedSuccess = false
        defer {
            observations?.demandFetch(
                layer: layer, queued: competitors.queued, running: competitors.running,
                nanoseconds: DispatchTime.now().uptimeNanoseconds - observationStarted,
                bytes: observationBytes, succeeded: observedSuccess)
        }
        // The previous call's records stay pinned with `keepingLast` (their GPU work may still run); earlier calls' finished.
        releasePins(keeping: keepingLast ? 1 : 0)
        freeStalledBatches()
        pins.append([])
        replicaPins.append([])
        if decode && pools.count > 1 { decodeAccesses += experts.count }
        let e = weights.geometry.experts
        // Per asked expert: its slot, and where its pinned record is (a repeated expert's may be at its first index only).
        var slots = [Int](repeating: -1, count: experts.count)
        var located = [(storage: ExpertSlabStorage, offset: Int)?](repeating: nil, count: experts.count)
        let neededKeys = Set(experts.map { layer * e + $0 })
        settle(keeping: neededKeys)
        if decode, let guess = guessed, guess.layer == layer {
            counters.predicted += experts.count
            counters.predictedRight += experts.filter(guess.experts.contains).count
            guessed = nil
            prefetchPolicy.observe(
                layer: layer, predicted: guess.ranked, needed: Set(experts), absent: guess.absent,
                seconds: Double(DispatchTime.now().uptimeNanoseconds - guess.began) / 1e9)
            for expert in Set(experts) { if let rank = guess.issued[expert] { Self.count(&counters.guessUsed, rank) } }
        }
        // Slots this call takes are stamped after `start`, which is how the victim scan skips them.
        let start = locked {
            let start = keepingLast ? policy.lastStart : policy.clock
            policy.lastStart = policy.clock
            return start
        }
        var awaited: [Int] = []
        var resident: [Int] = []
        // Seconds a byte of reads may be spun for (`spins`): twice the slower drive's measured rate (the GGUF's alone
        // without a hot copy, whose rate is then never measured).
        let spinPerByte =
            decode && Self.spins && !ProcessInfo.processInfo.isLowPowerModeEnabled
            ? 2 / (hot == nil ? bandwidth.gguf : min(bandwidth.hot, bandwidth.gguf)) : nil
        try lookUp(
            layer: layer, experts: experts, pool: p, decode: decode, slots: &slots, located: &located, awaited: &awaited,
            resident: &resident)
        var readyError: Error?
        func describe(_ i: Int) -> Slot {
            let at = located[i] ?? located[experts.indices.first { slots[$0] == slots[i] && located[$0] != nil }!]!
            return self.describe(at.storage, at.offset, layer: layer)
        }
        func runReady() {
            guard let ready, !resident.isEmpty, resident.count < experts.count else { return }
            do { try ready(resident, resident.map(describe)) } catch { readyError = error }
        }
        var prompt = decode ? nil : locked { policy.passed(usable(p), start: start, pool: p, core) }
        // Reads the records of `experts[indices]` not yet placed (`first`: with `ready` and `submitted`).
        func readMissing(_ indices: [Int], first: Bool) throws {
            var missing: [(index: Int, slot: Int, expert: Int, ticket: ExpertSlotTicket)] = []
            try acquireMissing(
                indices, layer: layer, experts: experts, pool: p, decode: decode, start: start, awaited: awaited, slots: &slots,
                prompt: &prompt, missing: &missing)
            guard !missing.isEmpty else {
                if first { runReady() }
                return
            }
            let started = Date()
            let files: [ExactFile]
            nonisolated(unsafe) let targets: [(storage: ExpertSlabStorage, offset: Int)]  // each job writes its own slot
            do {
                files = try bank.tensors.map { try weights.file($0.file) }
                targets = try locked { try missing.map { try core.beginReadLocked($0.ticket) } }
            } catch {
                forgetMissing(missing, pool: p)
                throw error
            }
            // Publishes miss `n`'s record (every part read) and pins it for this call (under the core's lock).
            func verify(_ n: Int) throws {
                guard core.completeReadLocked(missing[n].ticket, verified: true) else { throw EngineError("Expert record left the cache") }
                located[missing[n].index] = try core.pinLocked(missing[n].ticket)
                pins[pins.count - 1].append(missing[n].ticket)
            }
            // One job per record read from the hot copy, three per other record (gate, up, down from the GGUF): both
            // drives read at once.
            let hot = self.hot
            let held = missing.filter { hot?.holds(layer * e + $0.expert) == true }.count
            var share = copyShare(held: held, other: missing.count - held, record: bank.record, probe: true)
            counters.demandHeldMainRecords += held - share
            var jobs: [(miss: Int, part: Int)] = []
            var heldMain: [Int] = []
            for (i, m) in missing.enumerated() {
                if share > 0, hot?.holds(layer * e + m.expert) == true {
                    jobs.append((i, -1))
                    share -= 1
                } else {
                    if hot?.holds(layer * e + m.expert) == true { heldMain.append(i) }
                    jobs += (0..<3).map { (i, $0) }
                }
            }
            let misses = missing
            let work = jobs
            let probe = readProbe
            let hedges = hedge(missing.map { layer * e + $0.expert }, jobs: jobs, bank: bank, files: files, probe: probe)
            defer {
                hedges?.traceWinners()
                for race in hedges?.races ?? [] where race?.isHedged == true {
                    counters.hedges += 1
                    counters.hedgeBytes += bank.record
                    if race!.winningReplica != nil {
                        counters.hedgeReplicaWins += 1
                    } else if race!.primarySucceeded {
                        counters.hedgePrimaryWins += 1
                    }
                    if race!.replicaFailed { counters.hedgeReplicaFailures += 1 }
                }
            }
            let progress = Progress(records: misses.count)
            let tracksSubmission = first && submitted != nil
            if tracksSubmission {
                for job in work { progress.pendingBytes[job.part == -1 ? 0 : 1] += job.part < 0 ? bank.record : bank.bytes[job.part] }
            }
            let queuedAt = DispatchTime.now().uptimeNanoseconds
            if decode, first, routerAwaitsDemand, routerAt > 0 {
                routerAwaitsDemand = false
                let seconds = Double(queuedAt &- min(queuedAt, routerAt)) / 1e9
                counters.routerToDemandSeconds += seconds
                counters.routerToDemandLayers += 1
                counters.routerToDemandMaximum = max(counters.routerToDemandMaximum, seconds)
            }
            let read: @Sendable (Int) -> Void = { job in
                let (i, part) = work[job]
                let m = misses[i]
                let (storage, base) = targets[i]
                let begun = DispatchTime.now().uptimeNanoseconds
                var deliveredFromHot = false
                defer {
                    let ended = DispatchTime.now().uptimeNanoseconds
                    let d = deliveredFromHot ? 0 : 1
                    progress.lock.lock()
                    let scheduledSource = part == -1 ? 0 : 1
                    progress.queueSeconds[scheduledSource] += Double(begun - queuedAt) / 1e9
                    progress.jobs[scheduledSource] += 1
                    if tracksSubmission { progress.pendingBytes[part == -1 ? 0 : 1] -= part < 0 ? bank.record : bank.bytes[part] }
                    progress.fromHot[i] = deliveredFromHot
                    progress.first[d] = min(progress.first[d], begun)
                    progress.last[d] = max(progress.last[d], ended)
                    progress.recordFirst[i] = min(progress.recordFirst[i], begun)
                    progress.recordLast[i] = max(progress.recordLast[i], ended)
                    progress.lock.unlock()
                }
                do {
                    try probe?(part == -1 ? .internalSSD : .externalSSD, false)
                    deliveredFromHot = try Self.read(
                        part, layer: layer, expert: m.expert, of: e, bank: bank, files: files, hot: hot,
                        into: storage.buffer.contents() + base)
                } catch {
                    // A hedged record's failed read is its race's to settle (the replica may still deliver it).
                    if let hedges, hedges.races[i] != nil {
                        hedges.fail(i, error)
                    } else {
                        progress.lock.lock()
                        progress.error = error
                        progress.lock.unlock()
                    }
                }
            }
            let spin = spinPerByte.map { $0 * Double(misses.count * bank.record) }
            let afterSubmission: (() -> Void)? = (first ? submitted : nil).map { callback in
                {
                    progress.lock.lock()
                    let pending = progress.pendingBytes
                    progress.lock.unlock()
                    self.inDemandSubmission = true
                    self.counters.earlyBatches += 1
                    self.counters.earlyOutstandingBatches += pending.reduce(0, +) > 0 ? 1 : 0
                    self.counters.earlyDemandBytes += pending.reduce(0, +)
                    self.counters.earlyDemandHotBytes += pending[0]
                    defer { self.inDemandSubmission = false }
                    callback()
                }
            }
            // A batch that outlives its bound fails the request (`ExpertReadStall`); its slots stay `reading` until its
            // jobs have ended, then are freed off this thread.
            let batchBytes = work.reduce(0) { $0 + ($1.part < 0 ? bank.record : bank.bytes[$1.part]) }
            let ended = StalledBatch(missing: missing, pool: p)
            do {
                try Self.perform(
                    work.count, on: reads, source: { work[$0].part == -1 ? .internalSSD : .externalSSD },
                    bytes: { work[$0].part < 0 ? bank.record : bank.bytes[work[$0].part] }, spin: spin,
                    submitted: afterSubmission, meanwhile: { if first { runReady() } }, hedging: hedges?.hedging,
                    stall: ExpertReadStall.bound(bytes: batchBytes, bytesPerSecond: min(bandwidth.hot, bandwidth.gguf)),
                    stalled: {
                        hedges?.joinAll()
                        ended.end()
                    }, read)
            } catch {
                if (error as? EngineError)?.code == "drive_stalled" {
                    // The stuck slots stay `reading` (never evicted or retired) until `stalled` frees them; their keys
                    // leave the lookup now, so a later request reads those experts into other slots.
                    locked {
                        for m in missing where core.lookup[layer * e + m.expert] == m.ticket.slot {
                            core.forgetLocked(layer * e + m.expert)
                        }
                    }
                    stalledBatches.append(ended)
                    throw error
                }
                hedges?.joinAll()
                forgetMissing(missing, pool: p)
                throw error
            }
            // No read into a missing slot may still run when its slot is freed.
            if let error = progress.lock.withLock({ progress.error }) ?? hedges?.failure {
                hedges?.joinAll()
                forgetMissing(missing, pool: p)
                throw error
            }
            // Records a replica won: once their primary ended (it can no longer write) its bytes go into the slot,
            // verified as any read's; while their primary still reads, the lease is used in place (`orphans`).
            var won = Set<Int>()
            var orphans: [(miss: Int, lease: ExpertReplicaLease)] = []
            for n in missing.indices {
                guard let race = hedges?.races[n], let lease = race.winningReplica else { continue }
                won.insert(n)
                if race.primaryFinished || race.primaryDropped {
                    (targets[n].storage.buffer.contents() + targets[n].offset).copyMemory(
                        from: lease.bytes.baseAddress!, byteCount: bank.record)
                } else {
                    orphans.append((n, lease))
                }
            }
            do {
                try locked {
                    for n in missing.indices where !orphans.contains(where: { $0.miss == n }) { try verify(n) }
                }
            } catch {
                hedges?.joinAll()
                forgetMissing(missing, pool: p)
                throw error
            }
            for (n, lease) in orphans {
                // Its slot stays its primary's, read in the background as a demanded guess until it lands.
                located[missing[n].index] = (ExpertSlabStorage(buffer: lease.metalBuffer, retaining: lease), lease.offset)
                replicaPins[replicaPins.count - 1].append(lease)
                inFlight[layer * e + missing[n].expert] = Guess(
                    ticket: missing[n].ticket, losing: hedges!.races[n]!, done: hedges!.done[n]!)
            }
            if !heldMain.isEmpty {
                // Counter C: had each such record gone to the copy after the batch's copy reads, at the copy's live rate.
                let rate = driveRates.current(.internalSSD) ?? bandwidth.hot
                progress.lock.withLock {
                    var finish = Double(max(progress.last[0], queuedAt))
                    for i in heldMain.sorted(by: { progress.recordLast[$0] < progress.recordLast[$1] })
                    where progress.recordLast[i] > progress.recordFirst[i] {
                        finish += Double(bank.record) / rate * 1e9
                        counters.heldMainSeconds += Double(progress.recordLast[i] - progress.recordFirst[i]) / 1e9
                        counters.heldMainCopySoonerSeconds += max(0, (Double(progress.recordLast[i]) - finish) / 1e9)
                    }
                }
            }
            recordDemandReads(
                progress, won: won, work: work, misses: misses, missing: missing, bank: bank, pool: p, decode: decode, started: started,
                observations: observations, observationBytes: &observationBytes)
        }
        try readMissing(Array(experts.indices), first: true)
        var unverified: [Int] = []
        for key in awaited where inFlight[key] != nil {
            if inFlight[key]!.done.wait(timeout: .now()) == .timedOut {
                if inFlight[key]!.ahead {
                    counters.readAheadLate += 1
                } else if inFlight[key]!.hedge == nil {
                    counters.prefetchLate += 1
                }
                if let spinPerByte, let guess = inFlight[key] {
                    Self.spin(spinPerByte * Double(bank.record)) { guess.done.wait(timeout: .now()) == .success }
                }
                // Its read keeps its slot in `inFlight` until it lands, whenever that is (`ExpertReadStall`).
                let stall = ExpertReadStall.bound(bytes: bank.record, bytesPerSecond: min(bandwidth.hot, bandwidth.gguf))
                if inFlight[key]!.done.wait(timeout: .now() + stall) == .timedOut {
                    throw ExpertReadStall.error(bytes: bank.record, seconds: stall)
                }
            }
            _ = land(key)
            let asked = experts.indices.filter { layer * e + experts[$0] == key }
            if entry(key) == nil {
                // Its read failed or was cancelled after this call adopted it: its slot went back to free.
                for i in asked { slots[i] = -1 }
                unverified.append(key)
            } else if asked.allSatisfy({ located[$0] == nil }) {
                located[asked[0]] = try pin(key)
            }
        }
        if !unverified.isEmpty { try readMissing(experts.indices.filter { slots[$0] < 0 }, first: false) }
        if let readyError { throw readyError }
        let result = experts.indices.map(describe)
        observedSuccess = true
        return result
    }

    /// `fetch`'s lookup of each asked expert: a resident record is pinned and counted at once, a guessed one is awaited
    /// (or, when its read was cancelled, left to be read as a miss).
    private func lookUp(
        layer: Int, experts: [Int], pool p: Int, decode: Bool, slots: inout [Int],
        located: inout [(storage: ExpertSlabStorage, offset: Int)?],
        awaited: inout [Int], resident: inout [Int]
    ) throws {
        let e = weights.geometry.experts
        // A hit brought up to date (under the core's lock): a guessed record asked for counts as the miss it saved.
        func hit(_ key: Int, slot: Int) {
            if decode { policy.scan[slot] = false }
            if let ahead = unread.removeValue(forKey: key) {
                policy.touch(slot, heat: 1)
                missed[key, default: 0] += decode ? 4 : 1
                if ahead {
                    counters.readAheadUsed += 1
                } else {
                    counters.prefetchUsed += 1
                    counters.guessDrives[guessDrive[key] ?? 1].used += 1
                }
                guessDrive[key] = nil
            } else {
                policy.touch(slot, heat: policy.heat[slot] * exp2(-Double(policy.clock - policy.stamp[slot]) / policy.halfLife) + 1)
                counters.hits += 1
            }
        }
        for (i, expert) in experts.enumerated() {
            let key = layer * e + expert
            if decode { uses[key, default: 0] += 1 }
            // Its slot (numbered across pools); a record no guess is reading is pinned and counted at once, under one lock.
            let found = try locked { () throws -> (slot: Int, guessed: Bool)? in
                policy.clock += 1
                guard let index = core.lookup[key] else { return nil }
                let slot = pools[p].first + index
                guard inFlight[key] == nil else { return (slot, true) }
                let ticket = ExpertSlotTicket(sizeClass: p, slot: index, generation: core.classes[p].slots[index]!.generation, key: key)
                located[i] = try core.pinLocked(ticket)
                pins[pins.count - 1].append(ticket)
                hit(key, slot: slot)
                return (slot, false)
            }
            guard let found else { continue }
            let slot = found.slot
            guard found.guessed else {
                slots[i] = slot
                resident.append(i)
                continue
            }
            let guess = inFlight[key]!
            let ended = guess.done.wait(timeout: .now()) == .success
            if ended, guess.cancelled {
                _ = land(key)  // no bytes were read: read below as a miss
                continue
            }
            slots[i] = slot
            awaited.append(key)
            if ended, guess.failure == nil {
                // Its bytes are complete: verified now, so it can be delivered with the resident records.
                _ = land(key)
                located[i] = try pin(key)
                resident.append(i)
            }
            locked { hit(key, slot: slot) }
        }
    }

    /// A demand batch whose reads outlived their bound (`ExpertReadStall`): its slots stay `reading` until every job
    /// ended (`end`, from the thread that joined them), then the owner frees them (`freeStalledBatches`).
    final class StalledBatch: @unchecked Sendable {
        let missing: [(index: Int, slot: Int, expert: Int, ticket: ExpertSlotTicket)]
        let pool: Int
        private let lock = NSLock()
        private var done = false
        init(missing: [(index: Int, slot: Int, expert: Int, ticket: ExpertSlotTicket)], pool: Int) {
            self.missing = missing
            self.pool = pool
        }
        func end() { lock.withLock { done = true } }
        var ended: Bool { lock.withLock { done } }
    }
    private var stalledBatches: [StalledBatch] = []
    /// Batches still read since their stall (their slots are not free yet).
    package var stalledReads: Int { stalledBatches.count }
    /// Frees the slots of stalled batches whose reads have all returned.
    package func freeStalledBatches() {
        let ended = stalledBatches.filter(\.ended)
        for batch in ended { forgetMissing(batch.missing, pool: batch.pool) }
        stalledBatches.removeAll { batch in ended.contains { $0 === batch } }
    }

    // A miss that is not read in full must not stay a hit: its slot goes back to free (this call's pin included).
    private func forgetMissing(_ missing: [(index: Int, slot: Int, expert: Int, ticket: ExpertSlotTicket)], pool p: Int) {
        locked {
            for m in missing {
                guard let slot = core.classes[p].slots[m.ticket.slot], slot.generation == m.ticket.generation else { continue }
                // A record this call verified is pinned by it: unpinned first, so its slot frees clean.
                for _ in 0..<slot.pins { try? core.unpinLocked(m.ticket) }
                core.freeLocked(m.ticket.slot, sizeClass: p)
            }
        }
    }

    /// `readMissing`'s slots for the misses of `experts[indices]`, each taken from the pool's policy (or found, for an
    /// expert asked for twice).
    private func acquireMissing(
        _ indices: [Int], layer: Int, experts: [Int], pool p: Int, decode: Bool, start: UInt64, awaited: [Int],
        slots: inout [Int], prompt: inout FamilyEviction.Passed?,
        missing: inout [(index: Int, slot: Int, expert: Int, ticket: ExpertSlotTicket)]
    ) throws {
        let e = weights.geometry.experts
        for i in indices where slots[i] < 0 {
            let expert = experts[i]
            let key = layer * e + expert
            var request: FamilyEviction.Request? = FamilyEviction.Request(
                pool: p, start: start, usable: usable(p), limit: limits[p], decode: decode, prompt: prompt)
            prompt = nil  // the request owns it (one owner, so taking a victim copies nothing)
            // In one lock: the same expert twice in one call finds its slot, else the policy gets the request.
            if let index = locked({ () -> Int? in
                if let index = core.lookup[key] { return index }
                swap(&policy.request, &request)
                return nil
            }) {
                prompt = request!.prompt
                slots[i] = pools[p].first + index
                continue
            }
            var acquired: ExpertSlotAcquisition?
            var waitedForSlab = false
            while true {
                var refused: Error?
                do { acquired = try core.acquire(key) } catch { refused = error }
                if acquired != nil { break }  // the request comes back below, with the new record's bookkeeping
                locked { swap(&policy.request, &request) }
                switch request!.failure {
                case .land?:
                    // Guesses still read hold their slots (a `keepingLast` wave has no room to spare): wait for them to land.
                    for (key, guess) in inFlight where !awaited.contains(key) {
                        guard landed(key, guess) else {
                            forgetMissing(missing, pool: p)
                            throw ExpertReadStall.error(bytes: banks[layer]?.record ?? 0, seconds: guessStall(key))
                        }
                        _ = land(key)
                    }
                    request!.failure = nil
                    locked { swap(&policy.request, &request) }
                case .exhausted?:
                    forgetMissing(missing, pool: p)
                    throw EngineError("Layer \(layer) asked for more experts at once than its \(limits[p]) slots")
                case .denied(let error)?:
                    // Backpressure, as the state's admission: once, wait (bounded) for the slab to fit, then try it again.
                    if !waitedForSlab, slabMayFit(pool: p) {
                        waitedForSlab = true
                        request!.stage = .landed
                        request!.failure = nil
                        request!.denied = nil
                        locked { swap(&policy.request, &request) }
                        continue
                    }
                    forgetMissing(missing, pool: p)
                    throw error
                case nil:
                    // A slab allocation failed for another reason than memory.
                    forgetMissing(missing, pool: p)
                    throw refused!
                }
            }
            let ticket = acquired!.ticket
            let victim = pools[p].first + ticket.slot
            locked {
                swap(&policy.request, &request)
                policy.clock += 1
                policy.touch(victim, heat: 1)
                policy.scan[victim] = !decode
                policy.placed(key, at: victim)
            }
            prompt = request!.prompt
            slots[i] = victim
            missing.append((i, victim, expert, ticket))
            counters.misses += 1
            missed[key, default: 0] += decode ? 4 : 1
        }
    }

    // The first error, and when each drive's jobs started first and ended last (hot copy, GGUF).
    private final class Progress: @unchecked Sendable {
        var error: Error?
        var first = [UInt64.max, .max], last: [UInt64] = [0, 0]
        var recordFirst: [UInt64], recordLast: [UInt64], fromHot: [Bool]
        var queueSeconds = [0.0, 0.0], jobs = [0, 0]
        var pendingBytes = [0, 0]
        let lock = NSLock()
        init(records: Int) {
            recordFirst = Array(repeating: .max, count: records)
            recordLast = Array(repeating: 0, count: records)
            fromHot = Array(repeating: false, count: records)
        }
    }

    /// Drive `d`'s (0 the copy's, 1 the GGUF's) estimates after demand reads of `bytes` over `span` seconds, `records`
    /// reads of `recordBytes` each taking `service` seconds in all: its moving rate and latency, current and healthy rate,
    /// and the GGUF's slow-read tally.
    private func learn(drive d: Int, bytes: Int, span: Double, service: Double, records: Int, recordBytes: Int) {
        let oldRate = d == 0 ? bandwidth.hot : bandwidth.gguf
        let rate = min(64e9, max(1e6, min(oldRate * 4, max(oldRate / 4, Double(bytes) / span))))
        driveRates.add(d == 0 ? .internalSSD : .externalSSD, bytes: Double(bytes), seconds: span)
        let latency = records > 0 ? min(0.05, max(0, service / Double(records) - Double(recordBytes) / oldRate)) : 0
        if d == 0 {
            counters.demandHotServiceSeconds += service
            counters.demandHotRecords += records
            counters.demandHotSpanSeconds += span
            bandwidth.hot += (rate - bandwidth.hot) / 8
            readLatency.hot += (latency - readLatency.hot) / 8
            driveRates.observe(.internalSSD, average: bandwidth.hot)
        } else {
            counters.demandMainServiceSeconds += service
            counters.demandMainRecords += records
            counters.demandMainSpanSeconds += span
            bandwidth.gguf += (rate - bandwidth.gguf) / 8
            readLatency.gguf += (latency - readLatency.gguf) / 8
            driveRates.observe(.externalSSD, average: bandwidth.gguf)
            modelDriveReads.add(bytes: Double(bytes), seconds: span)
        }
    }

    /// `readMissing`'s accounting once its reads ended: per-drive rates, latencies and counters.
    private func recordDemandReads(
        _ progress: Progress, won: Set<Int>, work: [(miss: Int, part: Int)],
        misses: [(index: Int, slot: Int, expert: Int, ticket: ExpertSlotTicket)],
        missing: [(index: Int, slot: Int, expert: Int, ticket: ExpertSlotTicket)], bank: Bank, pool p: Int, decode: Bool, started: Date,
        observations: FamilyPrefetchObservations?, observationBytes: inout Int
    ) {
        // A record a replica won counts toward the drive it came from, and toward no drive's speed.
        let replicaFromHot = Set(won.filter { n in !work.contains { $0.miss == n && $0.part == -1 } })
        progress.lock.lock()
        defer { progress.lock.unlock() }
        for n in won { progress.fromHot[n] = replicaFromHot.contains(n) }
        counters.demandHotQueueSeconds += progress.queueSeconds[0]
        counters.demandMainQueueSeconds += progress.queueSeconds[1]
        counters.demandHotJobs += progress.jobs[0]
        counters.demandMainJobs += progress.jobs[1]
        let fromHot = progress.fromHot.filter { $0 }.count
        let measuredHot = misses.indices.filter { progress.fromHot[$0] && !won.contains($0) }.count
        // Measured with or without a hot copy: decode's spin bound needs the GGUF drive's rate either way.
        for (d, bytes) in [measuredHot, missing.count - won.count - measuredHot].map({ $0 * bank.record }).enumerated()
        where bytes > 0 && progress.last[d] > progress.first[d] {
            var service = 0.0
            var records = 0
            for i in misses.indices
            where progress.fromHot[i] == (d == 0) && !won.contains(i) && progress.recordLast[i] > progress.recordFirst[i] {
                service += Double(progress.recordLast[i] - progress.recordFirst[i]) / 1e9
                records += 1
            }
            learn(
                drive: d, bytes: bytes, span: Double(progress.last[d] - progress.first[d]) / 1e9, service: service, records: records,
                recordBytes: bank.record)
        }
        counters.hotReads += fromHot
        counters.hotBytes += fromHot * bank.record
        counters.readBytes += missing.count * bank.record
        counters.copyBytes += fromHot * bank.record
        counters.modelBytes += (missing.count - fromHot) * bank.record
        if observations != nil { observationBytes += missing.count * bank.record }
        let elapsed = Date().timeIntervalSince(started)
        counters.readSeconds += elapsed
        if decode && pools.count > 1 {
            // Use the already collected I/O timestamps: `elapsed` also includes the overlapped ready callback.
            let first = progress.first.min()!
            let last = progress.last.max()!
            if last > first {
                let cost = Double(last - first) / (1e9 * Double(missing.count))
                readCost[p] = readCost[p] > 0 ? readCost[p] + (cost - readCost[p]) / 8 : cost
            }
        }
    }

    /// A start's warm-up while no request runs (`FamilyServingBackend.warm`): the records of `keys` (layer * experts +
    /// expert, most wanted first) read into empty slots, never evicting, lowest slot first, in batches of a token's
    /// experts per layer with `fetch`'s reads (the hot copy's share, or the GGUF's three parts). Each pool keeps
    /// two prompt waves' slots empty (waves as `FamilyRunner.moe` cuts them), so the first prompt reads into those and
    /// then into its own passed records (`Passed`), not these. They enter as decode records (heat 1, not `scan`), not as
    /// misses, the most wanted stamped last so the heat rule drops the least wanted first. `stop` is asked before each
    /// batch. A record whose read fails is left out. The records placed.
    package func warm(keys: [Int], stop: () -> Bool = { false }) -> Int {
        let g = weights.geometry
        let e = g.experts
        var empty = pools.indices.map { p in
            locked { Array(usable(p).filter { policy.owner($0, pool: p, core) == nil && policy.idle($0) }.reversed()) }
        }
        var room = pools.indices.map { empty[$0].count - 2 * min(e, max(1, limits[$0] / 2)) }
        var plan: [(key: Int, slot: Int)] = []
        var planned = Set<Int>()
        for key in keys where key >= 0 && poolOf[key / e] != nil && entry(key) == nil && !planned.contains(key) {
            guard let p = poolOf[key / e], room[p] > 0 else { continue }
            room[p] -= 1
            planned.insert(key)
            plan.append((key, empty[p].removeLast()))
        }
        let base = locked {
            let base = policy.clock
            policy.clock += UInt64(plan.count)
            return base
        }
        var placed = 0
        for first in stride(from: 0, to: plan.count, by: g.expertsPerToken) {
            guard !stop(), ledger?.optionalGrowthAllowed != false else { break }
            let batch = Array(plan[first..<min(plan.count, first + g.expertsPerToken)])
            // Each record's slot reserved, then read: no lookup finds it before its read is verified.
            var tickets: [ExpertSlotTicket] = []
            do {
                try locked {
                    for r in batch {
                        let p = poolOf[r.key / e]!
                        try allocateLocked(pool: p, through: r.slot - pools[p].first)
                    }
                    guard core.nextGeneration <= UInt64.max - UInt64(batch.count) else { throw EngineError("Cache generation exhausted") }
                    tickets = batch.map { r in
                        let p = poolOf[r.key / e]!
                        let ticket = core.reserveLocked(r.slot - pools[p].first, key: r.key)
                        policy.didReserve(ticket.slot, sizeClass: p)
                        return ticket
                    }
                }
            } catch { break }
            nonisolated(unsafe) let targets = tickets.map { try? core.beginRead($0) }  // each job writes its own slot
            let hot = self.hot
            let held = batch.filter { hot?.holds($0.key) == true }.count
            var share = copyShare(
                held: held, other: batch.count - held,
                record: batch.reduce(0) { $0 + self.banks[$1.key / e]!.record } / batch.count)
            var jobs: [(record: Int, part: Int)] = []
            for (i, r) in batch.enumerated() {
                if share > 0, hot?.holds(r.key) == true {
                    jobs.append((i, -1))
                    share -= 1
                } else {
                    jobs += (0..<3).map { (i, $0) }
                }
            }
            let banks = self.banks
            let weights = self.weights
            let work = jobs
            final class Failed: @unchecked Sendable {
                var records = Set<Int>(), fromHot = Set<Int>()
                let lock = NSLock()
            }
            let failed = Failed()
            let probe = backgroundProbe
            DispatchQueue.concurrentPerform(iterations: work.count) { job in
                let (i, part) = work[job]
                let key = batch[i].key
                let bank = banks[key / e]!
                probe?(key)
                do {
                    guard let target = targets[i] else { throw EngineError("Expert slot is not reserved") }
                    if try Self.read(
                        part, layer: key / e, expert: key % e, of: e, bank: bank, files: try bank.tensors.map { try weights.file($0.file) },
                        hot: hot, into: target.storage.buffer.contents() + target.offset)
                    {
                        failed.lock.lock()
                        failed.fromHot.insert(i)
                        failed.lock.unlock()
                    }
                } catch {
                    failed.lock.lock()
                    failed.records.insert(i)
                    failed.lock.unlock()
                }
            }
            for (i, r) in batch.enumerated() {
                let record = banks[r.key / e]!.record
                if failed.fromHot.contains(i) { counters.copyBytes += record } else { counters.modelBytes += record }
                guard !failed.records.contains(i) else {
                    // A record whose read failed is left out: its slot is free again, as if never planned.
                    forgetting = r.key
                    if !core.completeRead(tickets[i], verified: false) { try? core.cancel(tickets[i]) }
                    forgetting = nil
                    continue
                }
                core.completeRead(tickets[i], verified: true)
                let s = r.slot
                let stamped = base + UInt64(plan.count - first - i)
                locked {
                    policy.heat[s] = 1
                    policy.stamp[s] = stamped
                    policy.level[s] = Double(stamped) / policy.halfLife
                    policy.scan[s] = false
                    policy.placed(r.key, at: s)
                }
                placed += 1
            }
        }
        return placed
    }

    /// Part `part` of one record into `record` (where its slot's record starts): -1 the whole record from the hot copy,
    /// -2 the whole record from the GGUF, 0–2 one projection from it (`files`, the bank's).
    /// Returns whether the completed bytes came from the hot copy. A failed hot attempt belongs to the fallback's
    /// inclusive service time, never to the hot drive's successful-byte throughput.
    @discardableResult private static func read(
        _ part: Int, layer: Int, expert: Int, of e: Int, bank: Bank, files: [ExactFile], hot: HotCopySet?,
        into record: UnsafeMutableRawPointer
    ) throws -> Bool {
        if part == -1, let hot,
            (try? hot.read(layer * e + expert, into: UnsafeMutableRawBufferPointer(start: record, count: bank.record))) != nil
        {
            return true
        }
        // A hot copy that cannot be read (its drive gone, or a record failing its checksum) gives way to the same bytes
        // from the GGUF.
        for p in part >= 0 ? [part] : [0, 1, 2] {
            try files[p].read(
                offset: bank.tensors[p].offset + Int64(expert * bank.bytes[p]),
                into: UnsafeMutableRawBufferPointer(start: record + bank.bytes.prefix(p).reduce(0, +), count: bank.bytes[p]))
        }
        return false
    }

    /// One more at `rank` of a per-rank counter.
    private static func count(_ ranks: inout [Int], _ rank: Int) {
        if ranks.count <= rank { ranks += repeatElement(0, count: rank + 1 - ranks.count) }
        ranks[rank] += 1
    }
    /// `key` leaves the lookup (under the core's lock, from `FamilyEviction.willForget`): a guessed record no fetch asked
    /// for was wasted when evicted; a guess `forget` drops was never evicted.
    private func forgot(_ key: Int) {
        let quiet = forgetting == key
        let drive = guessDrive.removeValue(forKey: key)
        if unread.removeValue(forKey: key) == false, !quiet {
            counters.prefetchWastedBytes += banks[key / weights.geometry.experts]!.record
            counters.guessDrives[drive ?? 1].wastedBytes += banks[key / weights.geometry.experts]!.record
        }
    }
    /// Drops `key`'s guess (cancelled or failed): the state machine returns its slot to free, its record never found.
    private func forget(_ key: Int, _ guess: Guess) {
        unread.removeValue(forKey: key)
        guessDrive.removeValue(forKey: key)
        if inFlight.removeValue(forKey: key) != nil {
            prefetchObservations?.finished(guess.observation, outcome: .unavailable)
            guessTickets.removeValue(forKey: key)
        }
        forgetting = key
        core.completeRead(guess.ticket, verified: false)
        forgetting = nil
    }
    /// Takes back the counts of a guess cancelled before its read began: nothing was read.
    private func uncount(_ key: Int, ahead: Bool) {
        let record = banks[key / weights.geometry.experts]!.record
        if ahead {
            counters.readAheads -= 1
            counters.readBytes -= record
        } else {
            counters.prefetchCancelled += 1
            counters.prefetchBytes -= record
        }
    }
    /// How long one token's worth of the largest records may take to read (`ExpertReadStall`, as at 1 GB/s): what a
    /// request waits for background reads that hold the model (the warm-up's batch, a record the keeper copies). Reads
    /// only what never changes, so any thread may ask.
    package var batchStall: Double {
        ExpertReadStall.bound(bytes: weights.geometry.expertsPerToken * (banks.values.map(\.record).max() ?? 0), bytesPerSecond: 1e9)
    }
    /// How long a guess's read may take (`ExpertReadStall`), at the slower drive's rate.
    private func guessStall(_ key: Int) -> Double {
        ExpertReadStall.bound(bytes: banks[key / weights.geometry.experts]?.record ?? 0, bytesPerSecond: min(bandwidth.hot, bandwidth.gguf))
    }
    /// Waits for a guess's read to end, within its bound (ROADMAP R2); false when it did not (it stays in `inFlight`,
    /// its slot reading, until it lands). A guess that outlived its bound once is not waited for again.
    private func landed(_ key: Int, _ guess: Guess) -> Bool {
        if guess.done.wait(timeout: guess.stalled ? .now() : .now() + guessStall(key)) == .success { return true }
        guess.stalled = true
        return false
    }
    /// A guess whose read ended leaves `inFlight`: a complete one is verified (found from now on); a failed one is
    /// forgotten and returns why, and one that cancelled itself before its read began is forgotten and counted as
    /// `settle` counts a guess it stops.
    private func land(_ key: Int) -> Error? {
        guard let guess = inFlight.removeValue(forKey: key) else { return nil }
        if !guess.ahead {
            let delivered: FamilyPrefetchObservations.Source = guess.deliveredFromHot ? .hot : .gguf
            prefetchObservations?.finished(
                guess.observation,
                outcome: guess.cancelled ? .cancelled : guess.error == nil ? .completed : .failed,
                delivered: delivered, bytes: banks[key / weights.geometry.experts]?.record ?? 0)
        }
        guessTickets.removeValue(forKey: key)
        if guess.ahead, guess.error == nil, guess.deliveredFromHot {
            counters.hotReads += 1
            counters.hotBytes += banks[key / weights.geometry.experts]!.record
        }
        // A hedge's losing primary was counted by its fetch.
        if !guess.cancelled, guess.hedge == nil {
            let record = banks[key / weights.geometry.experts]!.record
            if guess.deliveredFromHot { counters.copyBytes += record } else { counters.modelBytes += record }
        }
        if guess.cancelled {
            uncount(key, ahead: guess.ahead)
            forget(key, guess)
            return nil
        }
        guard let error = guess.failure else {
            if !guess.ahead, guess.hedge == nil, guess.routerAt > 0, let drive = guessDrive[key] {
                let ended = guessLock.withLock { guess.endedAt }
                counters.guessDrives[drive].completeSeconds += Double(ended &- min(ended, guess.routerAt)) / 1e9
                counters.guessDrives[drive].completions += 1
            }
            core.completeRead(guess.ticket, verified: true)
            return nil
        }
        forget(key, guess)
        return error
    }
    /// Lands the guesses whose reads ended; with `keeping` (a fetch's keys) also cancels and forgets those not begun
    /// that it does not hold.
    private func settle(keeping: Set<Int>?) {
        for (key, guess) in inFlight {
            let demanded = keeping?.contains(key) == true
            guessLock.lock()
            // A real demand promotes every projection. Already started records must finish atomically even when
            // pressure arrives; partially read gate/up/down may never be published as a resident expert.
            if demanded, !guess.cancelled { guess.demanded = true }
            // A queued guess holds a slot that already exists: memory pressure alone gains nothing by dropping it.
            let stop = !guess.started && (guess.cancelled || (!demanded && keeping != nil))
            if stop { guess.cancelled = true }
            guessLock.unlock()
            if stop {
                // Drop its queued reads; one already taken by a worker sees `cancelled` and reads nothing.
                for ticket in guessTickets[key] ?? [] { reads.cancelQueued(ticket) }
                uncount(key, ahead: guess.ahead)
                if !guess.ahead { prefetchObservations?.finished(guess.observation, outcome: .cancelled) }
                forget(key, guess)
                continue
            }
            if demanded { for ticket in guessTickets[key] ?? [] { reads.promote(ticket) } }
            settleProbe?()
            if guess.done.wait(timeout: .now()) == .success { _ = land(key) }
        }
    }

    /// Decode's guess at the next MoE layer: starts reading, without waiting, the records of `experts` of `layer` that
    /// are not resident or already read, on `reads` at prefetch priority, each copy-held one from the drive `copyShare` gives it. Their slots
    /// are victims as in `fetch` but never one the last fetch returned (its GPU work may still run, so its pins stay;
    /// every earlier call's finished behind the router's waited flush, so theirs go), one being read, or one holding a
    /// guessed record. Best effort: a slot that cannot be had stops it; a failed read frees its slot when it lands. The
    /// bytes are the ones `fetch` reads, so what runs is unchanged.
    package func prefetch(layer: Int, experts asked: [Int]) { guess(layer: layer, experts: asked, ahead: false) }

    /// A long prompt's read of the next MoE layer's first wave while this layer's last runs (`FamilyRunner.moe`):
    /// `prefetch`'s reads and victims, but its records are a prompt's (`scan`, heat 1, `Passed` victims before the first
    /// key read), and decode's guess and recall stay decode's.
    package func readAhead(layer: Int, experts asked: [Int]) { guess(layer: layer, experts: asked, ahead: true) }

    private func guess(layer: Int, experts asked: [Int], ahead: Bool) {
        guard let bank = banks[layer], let p = poolOf[layer], let files = try? bank.tensors.map({ try weights.file($0.file) }) else {
            return
        }
        // The enclosing fetch still owns awaited predictions and their error handling. Landing one here could
        // discard its failed-read result before fetch joins it. Leave all of them to that owner.
        if !inDemandSubmission { settle(keeping: nil) }
        releasePins(keeping: 1)
        let e = weights.geometry.experts
        let hot = self.hot
        let lock = guessLock
        let now = DispatchTime.now().uptimeNanoseconds
        let absent = asked.filter { entry(layer * e + $0) == nil }
        if !ahead { guessed = (layer, Set(asked), asked, Set(absent), [:], now) }
        if !ahead, let prefetchObservations {
            let current = observationResidency(layer: layer)
            prefetchObservations.predict(layer: layer, ranked: asked, resident: current.ready, inFlight: current.inFlight)
        }
        // Pressure, live or not, does not stop a guess or read-ahead: it only fills a slot the pool already has (skipping it
        // frees nothing and costs a read on demand later). Only a new slab waits for `optionalGrowthAllowed` (below).
        // The asked records already here stay: pinned while this guess chooses its victims.
        let kept = asked.compactMap { expert -> ExpertSlotTicket? in
            let key = layer * e + expert
            guard let phase = entry(key)?.phase, [.verified, .evictable, .pinned].contains(phase), let ticket = ticket(key) else {
                return nil
            }
            return (try? core.pin(ticket)) == nil ? nil : ticket
        }
        defer { for ticket in kept { try? core.unpin(ticket) } }
        let depth = ahead ? asked.count : guessDepth ?? asked.count
        let fresh = Array(absent.prefix(depth))
        if !ahead { prefetchObservations?.depthLimited(layer: layer, count: absent.count - fresh.count) }
        let held = fresh.filter { hot?.holds(layer * e + $0) == true }.count
        var share = copyShare(held: held, other: fresh.count - held, record: bank.record, priority: .prefetch)
        let lastStart = locked { policy.lastStart }
        var prompt = ahead ? locked { policy.passed(usable(p), start: lastStart, pool: p, core) } : nil
        let bound = layer * e + (fresh.first ?? 0)
        let probe = backgroundProbe
        for (freshIndex, expert) in fresh.enumerated() {
            let key = layer * e + expert
            let fromHot = share > 0 && hot?.holds(key) == true
            let queued = reads.snapshot().drives.first { $0.source == (fromHot ? .internalSSD : .externalSSD) }!.outstandingBytes
            let service = (fromHot ? readLatency.hot : readLatency.gguf) + Double(bank.record) / (fromHot ? bandwidth.hot : bandwidth.gguf)
            if !ahead, adaptiveGuesses {
                let decision = prefetchPolicy.decide(
                    layer: layer, rank: asked.firstIndex(of: expert) ?? 0,
                    queuedRecords: inFlight.values.filter { $0.done.wait(timeout: .now()) != .success }.count,
                    maximumRecords: max(2, weights.geometry.expertsPerToken),
                    queuedSeconds: Double(queued) / (fromHot ? bandwidth.hot : bandwidth.gguf), serviceSeconds: service)
                switch decision {
                case .admit: break
                case .confidence:
                    counters.guessRejectedConfidence += 1
                    counters.guessDrives[fromHot ? 0 : 1].refusedConfidence += 1
                    prefetchObservations?.rejected(layer: layer, expert: expert, reason: .confidence)
                    continue
                case .queue:
                    counters.guessRejectedQueue += 1
                    counters.guessDrives[fromHot ? 0 : 1].refusedQueue += 1
                    prefetchObservations?.rejected(layer: layer, expert: expert, reason: .queue)
                    continue
                case .deadline:
                    counters.guessRejectedDeadline += 1
                    counters.guessDrives[fromHot ? 0 : 1].refusedDeadline += 1
                    prefetchObservations?.rejected(layer: layer, expert: expert, reason: .deadline)
                    continue
                }
            }
            // Early dispatch reuses owned storage only: no new slab or relief callback while demand readers own
            // their targets. Current-wave stamps, busy guesses, and retained prediction records stay protected.
            var request: FamilyEviction.Request? = FamilyEviction.Request(
                pool: p, start: lastStart, usable: usable(p), limit: limits[p], prompt: prompt, guess: true, bound: bound,
                eligible: inDemandSubmission ? pools[p].first..<(pools[p].first + allocatedSlots(layer: layer)) : usable(p),
                grows: ledger?.optionalGrowthAllowed != false)
            prompt = nil  // the request owns it while a victim is taken
            locked { swap(&policy.request, &request) }
            let acquired = try? core.acquire(key)
            locked { swap(&policy.request, &request) }
            prompt = request?.prompt
            guard let ticket = acquired?.ticket, let into = try? core.beginRead(ticket) else {
                if let ticket = acquired?.ticket { try? core.cancel(ticket) }
                // Growth refused and no existing slot free: a pressure refusal, live when the governor says pressure is
                // live now, else the budget below its target.
                let pressure = ledger?.optionalGrowthAllowed == false
                if pressure { ledger?.noteOptionalRefusal(live: ledger?.optionalReadsAllowed == false, count: fresh.count - freshIndex) }
                if !ahead {
                    if pressure { counters.guessRejectedPressure += fresh.count - freshIndex }
                    var left = share
                    for skipped in fresh[freshIndex...] {
                        if pressure {
                            let copy = left > 0 && hot?.holds(layer * e + skipped) == true
                            if copy { left -= 1 }
                            counters.guessDrives[copy ? 0 : 1].refusedPressure += 1
                        }
                        prefetchObservations?.rejected(layer: layer, expert: skipped, reason: pressure ? .pressure : .noVictimOrCapacity)
                    }
                }
                return
            }
            let victim = pools[p].first + ticket.slot
            locked {
                policy.clock += 1
                policy.touch(victim, heat: ahead ? 1 : Self.guessHeat)
                policy.scan[victim] = ahead
                policy.placed(key, at: victim)
            }
            if fromHot { share -= 1 }
            let parts = fromHot ? [-1] : [0, 1, 2]
            let guess = Guess(ticket: ticket, ahead: ahead, parts: parts.count)
            if !ahead {
                guess.observation = prefetchObservations?.issued(
                    layer: layer, expert: expert,
                    planned: fromHot ? .hot : .gguf)
            }
            inFlight[key] = guess
            unread[key] = ahead
            if ahead {
                counters.readAheads += 1
                counters.readBytes += bank.record
            } else {
                counters.prefetches += 1
                counters.prefetchBytes += bank.record
                counters.guessDrives[fromHot ? 0 : 1].issued += 1
                guessDrive[key] = fromHot ? 0 : 1
                if routerAt > 0 {
                    let issued = DispatchTime.now().uptimeNanoseconds
                    guess.routerAt = routerAt
                    counters.guessDrives[fromHot ? 0 : 1].issueSeconds += Double(issued &- min(issued, routerAt)) / 1e9
                    counters.guessDrives[fromHot ? 0 : 1].issues += 1
                }
                if let rank = asked.firstIndex(of: expert) {
                    guessed?.issued[expert] = rank
                    Self.count(&counters.guessIssued, rank)
                }
            }
            // The slab stays alive with the read; only this guess writes its slot, and its ticket cannot publish a
            // recycled one.
            nonisolated(unsafe) let target = into
            // A GGUF record's gate, up and down are read at once, as `fetch` reads them; once one began none is cancelled.
            let deadline =
                !ahead && adaptiveGuesses ? prefetchPolicy.horizon(layer: layer).map { now + UInt64($0 * 1e9) } : nil
            var tickets: [IOTicket] = []
            for (n, part) in parts.enumerated() {
                do {
                    tickets.append(
                        try reads.submit(
                            source: part == -1 ? .internalSSD : .externalSSD, priority: .prefetch,
                            generation: reads.currentGeneration, byteCount: part < 0 ? bank.record : bank.bytes[part],
                            operation: { _ in
                                lock.lock()
                                if !guess.started, !guess.demanded, deadline.map({ DispatchTime.now().uptimeNanoseconds >= $0 }) == true {
                                    guess.cancelled = true
                                }
                                let go = !guess.cancelled
                                guess.started = guess.started || go
                                lock.unlock()
                                guard go else { return }
                                probe?(key)
                                do {
                                    let deliveredFromHot = try Self.read(
                                        part, layer: layer, expert: expert, of: e, bank: bank, files: files, hot: hot,
                                        into: target.storage.buffer.contents() + target.offset)
                                    if deliveredFromHot {
                                        lock.lock()
                                        guess.deliveredFromHot = true
                                        lock.unlock()
                                    }
                                } catch {
                                    lock.lock()
                                    guess.error = error
                                    lock.unlock()
                                }
                            },
                            completion: { _, outcome in
                                // Cancelled while queued: it read nothing. Unstarted, the guess is cancelled as `settle`
                                // cancels one; begun by another part, its record is incomplete, so it fails.
                                lock.lock()
                                if case .cancelled = outcome {
                                    if guess.started { guess.error = guess.error ?? IOOperationCancelled() } else { guess.cancelled = true }
                                }
                                guess.endedAt = max(guess.endedAt, DispatchTime.now().uptimeNanoseconds)
                                lock.unlock()
                                guess.done.leave()
                            }))
                } catch {
                    // Admission refused: this part and the later ones never run.
                    lock.lock()
                    if guess.started { guess.error = guess.error ?? error } else { guess.cancelled = true }
                    lock.unlock()
                    for _ in n..<parts.count { guess.done.leave() }
                    break
                }
            }
            guessTickets[key] = tickets
        }
    }

    /// Experts of `layer` resident now (for miss substitution and fast decode): records whose read was verified;
    /// guessed records still being read are not.
    package func resident(layer: Int) -> Set<Int> {
        settle(keeping: nil)
        let e = weights.geometry.experts
        return Set(
            (0..<e).filter { expert in entry(layer * e + expert).map { [.verified, .pinned, .evictable].contains($0.phase) } == true })
    }

    /// The phase of `expert`'s slot of `layer` (nil: not in the lookup), for checks.
    package func phase(layer: Int, expert: Int) -> ExpertSlotPhase? { entry(layer * weights.geometry.experts + expert)?.phase }

    /// Checks only: the first slot whose phase `FamilyEviction` mirrors differently from the core, nil when all agree.
    package func evictionMirrorMismatch() -> String? { locked { policy.stateMismatch(core) } }

    /// The slot core itself, for checks (tickets, generations).
    package var slotCore: ExpertSlotCore<FamilyEviction> { core }
}
