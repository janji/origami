import Foundation
import Metal
import OrigamiEngine
import Synchronization

/// `engine-checks --bench-deepseek-decode <shard 1 of a DeepSeek-V4.1-Flash GGUF> [contexts] [read µs] [mode]`: one decode
/// token (n = 1) of `DeepSeekModel.run` at each context (default 1024,8192,20480) with the GGUF header's shapes and ggml
/// types, random weights and state (no tensor data is read), the engine's own kernels and dispatches, and its commits (the
/// router's, the shared expert's and the routed experts' each MoE layer, the head's). Prints the GPU time a token takes as the engine runs it (the
/// `gpu_seconds` counter) and, from a second pass with every dispatch in its own compute pass between timestamp samples,
/// where that time goes by kind of dispatch. Then the GPU's idle time a token, with the engine's host work at each MoE
/// layer (the router's top-k, as `FamilyRunner.moe` picks, and a stand-in for misses' reads, `read µs` of spinning by
/// default, other waits and real reads by its suffix below) between the router's wait and the experts' encoding, with
/// `FamilyRunner.overlapsRouting` off and on, and the process's CPU time a token. A fourth argument, options joined by
/// commas: `keepalive` (or `keepalive-short`, its short poll) runs every token under Qwen's `RequestGPUKeepAlive`, which the
/// family engine does not use; `cpu-spin` one more thread (user-interactive) spinning on the CPU throughout; `qos-ui`,
/// `qos-ut` and `rt` give this thread user-interactive or utility QoS or a real-time policy; `router-spin` polls the
/// router's buffer instead of waiting for it; `early` compares `FamilyRunner.runsResidentFirst` off and on in place of the
/// overlap (with real reads: the layer's other experts resident, run while the reads go on) ("-": none).
/// Model-free: weights live in a few rotating sets (about 2 GB); these are synthetic operator timings, not production
/// throughput or quality measurements. Engram table I/O and the expert-cache policy are not represented. Mirror
/// changes to `DeepSeekModel.run` for n = 1 here.
func deepSeekDecodeBench(_ arguments: [String]) throws {
    guard let path = arguments.first else {
        throw EngineError("Usage: engine-checks --bench-deepseek-decode <shard1.gguf> [contexts] [read µs and mode] [options] [read file]")
    }
    let contextFields =
        arguments.count > 1
        ? arguments[1].split(separator: ",", omittingEmptySubsequences: false).map(String.init) : ["1024", "8192", "20480"]
    let contexts = contextFields.compactMap(Int.init)
    guard !contexts.isEmpty, contexts.count == contextFields.count, contexts.allSatisfy({ $0 >= 0 && $0 < Int(Int32.max) }) else {
        throw EngineError("Decode benchmark contexts must be nonnegative integer positions fitting Int32")
    }
    // "<µs><mode>[<bound µs>]": mode "" spins on this thread, "s" sleeps (usleep) on it, "m" sleeps to a hard deadline
    // (mach_wait_until); "b", "f" and "h" read on a lane (it sleeps to the deadline) while this thread blocks on a
    // semaphore ("b"), spins on a flag ("f") or spins up to <bound> µs then blocks ("h"); "B", "F" and "H" the same with
    // the lane spinning for the µs (no timer: this process's timers fire ~8 ms late). With a fifth argument, a file (on
    // the drive the experts are read from, opened uncached), <µs> is instead the records a MoE layer reads (the
    // layer's experts' sizes, at spread offsets): "p" as `FamilyExperts.fetch` (concurrentPerform, three parts each,
    // this thread one of the readers), "q" on lanes while this thread blocks, "r" while it spins on the count of parts
    // read (up to <bound> µs, then blocks; 0: no bound). "p" spins as decode's fetch does unless ORIGAMI_FAMILY_SPIN=0.
    let readArg = arguments.count > 2 ? arguments[2] : "0"
    let readMicros = UInt32(readArg.prefix { $0.isNumber }) ?? 0
    let readMode = String(readArg.drop { $0.isNumber }.prefix { !$0.isNumber })
    let spinBound = UInt64(readArg.drop { $0.isNumber }.drop { !$0.isNumber }) ?? 0
    let options = Set(arguments.count > 3 ? arguments[3].split(separator: ",").map(String.init) : [])
    let policy = options.compactMap { ["keepalive": RequestGPUKeepAlive.Policy.original, "keepalive-short": .shortRegisterAttention][$0] }
        .first
    // The owner must outlive the session: dropping it closes the session (its deinit).
    let owner = try policy.map { try RequestGPUKeepAlive(ledger: MemoryLedger(ceiling: 1 << 30, transientMargin: 1), policy: $0) }
    let keepAlive = try owner?.begin()
    defer { withExtendedLifetime(owner) { _ = keepAlive?.close() } }
    // An atomic flag: a plain one let the compiler drop the spinner's loads (the earlier cpu-spin runs spun nothing).
    final class Flag: Sendable { let stop = Atomic<Bool>(false) }
    let spinning = Flag()
    if options.contains("cpu-spin") {
        let spinner = Thread { while !spinning.stop.load(ordering: .relaxed) {} }
        spinner.qualityOfService = .userInteractive
        spinner.start()
    }
    defer { spinning.stop.store(true, ordering: .relaxed) }
    if options.contains("qos-ui") { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) }
    if options.contains("qos-ut") { pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0) }
    if options.contains("rt") {
        // A real-time (time-constraint) thread, as audio threads are: 5 ms period, 2 ms computation, 5 ms constraint.
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        let ticks = { (ms: Double) in UInt32(ms * 1e6 * Double(info.denom) / Double(info.numer)) }
        var tc = thread_time_constraint_policy_data_t(period: ticks(5), computation: ticks(2), constraint: ticks(5), preemptible: 1)
        let r = withUnsafeMutablePointer(to: &tc) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 4) {
                thread_policy_set(pthread_mach_thread_np(pthread_self()), thread_policy_flavor_t(THREAD_TIME_CONSTRAINT_POLICY), $0, 4)
            }
        }
        print("== real-time policy: \(r == KERN_SUCCESS ? "set" : "failed \(r)")")
    }
    print(
        "== serving thread QoS \(qos_class_self().rawValue), read stand-in \(readMicros) µs mode '\(readMode)' bound \(spinBound) µs, options \(options.sorted())"
    )
    let gguf = try GGUFSource(URL(fileURLWithPath: path))
    let geo = try ModelGeometry(gguf: gguf)
    guard geo.family == .deepseek, let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw EngineError("Needs a DeepSeek-V4.1-Flash GGUF and a Metal device")
    }
    let readFile = arguments.count > 4 ? try ExactFile(url: URL(fileURLWithPath: arguments[4])) : nil
    let bench = try DeepSeekDecodeBench(gguf: gguf, geo: geo, device: device, queue: queue, context: (contexts.max() ?? 1024) + 1)
    bench.readFile = readFile
    bench.readMicros = readMicros
    bench.readMode = readMode
    bench.spinBound = spinBound
    bench.routerSpin = options.contains("router-spin")
    let early = options.contains("early")
    let compared = early ? "resident first" : "overlap"
    for p in contexts {
        // Warm the pipelines and clocks, then the median of five tokens of each pass.
        for _ in 0..<2 { _ = try bench.token(at: p, profile: false) }
        try bench.validateFiniteOutput()
        let plain = try (0..<5).map { _ in try bench.token(at: p, profile: false).gpu }.sorted()[2]
        // Counter buffers and per-dispatch passes have their own first-use costs, separate from normal kernels.
        _ = try bench.token(at: p, profile: true)
        var kinds: [String: (ms: Double, dispatches: Int)] = [:]
        var profiled: [Double] = []
        var dispatches = 0
        var commits = 0
        for _ in 0..<3 {
            let r = try bench.token(at: p, profile: true)
            profiled.append(r.groups.reduce(0) { $0 + $1.ms })
            if kinds.isEmpty { for g in r.groups { kinds[g.kind, default: (0, 0)].dispatches += g.dispatches } }
            for g in r.groups { kinds[g.kind, default: (0, 0)].ms += g.ms / 3 }
            dispatches = r.dispatches
            commits = r.commits
        }
        try bench.validateFiniteOutput()
        print("\n== context \(p): outputs \(bench.digest())")
        print(
            "== context \(p): GPU \(String(format: "%.1f", plain * 1000)) ms per token as the engine runs it (median of 5); "
                + "\(dispatches) dispatches, \(commits) command buffers; profiled sum \(String(format: "%.1f", profiled.sorted()[1])) ms")
        for (kind, v) in kinds.sorted(by: { $0.value.ms > $1.value.ms }) {
            print(String(format: "%8.2f ms %6d  %@", v.ms, v.dispatches, kind))
        }
        // Idle: wall less GPU, the median of nine tokens of each layout, interleaved.
        var runs: [Bool: [(wall: Double, gpu: Double, commits: Int, cpu: Double)]] = [:]
        var digests: [Bool: String] = [:]
        bench.early = false
        func cpuSeconds() -> Double {
            var u = rusage()
            getrusage(RUSAGE_SELF, &u)
            return Double(u.ru_utime.tv_sec + u.ru_stime.tv_sec) + Double(u.ru_utime.tv_usec + u.ru_stime.tv_usec) / 1e6
        }
        for _ in 0..<9 {
            for overlap in [false, true] {
                if early { bench.early = overlap } else { bench.overlap = overlap }
                let cpu = cpuSeconds()
                let r = try bench.token(at: p, profile: false)
                runs[overlap, default: []].append((r.wall, r.gpu, r.commits, cpuSeconds() - cpu))
                digests[overlap] = bench.digest()
            }
        }
        if digests[false] != digests[true] {
            throw EngineError("\(compared) changed the outputs: \(digests[false]!) against \(digests[true]!)")
        }
        bench.overlap = true
        bench.early = false
        for overlap in [false, true] {
            let all = runs[overlap]!
            let wall = all.map(\.wall).sorted()[4]
            let gpu = all.map(\.gpu).sorted()[4]
            let idle = all.map { $0.wall - $0.gpu }.sorted()[4]
            let cpu = all.map(\.cpu).sorted()[4]
            print(
                String(
                    format:
                        "== context %d, %@ %@, reads %d µs a layer: wall %.1f ms, GPU %.1f ms, GPU idle %.1f ms, CPU %.1f ms per token (medians of 9); %d command buffers",
                    p, compared, overlap ? "on " : "off", readMicros, wall * 1000, gpu * 1000, idle * 1000, cpu * 1000, all[0].commits))
        }
    }
}

final class DeepSeekDecodeBench {
    typealias Ref = FamilyKernels.Ref
    typealias Matrix = FamilyWeights.Matrix
    let device: MTLDevice, queue: MTLCommandQueue, kernels: FamilyKernels, matrices: GGMLMatrixKernels
    let gguf: GGUFSource, geo: ModelGeometry
    let h: Int, layers: Int, heads: Int, d: Int, rope: Int, qRank: Int, groups: Int, oRank: Int, window: Int, ring: Int
    let indexHeads: Int, indexDim: Int, indexTop: Int, candidateSource: Int, candidateBlock: Int, candidateBlocks: Int
    let ratios: [Int], kvSource: [Int], kvSources: Set<Int>, indexSources: Set<Int>, engramLayers: [Int], engramWidth: Int
    let contextLimit: Int, sinkhorn: Int
    /// `FamilyRunner.overlapsRouting`, and the spin (or with `sleeps` the sleep) standing in for a MoE layer's reads (µs).
    var overlap = true, readMicros: UInt32 = 0, readMode = "", spinBound: UInt64 = 0, routerSpin = false
    /// `FamilyRunner.runsResidentFirst`: with real reads (`readFile`), the experts not read run while the reads go on.
    var early = false
    var readFile: ExactFile?
    private var readTarget: MTLBuffer?, readCount = 0
    private var readScheduler: IOScheduler?
    private final class ReadFlag: Sendable {
        let done = Atomic<Int>(0)
        let signal = DispatchSemaphore(value: 0)
    }
    private let flag = ReadFlag()
    /// The stand-in for a MoE layer's reads (`readMode`, see `deepSeekDecodeBench`).
    func read(layer l: Int, meanwhile: (() throws -> Void)? = nil) throws {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        let deadline = mach_absolute_time() + UInt64(readMicros) * 1000 * UInt64(info.denom) / UInt64(info.numer)
        let flag = self.flag
        flag.done.store(0, ordering: .relaxed)
        if let file = readFile {
            let parts = try ["gate", "up", "down"].map { role -> Int in
                guard let t = gguf.tensors["blk.\(l).ffn_\(role)_exps.weight"] else {
                    throw EngineError("The header lacks layer \(l)'s \(role) experts")
                }
                return Int(t.length) / geo.experts
            }
            let record = (parts.reduce(0, +) + 16383) / 16384 * 16384
            let records = Int(readMicros)
            let slots = Int(file.identity.size) / record
            guard slots > 0 else { throw EngineError("Decode benchmark read file cannot hold one expert record") }
            let (targetBytes, overflow) = records.multipliedReportingOverflow(by: record)
            guard !overflow, targetBytes > 0 else { throw EngineError("Invalid decode benchmark read size") }
            if readTarget == nil || readTarget!.length < targetBytes {
                guard let buffer = device.makeBuffer(length: targetBytes, options: .storageModeShared) else {
                    throw EngineError("Cannot allocate decode benchmark read target")
                }
                readTarget = buffer
            }
            nonisolated(unsafe) let target = readTarget!.contents()
            let base = readCount
            readCount += records
            // `FamilyExperts.perform`, as decode's `fetch` reads: "p" with its bound (twice the bytes at the unmeasured 2 GB/s;
            // ORIGAMI_FAMILY_SPIN=0: blocking), "q" and "r" spinning 0 and <bound> µs.
            let spin: Double? =
                readMode == "p"
                ? (FamilyExperts.spins ? 2 * Double(records * record) / 2e9 : nil)
                : readMode == "q" ? 0 : spinBound == 0 ? 1 : Double(spinBound) * 1e-6
            final class Failure: @unchecked Sendable {
                let lock = NSLock()
                var error: Error?
                func record(_ value: Error) {
                    lock.lock()
                    if error == nil { error = value }
                    lock.unlock()
                }
            }
            let failure = Failure()
            if readScheduler == nil { readScheduler = try FamilyExperts.makeReadScheduler(ledger: MemoryLedger(), jobs: records * 3) }
            try FamilyExperts.perform(
                records * 3, on: readScheduler!, spin: spin, meanwhile: { do { try meanwhile?() } catch { failure.record(error) } }
            ) { j in
                let r = j / 3
                let p = j % 3
                let at = (base + r) &* 7919 % slots * record + parts.prefix(p).reduce(0, +)
                do {
                    try file.read(
                        offset: Int64(at),
                        into: UnsafeMutableRawBufferPointer(start: target + r * record + parts.prefix(p).reduce(0, +), count: parts[p]))
                } catch { failure.record(error) }
            }
            if let error = failure.error { throw error }
            return
        }
        try meanwhile?()
        switch readMode {
        case "s": usleep(readMicros)
        case "m": mach_wait_until(deadline)
        case "b", "f", "h", "B", "F", "H":
            let precise = readMode.uppercased() == readMode
            DispatchQueue.global(qos: .userInitiated).async {
                if precise { while mach_absolute_time() < deadline {} } else { mach_wait_until(deadline) }
                flag.done.store(1, ordering: .releasing)
                flag.signal.signal()
            }
            let until =
                DispatchTime.now().uptimeNanoseconds
                + (readMode.lowercased() == "f" ? UInt64.max / 2 : readMode.lowercased() == "h" ? spinBound * 1000 : 0)
            while flag.done.load(ordering: .acquiring) == 0 && DispatchTime.now().uptimeNanoseconds < until {}
            flag.signal.wait()
        default: while mach_absolute_time() < deadline {}
        }
    }
    /// Rotating weight sets: layer l reads set l % sets of each (role, type, shape), experts one of `expertSets`.
    let sets = 4, expertSets = 24
    private var weights: [String: Matrix] = [:], floats: [String: MTLBuffer] = [:], scratch: [String: MTLBuffer] = [:]
    private var rng: UInt64 = 0x9E37_79B9_7F4A_7C15
    private var ringKV: [MTLBuffer] = [], latent: [Int: MTLBuffer] = [:], indexKeys: [Int: MTLBuffer] = [:], compKV: [Int: MTLBuffer] = [:],
        compScore: [Int: MTLBuffer] = [:]
    private let noPositionBias: MTLBuffer

    init(gguf: GGUFSource, geo: ModelGeometry, device: MTLDevice, queue: MTLCommandQueue, context: Int) throws {
        self.gguf = gguf
        self.geo = geo
        self.device = device
        self.queue = queue
        kernels = try FamilyKernels(device: device)
        matrices = try GGMLMatrixKernels(device: device)
        h = geo.hidden
        layers = geo.layers
        heads = try geo.int("num_attention_heads", 64, 1...64)
        d = try geo.int("head_dim", 512, 4...4096)
        rope = try geo.int("qk_rope_head_dim", 64, 0...4096)
        qRank = try geo.int("q_lora_rank", 1280, 1...1 << 20)
        groups = try geo.int("o_groups", 8, 1...64)
        oRank = try geo.int("o_lora_rank", 1024, 1...1 << 20)
        window = try geo.int("sliding_window", 128, 1...1 << 20)
        ring = window + 256
        indexHeads = try geo.int("index_n_heads", 32, 1...1024)
        indexDim = try geo.int("index_head_dim", 128, 4...4096)
        indexTop = try geo.int("index_topk", 512, 1...1 << 20)
        candidateSource = try geo.int("candidate_source_layer_id", -1, -1...1000)
        candidateBlocks = try geo.int("candidate_topk_blocks", 2048, 1...1 << 20)
        candidateBlock = try geo.int("candidate_block_size", 8, 1...1 << 20)
        sinkhorn = try geo.int("hc_sinkhorn_iters", 20, 0...1000)
        ratios = geo.lists["compress_ratios"] ?? []
        kvSources = Set(geo.lists["kv_source_layer_ids"] ?? [])
        indexSources = Set(geo.lists["index_source_layer_ids"] ?? [])
        engramLayers = geo.lists["engram_layer_ids"] ?? []
        engramWidth =
            (try geo.int("engram_max_ngram_size", 4, 2...16) - 1) * (try geo.int("engram_n_heads", 8, 1...1024))
            * (try geo.int("engram_head_dim", 256, 1...1 << 20))
        var last = -1
        var sources: [Int] = []
        for l in 0..<layers {
            if kvSources.contains(l) { last = l }
            sources.append(last)
        }
        kvSource = sources
        contextLimit = context
        guard let zero = device.makeBuffer(length: d * 8 * 4, options: .storageModeShared) else { throw EngineError("Out of memory") }
        memset(zero.contents(), 0, zero.length)
        noPositionBias = zero
        for _ in 0..<layers { ringKV.append(try random(ring * d)) }
        for l in kvSources.sorted() {
            let entries = context / ratios[l] + 1
            latent[l] = try random(entries * d)
            indexKeys[l] = try random(entries * indexDim)
            if ratios[l] > 1 {
                compKV[l] = try random(ratios[l] * d)
                compScore[l] = try random(ratios[l] * d)
            }
        }
    }

    // MARK: random weights of the header's types

    private func next() -> UInt64 {
        rng ^= rng << 13
        rng ^= rng >> 7
        rng ^= rng << 17
        return rng
    }
    /// Floats in [-scale, scale).
    private func random(_ count: Int, scale: Float = 0.05) throws -> MTLBuffer {
        guard let b = device.makeBuffer(length: max(16, count * 4), options: .storageModeShared) else { throw EngineError("Out of memory") }
        let p = b.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count { p[i] = (Float(next() >> 40) / Float(1 << 24) * 2 - 1) * scale }
        return b
    }
    /// The f16 scale fields of each block (byte offsets), set to a small value so random blocks decode to finite weights.
    private static let scaleFields: [Int: [Int]] = [
        8: [0], 14: [208], 13: [0, 2], 12: [0, 2], 11: [108], 10: [80, 82], 16: [0], 19: [0], 29: [], 17: [0], 22: [0], 18: [0], 21: [0],
        20: [0], 23: [0],
    ]
    private func matrix(_ type: Int, width: Int, outputs: Int) throws -> Matrix {
        let (block, bytes) = GGUFQuant.sizes[type]!
        let length = outputs * (width / block) * bytes
        guard let b = device.makeBuffer(length: length, options: .storageModeShared) else { throw EngineError("Out of memory") }
        let words = b.contents().bindMemory(to: UInt64.self, capacity: length / 8)
        for i in 0..<length / 8 { words[i] = next() }
        let raw = b.contents()
        if type == 0 {
            let p = raw.bindMemory(to: Float.self, capacity: length / 4)
            for i in 0..<length / 4 { p[i] = Float(Int32(truncatingIfNeeded: Int64(bitPattern: next())) >> 8) * 0x1p-30 }
        } else if type == 30 || type == 1 {
            let p = raw.bindMemory(to: UInt16.self, capacity: length / 2)
            for i in 0..<length / 2 {
                let f = Float(Int32(truncatingIfNeeded: Int64(bitPattern: next())) >> 8) * 0x1p-30
                p[i] = type == 30 ? UInt16(f.bitPattern >> 16) : Float16(f).bitPattern
            }
        } else {
            guard type == 39 || Self.scaleFields[type] != nil else {
                throw EngineError("Decode benchmark lacks a finite fixture generator for quantization type \(type)")
            }
            let small = Float16(0.002).bitPattern
            for blockAt in stride(from: 0, to: length, by: bytes) {
                // MXFP4's E8M0 scale is one exponent byte, not an f16. Random bytes include NaN and enormous
                // exponents; they made every logit non-finite and rendered digest equality vacuous.
                if type == 39 { raw.storeBytes(of: UInt8(119), toByteOffset: blockAt, as: UInt8.self) }
                for f in Self.scaleFields[type] ?? [] { raw.storeBytes(of: small, toByteOffset: blockAt + f, as: UInt16.self) }
            }
        }
        return Matrix(buffer: b, offset: 0, type: type, width: width, outputs: outputs, name: "bench")
    }
    /// Layer l's tensor `name` ({L}) as a matrix of its header type and shape, from the rotating set `set`.
    private func weight(_ name: String, layer l: Int, set: Int? = nil) throws -> Matrix {
        let resolved = name.replacingOccurrences(of: "{L}", with: "\(l)")
        guard let t = gguf.tensors[resolved], let width = t.shape.last else { throw EngineError("The header lacks \(resolved)") }
        let outputs = t.shape.dropLast().reduce(1, *)
        let key = "\(name)/\(t.type)/\(width)x\(outputs)/\(set ?? l % sets)"
        if let m = weights[key] { return m }
        let m = try matrix(t.type, width: width, outputs: outputs)
        weights[key] = m
        return m
    }
    /// Routed expert z of layer l (one of `expertSets` per type and shape): rows of one expert of a stacked tensor.
    private func expert(_ role: String, layer l: Int, _ z: Int) throws -> Matrix {
        guard let t = gguf.tensors["blk.\(l).ffn_\(role)_exps.weight"], t.shape.count == 3 else {
            throw EngineError("The header lacks layer \(l)'s \(role) experts")
        }
        let key = "expert-\(role)/\(t.type)/\((l * 6 + z) % expertSets)"
        if let m = weights[key] { return m }
        let m = try matrix(t.type, width: t.shape[2], outputs: t.shape[1])
        weights[key] = m
        return m
    }
    private func vector(_ key: String, _ count: Int) throws -> MTLBuffer {
        if let b = floats[key] { return b }
        let b = try random(count, scale: 1)
        floats[key] = b
        return b
    }
    func validateFiniteOutput() throws {
        guard let logits = scratch["logits"] else { throw EngineError("Decode benchmark produced no logits") }
        let values = UnsafeBufferPointer(start: logits.contents().assumingMemoryBound(to: Float.self), count: geo.vocab)
        guard values.allSatisfy(\.isFinite), values.contains(where: { $0 != 0 }) else {
            throw EngineError(
                "Decode benchmark produced non-finite or all-zero logits; timings are not accepted as valid synthetic evidence")
        }
    }
    /// FNV-1a of the last token's logits, attention output and top-k rows, and the logits' non-finite count: two builds
    /// whose kernels give the same bits print the same digest.
    func digest() -> String {
        func fnv(_ name: String) -> String {
            guard let b = scratch[name] else { return "-" }
            var hash: UInt64 = 0xcbf2_9ce4_8422_2325
            let p = b.contents().bindMemory(to: UInt8.self, capacity: b.length)
            for i in 0..<b.length { hash = (hash ^ UInt64(p[i])) &* 0x100_0000_01b3 }
            return String(hash, radix: 16)
        }
        let logits =
            scratch["logits"].map {
                Array(UnsafeBufferPointer(start: $0.contents().bindMemory(to: Float.self, capacity: geo.vocab), count: geo.vocab))
            } ?? []
        return "logits \(fnv("logits")) (\(logits.filter { !$0.isFinite }.count) not finite, max \(logits.filter(\.isFinite).max() ?? 0)), "
            + "attention \(fnv("attn-out")), top-k \(fnv("ds-topk")), mixes \(fnv("hc-comb"))"
    }
    private func work(_ name: String, _ count: Int) throws -> MTLBuffer {
        if let b = scratch[name], b.length >= count * 4 { return b }
        let b = try random(count)
        scratch[name] = b
        return b
    }

    // MARK: encoding, as FamilyRunner does it

    private var command: MTLCommandBuffer!, encoder: MTLComputeCommandEncoder!, buffers: [MTLCommandBuffer] = []
    /// Profiled passes sample into these in turn (4,096 samples each, Metal's limit).
    private var profile = false, sampleBuffers: [MTLCounterSampleBuffer] = [], samples = 0
    private static let perSampleBuffer = 4096
    private var groupKinds: [(kind: String, dispatches: Int)] = []
    private var kind = ""
    private func open() {
        command = queue.makeCommandBuffer()!
        if !profile { encoder = command.makeComputeCommandEncoder()! }
    }
    /// Profiled: each dispatch in its own compute pass between two timestamp samples.
    private func pass(_ body: () throws -> Void) throws {
        guard profile else { return try body() }
        let pd = MTLComputePassDescriptor()
        pd.dispatchType = .serial
        let set = device.counterSets!.first { $0.name == MTLCommonCounterSet.timestamp.rawValue }!
        if samples / Self.perSampleBuffer >= sampleBuffers.count {
            let desc = MTLCounterSampleBufferDescriptor()
            desc.counterSet = set
            desc.storageMode = .shared
            desc.sampleCount = Self.perSampleBuffer
            sampleBuffers.append(try device.makeCounterSampleBuffer(descriptor: desc))
        }
        pd.sampleBufferAttachments[0].sampleBuffer = sampleBuffers[samples / Self.perSampleBuffer]
        pd.sampleBufferAttachments[0].startOfEncoderSampleIndex = samples % Self.perSampleBuffer
        pd.sampleBufferAttachments[0].endOfEncoderSampleIndex = samples % Self.perSampleBuffer + 1
        samples += 2
        encoder = command.makeComputeCommandEncoder(descriptor: pd)!
        try body()
        encoder.endEncoding()
        groupKinds.append((kind, 1))
    }
    private func flush(wait: Bool, reopen: Bool = true) {
        if !profile { encoder.endEncoding() }
        command.commit()
        buffers.append(command)
        if wait { command.waitUntilCompleted() }
        if reopen { open() }
    }
    private func dispatch(
        _ name: String, _ refs: [Ref], ints: [Int32] = [], floats: [Float] = [], grid: MTLSize, group: MTLSize, as label: String? = nil
    ) throws {
        kind = label ?? name
        try pass { try kernels.dispatch(encoder, name, refs, ints: ints, floats: floats, grid: grid, group: group) }
    }
    private func mm(_ m: Matrix, _ input: Ref, _ out: Ref, _ what: String) throws {
        kind = "mm \(what) \(GGUFQuant.name(m.type)) \(m.width)x\(m.outputs)"
        try pass {
            try matrices.encode(
                encoder, type: m.type, weights: m.buffer, weightOffset: m.offset, width: m.width, outputs: m.outputs,
                input: input.buffer, inputOffset: input.offset, rows: 1, output: out.buffer, outputOffset: out.offset)
        }
    }
    /// FamilyRunner.mmBatch for one row.
    private func mmBatch(_ ms: [Matrix], _ input: Ref, inputStride: Int, _ out: Ref, outputStride: Int, _ what: String) throws {
        let m = ms[0]
        guard matrices.batches(type: m.type) else {
            for (z, x) in ms.enumerated() {
                try mm(x, Ref(input.buffer, input.offset + z * inputStride), Ref(out.buffer, out.offset + z * outputStride), what)
            }
            return
        }
        kind = "mm-batch \(what) \(GGUFQuant.name(m.type)) \(ms.count)x \(m.width)x\(m.outputs)"
        try pass {
            try matrices.encodeBatch(
                encoder, type: m.type, matrices: ms.map { ($0.buffer, $0.offset) }, width: m.width, outputs: m.outputs,
                input: input.buffer, inputOffset: input.offset, inputStride: inputStride, rows: 1,
                output: out.buffer, outputOffset: out.offset, outputStride: outputStride)
        }
    }
    private func grid2(_ n: Int, _ rows: Int) -> (MTLSize, MTLSize) {
        (MTLSize(width: (n + 255) / 256, height: rows, depth: 1), MTLSize(width: 256, height: 1, depth: 1))
    }
    private func rms(_ x: Ref, _ w: MTLBuffer?, _ y: Ref, n: Int, _ what: String) throws {
        try dispatch(
            "rms_norm", [x, Ref(w ?? x.buffer), y], ints: [Int32(n), w == nil ? 0 : 1], floats: [1e-20],
            grid: MTLSize(width: 1, height: 1, depth: 1),
            group: MTLSize(width: min(1024, max(32, (n + 31) / 32 * 32)), height: 1, depth: 1), as: "rms_norm \(what)")
    }
    private func rope(_ x: Ref, dim: Int, vectors: Int, first: Int, step: Int = 1, inverse: Bool = false, _ what: String) throws {
        try dispatch(
            "rope_tail", [x, Ref(try vector("freqs", rope / 2))],
            ints: [Int32(dim), Int32(rope), Int32(vectors), Int32(first), Int32(step), inverse ? 1 : 0],
            grid: MTLSize(width: 1, height: vectors, depth: 1), group: MTLSize(width: rope / 2, height: 1, depth: 1),
            as: "rope_tail \(what)")
    }
    private func mixes(_ part: String, layer l: Int, streams: Ref, pre: Ref, post: Ref, comb: Ref, _ what: String) throws {
        let flat = try work("hc-flat", 4 * h)
        let m = try work("hc-mixes", 24)
        try rms(streams, nil, Ref(flat), n: 4 * h, "hc streams")
        try mm(try weight("blk.{L}.hc_\(part)_fn.weight", layer: l), Ref(flat), Ref(m), "hc_fn")
        try dispatch(
            "hc_prepare", [Ref(m), Ref(try vector("hc-scale", 3)), Ref(try vector("hc-base", 24)), pre, post, comb],
            ints: [1, Int32(sinkhorn)], floats: [1e-6],
            grid: MTLSize(width: 1, height: 1, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
    }
    private func collapse(_ streams: Ref, pre: Ref, out: Ref) throws {
        let (grid, group) = grid2(h, 1)
        try dispatch("hc_collapse", [streams, pre, out], ints: [Int32(h), 1], grid: grid, group: group)
    }

    /// One decode token at position `p`: GPU seconds summed over its command buffers (as `gpu_seconds`), and with
    /// `profile` each dispatch's GPU milliseconds by kind.
    func token(at p: Int, profile: Bool) throws -> (
        gpu: Double, wall: Double, groups: [(kind: String, ms: Double, dispatches: Int)], dispatches: Int, commits: Int
    ) {
        self.profile = profile
        buffers = []
        groupKinds = []
        samples = 0
        guard
            !profile
                || device.supportsCounterSampling(.atStageBoundary)
                    && device.counterSets?.contains(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue }) == true
        else {
            throw EngineError("This GPU cannot sample timestamps at stage boundaries")
        }
        let started = Date()
        try autoreleasepool {
            open()
            do { try encodeToken(at: p) } catch {
                if !profile { encoder.endEncoding() }
                throw error
            }
            flush(wait: true, reopen: false)
        }
        command = nil
        encoder = nil
        let wall = Date().timeIntervalSince(started)
        let gpu = buffers.reduce(0) { $0 + $1.gpuEndTime - $1.gpuStartTime }
        var groups: [(kind: String, ms: Double, dispatches: Int)] = []
        if profile {
            var t: [UInt64] = []
            for (i, sb) in sampleBuffers.enumerated() where i * Self.perSampleBuffer < samples {
                let data = try sb.resolveCounterRange(0..<min(Self.perSampleBuffer, samples - i * Self.perSampleBuffer))!
                data.withUnsafeBytes { t += $0.bindMemory(to: UInt64.self) }
            }
            do {
                // Timestamp ticks to ms: the token's first and last sample against its buffers' GPU start and end.
                let msPerTick = (buffers.last!.gpuEndTime - buffers[0].gpuStartTime) * 1000 / Double(t[samples - 1] - t[0])
                for (i, g) in groupKinds.enumerated() {
                    let a = t[2 * i]
                    let b = t[2 * i + 1]
                    groups.append((g.kind, b > a ? Double(b - a) * msPerTick : 0, g.dispatches))
                }
            }
        }
        return (gpu, wall, groups, groupKinds.count, buffers.count)
    }

    /// `DeepSeekModel.run` for one token (n = 1) at position p, as `FamilyRunner` encodes it.
    private func encodeToken(at p: Int) throws {
        var x = try work("streams-a", 4 * h)
        var y = try work("streams-b", 4 * h)
        let prevPre = try work("pre-ffn", 4)
        let attnPre = try work("pre-attn", 4)
        let post = try work("hc-post", 4)
        let comb = try work("hc-comb", 16)
        // The first attention collapses with the identity, as `DeepSeekModel.run` sets it, so every token at p gives the same bits.
        for (i, v) in [Float(1), 0, 0, 0].enumerated() { prevPre.contents().storeBytes(of: v, toByteOffset: i * 4, as: Float.self) }
        do {
            let (grid, group) = grid2(h, 1)
            try dispatch("hc_broadcast", [Ref(try work("embedded", h)), Ref(x)], ints: [Int32(h), 1], grid: grid, group: group)
        }
        for l in 0..<layers {
            if engramLayers.contains(l) {
                let kv = try work("engram-kv", 5 * h)
                try mm(try weight("blk.{L}.engram_kv.weight", layer: l), Ref(try work("engram-rows", engramWidth)), Ref(kv), "engram_kv")
                try dispatch(
                    "engram_gate", [Ref(x), Ref(kv), Ref(try vector("engram-w", 4 * h))], ints: [Int32(h), 4], floats: [1e-20],
                    grid: MTLSize(width: 4, height: 1, depth: 1), group: MTLSize(width: 1024, height: 1, depth: 1))
            }
            try mixes("attn", layer: l, streams: Ref(x), pre: Ref(attnPre), post: Ref(post), comb: Ref(comb), "attn")
            let cur = try work("attn-in", h)
            let out = try work("attn-out", h)
            try collapse(Ref(x), pre: Ref(prevPre), out: Ref(cur))
            try rms(Ref(cur), try vector("norm-h", h), Ref(cur), n: h, "hidden")
            try attention(layer: l, input: cur, output: out, start: p)
            do {
                let (grid, group) = grid2(h, 1)
                try dispatch("hc_expand", [Ref(out), Ref(x), Ref(post), Ref(comb), Ref(y)], ints: [Int32(h), 1], grid: grid, group: group)
            }
            swap(&x, &y)
            let ffnIn = try work("ffn-in", h)
            let mixed = try work("ffn-out", h)
            try mixes("ffn", layer: l, streams: Ref(x), pre: Ref(prevPre), post: Ref(post), comb: Ref(comb), "ffn")
            try collapse(Ref(x), pre: Ref(attnPre), out: Ref(ffnIn))
            try rms(Ref(ffnIn), try vector("norm-h", h), Ref(ffnIn), n: h, "hidden")
            var guessIn: MTLBuffer?
            if l + 1 < layers {
                let guessPre = try work("guess-pre", 4)
                let guess = try work("guess-in", h)
                try mixes(
                    "attn", layer: l + 1, streams: Ref(x), pre: Ref(guessPre), post: Ref(try work("guess-post", 4)),
                    comb: Ref(try work("guess-comb", 16)), "guess")
                try collapse(Ref(x), pre: Ref(guessPre), out: Ref(guess))
                try rms(Ref(guess), try vector("norm-h", h), Ref(guess), n: h, "hidden")
                guessIn = guess
            }
            try moe(layer: l, input: ffnIn, output: mixed, guessIn: guessIn)
            let (grid, group) = grid2(h, 1)
            try dispatch("hc_expand", [Ref(mixed), Ref(x), Ref(post), Ref(comb), Ref(y)], ints: [Int32(h), 1], grid: grid, group: group)
            swap(&x, &y)
        }
        let normed = try work("head-in", h)
        try collapse(Ref(x), pre: Ref(prevPre), out: Ref(normed))
        try rms(Ref(normed), try vector("norm-h", h), Ref(normed), n: h, "hidden")
        try mm(try weight("output.weight", layer: 0, set: 0), Ref(normed), Ref(try work("logits", geo.vocab)), "head")
    }

    private func attention(layer l: Int, input: MTLBuffer, output: MTLBuffer, start: Int) throws {
        let fresh = try work("ds-kv", d)
        let ringBuffer = ringKV[l]
        try mm(try weight("blk.{L}.attn_kv.weight", layer: l), Ref(input), Ref(fresh), "attn_kv")
        try rms(Ref(fresh), try vector("norm-d", d), Ref(fresh), n: d, "kv")
        try rope(Ref(fresh), dim: d, vectors: 1, first: start, "kv")
        do {
            let (grid, group) = grid2(d, 1)
            try dispatch("copy_rows", [Ref(fresh), Ref(ringBuffer, start % ring * d * 4)], ints: [Int32(d), 1], grid: grid, group: group)
        }
        let qr = try work("ds-qr", qRank)
        let q = try work("ds-q", heads * d)
        try mm(try weight("blk.{L}.attn_q_a.weight", layer: l), Ref(input), Ref(qr), "q_a")
        try rms(Ref(qr), try vector("norm-q", qRank), Ref(qr), n: qRank, "q")
        try mm(try weight("blk.{L}.attn_q_b.weight", layer: l), Ref(qr), Ref(q), "q_b")
        try rope(Ref(q), dim: d, vectors: heads, first: start, "q")
        let compressed = ratios[l] > 0
        let selected = window + (compressed ? indexTop : 0)
        let sel = try work("ds-sel", selected)
        try dispatch(
            "ring_select", [Ref(sel)], ints: [Int32(window), Int32(ring), Int32(selected), Int32(start)],
            grid: MTLSize(width: (window + 63) / 64, height: 1, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
        var second = ringBuffer
        let topk = try work("ds-topk", indexTop)
        if compressed {
            if kvSources.contains(l) { try compress(layer: l, input: input, first: start) }
            if indexSources.contains(l) { try index(layer: l, input: input, qr: qr, first: start, topk: topk) }
            try dispatch(
                "copy_int_rows", [Ref(topk), Ref(sel, window * 4)], ints: [Int32(indexTop), 1, Int32(indexTop), Int32(selected)],
                grid: MTLSize(width: (indexTop + 63) / 64, height: 1, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
            second = latent[kvSource[l]]!
        }
        let scores = try work("ds-scores", heads * selected)
        let o = try work("ds-o", heads * d)
        try dispatch(
            heads % 8 == 0 ? "attend_scores_heads8" : "attend_scores_heads", [Ref(q), Ref(ringBuffer), Ref(second), Ref(sel), Ref(scores)],
            ints: [Int32(heads), Int32(d), Int32(selected), Int32(heads * d), Int32(d), 0, 0, Int32(ring)],
            floats: [1 / Float(d).squareRoot()],
            grid: MTLSize(width: (selected + 63) / 64, height: 1, depth: heads % 8 == 0 ? heads / 8 : heads),
            group: MTLSize(width: 64, height: 1, depth: 1), as: "attend_scores \(selected)")
        try dispatch(
            "softmax_rows", [Ref(scores), Ref(try vector("sinks", heads))], ints: [Int32(selected), 1, Int32(heads)],
            grid: MTLSize(width: heads, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1),
            as: "softmax_rows \(selected)")
        try dispatch(
            heads % 8 == 0 ? "attend_values_heads8" : "attend_values_heads", [Ref(scores), Ref(ringBuffer), Ref(second), Ref(sel), Ref(o)],
            ints: [Int32(heads), Int32(d), Int32(selected), 0, 0, Int32(heads * d), Int32(d), Int32(ring)], floats: [0],
            grid: MTLSize(width: (d + 63) / 64, height: 1, depth: heads % 8 == 0 ? heads / 8 : heads),
            group: MTLSize(width: 64, height: 1, depth: 1), as: "attend_values \(selected)")
        try rope(Ref(o), dim: d, vectors: heads, first: start, inverse: true, "out")
        let perGroup = heads / groups * d
        let og = try work("ds-og", groups * perGroup)
        let low = try work("ds-low", groups * oRank)
        let lowRows = try work("ds-low-rows", groups * oRank)
        try dispatch(
            "permute3", [Ref(o), Ref(og)], ints: [1, Int32(groups), Int32(perGroup)],
            grid: MTLSize(width: (perGroup + 63) / 64, height: groups, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
        let outA = try weight("blk.{L}.attn_output_a.weight", layer: l)
        let (block, bytes) = GGUFQuant.sizes[outA.type]!
        let parts = (0..<groups).map {
            Matrix(
                buffer: outA.buffer, offset: $0 * oRank * (outA.width / block * bytes), type: outA.type, width: outA.width, outputs: oRank,
                name: "")
        }
        try mmBatch(parts, Ref(og), inputStride: perGroup * 4, Ref(low), outputStride: oRank * 4, "output_a")
        try dispatch(
            "permute3", [Ref(low), Ref(lowRows)], ints: [Int32(groups), 1, Int32(oRank)],
            grid: MTLSize(width: (oRank + 63) / 64, height: 1, depth: Int(groups)), group: MTLSize(width: 64, height: 1, depth: 1))
        try mm(try weight("blk.{L}.attn_output_b.weight", layer: l), Ref(lowRows), Ref(output), "output_b")
    }

    private func compress(layer l: Int, input: MTLBuffer, first: Int) throws {
        let ratio = ratios[l]
        let latents = latent[l]!
        let keys = indexKeys[l]!
        let firstGroup: Int
        let groupsDone: Int
        if ratio == 1 {
            try mm(try weight("blk.{L}.attn_compressor_kv.weight", layer: l), Ref(input), Ref(latents, first * d * 4), "compressor_kv")
            firstGroup = first
            groupsDone = 1
        } else {
            let carry = first % ratio
            try mm(try weight("blk.{L}.attn_compressor_kv.weight", layer: l), Ref(input), Ref(compKV[l]!, carry * d * 4), "compressor_kv")
            try mm(
                try weight("blk.{L}.attn_compressor_gate.weight", layer: l), Ref(input), Ref(compScore[l]!, carry * d * 4),
                "compressor_gate")
            firstGroup = first / ratio
            groupsDone = (first + 1) / ratio - firstGroup
            guard groupsDone > 0 else { return }
            try dispatch(
                "kpool_pool", [Ref(compKV[l]!), Ref(compScore[l]!), Ref(noPositionBias), Ref(latents, firstGroup * d * 4)],
                ints: [Int32(d), 0, Int32(groupsDone), Int32(ratio)],
                grid: MTLSize(width: (d + 63) / 64, height: groupsDone, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
        }
        let rowsRef = Ref(latents, firstGroup * d * 4)
        let keyRef = Ref(keys, firstGroup * indexDim * 4)
        try rms(rowsRef, try vector("norm-d", d), rowsRef, n: d, "kv")
        try mm(try weight("blk.{L}.indexer.attn_k.weight", layer: l), rowsRef, keyRef, "index_k")
        try rms(keyRef, try vector("norm-i", indexDim), keyRef, n: indexDim, "index key")
        try rope(keyRef, dim: indexDim, vectors: 1, first: firstGroup * ratio, step: ratio, "index key")
        try rope(rowsRef, dim: d, vectors: 1, first: firstGroup * ratio, step: ratio, "kv")
    }

    private func index(layer l: Int, input: MTLBuffer, qr: MTLBuffer, first: Int, topk: MTLBuffer) throws {
        let ratio = ratios[l]
        let source = kvSource[l]
        let entries = (first + 1) / ratio
        guard entries > 0 else {
            try dispatch(
                "fill_int", [Ref(topk)], ints: [Int32(indexTop), -1], grid: MTLSize(width: (indexTop + 255) / 256, height: 1, depth: 1),
                group: MTLSize(width: 256, height: 1, depth: 1))
            return
        }
        let iq = try work("ds-iq", indexHeads * indexDim)
        let w = try work("ds-iw", indexHeads)
        try mm(try weight("blk.{L}.indexer.attn_q_b.weight", layer: l), Ref(qr), Ref(iq), "index_q_b")
        try rope(Ref(iq), dim: indexDim, vectors: indexHeads, first: first, "index q")
        try mm(try weight("blk.{L}.indexer.proj.weight", layer: l), Ref(input), Ref(w), "index_proj")
        try dispatch(
            "scale_inplace", [Ref(w)], ints: [Int32(indexHeads)], floats: [1 / Float(indexDim * indexHeads).squareRoot()],
            grid: MTLSize(width: 1, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
        let scores = try work("ds-iscores", entries)
        let blocks = (entries + candidateBlock - 1) / candidateBlock
        let stride = (contextLimit + candidateBlock - 1) / candidateBlock
        let keep = try work("ds-candidates", stride)
        let candidates = candidateSource >= 0 && l >= candidateSource
        let masked = candidates && l > candidateSource
        var scoreRefs = [Ref(iq), Ref(indexKeys[source]!), Ref(w), Ref(scores)]
        var scoreInts = [Int32(indexHeads), Int32(indexDim), Int32(entries), Int32(ratio), Int32(first)]
        if masked {
            scoreRefs.append(Ref(keep))
            scoreInts += [Int32(candidateBlock), Int32(stride)]
        }
        try dispatch(
            masked ? "index_scores_masked" : "index_scores", scoreRefs, ints: scoreInts,
            grid: MTLSize(width: (entries + 63) / 64, height: 1, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
        if candidates && l == candidateSource {
            try dispatch(
                "candidate_blocks", [Ref(scores), Ref(keep)],
                ints: [
                    Int32(entries), Int32(candidateBlock), Int32(blocks), Int32(candidateBlocks), Int32(first), Int32(ratio), Int32(stride),
                ],
                grid: MTLSize(width: 1, height: 1, depth: 1), group: MTLSize(width: 256, height: 1, depth: 1))
        }
        try dispatch(
            "index_select", [Ref(scores), Ref(topk)],
            ints: [Int32(entries), Int32(ratio), Int32(indexTop), Int32(indexTop), Int32(first), 0, 1, Int32(ring), Int32(indexTop)],
            grid: MTLSize(width: 1, height: 1, depth: 1), group: MTLSize(width: 1024, height: 1, depth: 1))
    }

    /// FamilyRunner.moe for one row: the router and the next layer's guess, the waited flush, the shared expert and its
    /// flush (`overlap`: committed behind the router before the wait), the host's top-k and `readMicros`, then
    /// `runExperts` for six experts (`overlap`: committed at once).
    private func moe(layer l: Int, input: MTLBuffer, output: MTLBuffer, guessIn: MTLBuffer?) throws {
        let e = geo.experts
        let router = try work("router", e)
        try mm(try weight("blk.{L}.ffn_gate_inp.weight", layer: l), Ref(input), Ref(router), "router")
        if let guessIn {
            try mm(try weight("blk.{L}.ffn_gate_inp.weight", layer: l + 1), Ref(guessIn), Ref(try work("router-next", e)), "router")
        }
        let routed = command!
        flush(wait: !overlap)
        let width = geo.expertWidth
        let gate = try work("ffn-gate", width)
        let up = try work("ffn-up", width)
        try mm(try weight("blk.{L}.ffn_gate_shexp.weight", layer: l), Ref(input), Ref(gate), "shared gate/up")
        try mm(try weight("blk.{L}.ffn_up_shexp.weight", layer: l), Ref(input), Ref(up), "shared gate/up")
        do {
            let (grid, group) = grid2(width, 1)
            try dispatch(
                "swiglu", [Ref(gate), Ref(up), Ref(gate)], ints: [Int32(width), 1, Int32(width), Int32(width), Int32(width)], floats: [10],
                grid: grid, group: group)
        }
        try mm(try weight("blk.{L}.ffn_down_shexp.weight", layer: l), Ref(gate), Ref(output), "shared down")
        flush(wait: false)
        if overlap {
            if routerSpin { while routed.status != .completed && routed.status != .error {} } else { routed.waitUntilCompleted() }
        }
        // The host's work before the experts: DeepSeek's top-k as `moe` picks (its weights unused here), then the reads.
        let logits = router.contents().bindMemory(to: Float.self, capacity: e)
        let probs = (0..<e).map { i -> Float in
            let v = logits[i]
            return (v > 20 ? v : log1p(exp(v))).squareRoot()
        }
        _ = FamilyRouting.select(
            probs: probs, bias: [Float](repeating: 0, count: e), k: geo.expertsPerToken, dial: .exact, resident: nil, normalize: true,
            scale: 1, sum: { $0 + 1e-20 })
        // usleep ("s") coalesces short sleeps' timers (300 µs sleeps ~3 ms), where a read wakes on its I/O.
        // `early` with real reads: the records read are experts spread over the layer's six, the others resident and run
        // (and committed) while the reads go on, as `FamilyRunner.moe` does with `runsResidentFirst`.
        let k = geo.expertsPerToken
        let reads = early && readFile != nil ? min(k, Int(readMicros)) : 0
        let missed = Set((0..<reads).map { (l + $0 * k / reads) % k })
        let resident = (0..<k).filter { !missed.contains($0) }
        let splits = !missed.isEmpty && !resident.isEmpty
        if readMicros > 0 {
            try read(
                layer: l,
                meanwhile: splits
                    ? {
                        try self.expertOutputs(layer: l, resident, first: 0, input: input)
                        self.flush(wait: false)
                    } : nil)
        }
        let out = try work("exp-out", k * h)
        let w = Array(repeating: Float(0.1), count: k) + Array(repeating: 0, count: 64 - k)
        let (grid, group) = grid2(h, 1)
        if splits {
            let rest = missed.sorted()
            let order = resident + rest
            try expertOutputs(layer: l, rest, first: resident.count, input: input)
            var row = [Int32](repeating: 0, count: 64)
            for (r, z) in order.enumerated() { row[z] = Int32(r) }
            try dispatch(
                "add_weighted_rows_at", [Ref(out), Ref(output)], ints: [Int32(h), Int32(k)] + row, floats: w, grid: grid, group: group)
        } else {
            try expertOutputs(layer: l, Array(0..<k), first: 0, input: input)
            try dispatch("add_weighted_rows", [Ref(out), Ref(output)], ints: [Int32(h), Int32(k)], floats: w, grid: grid, group: group)
        }
        if overlap { flush(wait: false) }
    }

    /// `FamilyRunner.expertOutputs`: routed experts `zs` of layer l into rows `first ..< first + zs.count` of the layer's six.
    private func expertOutputs(layer l: Int, _ zs: [Int], first: Int, input: MTLBuffer) throws {
        let k = geo.expertsPerToken
        let width = geo.expertWidth
        let n = zs.count
        let buffer = try work("exp-gu", 2 * k * width)
        let out = try work("exp-out", k * h)
        let gu = Ref(buffer, 2 * first * width * 4)
        let up = Ref(buffer, (2 * first + n) * width * 4)
        let gates = try zs.map { try expert("gate", layer: l, $0) }
        let ups = try zs.map { try expert("up", layer: l, $0) }
        if gates[0].type == ups[0].type {
            try mmBatch(gates + ups, Ref(input), inputStride: 0, gu, outputStride: width * 4, "experts gate+up")
        } else {
            try mmBatch(gates, Ref(input), inputStride: 0, gu, outputStride: width * 4, "experts gate")
            try mmBatch(ups, Ref(input), inputStride: 0, up, outputStride: width * 4, "experts up")
        }
        let (grid, group) = grid2(width, n)
        try dispatch(
            "swiglu", [gu, up, gu], ints: [Int32(width), Int32(n), Int32(width), Int32(width), Int32(width)], floats: [10], grid: grid,
            group: group)
        try mmBatch(
            try zs.map { try expert("down", layer: l, $0) }, gu, inputStride: width * 4, Ref(out, first * h * 4), outputStride: h * 4,
            "experts down")
    }
}
