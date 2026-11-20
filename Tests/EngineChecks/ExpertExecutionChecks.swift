import Dispatch
import Foundation
import Metal
import OrigamiEngine

private final class ExpertExecutionReads: @unchecked Sendable {
    private let lock = NSLock()
    private var reads: [(ExpertKey, IOSource)] = []
    func add(_ key: ExpertKey, _ source: IOSource) {
        lock.lock()
        reads.append((key, source))
        lock.unlock()
    }
    var values: [(ExpertKey, IOSource)] {
        lock.lock()
        defer { lock.unlock() }
        return reads
    }
}
private func executionRequire(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    guard condition() else { throw EngineError(message) }
}
private func executionDevice() throws -> MTLDevice {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal device") }
    return device
}

func expertExecutionWindowIsBoundedAndReusesRecords() throws {
    let device = try executionDevice()
    for slots in [8, 64] {
        let ledger = try MemoryLedger(ceiling: 300_000_000, transientMargin: 1_000_000)
        let reads = ExpertExecutionReads()
        let config = try ExpertExecutionConfiguration(
            mode: .slabPackedPreferred,
            cacheBytes: UInt64(slots * PackStore.expertBytes + 123), demandWindow: 8, readConcurrency: 4)
        try executionRequire(
            config.cacheSlots == slots && config.cacheBytes == UInt64(slots * PackStore.expertBytes), "Cache budget was not rounded down")
        weak var releasedEngine: ExpertExecution?
        try autoreleasepool {
            var engine: ExpertExecution? = try ExpertExecution(
                device: device, ledger: ledger, configuration: config, packedAvailable: { false }
            ) {
                key, source, destination in
                reads.add(key, source)
                destination.initializeMemory(as: UInt8.self, repeating: UInt8(key.expert))
            }
            try engine!.beginSequence()
            var window: ExpertDemandWindow? = try engine!.makeWindow(layer: 0, experts: Array((0..<12).reversed()) + [3, 1])
            try executionRequire(window!.keys.map(\.expert) == Array(0..<12), "Layer union was not sorted/deduplicated")
            for expert in 0..<12 {
                try executionRequire(window!.pendingCount <= 8, "Demand window exceeded its pin bound")
                try window!.consume(expert: expert) { buffer in
                    try executionRequire(
                        buffer.bytes[0] == UInt8(expert) && buffer.bytes[PackStore.expertBytes - 1] == UInt8(expert),
                        "Wrong expert reached consumer")
                }
            }
            try executionRequire(window!.isClosed && engine!.snapshot().cache.pinnedSlots == 0, "Consumed window retained a pin")
            window = nil
            try engine!.finishSequence()
            let before = reads.values.count
            try engine!.beginSequence()
            var reused: ExpertDemandWindow? = try engine!.makeWindow(layer: 0, experts: [11])
            _ = try reused!.consume(expert: 11) { $0.bytes[0] }
            reused = nil
            try engine!.finishSequence()
            try executionRequire(
                reads.values.count == before && reads.values.allSatisfy { $0.1 == .internalSSD },
                "Packless reads left the original files, or cross-sequence cache reuse failed")
            try executionRequire(engine!.snapshot().cache.allocatedBytes <= config.cacheBytes, "Slabs exceeded configured budget")
            releasedEngine = engine
            engine = nil
        }
        // Match the executor's per-layer / server-worker autorelease boundary:
        // the raw Metal resource must remain charged until that boundary drains.
        // Physical resources remain charged through final worker/ARC cleanup,
        // which may follow the ticket's completed outcome by a few instructions.
        for _ in 0..<1_000 {
            if ledger.snapshot().reserved == 0 { break }
            Thread.sleep(forTimeInterval: 0.001)
        }
        try executionRequire(
            ledger.snapshot().reserved == 0,
            "Expert helper reservation remained after autorelease drain: engineAlive=\(releasedEngine != nil), categories=\(ledger.snapshot().categories)"
        )
    }
}

func expertExecutionPackedFallbackAndExceptionalConsumerCleanup() throws {
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
    let reads = ExpertExecutionReads()
    let config = ExpertExecutionConfiguration(
        mode: .slabPackedPreferred, cacheSlots: 8, slotsPerSlab: 8, demandWindow: 2, readConcurrency: 2)
    let engine = try ExpertExecution(device: executionDevice(), ledger: ledger, configuration: config) { key, source, destination in
        reads.add(key, source)
        if source == .externalSSD { throw EngineError("Injected external read failure") }
        destination.initializeMemory(as: UInt8.self, repeating: 0x42)
    }
    try engine.beginSequence()
    let first = try engine.makeWindow(layer: 1, experts: [7])
    try first.consume(expert: 7) { buffer in try executionRequire(buffer.bytes[0] == 0x42, "Fallback bytes missing") }
    try executionRequire(reads.values.map(\.1) == [.externalSSD, .internalSSD], "Packed fallback did not change scheduler drive")
    let failing = try engine.makeWindow(layer: 1, experts: [8, 9])
    do {
        try failing.consume(expert: 8) { _ in throw EngineError("Injected consumer failure") }
        throw ServingFailure("Consumer failure was swallowed")
    } catch let error as EngineError { try executionRequire(error.description == "Injected consumer failure", "Wrong consumer failure") }
    try executionRequire(
        failing.isClosed && engine.snapshot().cache.pinnedSlots == 0 && engine.snapshot().cache.inFlightReads == 0,
        "Exceptional consumer left pins or writers")
    try engine.finishSequence()
}

func expertExecutionCancellationJoinsBeforeReuse() throws {
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
    let control = GenerationControl()
    let started = DispatchSemaphore(value: 0)
    let release = DispatchSemaphore(value: 0)
    let config = ExpertExecutionConfiguration(
        mode: .slabPackedPreferred, cacheSlots: 8, slotsPerSlab: 8, demandWindow: 1, readConcurrency: 1)
    let engine = try ExpertExecution(device: executionDevice(), ledger: ledger, configuration: config, packedAvailable: { false }) {
        _, _, destination in
        started.signal()
        guard release.wait(timeout: .now() + 5) == .success else { throw EngineError("Cancellation producer timeout") }
        destination.initializeMemory(as: UInt8.self, repeating: 0xca)
    }
    try engine.beginSequence()
    let window = try engine.makeWindow(layer: 2, experts: [1, 2], shouldContinue: { !control.isCancelled })
    let group = DispatchGroup()
    group.enter()
    DispatchQueue.global().async {
        _ = started.wait(timeout: .now() + 5)
        control.cancel()
        Thread.sleep(forTimeInterval: 0.05)
        release.signal()
        group.leave()
    }
    do {
        try window.consume(expert: 1) { _ in throw EngineError("Cancelled expert became visible") }
        throw ServingFailure("Cancelled window returned")
    } catch is IOOperationCancelled {}
    group.wait()
    try executionRequire(
        window.isClosed && engine.snapshot().cache.inFlightReads == 0 && engine.snapshot().cache.pinnedSlots == 0,
        "Cancellation returned before writer/pins retired")
    try engine.finishSequence()
}

func expertExecutionBufferNativeMatchesIndividualPinnedQMV() throws {
    let ledger = try MemoryLedger(ceiling: 100_000_000, transientMargin: 1_000_000)
    let kernels = try ReferenceKernels()
    let config = ExpertExecutionConfiguration(
        mode: .slabPackedPreferred, cacheSlots: 8, slotsPerSlab: 8, demandWindow: 2, readConcurrency: 2)
    let engine = try ExpertExecution(device: executionDevice(), ledger: ledger, configuration: config, packedAvailable: { false }) {
        key, _, destination in
        for kind in ExpertProjection.allCases {
            let words = destination.baseAddress!.advanced(by: kind.weightOffset).assumingMemoryBound(to: UInt32.self)
            let scales = destination.baseAddress!.advanced(by: kind.scaleOffset).assumingMemoryBound(to: UInt16.self)
            let biases = destination.baseAddress!.advanced(by: kind.biasOffset).assumingMemoryBound(to: UInt16.self)
            for index in 0..<(819_200 / 4) { words[index] = 0x7654_3210 ^ UInt32((index + key.expert) & 7) }
            for index in 0..<(51_200 / 2) {
                scales[index] = BF16.encode(0.03125)
                biases[index] = BF16.encode(-0.0625)
            }
        }
    }
    try engine.beginSequence()
    let window = try engine.makeWindow(layer: 3, experts: [0, 1])
    _ = try window.consume(expert: 0) { $0.bytes[0] }
    try window.consume(expert: 1) { buffer in
        try executionRequire(buffer.offset != 0, "Fixture did not exercise a nonzero slab offset")
        func oldProjection(_ input: [Float], _ kind: ExpertProjection) throws -> [Float] {
            let words = Array(
                UnsafeBufferPointer(
                    start: buffer.bytes.baseAddress!.advanced(by: kind.weightOffset).assumingMemoryBound(to: UInt32.self),
                    count: 819_200 / 4))
            let scales = Array(
                UnsafeBufferPointer(
                    start: buffer.bytes.baseAddress!.advanced(by: kind.scaleOffset).assumingMemoryBound(to: UInt16.self), count: 51_200 / 2)
            )
            let biases = Array(
                UnsafeBufferPointer(
                    start: buffer.bytes.baseAddress!.advanced(by: kind.biasOffset).assumingMemoryBound(to: UInt16.self), count: 51_200 / 2))
            var result: [Float] = []
            for lo in stride(from: 0, to: input.count, by: kind.width) {
                result += try kernels.qlinearPinnedBF16(
                    x: Array(input[lo..<(lo + kind.width)]), width: kind.width,
                    words: words, scales: scales, biases: biases
                ).output
            }
            return result
        }
        for rows in [1, 11] {
            let input = (0..<(rows * 2560)).map { BF16.decode(BF16.encode(Float($0 % 13 - 6) * 0.0625)) }
            let gate = try oldProjection(input, .gate)
            let up = try oldProjection(input, .up)
            let product = zip(gate, up).map { LayerArithmetic.round(LayerArithmetic.silu($0) * $1) }
            let expected = try oldProjection(product, .down)
            let actual = try ExpertExecution.evaluate(input, buffer: buffer, kernels: kernels)
            try executionRequire(
                actual.map(\.bitPattern) == expected.map(\.bitPattern),
                "Buffer-native grouped QMV differs from per-token pinned QMV for \(rows) rows")
        }
    }
    try engine.finishSequence()
}
