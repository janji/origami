import Foundation
import Metal
import OrigamiEngine

/// Model-free checks of `QSAFileStaging` at the 262,143-token geometry: verified bytes at the maximum
/// addresses, a full LRU cache with eviction, busy/foreign-lease refusals and frame lifetime.
package enum QSAFileStagingSelfChecks {
    nonisolated(unsafe) package private(set) static var reports: [[String: Any]] = []
    private static func require(_ value: Bool, _ message: String) throws { if !value { throw EngineError(message) } }
    private static func rejected(_ body: () throws -> Void) throws {
        do { try body() } catch is EngineError { return }
        throw EngineError("File-KV staging check expected a refusal")
    }
    package static func maximumCacheAndReaders() throws {
        let ledger = try MemoryLedger(ceiling: 1_300_000_000, transientMargin: 10_000_000)
        var escaped: MTLBuffer?
        try autoreleasepool {
            guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
                throw EngineError("Metal unavailable")
            }
            let arrays = try CheckpointMemoryLease(ledger: ledger, bytes: 32_000_000, category: "file_kv_stage_fixture_arrays")
            let parent = CheckFixtures.scratch
            let store = try QSAFilePageStore(parent: parent, diskBytes: 768 * 1024 * 1024, ledger: ledger)
            let denied = try MemoryLedger(ceiling: QSAFileStaging.admittedBytes - 1, transientMargin: 0)
            try rejected { _ = try QSAFileStaging(device: device, ledger: denied, store: store) }
            try require(denied.snapshot().reserved == 0, "Denied file-KV staging allocation leaked")
            let staging = try QSAFileStaging(device: device, ledger: ledger, store: store)
            var k = QSAFileVector(store: store)
            var v = QSAFileVector(store: store)
            let old = 262143
            func bits(_ stream: Int, _ scalar: Int) -> UInt16 { UInt16(truncatingIfNeeded: stream * 101 + scalar * 17) }
            for start in stride(from: 0, to: old, by: 256) {
                let count = min(256, old - start) * 512
                k = try k.appending((0..<count).map { bits(0, start * 512 + $0) })
                v = try v.appending((0..<count).map { bits(1, start * 512 + $0) })
            }
            guard let output = device.makeBuffer(length: 16384, options: .storageModeShared) else {
                throw EngineError("Staging check output unavailable")
            }
            func drive(swapped: Bool, selected: [[UInt32]]?) throws {
                try autoreleasepool {
                    guard let command = queue.makeCommandBuffer(), let busy = queue.makeCommandBuffer() else {
                        throw EngineError("Staging check command unavailable")
                    }
                    let lease = try staging.stage(
                        command: command, keys: swapped ? v : k, values: swapped ? k : v, rows: 1, selected: selected)
                    if let selected { try lease.validateSelection(selected, totalTokens: old + 1) }
                    try rejected { _ = try staging.stage(command: busy, keys: k, values: v, rows: 1, selected: [[0]]) }
                    try rejected { try lease.finish(command: command, success: true) }
                    let positions = selected?.first ?? [0, UInt32(old - 1)]
                    for token in positions where Int(token) < old {
                        for column in [0, 511] {
                            let scalar = Int(token) * 512 + column
                            try require(
                                lease.buffer.contents().load(fromByteOffset: scalar * 2, as: UInt16.self) == bits(swapped ? 1 : 0, scalar)
                                    && lease.buffer.contents().load(
                                        fromByteOffset: QSAFileStaging.frameBytes / 2 + scalar * 2, as: UInt16.self)
                                        == bits(swapped ? 0 : 1, scalar),
                                "Verified staged bytes differ at the maximum address geometry")
                        }
                    }
                    guard let encoder = command.makeBlitCommandEncoder() else { throw EngineError("Staging reader encoder unavailable") }
                    let token = Int(positions.first ?? 0)
                    encoder.copy(from: lease.buffer, sourceOffset: token * 1024, to: output, destinationOffset: 0, size: 16)
                    encoder.endEncoding()
                    command.commit()
                    command.waitUntilCompleted()
                    try require(
                        command.status == .completed && output.contents().load(as: UInt16.self) == bits(swapped ? 1 : 0, token * 512),
                        "Maximum file-KV GPU reader failed")
                    try lease.finish(command: command, success: true)
                    try rejected { try lease.validateSelection([[0]], totalTokens: old + 1) }
                    escaped = lease.buffer
                }
            }
            // Full prefill domain first, then fill the entire 480 MiB cache,
            // exercising real LRU eviction rather than only its empty shape.
            try drive(swapped: false, selected: nil)
            for (step, begin) in stride(from: 0, to: 32768, by: 2051).enumerated() {
                try drive(swapped: step % 2 == 0, selected: [(begin..<min(32768, begin + 2051)).map { UInt32($0 * 8) }])
            }
            let full = staging.snapshot()
            try require(full.cacheEntries == 61440 && !full.active, "Maximum chunk cache did not reach its bounded geometry")
            try drive(swapped: true, selected: [[262136]])
            try require(staging.snapshot().cacheHitBytes > full.cacheHitBytes, "Cross-layer verified cache did not reuse immutable bytes")
            let beforeEvicted = staging.snapshot().fileReadBytes
            try drive(swapped: false, selected: [[0]])
            try require(
                staging.snapshot().fileReadBytes > beforeEvicted && staging.snapshot().cacheEntries == 61440,
                "LRU eviction did not replace an entry within its fixed capacity")
            // An initialized chunk cannot authorize a different missing span.
            try autoreleasepool {
                let command = queue.makeCommandBuffer()!
                let lease = try staging.stage(command: command, keys: v, values: k, rows: 1, selected: [[0]])
                try rejected { try lease.validateSelection([[16]], totalTokens: old + 1) }
                try rejected { try lease.validateSelection([[UInt32(old + 1)]], totalTokens: old + 1) }
                try lease.finish(command: command, success: false)
            }
            let last = staging.snapshot()
            reports.append([
                "capacity": old + 1, "stagingBytes": last.admittedBytes, "cacheEntries": last.cacheEntries,
                "fileReadBytes": last.fileReadBytes, "cacheHitBytes": last.cacheHitBytes, "transactions": last.transactions,
                "ledgerHighWater": ledger.snapshot().reservationHighWater,
            ])
            withExtendedLifetime(arrays) {}
        }
        try require(
            escaped != nil && ledger.snapshot().reserved == QSAFileStaging.admittedBytes,
            "Escaped file-KV Metal frame lost its complete allocation/cache charge")
        autoreleasepool { escaped = nil }
        try require(ledger.snapshot().reserved == 0, "File-KV frame allocation outlived its final GPU reader")
    }
}
