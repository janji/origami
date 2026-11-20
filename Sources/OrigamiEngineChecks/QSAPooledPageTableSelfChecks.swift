import Foundation
import Metal
import OrigamiEngine

/// The indexer reading the pooled index in place through a page table scores exactly as it did reading a whole
/// FP32 copy: paged and host-held states, decode rows with and without a new block, a prefill stripe, and a table
/// rebound as the index grows (only the last page changes).
package enum QSAPooledPageTableSelfChecks {
    package static func pagedScoresMatchCopy() throws {
        let ledger = try MemoryLedger()
        try autoreleasepool {
            guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
                throw EngineError("Metal unavailable")
            }
            let indexer = try PinnedIndexer(ledger: ledger)
            let table = try QSAPooledPageTable(device: device, ledger: ledger, arguments: indexer.pooledArguments)
            var random = SystemRandomNumberGenerator()
            func bits(_ count: Int) -> [UInt16] { (0..<count).map { _ in BF16.encode(Float.random(in: -2...2, using: &random)) } }
            func floats(_ count: Int) -> [Float] { (0..<count).map { _ in Float.random(in: -2...2, using: &random) } }
            func buffer(_ values: [Float]) throws -> MTLBuffer {
                guard let result = device.makeBuffer(length: max(4, values.count * 4), options: .storageModeShared) else {
                    throw EngineError("Metal buffer")
                }
                values.withUnsafeBytes { if !$0.isEmpty { result.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) } }
                return result
            }
            func compare(_ state: QSAState, rows: Int, label: String) throws {
                let blocks = (state.offset + rows) / 4
                let old = state.offset / 4
                let query = try buffer(floats(rows * 512))
                let fresh = try buffer(floats((blocks - old) * 128))
                let whole = try buffer(
                    state.pooledKeys.map { Float(bitPattern: UInt32($0) << 16) }
                        + Array(
                            UnsafeBufferPointer(start: fresh.contents().assumingMemoryBound(to: Float.self), count: (blocks - old) * 128)))
                let dots = try buffer([Float](repeating: 0, count: rows * 4 * blocks))
                var results: [[UInt32]] = []
                for paged in [false, true] {
                    let output = try buffer([Float](repeating: .nan, count: rows * blocks))
                    guard let command = queue.makeCommandBuffer() else { throw EngineError("Metal command") }
                    if paged {
                        let pages = try state.pooledPages(device: device, host: indexer.copyPooled)
                        let keys = PinnedIndexer.PagedKeys(
                            table: table.table, resources: try table.bind(pages, command: command), fresh: fresh, oldBlocks: old)
                        try indexer.encodeScores(command, query: query, paged: keys, dots: dots, output: output, rows: rows, blocks: blocks)
                    } else {
                        try indexer.encodeScores(
                            command, query: query, pooledKeys: whole, dots: dots, output: output, rows: rows, blocks: blocks)
                    }
                    command.commit()
                    command.waitUntilCompleted()
                    guard command.status == .completed else { throw EngineError.gpu("Pooled page-table check GPU failed", command.error) }
                    results.append(
                        Array(UnsafeBufferPointer(start: output.contents().assumingMemoryBound(to: UInt32.self), count: rows * blocks)))
                }
                guard results[0] == results[1] else { throw EngineError("Paged pooled scores differ from the fp32 copy (\(label))") }
            }
            // One growing paged conversation: 600 blocks over 10 pages, then a group closes and a block is published.
            let arena = try QSAPageArena(ledger: ledger, maximumBytes: 64 << 20)
            var storage = QSAPagedStorage(arena: arena)
            func grow(to tokens: Int, from current: Int) throws -> QSAState {
                storage.keys = try storage.keys.appending(bits((tokens - current) * 512))
                storage.values = try storage.values.appending(bits((tokens - current) * 512))
                storage.pooled = try storage.pooled.appending(bits((tokens / 4 - current / 4) * 128))
                return try QSAState(offset: tokens, pages: storage, rawTail: bits(tokens % 4 * 128))
            }
            try compare(try grow(to: 2400, from: 0), rows: 1, label: "decode, no new block")
            try compare(try grow(to: 2403, from: 2400), rows: 1, label: "decode, a new block")
            try compare(try grow(to: 2404, from: 2403), rows: 1, label: "decode after a block was published")
            try compare(try grow(to: 4611, from: 2404), rows: 1, label: "decode, 18 full pages")
            var prefill = QSAPagedStorage(arena: arena)
            prefill.keys = try prefill.keys.appending(bits(2304 * 512))
            prefill.values = try prefill.values.appending(bits(2304 * 512))
            prefill.pooled = try prefill.pooled.appending(bits(576 * 128))
            try compare(try QSAState(offset: 2304, pages: prefill, rawTail: []), rows: 256, label: "prefill stripe")
            try compare(
                try QSAState(
                    offset: 2403, keys: bits(2403 * 512), values: bits(2403 * 512), pooledKeys: bits(600 * 128),
                    rawTail: bits(3 * 128)), rows: 1, label: "host-held index")
        }
        guard ledger.snapshot().reserved == 0 else { throw EngineError("Pooled page-table check leaked a reservation") }
    }
}
