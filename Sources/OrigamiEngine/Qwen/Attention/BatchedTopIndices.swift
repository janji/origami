import Darwin
import Foundation
import Metal
import ObjectiveC

private final class BatchedSelectionAllocation {
    let ledger: MemoryLedger
    let reservation: MemoryLedger.Reservation
    init(ledger: MemoryLedger, bytes: UInt64) throws {
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: bytes, category: "batched-selection-buffers")
    }
    deinit { try? ledger.release(reservation) }
}
private enum BatchedSelectionAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// Exact batched counterpart of ReferenceKernels.topIndices. Comparison uses
/// the original Float32 BIT pattern: NaNs sort after -infinity; signed zeros
/// and subnormals tie with +zero; equal keys retain ascending column index.
/// No floating-point arithmetic, score conversion, or approximate top-k occurs.
///
/// Up to 1024 entries are sorted in each 8 KiB threadgroup workspace. Sorted
/// top-k chunks are reduced through stable parallel-rank merges. At the maximum
/// 65536-column domain this requires six merge encoders and one final CPU wait.
/// Temporary Metal buffers own page-rounded ledger charges until the underlying
/// resources retire. Input scores and returned index arrays remain caller-owned.
/// The same schedule selects the QSA indexer's 512 blocks and the MoE router's 10 of 512 experts
/// (one 512-column chunk, no merge).
public final class BatchedTopIndices {
    public static let maximumRows = 256
    public static let maximumColumns = 65_536
    public static let maximumCount = 512
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let chunkPipeline, mergePipeline, maskPipeline: MTLComputePipelineState
    private let ledger: MemoryLedger

    public init(ledger: MemoryLedger) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("Batched selection Metal runtime unavailable")
        }
        try ModelContract.validate(device: device)
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        let library = try device.makeLibrary(source: Self.source, options: options)
        guard let chunk = library.makeFunction(name: "selection_sorted_chunks"),
            let merge = library.makeFunction(name: "selection_merge_chunks"),
            let mask = library.makeFunction(name: "selection_mask_columns")
        else {
            throw EngineError("Missing batched selection kernel")
        }
        chunkPipeline = try device.makeComputePipelineState(function: chunk)
        mergePipeline = try device.makeComputePipelineState(function: merge)
        maskPipeline = try device.makeComputePipelineState(function: mask)
        guard chunkPipeline.maxTotalThreadsPerThreadgroup >= 256 else {
            throw EngineError("Batched selection requires 256-thread groups")
        }
        self.device = device
        self.queue = queue
        self.ledger = ledger
    }

    /// Causal masking precedes full-domain selection. Visibility filtering after
    /// sorting preserves future -infinity versus visible NaN behavior exactly.
    public func selectedPositions(scores: [Float], positions: [Int], kvLength: Int) throws -> [[UInt32]] {
        let columns = kvLength / 4
        guard kvLength > 0, kvLength <= ModelContract.maximumContext, !positions.isEmpty,
            positions.count <= Self.maximumRows, positions.allSatisfy({ $0 >= 0 && $0 < kvLength }),
            scores.count == positions.count * columns
        else { throw EngineError("Invalid batched causal indexer domain") }
        if kvLength <= 2048 { return positions.map { (0...$0).map(UInt32.init) } }
        var masked = scores
        for row in positions.indices {
            for column in 0..<columns where column * 4 + 3 > positions[row] { masked[row * columns + column] = -.infinity }
        }
        let selected = try topIndices(scores: masked, columns: columns, count: min(512, columns))
        return positions.indices.map { row in
            var result: [UInt32] = []
            for block in selected[row] where Int(block) * 4 + 3 <= positions[row] {
                result.append(contentsOf: (0..<4).map { block * 4 + UInt32($0) })
            }
            let tail = (positions[row] + 1) / 4 * 4
            if tail <= positions[row] { result.append(contentsOf: (tail...positions[row]).map(UInt32.init)) }
            return result.sorted()
        }
    }

    /// Scores are row-major. Count is shared by all rows and must not exceed
    /// the column count. Returned rows each contain exactly `count` indices.
    public func topIndices(scores: [Float], columns: Int, count: Int) throws -> [[UInt32]] {
        guard (1...Self.maximumColumns).contains(columns), !scores.isEmpty,
            scores.count % columns == 0, scores.count / columns <= Self.maximumRows,
            (1...min(Self.maximumCount, columns)).contains(count)
        else {
            throw EngineError("Invalid batched selection row, column, or top-k geometry")
        }
        let rows = scores.count / columns
        var chunkWidth = 1
        while chunkWidth < min(columns, 1024) { chunkWidth *= 2 }
        let chunkCount = (columns + chunkWidth - 1) / chunkWidth
        let threads = min(256, chunkWidth)
        return try autoreleasepool {
            let input = try allocate(bytes: scores.count * MemoryLayout<Float>.stride)
            scores.withUnsafeBytes { input.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
            // Each sorted entry is a uint2(key, original-column-index).
            let first = try allocate(bytes: rows * chunkCount * count * 8)
            // Later levels only shrink, so the first merge's output is the
            // largest second backing needed by the ping-pong schedule.
            let second: MTLBuffer?
            if chunkCount > 1 { second = try allocate(bytes: rows * ((chunkCount + 1) / 2) * count * 8) } else { second = nil }
            guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
                throw EngineError("Batched selection command allocation failed", code: "insufficient_memory")
            }
            command.label = "selection-exact-chunk-sort-merge"
            encoder.setComputePipelineState(chunkPipeline)
            encoder.setBuffer(input, offset: 0, index: 0)
            encoder.setBuffer(first, offset: 0, index: 1)
            let shape = [rows, columns, count, chunkCount, chunkWidth, threads].map(UInt32.init)
            shape.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 2) }
            encoder.dispatchThreadgroups(
                MTLSize(width: chunkCount, height: rows, depth: 1),
                threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
            encoder.endEncoding()
            var current = first
            var currentChunks = chunkCount
            while currentChunks > 1 {
                let outputChunks = (currentChunks + 1) / 2
                let destination = current === first ? second! : first
                guard let mergeEncoder = command.makeComputeCommandEncoder() else {
                    throw EngineError("Batched selection merge encoder allocation failed", code: "insufficient_memory")
                }
                mergeEncoder.setComputePipelineState(mergePipeline)
                mergeEncoder.setBuffer(current, offset: 0, index: 0)
                mergeEncoder.setBuffer(destination, offset: 0, index: 1)
                let mergeShape = [count, currentChunks, outputChunks].map(UInt32.init)
                mergeShape.withUnsafeBytes { mergeEncoder.setBytes($0.baseAddress!, length: $0.count, index: 2) }
                mergeEncoder.dispatchThreads(
                    MTLSize(width: rows * outputChunks * count * 2, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(256, mergePipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                mergeEncoder.endEncoding()
                current = destination
                currentChunks = outputChunks
            }
            command.commit()
            try command.waitBounded()
            guard command.status == .completed else {
                throw EngineError.gpu("Batched selection dispatch failed", command.error)
            }
            let pairs = current.contents().assumingMemoryBound(to: UInt32.self)
            var output = [[UInt32]]()
            output.reserveCapacity(rows)
            for row in 0..<rows {
                var indices = [UInt32]()
                indices.reserveCapacity(count)
                for rank in 0..<count {
                    let index = pairs[(row * count + rank) * 2 + 1]
                    guard index < UInt32(columns) else { throw EngineError("Batched selection published a padding index") }
                    indices.append(index)
                }
                output.append(indices)
            }
            return output
        }
    }

    /// Chunk geometry and the two ping-pong pair buffers `encodeTopIndices` needs.
    package struct BufferPlan {
        let rows, columns, count, chunkWidth, chunkCount, threads: Int
        package var firstBytes: Int { rows * chunkCount * count * 8 }
        package var secondBytes: Int { chunkCount > 1 ? rows * ((chunkCount + 1) / 2) * count * 8 : 0 }
    }

    package static func bufferPlan(rows: Int, columns: Int, count: Int) throws -> BufferPlan {
        guard (1...maximumRows).contains(rows), (1...maximumColumns).contains(columns),
            (1...min(maximumCount, columns)).contains(count)
        else { throw EngineError("Invalid encoded selection geometry") }
        var width = 1
        while width < min(columns, 1024) { width *= 2 }
        return BufferPlan(
            rows: rows, columns: columns, count: count, chunkWidth: width,
            chunkCount: (columns + width - 1) / width, threads: min(256, width))
    }

    /// Encode precisely the existing chunk-sort/merge schedule. Returned pairs
    /// are still in score order. No causal expansion, validation, or command
    /// completion is implied by returning this provisional device buffer.
    package func encodeTopIndices(
        _ command: MTLCommandBuffer, scores: MTLBuffer, first: MTLBuffer,
        second: MTLBuffer?, rows: Int, columns: Int, count: Int
    ) throws -> MTLBuffer {
        let p = try Self.bufferPlan(rows: rows, columns: columns, count: count)
        var spans = [(scores, rows * columns * 4), (first, p.firstBytes)]
        if p.secondBytes > 0 {
            guard let second else { throw EngineError("Encoded selection missing merge scratch") }
            spans.append((second, p.secondBytes))
        }
        guard command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued,
            spans.allSatisfy({ $0.0.device.registryID == device.registryID && $0.0.length >= $0.1 }),
            Set(spans.map { ObjectIdentifier($0.0) }).count == spans.count,
            let e = command.makeComputeCommandEncoder()
        else { throw EngineError("Invalid encoded selection spans/command") }
        e.setComputePipelineState(chunkPipeline)
        e.setBuffer(scores, offset: 0, index: 0)
        e.setBuffer(first, offset: 0, index: 1)
        let shape = [rows, columns, count, p.chunkCount, p.chunkWidth, p.threads].map(UInt32.init)
        shape.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: 2) }
        e.dispatchThreadgroups(
            MTLSize(width: p.chunkCount, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: p.threads, height: 1, depth: 1))
        e.endEncoding()
        var current = first
        var chunks = p.chunkCount
        while chunks > 1 {
            let outputChunks = (chunks + 1) / 2
            let destination = current === first ? second! : first
            guard let merge = command.makeComputeCommandEncoder() else { throw EngineError("Encoded selection merge unavailable") }
            merge.setComputePipelineState(mergePipeline)
            merge.setBuffer(current, offset: 0, index: 0)
            merge.setBuffer(destination, offset: 0, index: 1)
            let shape = [count, chunks, outputChunks].map(UInt32.init)
            shape.withUnsafeBytes { merge.setBytes($0.baseAddress!, length: $0.count, index: 2) }
            merge.dispatchThreads(
                MTLSize(width: rows * outputChunks * count * 2, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(256, mergePipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            merge.endEncoding()
            current = destination
            chunks = outputChunks
        }
        return current
    }

    /// Columns `eligible..<columns` of each row become NaN, which ranks below every score and ties go to the lower column,
    /// so a selection of at most `eligible` never takes them (a pruned GGUF's router columns past its experts). No change
    /// when every column is eligible.
    package static func maskColumns(_ scores: inout [Float], columns: Int, eligible: Int) {
        guard eligible < columns else { return }
        for row in 0..<(scores.count / columns) { for column in eligible..<columns { scores[row * columns + column] = .nan } }
    }
    /// `maskColumns` on the GPU, behind whatever wrote `scores` in `command`.
    func encodeMaskColumns(_ command: MTLCommandBuffer, scores: MTLBuffer, rows: Int, columns: Int, eligible: Int) throws {
        guard (1...Self.maximumRows).contains(rows), (1..<columns).contains(eligible), scores.length >= rows * columns * 4,
            let e = command.makeComputeCommandEncoder()
        else { throw EngineError("Invalid selection mask") }
        e.setComputePipelineState(maskPipeline)
        e.setBuffer(scores, offset: 0, index: 0)
        let shape = [columns, eligible].map(UInt32.init)
        shape.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: 1) }
        e.dispatchThreads(
            MTLSize(width: columns - eligible, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, columns - eligible), height: 1, depth: 1))
        e.endEncoding()
    }

    private func allocate(bytes: Int) throws -> MTLBuffer {
        guard bytes > 0, bytes <= device.maxBufferLength else {
            throw EngineError("Batched selection buffer exceeds the Metal allocation bound")
        }
        let page = UInt64(getpagesize())
        let charged = (UInt64(bytes) + page - 1) / page * page
        let owner = try BatchedSelectionAllocation(ledger: ledger, bytes: charged)
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw EngineError("Batched selection buffer allocation failed", code: "insufficient_memory")
        }
        objc_setAssociatedObject(buffer, &BatchedSelectionAssociation.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        guard UInt64(buffer.allocatedSize) <= charged else {
            throw EngineError("Batched selection allocation exceeded its reserved page bound")
        }
        return buffer
    }

    private static let source = #"""
        #include <metal_stdlib>
        using namespace metal;
        // Order-preserving uint key of a Float32 bit pattern: NaN -> 0 (below -infinity), subnormals and
        // -0 -> the +0 key, otherwise the usual sign-flip transform. Larger key = better score.
        inline uint selection_key_bits(uint bits) {
            uint mag = bits & 0x7fffffffu;
            if (mag > 0x7f800000u) return 0;
            if (mag < 0x00800000u) bits = 0;
            return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
        }
        // Total order on (key, column): higher key first, then lower column. Columns are unique, so no ties.
        inline bool selection_better(uint2 a, uint2 b) {
            return a.x > b.x || (a.x == b.x && a.y < b.y);
        }
        // One threadgroup (<= 256 threads) per (chunk of <= 1024 columns, row): bitonic sort of
        // uint2(key, column) in 8 KiB of threadgroup memory; writes the chunk's best `count` in order.
        kernel void selection_sorted_chunks(device const uint *scoreBits [[buffer(0)]],
            device uint2 *output [[buffer(1)]], constant uint *shape [[buffer(2)]],
            uint tid [[thread_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) {
            threadgroup uint2 entries[1024];
            uint columns = shape[1], count = shape[2], chunks = shape[3], width = shape[4], threads = shape[5];
            uint chunk = group.x, row = group.y;
            for (uint i = tid; i < width; i += threads) {
                uint column = chunk * width + i;
                // Padding has the lowest key but a unique index after every valid
                // column. Valid NaNs therefore win every tie against padding, and
                // even padding remains totally ordered in intermediate merges.
                uint key = column < columns ? selection_key_bits(scoreBits[row * columns + column]) : 0;
                entries[i] = uint2(key, column);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint span = 2; span <= width; span <<= 1) {
                for (uint distance = span >> 1; distance > 0; distance >>= 1) {
                    for (uint i = tid; i < width; i += threads) {
                        uint other = i ^ distance;
                        // Each disjoint pair has exactly one writer. No other
                        // pair reads either location before the next barrier.
                        if (other > i) {
                            uint2 a = entries[i], b = entries[other];
                            bool descending = (i & span) == 0;
                            bool swap = descending ? selection_better(b, a) : selection_better(a, b);
                            if (swap) { entries[i] = b; entries[other] = a; }
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            for (uint i = tid; i < count; i += threads) output[(row * chunks + chunk) * count + i] = entries[i];
        }
        // Merges sorted chunk pairs: each entry's output rank is its own rank plus the number of better
        // entries in the partner chunk (binary search); ranks >= count are dropped.
        kernel void selection_merge_chunks(device const uint2 *input [[buffer(0)]],
            device uint2 *output [[buffer(1)]], constant uint *shape [[buffer(2)]],
            uint index [[thread_position_in_grid]]) {
            uint count = shape[0], inputChunks = shape[1], outputChunks = shape[2];
            uint entry = index % (2 * count), group = index / (2 * count);
            uint row = group / outputChunks, pair = group % outputChunks;
            uint side = entry / count, ownRank = entry % count, chunk = 2 * pair + side;
            if (chunk >= inputChunks) return;
            uint2 value = input[(row * inputChunks + chunk) * count + ownRank];
            uint otherChunk = 2 * pair + (1 - side), rank = ownRank;
            if (otherChunk < inputChunks) {
                uint base = (row * inputChunks + otherChunk) * count;
                uint lo = 0, hi = count;
                while (lo < hi) {
                    uint mid = (lo + hi) / 2;
                    if (selection_better(input[base + mid], value)) lo = mid + 1;
                    else hi = mid;
                }
                rank += lo;
            }
            // Disjoint input chunks have unique original indices. Stable total
            // ordering therefore assigns exactly one writer to each retained rank.
            if (rank < count) output[(row * outputChunks + pair) * count + rank] = value;
        }
        // scores[row][eligible + x] = NaN (shape = [columns, eligible]).
        kernel void selection_mask_columns(device uint *scoreBits [[buffer(0)]], constant uint *shape [[buffer(1)]],
            uint2 p [[thread_position_in_grid]]) {
            if (p.x < shape[0] - shape[1]) scoreBits[p.y * shape[0] + shape[1] + p.x] = 0x7fc00000u;
        }
        """#
}
