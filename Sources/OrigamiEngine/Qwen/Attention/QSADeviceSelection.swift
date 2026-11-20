import Foundation
import Metal

/// The QSA key selection on the GPU, equal to `BatchedTopIndices.selectedPositions` (BITWISE): up to
/// 2048 keys every causal key; above, future blocks are masked to -infinity, the top 512 of keys/4
/// four-key blocks are chosen by score (ties to the lower block), the visible ones sorted ascending
/// and expanded to keys, plus the 0-3 keys of the query's partial last block (<= 2051 keys).
/// Output: ids [rows][2051] uint32 (unused tail 0xffffffff), counts [rows], one exception flag.
/// Counts are bounded by construction even on a malformed sort result; the
/// transaction must inspect exceptions before publishing dependent state.
final class QSADeviceSelection {
    /// A selection encoded into `command`; its buffers stay caller-owned until the command joins.
    struct Encoded {
        let command: MTLCommandBuffer
        let ids, counts, exceptions: MTLBuffer
        let rows, keys: Int
        fileprivate init(
            command: MTLCommandBuffer, ids: MTLBuffer, counts: MTLBuffer,
            exceptions: MTLBuffer, rows: Int, keys: Int
        ) {
            self.command = command
            self.ids = ids
            self.counts = counts
            self.exceptions = exceptions
            self.rows = rows
            self.keys = keys
        }
    }
    private let device: MTLDevice, selector: BatchedTopIndices
    private let mask, expand, dense: MTLComputePipelineState

    init(device: MTLDevice, selector: BatchedTopIndices) throws {
        self.device = device
        self.selector = selector
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .precise
        let library = try device.makeLibrary(source: Self.source, options: options)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else { throw EngineError("Missing QSA position kernel") }
            return try device.makeComputePipelineState(function: function)
        }
        mask = try pipeline("qsa_mask_positions")
        expand = try pipeline("qsa_expand_positions")
        dense = try pipeline("qsa_dense_positions")
        guard expand.maxTotalThreadsPerThreadgroup >= 256 else { throw EngineError("QSA position sort requires256threads") }
    }

    func encode(
        _ command: MTLCommandBuffer, positions: [Int], keys: Int, positionsBuffer: MTLBuffer,
        scores: MTLBuffer?, masked: MTLBuffer?, first: MTLBuffer?, second: MTLBuffer?,
        ids: MTLBuffer, counts: MTLBuffer, exceptions: MTLBuffer
    ) throws -> Encoded {
        let rows = positions.count
        guard (1...256).contains(rows), (rows...ModelContract.maximumContext).contains(keys),
            positions.allSatisfy({ (0..<keys).contains($0) }), positionsBuffer.storageMode == .shared,
            command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued
        else {
            throw EngineError("Invalid QSA position domain/command")
        }
        var spans = [
            (positionsBuffer, rows * 4), (ids, rows * PinnedAttention.maximumSelectedKeys * 4), (counts, rows * 4), (exceptions, 4),
        ]
        if keys > 2048 {
            let plan = try BatchedTopIndices.bufferPlan(rows: rows, columns: keys / 4, count: 512)
            guard let scores, let masked, let first else { throw EngineError("Missing QSA position score/sort backing") }
            spans += [(scores, rows * (keys / 4) * 4), (masked, rows * (keys / 4) * 4), (first, plan.firstBytes)]
            if plan.secondBytes > 0 {
                guard let second else { throw EngineError("Missing QSA merge backing") }
                spans.append((second, plan.secondBytes))
            }
        }
        guard spans.allSatisfy({ $0.0.device.registryID == device.registryID && $0.0.length >= $0.1 }),
            Set(spans.map { ObjectIdentifier($0.0) }).count == spans.count
        else { throw EngineError("QSA position buffers alias or exceed extent") }
        for (row, position) in positions.enumerated() {
            positionsBuffer.contents().assumingMemoryBound(to: UInt32.self)[row] = UInt32(position)
        }
        let shape = [UInt32(rows), UInt32(keys), UInt32(keys / 4)]
        func encoder(_ pipeline: MTLComputePipelineState, _ buffers: [MTLBuffer]) throws -> MTLComputeCommandEncoder {
            guard let e = command.makeComputeCommandEncoder() else { throw EngineError("QSA position encoder unavailable") }
            e.setComputePipelineState(pipeline)
            for (index, buffer) in buffers.enumerated() { e.setBuffer(buffer, offset: 0, index: index) }
            shape.withUnsafeBytes { e.setBytes($0.baseAddress!, length: $0.count, index: buffers.count) }
            return e
        }
        if keys <= 2048 {
            let e = try encoder(dense, [positionsBuffer, ids, counts, exceptions])
            e.dispatchThreadgroups(
                MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            e.endEncoding()
        } else {
            let e = try encoder(mask, [scores!, positionsBuffer, masked!])
            e.dispatchThreads(
                MTLSize(width: rows * (keys / 4), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(256, mask.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            e.endEncoding()
            let pairs = try selector.encodeTopIndices(
                command, scores: masked!, first: first!, second: second,
                rows: rows, columns: keys / 4, count: 512)
            let finish = try encoder(expand, [pairs, positionsBuffer, ids, counts, exceptions])
            finish.dispatchThreadgroups(
                MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            finish.endEncoding()
        }
        return Encoded(command: command, ids: ids, counts: counts, exceptions: exceptions, rows: rows, keys: keys)
    }

    private static let source = #"""
        #include <metal_stdlib>
        using namespace metal;
        // Copies the indexer scores, replacing blocks not wholly visible to the row's position with -infinity.
        kernel void qsa_mask_positions(device const uint *scores [[buffer(0)]], device const uint *positions [[buffer(1)]],
            device uint *masked [[buffer(2)]], constant uint *shape [[buffer(3)]], uint i [[thread_position_in_grid]]) {
            uint columns = shape[2];
            if (i < shape[0] * columns) masked[i] = (i % columns) * 4u + 3u <= positions[i / columns] ? scores[i] : 0xff800000u;
        }
        // Keys <= 2048: every key 0...position; flags a position outside the domain.
        kernel void qsa_dense_positions(device const uint *positions [[buffer(0)]], device uint *ids [[buffer(1)]],
            device uint *counts [[buffer(2)]], device atomic_uint &flag [[buffer(3)]], constant uint *shape [[buffer(4)]],
            uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
            uint position = positions[row];
            if (position >= shape[1] || shape[1] > 2048u) {
                if (lane == 0) { counts[row] = 0; atomic_store_explicit(&flag, 1u, memory_order_relaxed); } return;
            }
            if (lane == 0) counts[row] = position + 1u;
            for (uint i = lane; i < 2051u; i += 256u) ids[row * 2051u + i] = i <= position ? i : 0xffffffffu;
        }
        // One 256-thread group per row: drops invisible or invalid blocks of the top 512, bitonic-sorts the
        // block ids ascending, flags duplicates or out-of-range ids, expands blocks to keys and appends the tail.
        kernel void qsa_expand_positions(device const uint2 *pairs [[buffer(0)]], device const uint *positions [[buffer(1)]],
            device uint *ids [[buffer(2)]], device uint *counts [[buffer(3)]], device atomic_uint &flag [[buffer(4)]],
            constant uint *shape [[buffer(5)]], uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
            threadgroup uint blocks[512]; threadgroup uint complete;
            uint position = positions[row];
            if (position >= shape[1]) {
                if (lane == 0) { counts[row] = 0; atomic_store_explicit(&flag, 1u, memory_order_relaxed); } return;
            }
            for (uint i = lane; i < 512u; i += 256u) {
                uint b = pairs[row * 512u + i].y;
                if (b >= shape[2]) atomic_store_explicit(&flag, 1u, memory_order_relaxed);
                blocks[i] = b < shape[2] && b * 4u + 3u <= position ? b : 0xffffffffu;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Sort only integer addresses after the unchanged score-order top512.
            // This is the original host's ascending-position order, not a new
            // attention reduction lane or selected-key visitation schedule.
            for (uint k = 2; k <= 512u; k <<= 1) {
                for (uint j = k >> 1; j > 0; j >>= 1) {
                    for (uint i = lane; i < 512u; i += 256u) {
                        uint peer = i ^ j;
                        if (peer > i) {
                            uint a = blocks[i], b = blocks[peer]; bool ascending = (i & k) == 0;
                            blocks[i] = ascending ? min(a,b) : max(a,b);
                            blocks[peer] = ascending ? max(a,b) : min(a,b);
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            if (lane == 0) {
                uint n = 0; while (n < 512u && blocks[n] != 0xffffffffu) ++n;
                complete = n; counts[row] = n * 4u + (position + 1u) % 4u;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lane; i < complete; i += 256u) {
                if (i > 0 && blocks[i] == blocks[i-1]) atomic_store_explicit(&flag, 1u, memory_order_relaxed);
            }
            uint length = complete * 4u + (position + 1u) % 4u;
            for (uint i = lane; i < 2051u; i += 256u) {
                uint value = 0xffffffffu;
                if (i < complete * 4u) value = blocks[i / 4u] * 4u + i % 4u;
                else if (i < length) value = ((position + 1u) / 4u) * 4u + i - complete * 4u;
                ids[row * 2051u + i] = value;
            }
        }
        """#
}
