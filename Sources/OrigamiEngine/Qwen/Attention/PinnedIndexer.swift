import Foundation
import Metal
import ObjectiveC

private final class PinnedIndexerAllocation {
    let ledger: MemoryLedger
    let reservation: MemoryLedger.Reservation
    init(_ ledger: MemoryLedger, bytes: UInt64) throws {
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: bytes, category: "pinned_indexer_buffers")
    }
    deinit { try? ledger.release(reservation) }
}
private enum PinnedIndexerAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// The QSA indexer's block scores in MLX Steel GEMM's FP32 fragment order (BITWISE in exact mode): for each
/// query row, 4 index heads x 128 dims (FP32, already normalized and rotated) against every pooled
/// 4-token block key (128 dims), score = sum over heads of relu(q.k), divided by sqrt(128).
/// Output [rows][blocks] FP32 feeds the top-512 block selection.
public final class PinnedIndexer {
    private let device: MTLDevice, queue: MTLCommandQueue, ledger: MemoryLedger
    private let dotsKernel, finishKernel, pagedDotsKernel: MTLComputePipelineState
    /// Encodes the page table `origami_indexer_dots_paged` reads; `QSAPooledPageTable` owns one per attention layer.
    package let pooledArguments: MTLArgumentEncoder
    public init(ledger: MemoryLedger) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("Pinned indexer Metal unavailable")
        }
        try ModelContract.validate(device: device)
        self.device = device
        self.queue = queue
        self.ledger = ledger
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .fast
        let library = try device.makeLibrary(source: Self.source, options: options)
        guard let dots = library.makeFunction(name: "origami_indexer_dots"),
            let finish = library.makeFunction(name: "origami_indexer_finish"),
            let paged = library.makeFunction(name: "origami_indexer_dots_paged")
        else { throw EngineError("Pinned indexer kernels missing") }
        dotsKernel = try device.makeComputePipelineState(function: dots)
        finishKernel = try device.makeComputePipelineState(function: finish)
        pagedDotsKernel = try device.makeComputePipelineState(function: paged)
        pooledArguments = paged.makeArgumentEncoder(bufferIndex: 1)
    }
    public static func tileRows(rows: Int, blocks: Int) throws -> Int {
        guard (1...256).contains(rows), (512...65536).contains(blocks) else { throw EngineError("Invalid pinned indexer geometry") }
        return min(rows, 2_097_152 / blocks)
    }
    private func allocate(_ bytes: Int) throws -> MTLBuffer {
        guard bytes > 0, bytes <= 32 * 1024 * 1024 else { throw EngineError("Pinned indexer allocation exceeds32MiB") }
        let page = UInt64(getpagesize())
        let charged = (UInt64(bytes) + page - 1) / page * page
        let owner = try PinnedIndexerAllocation(ledger, bytes: charged)
        guard let result = device.makeBuffer(length: bytes, options: .storageModeShared), UInt64(result.allocatedSize) <= charged else {
            throw EngineError("Pinned indexer allocation failed", code: "insufficient_memory")
        }
        objc_setAssociatedObject(result, &PinnedIndexerAssociation.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        return result
    }
    private func copy(_ values: [Float]) throws -> MTLBuffer {
        let result = try allocate(values.count * 4)
        values.withUnsafeBytes { result.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return result
    }
    /// A host-held pooled index (no page arena) as one charged buffer the page table points into.
    package func copyPooled(_ values: [UInt16]) throws -> MTLBuffer {
        let result = try allocate(max(2, values.count * 2))
        values.withUnsafeBytes { if !$0.isEmpty { result.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) } }
        return result
    }

    /// The pooled index read in place, as a table of the conversation's BF16 pages (64 blocks a page) for the
    /// blocks before `oldBlocks`; this step's blocks come from `fresh` in FP32, as a copied index had them.
    package struct PagedKeys {
        let table: MTLBuffer, resources: [MTLResource], fresh: MTLBuffer, oldBlocks: Int
        package init(table: MTLBuffer, resources: [MTLResource], fresh: MTLBuffer, oldBlocks: Int) {
            self.table = table
            self.resources = resources
            self.fresh = fresh
            self.oldBlocks = oldBlocks
        }
    }

    /// Encodes one indexer stripe (dots, then finish) into `command` without allocation, submission
    /// or readback; the caller owns every span until the command joins. Same FP32 four-head
    /// dot/reduction schedule as `scores(query:pooledKeys:)`, including the host-computed Float
    /// denominator.
    package func encodeScores(
        _ command: MTLCommandBuffer, query: MTLBuffer, queryOffset: Int = 0,
        pooledKeys: MTLBuffer? = nil, paged: PagedKeys? = nil, dots: MTLBuffer, output: MTLBuffer,
        outputOffset: Int = 0, rows: Int, blocks: Int
    ) throws {
        let stripe = try Self.tileRows(rows: rows, blocks: blocks)
        guard rows <= stripe, (pooledKeys == nil) != (paged == nil), (0...blocks).contains(paged?.oldBlocks ?? 0),
            (paged?.oldBlocks ?? 0) <= QSAPooledPageTable.maximumPages * 64,
            command.commandQueue.device.registryID == device.registryID,
            command.status == .notEnqueued
        else {
            throw EngineError("Pinned indexer region exceeds its original stripe or device")
        }
        var spans: [(buffer: MTLBuffer, offset: Int, bytes: Int, writes: Bool)] = [
            (query, queryOffset, rows * 512 * 4, false),
            (dots, 0, rows * 4 * blocks * 4, true),
            (output, outputOffset, rows * blocks * 4, true),
        ]
        if let pooledKeys { spans.append((pooledKeys, 0, blocks * 128 * 4, false)) }
        if let paged {
            spans += [
                (paged.table, 0, pooledArguments.encodedLength, false), (paged.fresh, 0, (blocks - paged.oldBlocks) * 128 * 4, false),
            ]
        }
        for span in spans {
            guard span.buffer.device.registryID == device.registryID,
                span.offset >= 0, span.offset % 4 == 0,
                span.offset <= span.buffer.length,
                span.bytes <= span.buffer.length - span.offset
            else {
                throw EngineError("Pinned indexer region has an invalid buffer span")
            }
        }
        for i in spans.indices {
            for j in spans.indices where j > i {
                let a = spans[i]
                let b = spans[j]
                if a.writes || b.writes, a.buffer === b.buffer,
                    a.offset < b.offset + b.bytes, b.offset < a.offset + a.bytes
                {
                    throw EngineError("Pinned indexer region aliases a live output")
                }
            }
        }
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw EngineError("Pinned indexer region encoder unavailable")
        }
        encoder.setComputePipelineState(paged == nil ? dotsKernel : pagedDotsKernel)
        encoder.setBuffer(query, offset: queryOffset, index: 0)
        if let paged {
            encoder.setBuffer(paged.table, offset: 0, index: 1)
            encoder.setBuffer(paged.fresh, offset: 0, index: 4)
            encoder.useResources(paged.resources, usage: .read)
        } else {
            encoder.setBuffer(pooledKeys, offset: 0, index: 1)
        }
        encoder.setBuffer(dots, offset: 0, index: 2)
        let shape = [UInt32(rows * 4), UInt32(blocks), UInt32(paged?.oldBlocks ?? blocks)]
        shape.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 3) }
        encoder.dispatchThreadgroups(
            MTLSize(width: (blocks + 7) / 8, height: (rows * 4 + 7) / 8, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
        guard let finish = command.makeComputeCommandEncoder() else {
            throw EngineError("Pinned indexer region finish encoder unavailable")
        }
        finish.setComputePipelineState(finishKernel)
        finish.setBuffer(dots, offset: 0, index: 0)
        finish.setBuffer(output, offset: outputOffset, index: 1)
        var n = UInt32(blocks)
        var denominator = sqrt(Float(128))
        withUnsafeBytes(of: &n) { finish.setBytes($0.baseAddress!, length: $0.count, index: 2) }
        withUnsafeBytes(of: &denominator) { finish.setBytes($0.baseAddress!, length: $0.count, index: 3) }
        finish.dispatchThreads(
            MTLSize(width: rows * blocks, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        finish.endEncoding()
    }

    public func scores(query: [Float], pooledKeys: [Float]) throws -> [Float] {
        guard !query.isEmpty, query.count % 512 == 0, pooledKeys.count % 128 == 0 else {
            throw EngineError("Invalid pinned indexer tensor shape")
        }
        let rows = query.count / 512
        let blocks = pooledKeys.count / 128
        let tile = try Self.tileRows(rows: rows, blocks: blocks)
        return try autoreleasepool {
            let q = try copy(query)
            let k = try copy(pooledKeys)
            var result = [Float]()
            result.reserveCapacity(rows * blocks)
            for lo in stride(from: 0, to: rows, by: tile) {
                try autoreleasepool {
                    let count = min(tile, rows - lo)
                    let dots = try allocate(count * 4 * blocks * 4)
                    let out = try allocate(count * blocks * 4)
                    guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
                        throw EngineError("Pinned indexer command allocation failed")
                    }
                    command.label = "pinned-indexer"
                    encoder.setComputePipelineState(dotsKernel)
                    encoder.setBuffer(q, offset: lo * 512 * 4, index: 0)
                    encoder.setBuffer(k, offset: 0, index: 1)
                    encoder.setBuffer(dots, offset: 0, index: 2)
                    let shape = [UInt32(count * 4), UInt32(blocks)]
                    shape.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 3) }
                    encoder.dispatchThreadgroups(
                        MTLSize(width: (blocks + 7) / 8, height: (count * 4 + 7) / 8, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    encoder.endEncoding()
                    guard let finish = command.makeComputeCommandEncoder() else {
                        throw EngineError("Pinned indexer finish allocation failed")
                    }
                    finish.setComputePipelineState(finishKernel)
                    finish.setBuffer(dots, offset: 0, index: 0)
                    finish.setBuffer(out, offset: 0, index: 1)
                    var n = UInt32(blocks)
                    var denominator = sqrt(Float(128))
                    withUnsafeBytes(of: &n) { finish.setBytes($0.baseAddress!, length: $0.count, index: 2) }
                    withUnsafeBytes(of: &denominator) { finish.setBytes($0.baseAddress!, length: $0.count, index: 3) }
                    finish.dispatchThreads(
                        MTLSize(width: count * blocks, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
                    )
                    finish.endEncoding()
                    command.commit()
                    try command.waitBounded()
                    guard command.status == .completed else { throw EngineError.gpu("Pinned indexer GPU failed", command.error) }
                    result.append(
                        contentsOf: UnsafeBufferPointer(start: out.contents().assumingMemoryBound(to: Float.self), count: count * blocks))
                }
            }
            return result
        }
    }
    static let source = #"""
        #include <metal_stdlib>
        #include <metal_simdgroup_matrix>
        using namespace metal;
        // Same float8x8 fragments and increasing K-eight updates as pinned Steel.
        // Tiling changes only output ownership; no BF16 narrowing or K split occurs.
        kernel void origami_indexer_dots(device const float *q [[buffer(0)]],device const float *k [[buffer(1)]],device float *out [[buffer(2)]],constant uint *shape [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
            uint rows=shape[0],blocks=shape[1],rowBase=group.y*8,blockBase=group.x*8;
            threadgroup float aa[64],bb[64],cc[64];
            simdgroup_float8x8 a,b,c=make_filled_simdgroup_matrix<float,8,8>(0.0f);
            for(uint d=0;d<128;d+=8){
                for(uint i=lane;i<64;i+=32){uint row=i/8,col=i%8;aa[i]=rowBase+row<rows?q[(rowBase+row)*128+d+col]:0.0f;bb[i]=blockBase+col<blocks?k[(blockBase+col)*128+d+row]:0.0f;}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(a,aa,8);simdgroup_load(b,bb,8);simdgroup_multiply_accumulate(c,a,b,c);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            simdgroup_store(c,cc,8);threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint i=lane;i<64;i+=32){uint row=rowBase+i/8,block=blockBase+i%8;if(row<rows&&block<blocks)out[row*blocks+block]=cc[i];}
        }
        // The same fragments and K-eight order as origami_indexer_dots. Blocks before old are read in place from the
        // conversation's BF16 pages and widened exactly as the FP32 copy was; this step's blocks come from fresh.
        struct OrigamiPooledPages { array<device const ushort *, 1024> pages [[id(0)]]; };
        kernel void origami_indexer_dots_paged(device const float *q [[buffer(0)]],constant OrigamiPooledPages &pooled [[buffer(1)]],device float *out [[buffer(2)]],constant uint *shape [[buffer(3)]],device const float *fresh [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
            uint rows=shape[0],blocks=shape[1],old=shape[2],rowBase=group.y*8,blockBase=group.x*8;
            threadgroup float aa[64],bb[64],cc[64];
            simdgroup_float8x8 a,b,c=make_filled_simdgroup_matrix<float,8,8>(0.0f);
            for(uint d=0;d<128;d+=8){
                for(uint i=lane;i<64;i+=32){uint row=i/8,col=i%8,block=blockBase+col;aa[i]=rowBase+row<rows?q[(rowBase+row)*128+d+col]:0.0f;
                    bb[i]=block<old?as_type<float>(uint(pooled.pages[block/64][(block%64)*128+d+row])<<16):(block<blocks?fresh[(block-old)*128+d+row]:0.0f);}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(a,aa,8);simdgroup_load(b,bb,8);simdgroup_multiply_accumulate(c,a,b,c);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            simdgroup_store(c,cc,8);threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint i=lane;i<64;i+=32){uint row=rowBase+i/8,block=blockBase+i%8;if(row<rows&&block<blocks)out[row*blocks+block]=cc[i];}
        }
        // One thread per (row, block): relu of each head's dot (NaN kept, -0 made +0), summed head 0 then
        // h + sum for heads 1..3, divided by the host's sqrt(128).
        kernel void origami_indexer_finish(device const float *dots [[buffer(0)]],device float *out [[buffer(1)]],constant uint &blocks [[buffer(2)]],constant float &denominator [[buffer(3)]],uint i [[thread_position_in_grid]]) {
            uint row=i/blocks,block=i%blocks;float values[4];
            for(uint h=0;h<4;++h){float dot=dots[(row*4+h)*blocks+block];float positive=isnan(dot)?dot:(dot>0.0f?dot:0.0f);values[h]=positive+0.0f;}
            float sum=values[0];for(uint h=1;h<4;++h)sum=values[h]+sum;
            out[i]=sum/denominator;
        }
        """#
}
