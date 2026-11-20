import Foundation
import Metal
import ObjectiveC

private final class PinnedRoPEAllocation {
    let ledger: MemoryLedger
    let reservation: MemoryLedger.Reservation
    init(ledger: MemoryLedger, bytes: UInt64) throws {
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: bytes, category: "pinned_rope_buffers")
    }
    deinit { try? ledger.release(reservation) }
}
private enum PinnedRoPEAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// Pinned MLX 1f8e74e3 FP32 pow/multiply/cos/sin, followed by BF16
/// partial rotation. CPU libm tables differ at real model positions 64/128.
/// Safe algebra and fast elementary functions are separate, explicit choices.
/// Two ledger-owned table entries mirror MLX's bounded query/pool cache.
/// Theta 10,000,000 over the first 64 dimensions of each head (32 frequency pairs, half-rotation
/// d <-> d+32); the rest pass through. Table = [positions][32] cos then [positions][32] sin, BF16-rounded.
public final class PinnedRoPE {
    private let device: MTLDevice, queue: MTLCommandQueue, ledger: MemoryLedger
    private let inverse: MTLBuffer
    private let inverseKernel, anglesKernel, anglesPerPairKernel, trigKernel, rotateKernel: MTLComputePipelineState
    private var cache: [(positions: [Int], table: MTLBuffer)] = []
    /// A prompt with images: token indices map to M-RoPE positions. Tables are cached per token
    /// indices, so they are dropped whenever the map changes.
    public var positionMap: RoPEPositionMap? { didSet { if positionMap != oldValue { cache.removeAll() } } }
    public private(set) var tableHits = 0
    public private(set) var tableBuilds = 0

    public convenience init(ledger: MemoryLedger) throws {
        try self.init(ledger: ledger, source: Self.source, prefix: "origami_rope_")
    }
    // Internal fixture initializer selects exact pinned-header functors while
    // preserving the same host shapes and independently constructed library.
    init(ledger: MemoryLedger, source: String, prefix: String) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("Pinned RoPE Metal unavailable")
        }
        try ModelContract.validate(device: device)
        self.device = device
        self.queue = queue
        self.ledger = ledger
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        options.mathFloatingPointFunctions = .fast
        let library = try device.makeLibrary(source: source, options: options)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: prefix + name) else { throw EngineError("Missing pinned RoPE kernel") }
            return try device.makeComputePipelineState(function: function)
        }
        inverseKernel = try pipeline("inv")
        anglesKernel = try pipeline("angles")
        anglesPerPairKernel = try pipeline("angles_pairs")
        trigKernel = try pipeline("trig")
        rotateKernel = try pipeline("rotate")
        inverse = try Self.allocate(32 * 4, device: device, ledger: ledger)
        try autoreleasepool {
            let exponents = try copy((0..<32).map { -Float($0 * 2) / Float(64) })
            let command = try makeCommand()
            try encode(command, inverseKernel, [exponents, inverse], threads: 32)
            try complete(command)
        }
    }
    private static func allocate(_ bytes: Int, device: MTLDevice, ledger: MemoryLedger) throws -> MTLBuffer {
        guard bytes > 0, bytes <= 8 * 1024 * 1024 else { throw EngineError("Pinned RoPE allocation outside bounded geometry") }
        let page = Int(getpagesize())
        let rounded = ((bytes + page - 1) / page) * page
        let owner = try PinnedRoPEAllocation(ledger: ledger, bytes: UInt64(rounded))
        guard let buffer = device.makeBuffer(length: rounded, options: .storageModeShared), buffer.allocatedSize <= rounded else {
            throw EngineError("Pinned RoPE allocation failed", code: "insufficient_memory")
        }
        objc_setAssociatedObject(buffer, &PinnedRoPEAssociation.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        return buffer
    }
    private func allocate(_ bytes: Int) throws -> MTLBuffer { try Self.allocate(bytes, device: device, ledger: ledger) }
    private func copy<T>(_ values: [T]) throws -> MTLBuffer {
        let buffer = try allocate(values.count * MemoryLayout<T>.stride)
        _ = values.withUnsafeBytes { memcpy(buffer.contents(), $0.baseAddress!, $0.count) }
        return buffer
    }
    private func makeCommand() throws -> MTLCommandBuffer {
        guard let command = queue.makeCommandBuffer() else { throw EngineError("Pinned RoPE command allocation failed") }
        command.label = "pinned-rope"
        return command
    }
    private func encode(
        _ command: MTLCommandBuffer, _ pipeline: MTLComputePipelineState,
        _ buffers: [MTLBuffer], shape: [UInt32] = [], threads: Int
    ) throws {
        guard let encoder = command.makeComputeCommandEncoder() else { throw EngineError("Pinned RoPE encoder allocation failed") }
        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers.enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
        if !shape.isEmpty { shape.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: buffers.count) } }
        encoder.dispatchThreads(
            MTLSize(width: threads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        encoder.endEncoding()
    }
    private func complete(_ command: MTLCommandBuffer) throws {
        command.commit()
        try command.waitBounded()
        guard command.status == .completed else { throw EngineError.gpu("Pinned RoPE command failed", command.error) }
    }
    private func validate(_ positions: [Int]) throws {
        guard (1...256).contains(positions.count), positions.allSatisfy({ (0..<ModelContract.maximumContext).contains($0) }) else {
            throw EngineError("Pinned RoPE requires 1...256 valid absolute positions")
        }
    }
    private func table(_ positions: [Int], command: MTLCommandBuffer) throws -> MTLBuffer {
        if let index = cache.firstIndex(where: { $0.positions == positions }) {
            let entry = cache.remove(at: index)
            cache.append(entry)
            tableHits += 1
            return entry.table
        }
        let angles = try allocate(positions.count * 32 * 4)
        let result = try allocate(positions.count * 64 * 4)
        if let map = positionMap, map.affects(positions) {
            // M-RoPE: each rotary pair takes its own section's position; a text token's three are equal.
            let perPair = positions.flatMap { index -> [UInt32] in
                let p = map.position(index)
                let components = [p.t, p.h, p.w]
                return (0..<32).map { UInt32(components[RoPEPositionMap.component($0)]) }
            }
            try encode(command, anglesPerPairKernel, [inverse, try copy(perPair), angles], threads: positions.count * 32)
        } else {
            try encode(command, anglesKernel, [inverse, try copy(positions.map(UInt32.init)), angles], threads: positions.count * 32)
        }
        try encode(command, trigKernel, [angles, result], shape: [UInt32(positions.count * 32)], threads: positions.count * 32)
        return result
    }
    private func retainTable(_ table: MTLBuffer, positions: [Int]) {
        // Only completed tables become cache entries; failed commands never
        // publish unfinished contents for a later call to observe.
        guard !cache.contains(where: { $0.positions == positions }) else { return }
        if cache.count == 2 { cache.removeFirst() }
        cache.append((positions, table))
        tableBuilds += 1
    }
    public func clearTables() { cache.removeAll() }

    /// A table encoded into a caller-owned command is provisional. Retaining
    /// this ticket retains the command and its charged resources. Only joined
    /// success may publish it in the helper's existing two-entry cache.
    package struct EncodedTables {
        fileprivate let owner: PinnedRoPE
        fileprivate let command: MTLCommandBuffer
        fileprivate let positions: [Int]
        fileprivate let buffer: MTLBuffer
    }

    package func encodeTables(_ command: MTLCommandBuffer, positions: [Int]) throws -> EncodedTables {
        try validate(positions)
        guard command.commandQueue.device.registryID == device.registryID, command.status == .notEnqueued else {
            throw EngineError("Pinned RoPE region command uses another device")
        }
        return EncodedTables(
            owner: self, command: command, positions: positions,
            buffer: try table(positions, command: command))
    }

    /// The existing rotate kernel and dispatch, with no host activation copy.
    /// Q/K/indexer rotations may share the same provisional position table in
    /// one command. Input/output must not alias: paired rotation lanes read
    /// each other's old values before writing their own results.
    package func encodeRotation(
        _ command: MTLCommandBuffer, input: MTLBuffer, output: MTLBuffer,
        width: Int, heads: Int, tables: EncodedTables
    ) throws {
        guard tables.owner === self, tables.command === command, command.status == .notEnqueued,
            [128, 256].contains(width), [1, 2, 4, 24].contains(heads),
            input !== output
        else { throw EngineError("Invalid pinned RoPE region ownership or geometry") }
        let count = tables.positions.count * heads * width
        for buffer in [input, output] {
            guard buffer.device.registryID == device.registryID, count * 4 <= buffer.length else {
                throw EngineError("Pinned RoPE region buffer is too small or on another device")
            }
        }
        try encode(
            command, rotateKernel, [input, tables.buffer, output],
            shape: [UInt32(width), UInt32(heads), UInt32(tables.positions.count)], threads: count)
    }

    package func publishTables(_ tables: EncodedTables) throws {
        guard tables.owner === self, tables.command.status == .completed else {
            throw EngineError("Cannot publish unjoined or failed pinned RoPE tables")
        }
        retainTable(tables.buffer, positions: tables.positions)
    }

    public func rotation(_ input: [Float], width: Int, heads: Int, positions: [Int]) throws -> [Float] {
        try validate(positions)
        guard [128, 256].contains(width), [1, 2, 4, 24].contains(heads),
            input.count == positions.count * heads * width
        else { throw EngineError("Invalid pinned RoPE geometry") }
        return try autoreleasepool {
            let command = try makeCommand()
            let table = try table(positions, command: command)
            let x = try copy(input)
            let out = try allocate(input.count * 4)
            try encode(
                command, rotateKernel, [x, table, out],
                shape: [UInt32(width), UInt32(heads), UInt32(positions.count)], threads: input.count)
            try complete(command)
            retainTable(table, positions: positions)
            return Array(UnsafeBufferPointer(start: out.contents().assumingMemoryBound(to: Float.self), count: input.count))
        }
    }
    /// Bounded primitive diagnostic surface; model execution keeps tables on GPU.
    public func tables(positions: [Int]) throws -> (cosines: [Float], sines: [Float]) {
        try validate(positions)
        return try autoreleasepool {
            let command = try makeCommand()
            let table = try table(positions, command: command)
            try complete(command)
            retainTable(table, positions: positions)
            let count = positions.count * 32
            let values = table.contents().assumingMemoryBound(to: Float.self)
            return (
                Array(UnsafeBufferPointer(start: values, count: count)), Array(UnsafeBufferPointer(start: values + count, count: count))
            )
        }
    }
    static let source = #"""
        #include <metal_stdlib>
        using namespace metal;
        inline float bf16(float x) {
            uint bits = as_type<uint>(x);
            if ((bits & 0x7fffffff) > 0x7f800000) return as_type<float>(bits | 0x00400000);
            return as_type<float>((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
        }
        // inv[i] = pow(1e7, -2i/64) for the 32 frequency pairs.
        kernel void origami_rope_inv(device const float *exponents [[buffer(0)]], device float *out [[buffer(1)]], uint i [[thread_position_in_grid]]) {
            out[i] = metal::pow(10000000.0f, exponents[i]);
        }
        // angle = float(position) * inv, one position per row (text) or one per pair (M-RoPE, *_pairs).
        kernel void origami_rope_angles(device const float *inv [[buffer(0)]], device const uint *positions [[buffer(1)]], device float *out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
            out[i] = float(positions[i / 32]) * inv[i % 32];
        }
        kernel void origami_rope_angles_pairs(device const float *inv [[buffer(0)]], device const uint *positions [[buffer(1)]], device float *out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
            out[i] = float(positions[i]) * inv[i % 32];
        }
        // Precise cos/sin, each rounded to BF16; cosines first, sines at offset `count`.
        kernel void origami_rope_trig(device const float *angles [[buffer(0)]], device float *out [[buffer(1)]], constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
            out[i] = bf16(metal::precise::cos(angles[i])); out[count + i] = bf16(metal::precise::sin(angles[i]));
        }
        // out = bf16(bf16(x*cos) + bf16(rot*sin)) for d < 64, rot = -x[d+32] (d < 32) or x[d-32]; x otherwise.
        kernel void origami_rope_rotate(device const float *x [[buffer(0)]], device const float *table [[buffer(1)]], device float *out [[buffer(2)]], constant uint *shape [[buffer(3)]], uint i [[thread_position_in_grid]]) {
            uint width = shape[0], heads = shape[1], rows = shape[2], d = i % width, row = i / (width * heads);
            if (d >= 64) { out[i] = x[i]; return; }
            float rot = d < 32 ? -x[i + 32] : x[i - 32];
            float c = table[row * 32 + d % 32], s = table[rows * 32 + row * 32 + d % 32];
            out[i] = bf16(bf16(x[i] * c) + bf16(rot * s));
        }
        """#
}
