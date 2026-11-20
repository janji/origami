import CExactIO
import Darwin
import Foundation
import Metal
import ObjectiveC

private final class ResidentFloatMetadata {
    let ledger: MemoryLedger
    let reservation: MemoryLedger.Reservation
    init(ledger: MemoryLedger, maximumEntries: Int) throws {
        self.ledger = ledger
        reservation = try ledger.reserve(
            bytes: UInt64(maximumEntries * 512 + 65_536),
            category: "resident-float-metadata")
    }
    deinit { try? ledger.release(reservation) }
}
private final class ResidentFloatAllocation {
    let metadata: ResidentFloatMetadata
    let reservation: MemoryLedger.Reservation
    init(metadata: ResidentFloatMetadata, bytes: UInt64) throws {
        self.metadata = metadata
        reservation = try metadata.ledger.reserve(bytes: bytes, category: "resident-float-tensors")
    }
    deinit { try? metadata.ledger.release(reservation) }
}
private enum ResidentFloatAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// Immutable FP32 dense weight storage. The buffer is read-only by contract.
/// Retain through a synchronous BLAS call; a raw buffer escape also retains its
/// ledger reservation and metadata owner until the actual resource is destroyed.
public final class ResidentFloatTensorLease: @unchecked Sendable {
    public let name: String
    public let sourceSpan: TensorSpan
    public let buffer: MTLBuffer
    public let elementCount: Int
    /// Pages wired at admission (`wireResidentPages`); unwired when the last lease drops, while the buffer is alive.
    fileprivate let wiredBytes: Int
    fileprivate init(name: String, sourceSpan: TensorSpan, buffer: MTLBuffer, elementCount: Int, wiredBytes: Int) {
        self.name = name
        self.sourceSpan = sourceSpan
        self.buffer = buffer
        self.elementCount = elementCount
        self.wiredBytes = wiredBytes
    }
    deinit { if wiredBytes > 0 { _ = origami_unwire_pages(buffer.contents(), wiredBytes) } }
}

/// Optional persistent decoded weights for dense CPU BLAS projections only.
/// No eviction or speculative reads: a bounded miss bypasses only when its
/// configured byte/entry limit cannot admit the new matrix. Ledger, geometry,
/// source-identity and read failures remain errors. All hits revalidate source
/// identities. The default 384-entry limit covers all 276 checkpoint matrices.
public final class ResidentFloatTensorCache: @unchecked Sendable {
    /// Budget, decoded and source bytes, hit/miss/bypass counters, wired bytes and failed wires.
    public struct Snapshot: Codable, Sendable {
        public let byteLimit, allocatedBytes, decodedBytes, sourceReadBytes: UInt64
        public let maximumEntries, entries: Int
        public let hits, misses, bypasses: UInt64
        public let wiredBytes, wireFallbacks: UInt64
    }
    public static let maximumByteLimit: UInt64 = 600_000_000
    public static let entryLimit = 384
    public let byteLimit: UInt64
    public let maximumEntries: Int
    public let sourceManifestSHA256: String
    private let store: PackStore
    private let device: MTLDevice
    private let metadata: ResidentFloatMetadata
    private let wire: Bool
    private let lock = NSLock()
    private var entries: [String: ResidentFloatTensorLease] = [:]
    private var allocatedBytes: UInt64 = 0, decodedBytes: UInt64 = 0, sourceReadBytes: UInt64 = 0
    private var hits: UInt64 = 0, misses: UInt64 = 0, bypasses: UInt64 = 0
    private var wiredBytes: UInt64 = 0, wireFallbacks: UInt64 = 0

    /// `wire` (from `wire_decode_slabs`) wires each new matrix once it is decoded; the ledger charge is the same.
    public init(
        store: PackStore, device: MTLDevice, ledger: MemoryLedger, byteLimit: UInt64,
        maximumEntries: Int = ResidentFloatTensorCache.entryLimit, wire: Bool = false
    ) throws {
        guard byteLimit <= Self.maximumByteLimit, (1...Self.entryLimit).contains(maximumEntries) else {
            throw EngineError("Invalid dense resident byte or entry limit")
        }
        self.store = store
        self.device = device
        self.byteLimit = byteLimit
        self.maximumEntries = maximumEntries
        self.wire = wire
        sourceManifestSHA256 = store.modelFingerprint
        metadata = try ResidentFloatMetadata(ledger: ledger, maximumEntries: maximumEntries)
    }

    public func acquire(_ name: String) throws -> ResidentFloatTensorLease? {
        lock.lock()
        defer { lock.unlock() }
        let key = name.hasPrefix("language_model.") ? name : "language_model." + name
        try store.validateBatchBoundary()
        // A retained hit already passed the immutable name/geometry policy.
        // Keep the live source-identity check before returning its exact bytes.
        if let entry = entries[key] {
            hits &+= 1
            return entry
        }
        return try admit(key)
    }

    /// Background warm-up of one matrix a caller will acquire next (host-stage prefetch).
    /// The same admission as a miss (this cache never evicts); counts no hit, and returns
    /// whether the matrix is resident. The later `acquire` validates source identity.
    public func prefetch(_ name: String) throws -> (resident: Bool, read: Bool) {
        lock.lock()
        defer { lock.unlock() }
        let key = name.hasPrefix("language_model.") ? name : "language_model." + name
        if entries[key] != nil { return (true, false) }
        return try admit(key) == nil ? (false, false) : (true, true)
    }

    /// Caller holds the lock; `key` is not resident. Nil is the bounded bypass.
    private func admit(_ key: String) throws -> ResidentFloatTensorLease? {
        guard let span = store.tensors[key] else { throw EngineError("Missing dense resident tensor: \(key)") }
        // A matrix the engine keeps as floats may come quantized (an 8-bit router, a 4-bit block inject);
        // it is dequantized here. Other quantized tensors stay with their own kernels.
        let module = String(key.dropLast(7))
        let quantized: (format: AffineFormat, scales: TensorSpan, biases: TensorSpan)? = store.tensors[module + ".scales"].flatMap {
            scales in
            guard ModelContract.floatModules.contains(where: { module.hasSuffix($0) }), let biases = store.tensors[module + ".biases"]
            else { return nil }
            return (store.model.format(ofModule: module), scales, biases)
        }
        guard key.hasPrefix("language_model.model.") || key == "language_model.lm_head.weight",
            key.hasSuffix(".weight"), !key.contains(".switch_mlp."),
            !key.contains(".ple_embedding.ngram_embedding."),
            !key.hasPrefix("language_model.model.embed_tokens."),
            store.tensors[module + ".scales"] == nil || quantized != nil
        else {
            bypasses &+= 1
            return nil
        }
        guard span.shape.count == 2, span.shape.allSatisfy({ $0 > 0 }), span.offset >= 0, span.length > 0,
            quantized == nil
                ? span.dtype == "BF16" || span.dtype == "F32" : span.dtype == "U32" && [2, 4, 8].contains(quantized!.format.bits)
        else {
            throw EngineError("Invalid dense resident dtype or matrix geometry: \(key)")
        }
        let columns = quantized.map { span.shape[1] * 32 / $0.format.bits } ?? span.shape[1]
        if let quantized {
            guard columns % quantized.format.group == 0, quantized.scales.dtype == "BF16", quantized.biases.dtype == "BF16",
                quantized.scales.shape == [span.shape[0], columns / quantized.format.group],
                quantized.biases.shape == quantized.scales.shape
            else {
                throw EngineError(
                    "Quantized dense resident matrix \(key) differs from its \(quantized.format.bits)-bit group-\(quantized.format.group) layout"
                )
            }
        }
        let elements = span.shape[0].multipliedReportingOverflow(by: columns)
        guard !elements.overflow else { throw EngineError("Dense resident matrix shape overflows") }
        let decoded = elements.partialValue.multipliedReportingOverflow(by: MemoryLayout<Float>.stride)
        // Quantized: rows × words u32 codes (no overflow: a word holds at least four codes, so this is at most `elements`)
        // and a bf16 scale and bias per group, so the dequantization below stays inside what it read.
        let original =
            quantized == nil
            ? elements.partialValue.multipliedReportingOverflow(by: span.dtype == "BF16" ? 2 : 4)
            : (partialValue: span.shape[0] * span.shape[1] * 4, overflow: false)
        guard !decoded.overflow, !original.overflow, Int64(original.partialValue) == span.length,
            quantized.map({
                $0.scales.length == Int64(span.shape[0] * (columns / $0.format.group) * 2) && $0.biases.length == $0.scales.length
            }) ?? true
        else {
            throw EngineError("Dense resident matrix byte count differs from source span")
        }
        let page = UInt64(getpagesize())
        let bytes = UInt64(decoded.partialValue)
        let charged = (bytes + page - 1) / page * page
        guard charged <= byteLimit - allocatedBytes, entries.count < maximumEntries else {
            bypasses &+= 1
            return nil
        }
        guard decoded.partialValue <= device.maxBufferLength else {
            throw EngineError("Dense resident tensor exceeds Metal buffer size")
        }
        misses &+= 1
        let entry = try autoreleasepool { () throws -> ResidentFloatTensorLease in
            let owner = try ResidentFloatAllocation(metadata: metadata, bytes: charged)
            guard let buffer = device.makeBuffer(length: decoded.partialValue, options: .storageModeShared) else {
                throw EngineError("Dense resident Metal allocation failed", code: "insufficient_memory")
            }
            objc_setAssociatedObject(buffer, &ResidentFloatAssociation.key, owner, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
            guard UInt64(buffer.allocatedSize) <= charged else {
                throw EngineError("Dense resident allocation exceeded its page reservation")
            }
            if let quantized {
                // Plain float math (scale * code + bias): the value MLX's affine matmul uses.
                var words = [UInt32](repeating: 0, count: Int(span.length) / 4)
                var scales = [UInt16](repeating: 0, count: Int(quantized.scales.length) / 2)
                var biases = scales
                try words.withUnsafeMutableBytes { try store.readTensor(key, into: $0) }
                try scales.withUnsafeMutableBytes { try store.readTensor(module + ".scales", into: $0) }
                try biases.withUnsafeMutableBytes { try store.readTensor(module + ".biases", into: $0) }
                Self.dequantize(
                    words: words, scales: scales, biases: biases, rows: span.shape[0], columns: columns, format: quantized.format,
                    into: buffer.contents().assumingMemoryBound(to: Float.self))
            } else {
                try store.readTensor(key, into: UnsafeMutableRawBufferPointer(start: buffer.contents(), count: original.partialValue))
            }
            if quantized == nil && span.dtype == "BF16" {
                // Read into the first half of the final FP32 backing and widen
                // backwards. Each write lies above all unread BF16 elements;
                // no second buffer, host Data or transient typed array is needed.
                // The bit shift preserves signed zero and every NaN payload.
                for index in stride(from: elements.partialValue - 1, through: 0, by: -1) {
                    let bits = buffer.contents().load(fromByteOffset: index * 2, as: UInt16.self)
                    buffer.contents().storeBytes(of: UInt32(bits) << 16, toByteOffset: index * 4, as: UInt32.self)
                }
            }
            let wired = wire ? wireResidentPages(buffer) : 0
            if let wired { wiredBytes += UInt64(wired) } else { wireFallbacks &+= 1 }
            return ResidentFloatTensorLease(
                name: key, sourceSpan: span, buffer: buffer, elementCount: elements.partialValue, wiredBytes: wired ?? 0)
        }
        entries[key] = entry
        allocatedBytes += charged
        decodedBytes += bytes
        sourceReadBytes += UInt64(span.length) + (quantized.map { UInt64($0.scales.length + $0.biases.length) } ?? 0)
        return entry
    }

    /// An MLX affine matrix as floats: `format.bits`-bit codes packed low bits first into u32 words, a bf16 scale and
    /// bias per `format.group` columns, each value scale * code + bias in plain float math (the value MLX's affine
    /// matmul uses). The caller has checked that `words`, `scales` and `biases` hold `rows` x `columns` of that layout.
    static func dequantize(
        words: [UInt32], scales: [UInt16], biases: [UInt16], rows: Int, columns: Int, format: AffineFormat,
        into output: UnsafeMutablePointer<Float>
    ) {
        let bits = format.bits
        let group = format.group
        let mask = UInt32((1 << bits) - 1)
        for index in 0..<(rows * columns) {
            let row = index / columns
            let column = index % columns
            let bit = column * bits
            let code = (words[row * (columns * bits / 32) + bit / 32] >> UInt32(bit % 32)) & mask
            let g = row * (columns / group) + column / group
            output[index] = BF16.decode(scales[g]) * Float(code) + BF16.decode(biases[g])
        }
    }

    public func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(
            byteLimit: byteLimit, allocatedBytes: allocatedBytes, decodedBytes: decodedBytes,
            sourceReadBytes: sourceReadBytes, maximumEntries: maximumEntries, entries: entries.count,
            hits: hits, misses: misses, bypasses: bypasses, wiredBytes: wiredBytes, wireFallbacks: wireFallbacks)
    }
}
