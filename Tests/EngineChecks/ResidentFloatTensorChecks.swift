import Darwin
import Foundation
import Metal
import OrigamiEngine

private func denseResidentRequire(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try condition() else { throw EngineError(message) }
}
private func denseResidentReject(_ body: () throws -> Void) throws {
    do { try body() } catch is EngineError { return }
    throw EngineError("Invalid dense resident operation unexpectedly succeeded")
}
private func denseResidentDevice() throws -> MTLDevice {
    guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("Missing Metal device") }
    return device
}

private final class DenseResidentFixture {
    let directory: URL, payload: URL, store: PackStore
    let exceptionalFloatBits: [UInt32]
    init() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("origami-dense-resident-" + UUID().uuidString)
        let payload = directory.appendingPathComponent("tensor.bin")
        let exceptionalFloatBits: [UInt32] = [
            0, 0x8000_0000, 0x7fc1_2345, 0xff81_2345,
            0x7f80_0000, 0xff80_0000, 1, 0x807f_ffff,
        ]
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        var data = Data()
        var tensors: [String: Any] = [:]
        func append<T>(_ name: String, _ dtype: String, _ shape: [Int], _ values: [T]) {
            let bytes = values.withUnsafeBytes { Data($0) }
            tensors["language_model.model.test." + name] = [
                "file": "tensor.bin", "dtype": dtype,
                "offset": data.count, "length": bytes.count, "shape": shape,
            ]
            data.append(bytes)
        }
        append("all.weight", "BF16", [256, 256], (0...65535).map(UInt16.init))
        let bf = (0..<32).map { BF16.encode(Float($0 - 16) / 8) }
        let fp = (0..<32).map { Float($0 - 15) / 16 }
        append("bf.weight", "BF16", [4, 8], bf)
        append("fp.weight", "F32", [4, 8], fp)
        append("special.weight", "F32", [2, 4], exceptionalFloatBits)
        let source = tensors["language_model.model.test.bf.weight"] as! [String: Any]
        for name in [
            "model.layers.0.mlp.switch_mlp.gate_proj.weight", "model.embed_tokens.weight",
            "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
        ] {
            tensors["language_model." + name] = source
        }
        tensors["language_model.model.test.quant.weight"] = source
        tensors["language_model.model.test.quant.scales"] = source
        var malformed = source
        malformed["shape"] = [4, 9]
        tensors["language_model.model.test.bad-shape.weight"] = malformed
        malformed["shape"] = [Int.max, 2]
        tensors["language_model.model.test.overflow.weight"] = malformed
        malformed = source
        malformed["dtype"] = "I64"
        tensors["language_model.model.test.bad-type.weight"] = malformed
        malformed = source
        malformed["shape"] = [32]
        tensors["language_model.model.test.vector.weight"] = malformed
        // A quantized float module (4-bit group 64: 2 rows of 8 words, one scale and bias each) whose weight or
        // scales span less than its shapes need.
        append("q.mlp.gate.weight", "U32", [2, 8], [UInt32](repeating: 0x7654_3210, count: 16))
        append("q.mlp.gate.scales", "BF16", [2, 1], [BF16.encode(1), BF16.encode(1)])
        append("q.mlp.gate.biases", "BF16", [2, 1], [BF16.encode(0), BF16.encode(0)])
        for (module, part, length) in [("short-weight", "weight", 4), ("short-scales", "scales", 2)] {
            for name in ["weight", "scales", "biases"] {
                var span = tensors["language_model.model.test.q.mlp.gate." + name] as! [String: Any]
                if name == part { span["length"] = length }
                tensors["language_model.model.test.\(module).mlp.gate." + name] = span
            }
        }
        try data.write(to: payload)
        store = try PackStore(model: ModelSource(fixture: directory, spans: tensors))
        self.directory = directory
        self.payload = payload
        self.exceptionalFloatBits = exceptionalFloatBits
    }
    deinit { try? FileManager.default.removeItem(at: directory) }
    func mutate() throws {
        let writer = try FileHandle(forWritingTo: payload)
        try writer.write(contentsOf: Data([0xff]))
        try writer.close()
    }
}

func denseResidentPreservesAllBF16BitsAndFloatPayloads() throws {
    let fixture = try DenseResidentFixture()
    let device = try denseResidentDevice()
    let ledger = try MemoryLedger(ceiling: 10_000_000, transientMargin: 1)
    try autoreleasepool {
        let cache = try ResidentFloatTensorCache(store: fixture.store, device: device, ledger: ledger, byteLimit: 1_000_000)
        guard let bf = try cache.acquire("model.test.all.weight"),
            let fp = try cache.acquire("model.test.special.weight")
        else { throw EngineError("Valid dense tensor bypassed") }
        try denseResidentRequire(bf.elementCount == 65536 && bf.buffer.length == 65536 * 4, "Dense BF16 decoded length changed")
        let bits = bf.buffer.contents().assumingMemoryBound(to: UInt32.self)
        for index in 0..<65536 {
            try denseResidentRequire(bits[index] == UInt32(index) << 16, "In-place BF16 expansion changed bits at \(index)")
        }
        let actual = Array(UnsafeBufferPointer(start: fp.buffer.contents().assumingMemoryBound(to: UInt32.self), count: 8))
        try denseResidentRequire(actual == fixture.exceptionalFloatBits, "Dense FP32 residency changed IEEE payloads")
        let hit = try cache.acquire("language_model.model.test.all.weight")
        try denseResidentRequire(hit?.buffer === bf.buffer, "Dense cache canonical alias created another buffer")
        let state = cache.snapshot()
        try denseResidentRequire(
            state.entries == 2 && state.misses == 2 && state.hits == 1 && state.sourceReadBytes == 131104 && state.decodedBytes == 262176
                && state.maximumEntries == 384,
            "Dense cache read, decoded-byte or entry accounting changed")
    }
    try denseResidentRequire(ledger.snapshot().reserved == 0, "Dense residency retained dead resources")
}

func denseResidentCapsAndEscapedBuffersRemainAccounted() throws {
    let fixture = try DenseResidentFixture()
    let device = try denseResidentDevice()
    let page = UInt64(getpagesize())
    let ledger = try MemoryLedger(ceiling: 10_000_000, transientMargin: 1)
    var escaped: MTLBuffer?
    try autoreleasepool {
        var cache: ResidentFloatTensorCache? = try ResidentFloatTensorCache(
            store: fixture.store, device: device,
            ledger: ledger, byteLimit: page, maximumEntries: 1)
        var lease = try cache!.acquire("model.test.bf.weight")
        escaped = lease!.buffer
        lease = nil
        try denseResidentRequire(try cache!.acquire("model.test.fp.weight") == nil, "Dense cache ignored byte/entry cap")
        let state = cache!.snapshot()
        try denseResidentRequire(
            state.entries == 1 && state.allocatedBytes == page && state.bypasses == 1,
            "Dense capacity bypass evicted or overallocated")
        cache = nil
    }
    try withExtendedLifetime(escaped) {
        try denseResidentRequire(
            escaped?.length == 128 && ledger.snapshot().categories["resident-float-tensors"] == page
                && ledger.snapshot().categories["resident-float-metadata"] != nil, "Raw dense buffer escape lost ownership charges")
    }
    escaped = nil
    try denseResidentRequire(ledger.snapshot().reserved == 0, "Dense raw buffer final release did not retire reservations")
    try autoreleasepool {
        let cache = try ResidentFloatTensorCache(
            store: fixture.store, device: device, ledger: ledger,
            byteLimit: page * 8, maximumEntries: 1)
        _ = try cache.acquire("model.test.bf.weight")
        try denseResidentRequire(try cache.acquire("model.test.fp.weight") == nil, "Dense entry cap was ignored with spare bytes")
    }
    try denseResidentRequire(ledger.snapshot().reserved == 0, "Dense entry-cap fixture leaked reservations")
}

func denseResidentGeometryIdentityAndLedgerFailuresAreFatal() throws {
    let device = try denseResidentDevice()
    let page = UInt64(getpagesize())
    let fixture = try DenseResidentFixture()
    let ledger = try MemoryLedger(ceiling: 10_000_000, transientMargin: 1)
    try autoreleasepool {
        let cache = try ResidentFloatTensorCache(store: fixture.store, device: device, ledger: ledger, byteLimit: 1_000_000)
        for name in [
            "model.layers.0.mlp.switch_mlp.gate_proj.weight", "model.embed_tokens.weight",
            "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", "model.test.quant.weight",
        ] {
            try denseResidentRequire(try cache.acquire(name) == nil, "Ineligible dense weight entered persistent residency")
        }
        try denseResidentRequire(
            try cache.acquire("model.test.q.mlp.gate.weight")?.elementCount == 128, "Quantized dense module was not dequantized")
        for name in [
            "missing", "model.test.bad-shape.weight", "model.test.overflow.weight",
            "model.test.bad-type.weight", "model.test.vector.weight",
            "model.test.short-weight.mlp.gate.weight", "model.test.short-scales.mlp.gate.weight",
        ] {
            try denseResidentReject { _ = try cache.acquire(name) }
        }
        try denseResidentReject {
            _ = try ResidentFloatTensorCache(store: fixture.store, device: device, ledger: ledger, byteLimit: 600_000_001)
        }
        try denseResidentReject {
            _ = try ResidentFloatTensorCache(store: fixture.store, device: device, ledger: ledger, byteLimit: page, maximumEntries: 385)
        }
    }
    for warm in [false, true] {
        let changed = try DenseResidentFixture()
        try autoreleasepool {
            let cache = try ResidentFloatTensorCache(store: changed.store, device: device, ledger: ledger, byteLimit: page)
            if warm { _ = try cache.acquire("model.test.bf.weight") }
            try changed.mutate()
            try denseResidentReject { _ = try cache.acquire("model.test.bf.weight") }
            try denseResidentRequire(cache.snapshot().entries == (warm ? 1 : 0), "Source mutation published stale dense weights")
        }
    }
    let metadataBytes: UInt64 = 512 + 65536
    let denied = try MemoryLedger(ceiling: metadataBytes + page, transientMargin: 1)
    try autoreleasepool {
        let cache = try ResidentFloatTensorCache(
            store: fixture.store, device: device, ledger: denied,
            byteLimit: page, maximumEntries: 1)
        try denseResidentReject { _ = try cache.acquire("model.test.bf.weight") }
        try denseResidentRequire(
            cache.snapshot().entries == 0 && cache.snapshot().bypasses == 0
                && denied.snapshot().categories["resident-float-tensors"] == nil, "Ledger denial silently bypassed or leaked dense weights")
    }
    try denseResidentRequire(
        denied.snapshot().reserved == 0 && ledger.snapshot().reserved == 0,
        "Dense failure fixture retained reservations")
}

func denseReferenceWeightsPreservesBLASAcrossLayersAndFallback() throws {
    let fixture = try DenseResidentFixture()
    let ledger = try MemoryLedger(ceiling: 30_000_000, transientMargin: 1)
    try autoreleasepool {
        let kernels = try ReferenceKernels()
        let baseline = try ReferenceWeights(store: fixture.store, kernels: kernels, ledger: ledger)
        let resident = try ReferenceWeights(store: fixture.store, kernels: kernels, ledger: ledger, denseResidentBytes: 1_000_000)
        let bypass = try ReferenceWeights(store: fixture.store, kernels: kernels, ledger: ledger, denseResidentBytes: 1)
        for rows in [1, 7, 256] {
            try baseline.enterLayer(rows % 48)
            try resident.enterLayer(rows % 48)
            try bypass.enterLayer(rows % 48)
            let x = (0..<(rows * 8)).map { Float($0 % 19 - 9) / 4 }
            for name in ["model.test.bf", "model.test.fp"] {
                for bf16 in [false, true] {
                    let expected = try baseline.linear(x, name: name, outputBF16: bf16).map(\.bitPattern)
                    let actual = try resident.linear(x, name: name, outputBF16: bf16).map(\.bitPattern)
                    let fallback = try bypass.linear(x, name: name, outputBF16: bf16).map(\.bitPattern)
                    try denseResidentRequire(
                        expected == actual && expected == fallback,
                        "Dense residency changed BLAS output for rows=\(rows), \(name), BF16=\(bf16)")
                }
            }
        }
        try denseResidentRequire(
            resident.denseResidentSnapshot?.entries == 2 && resident.denseResidentSnapshot?.sourceReadBytes == 192,
            "Changing layers discarded persistent decoded weights")
        try denseResidentRequire(
            bypass.denseResidentSnapshot?.entries == 0 && bypass.denseResidentSnapshot?.bypasses == 12,
            "Tiny dense cache did not preserve the original fallback")
        try fixture.mutate()
        try denseResidentReject { _ = try resident.linear([Float](repeating: 1, count: 8), name: "model.test.bf") }
    }
    try denseResidentRequire(ledger.snapshot().reserved == 0, "Dense ReferenceWeights leaked its cache or fallback data")
}
