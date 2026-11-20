import Darwin
import Foundation
import OrigamiEngine
import OrigamiEngineChecks

/// Exercises the complete optional residency integration, including a deliberate
/// whole-group bypass and a source mutation after successful cache hits.
func referenceWeightsResidencyPreservesFallbackAndRejectsMutation() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("origami-resident-linear-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let payload = directory.appendingPathComponent("tensor.bin")
    var bytes = Data()
    func append<T>(_ values: [T]) { values.withUnsafeBytes { bytes.append(contentsOf: $0) } }
    append((0..<32).map { UInt32(truncatingIfNeeded: $0 * 2_654_435_761) })
    append([UInt16](repeating: BF16.encode(0.0625), count: 4))
    append([UInt16](repeating: BF16.encode(-0.25), count: 4))
    try bytes.write(to: payload)
    let tensors: [String: Any] = [
        "language_model.fixture.weight": ["file": "tensor.bin", "dtype": "U32", "offset": 0, "length": 128, "shape": [4, 8]],
        "language_model.fixture.scales": ["file": "tensor.bin", "dtype": "BF16", "offset": 128, "length": 8, "shape": [4, 1]],
        "language_model.fixture.biases": ["file": "tensor.bin", "dtype": "BF16", "offset": 136, "length": 8, "shape": [4, 1]],
    ]
    let store = try PackStore(model: ModelSource(fixture: directory, spans: tensors))
    let ledger = try MemoryLedger()
    try autoreleasepool {
        let kernels = try ReferenceKernels()
        let page = UInt64(getpagesize())
        let baseline = try ReferenceWeights(store: store, kernels: kernels, ledger: ledger)
        let resident = try ReferenceWeights(store: store, kernels: kernels, ledger: ledger, residentBytes: 3 * page)
        let bypass = try ReferenceWeights(store: store, kernels: kernels, ledger: ledger, residentBytes: 2 * page)
        let x = (0..<64).map { Float(($0 % 9) - 4) / 8 }
        let expected = try baseline.linear(x, name: "fixture")
        for _ in 0..<2 {
            let hit = try resident.linear(x, name: "language_model.fixture")
            let fallback = try bypass.linear(x, name: "fixture")
            guard hit.map(\.bitPattern) == expected.map(\.bitPattern), fallback.map(\.bitPattern) == expected.map(\.bitPattern) else {
                throw EngineError("Optional tensor residency changed projection or bypass output")
            }
        }
        guard let cached = resident.residentSnapshot, let missed = bypass.residentSnapshot,
            cached.readBytes == 144, cached.hits == 3, cached.pinnedEntries == 0,
            missed.readBytes == 0, missed.bypasses == 2
        else {
            throw EngineError("Resident integration did not group-admit/cache/bypass exact tensors")
        }
        let writer = try FileHandle(forWritingTo: payload)
        try writer.write(contentsOf: Data([0xff]))
        try writer.close()
        var rejected = false
        do { _ = try resident.linear(x, name: "fixture") } catch { rejected = true }
        guard rejected, resident.residentSnapshot?.hits == cached.hits else {
            throw EngineError("Resident integration served a cached projection after source mutation")
        }
    }
    guard ledger.snapshot().reserved == 0 else { throw EngineError("Resident weight integration leaked a reservation") }
}

/// Exact full-domain widening, layer reentry, capacity fallback and source
/// invalidation for the small fixed-tensor access path. No model weights used.
func smallTensorResidencyPreservesBitsChargesAndMutation() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("origami-small-tensor-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let payload = directory.appendingPathComponent("tensor.bin")
    let bf16 = (0..<65_536).map { UInt16($0) }
    let f32: [UInt32] = [0, 0x8000_0000, 1, 0x007f_ffff, 0x3f80_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_0001, 0x7fa1_2345]
    var bytes = bf16.withUnsafeBytes { Data($0) }
    f32.withUnsafeBytes { bytes.append(contentsOf: $0) }
    try bytes.write(to: payload)
    let names = ["model.layers.0.linear_attn.norm.weight", "model.layers.1.linear_attn.A_log"]
    let constantNames = ["model.layers.0.constant_low", "model.layers.0.constant_high", names[1]]
    let tensors: [String: Any] = [
        "language_model." + constantNames[0]: ["file": "tensor.bin", "dtype": "BF16", "offset": 0, "length": 65536, "shape": [32768]],
        "language_model." + constantNames[1]: ["file": "tensor.bin", "dtype": "BF16", "offset": 65536, "length": 65536, "shape": [32768]],
        "language_model." + names[0]: [
            "file": "tensor.bin", "dtype": "BF16", "offset": 0, "length": bf16.count * 2, "shape": [bf16.count],
        ],
        "language_model." + names[1]: [
            "file": "tensor.bin", "dtype": "F32", "offset": bf16.count * 2, "length": f32.count * 4, "shape": [f32.count],
        ],
    ]
    let store = try PackStore(model: ModelSource(fixture: directory, spans: tensors))
    let ledger = try MemoryLedger(ceiling: 30_000_000, transientMargin: 1)
    try autoreleasepool {
        let kernels = try ReferenceKernels()
        let baseline = try ReferenceWeights(store: store, kernels: kernels, ledger: ledger)
        let resident = try ReferenceWeights(
            store: store, kernels: kernels, ledger: ledger, residentBytes: 1_000_000, residentSmallTensors: true)
        let bypass = try ReferenceWeights(store: store, kernels: kernels, ledger: ledger, residentBytes: 1, residentSmallTensors: true)
        for layer in [0, 1, 0] {
            for weights in [baseline, resident, bypass] { try weights.enterLayer(layer) }
            for (index, name) in names.enumerated() {
                let expected = index == 0 ? bf16.map { UInt32($0) << 16 } : f32
                for weights in [baseline, resident, bypass] {
                    guard try weights.tensor(name).map(\.bitPattern) == expected else {
                        throw EngineError("Small tensor residency changed source bits")
                    }
                }
            }
        }
        guard let snapshot = resident.residentSnapshot, snapshot.misses == 2, snapshot.hits == 4,
            snapshot.readBytes == UInt64(bytes.count), snapshot.pinnedEntries == 0,
            bypass.residentSnapshot?.bypasses == 6
        else { throw EngineError("Small tensor reentry/bypass did not preserve bounded cache access") }
        // Same-layer raw Data fallback must reuse the precharged host bound.
        let before = ledger.snapshot().reserved
        _ = try resident.data(names[0])
        guard ledger.snapshot().reserved == before else {
            throw EngineError("Small tensor/Data interop duplicated or replaced a live reservation")
        }
        for weights in [baseline, resident, bypass] { try weights.enterLayer(2) }
        guard ledger.snapshot().categories["reference_weight_and_transfer_bound"] == nil else {
            throw EngineError("Small tensor host charge survived its layer")
        }
        // Existing resident bytes remain; deny the host copy before allocation.
        let occupied = ledger.snapshot().reserved
        let blocker = try ledger.reserve(bytes: ledger.ceiling - ledger.transientMargin - occupied, category: "test-admission-block")
        var denied = false
        do { _ = try resident.tensor(names[0]) } catch let error as EngineError { denied = error.code == "insufficient_memory" }
        try ledger.release(blocker)
        guard denied, ledger.snapshot().reserved == occupied else {
            throw EngineError("Small tensor host copy bypassed admission or leaked")
        }
        try WholeExecutionSelfChecks.constantOwnership(
            store: store, names: constantNames,
            expected: [bf16.prefix(32768).map { UInt32($0) << 16 }, bf16.suffix(32768).map { UInt32($0) << 16 }, f32])
        let prior = resident.residentSnapshot!
        let writer = try FileHandle(forWritingTo: payload)
        try writer.write(contentsOf: Data([0xff]))
        try writer.close()
        var rejected = false
        do { _ = try resident.tensor(names[0]) } catch { rejected = true }
        guard rejected, resident.residentSnapshot?.hits == prior.hits else {
            throw EngineError("Small tensor cache hit ignored mutated source")
        }
    }
    guard ledger.snapshot().reserved == 0 else { throw EngineError("Small tensor ownership leaked after teardown") }
}

func smallTensorResidencyConfigurationIsBoundedAndBoolean() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-small-config-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func load(_ execution: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": execution]).write(
            to: url)
        return try RuntimeConfiguration.load(url)
    }
    guard try !load([:]).residentSmallTensors,
        try load(["resident_small_tensors": true, "resident_tensor_bytes": 1]).residentSmallTensors
    else { throw EngineError("Small tensor residency defaults differ") }
    let invalid: [[String: Any]] = [["resident_small_tensors": true], ["resident_small_tensors": 1], ["resident_small_tensors": "true"]]
    for execution in invalid {
        var rejected = false
        do { _ = try load(execution) } catch { rejected = true }
        guard rejected else { throw EngineError("Small tensor residency accepted invalid configuration") }
    }
}
