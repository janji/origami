import Foundation
import Metal
import OrigamiEngine

/// A group-32 checkpoint (mlx-community 4-bit) passes ReferenceWeights' resident guards: the HC entry with group-32
/// scales and a quantized block inject, and the GDN input projections with a quantized in_proj_a (matrices kept as
/// floats are dequantized at load). Zero weights: only the guards and the encoders' buffer checks are exercised.
func group32ResidentGuardsAcceptCheckpointShapes() throws {
    guard MTLCreateSystemDefaultDevice() != nil else { throw FixtureUnavailable(description: "no Metal device") }
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("origami-group32-guards-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    var spans: [String: Any] = [:]
    var formats: [String: AffineFormat] = [:]
    var length = 0
    func add(_ name: String, _ dtype: String, _ shape: [Int]) {
        let bytes = shape.reduce(1, *) * (dtype == "BF16" ? 2 : 4)
        spans[name] = ["file": "tensor.bin", "dtype": dtype, "offset": length, "length": bytes, "shape": shape]
        length += bytes
    }
    func quantized(_ module: String, width: Int, outputs: Int, keptAsFloat: Bool = false) {
        add(module + ".weight", "U32", [outputs, width / 8])
        add(module + ".scales", "BF16", [outputs, width / 32])
        add(module + ".biases", "BF16", [outputs, width / 32])
        if keptAsFloat { formats[module] = AffineFormat(bits: 4, group: 32) }
    }
    let hc = "language_model.model.layers.0.attn_hyper_connection"
    let gdn = "language_model.model.layers.0.linear_attn"
    add(hc + ".hc_norm.weight", "BF16", [10240])
    quantized(hc + ".input_mix_weight_down", width: 10240, outputs: 320)
    quantized(hc + ".input_mix_weight_up", width: 320, outputs: 10240)
    quantized(hc + ".block_inject_weight", width: 10240, outputs: 4, keptAsFloat: true)
    quantized(gdn + ".in_proj_qkv", width: 2560, outputs: 10240)
    quantized(gdn + ".in_proj_z", width: 2560, outputs: 6144)
    quantized(gdn + ".in_proj_a", width: 2560, outputs: 48, keptAsFloat: true)
    add(gdn + ".in_proj_b.weight", "BF16", [48, 2560])
    try Data(count: length).write(to: directory.appendingPathComponent("tensor.bin"))
    let store = try PackStore(model: ModelSource(fixture: directory, spans: spans, formats: formats))
    let ledger = try MemoryLedger(ceiling: 400_000_000, transientMargin: 1_000_000)
    try withGroup32 {
        try autoreleasepool {
            let kernels = try ReferenceKernels(pinnedUnary: true, ledger: ledger, pinnedRMS: true, quantizedTiled: true)
            let weights = try ReferenceWeights(
                store: store, kernels: kernels, ledger: ledger, simdMatrix: true,
                residentBytes: 64 << 20, denseResidentBytes: 8 << 20, pinnedDense: true, fusedHCDecode: true, gdnInputComposition: true)
            let mixed = try weights.hyperConnectionDecode(
                [Float](repeating: 0.5, count: 10240),
                base: "model.layers.0.attn_hyper_connection", useCombine: true, capture: false, shouldContinue: { true })
            guard mixed != nil else { throw EngineError("Group-32 HC entry bypassed its resident tensors") }
            let projected = try weights.gdnInputProjections(
                [Float](repeating: 0.5, count: 2560),
                base: "model.layers.0.linear_attn", shouldContinue: { true })
            guard projected != nil else { throw EngineError("Group-32 GDN input projections bypassed their resident tensors") }
        }
    }
    guard ledger.snapshot().reserved == 0 else { throw EngineError("Group-32 resident guard fixture leaked resources") }
}
