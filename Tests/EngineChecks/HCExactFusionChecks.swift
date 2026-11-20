import Foundation
import OrigamiEngine
import OrigamiEngineChecks

// `hc_exact_fusion` is default-off, boolean, and refuses a configuration without
// fused_hc_prefill or simd_matrix; the exactness checks live in HCExactFusionSelfChecks.

func hcExactFusionConfiguration() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-hc-exact-fusion-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func load(_ execution: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": execution]).write(
            to: url)
        return try RuntimeConfiguration.load(url)
    }
    func reject(_ execution: [String: Any], _ label: String) throws {
        do { _ = try load(execution) } catch is EngineError { return }
        throw EngineError("hc_exact_fusion accepted an invalid configuration: " + label)
    }
    let valid: [String: Any] = [
        "fused_hc_prefill": true, "pinned_unary": true, "pinned_dense": true, "pinned_reductions": true,
        "resident_tensor_bytes": 10_000_000, "dense_resident_bytes": 1_000_000, "simd_matrix": true, "hc_exact_fusion": true,
    ]
    guard try !load([:]).hcExactFusion else { throw EngineError("hc_exact_fusion is not default-off") }
    var off = valid
    off["hc_exact_fusion"] = false
    guard try !load(off).hcExactFusion, try load(valid).hcExactFusion else { throw EngineError("hc_exact_fusion selection ignored") }
    for key in ["fused_hc_prefill", "simd_matrix"] {
        var invalid = valid
        invalid[key] = false
        try reject(invalid, "without " + key)
    }
    try reject(["hc_exact_fusion": true], "alone")
    for value in [1, "true"] as [Any] {
        var invalid = valid
        invalid["hc_exact_fusion"] = value
        try reject(invalid, "non-boolean")
    }
}
