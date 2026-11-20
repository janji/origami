import Foundation
import OrigamiEngine
import OrigamiEngineChecks

// `hc_decode_fusion` is default-off, boolean, and refuses a configuration without
// fused_hc_decode; the exactness check lives in HCDecodeFusionSelfChecks.

func hcDecodeFusionConfiguration() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-hc-decode-fusion-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func load(_ execution: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: ["schema": "origami-config-v1", "profile": "exact", "execution": execution]).write(
            to: url)
        return try RuntimeConfiguration.load(url)
    }
    func reject(_ execution: [String: Any], _ label: String) throws {
        do { _ = try load(execution) } catch is EngineError { return }
        throw EngineError("hc_decode_fusion accepted an invalid configuration: " + label)
    }
    let valid: [String: Any] = [
        "fused_hc_decode": true, "pinned_reductions": true, "pinned_dense": true, "pinned_unary": true,
        "resident_tensor_bytes": 10_000_000, "dense_resident_bytes": 1_000_000, "hc_decode_fusion": true,
    ]
    guard try !load([:]).hcDecodeFusion else { throw EngineError("hc_decode_fusion is not default-off") }
    var off = valid
    off["hc_decode_fusion"] = false
    guard try !load(off).hcDecodeFusion, try load(valid).hcDecodeFusion else { throw EngineError("hc_decode_fusion selection ignored") }
    var without = valid
    without["fused_hc_decode"] = false
    try reject(without, "without fused_hc_decode")
    try reject(["hc_decode_fusion": true], "alone")
    for value in [1, "true"] as [Any] {
        var invalid = valid
        invalid["hc_decode_fusion"] = value
        try reject(invalid, "non-boolean")
    }
}
