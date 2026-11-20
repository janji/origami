import Foundation
import OrigamiEngine

/// `qsa_exact_kernels` defaults to false, is a boolean, requires batched pinned attention, and is
/// accepted on the 8K and 32K fixture configurations.
func qsaExactKernelsConfiguration() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("qsa-exact-kernels-config-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func load(_ object: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: object).write(to: url)
        return try RuntimeConfiguration.load(url)
    }
    func refused(_ object: [String: Any], _ fragment: String) throws -> Bool {
        // An empty fragment accepts any EngineError (an earlier dependency guard may name the refusal).
        do { _ = try load(object) } catch let error as EngineError { return fragment.isEmpty || error.description.contains(fragment) }
        return false
    }
    let minimal: [String: Any] = ["schema": "origami-config-v1", "profile": "exact"]
    guard try !load(minimal).qsaExactKernels else { throw EngineError("qsa_exact_kernels is not default-off") }
    guard try refused(["schema": "origami-config-v1", "profile": "exact", "execution": ["qsa_exact_kernels": true]], "qsa_exact_kernels")
    else {
        throw EngineError("qsa_exact_kernels was accepted without batched pinned attention")
    }
    guard
        try refused(
            [
                "schema": "origami-config-v1", "profile": "exact",
                "execution": ["qsa_exact_kernels": true, "pinned_attention": true],
            ], "qsa_exact_kernels")
    else {
        throw EngineError("qsa_exact_kernels was accepted with unbatched pinned attention")
    }
    for context in [8192, 32768] {
        let source = CheckFixtures.file("config/exact-per-layer-\(context).json")
        var object = try JSONSerialization.jsonObject(with: Data(contentsOf: source)) as! [String: Any]
        var execution = object["execution"] as! [String: Any]
        guard try !load(object).qsaExactKernels else {
            throw EngineError("qsa_exact_kernels is not default-off on the per-layer configuration")
        }
        execution["qsa_exact_kernels"] = true
        object["execution"] = execution
        let enabled = try load(object)
        guard enabled.qsaExactKernels && enabled.batchedPinnedAttention && enabled.pinnedAttention else {
            throw EngineError("qsa_exact_kernels flag was lost")
        }
        for (key, value, fragment) in [
            ("qsa_exact_kernels", 1 as Any, "boolean"), ("qsa_exact_kernels", "true" as Any, "boolean"),
            ("batched_pinned_attention", false as Any, ""),
        ] {
            var candidate = object
            var changed = execution
            changed[key] = value
            candidate["execution"] = changed
            guard try refused(candidate, fragment) else { throw EngineError("Invalid qsa_exact_kernels dependency was accepted: " + key) }
        }
        execution["qsa_exact_kernels"] = false
        object["execution"] = execution
        guard try !load(object).qsaExactKernels else { throw EngineError("qsa_exact_kernels false was not honoured") }
    }
}
