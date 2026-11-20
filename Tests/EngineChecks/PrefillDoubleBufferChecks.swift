import Foundation
import OrigamiEngine

/// `prefill_double_buffer` defaults to false, is a boolean and requires the whole schedule.
func prefillDoubleBufferConfiguration() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("prefill-double-buffer-config-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func load(_ object: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: object).write(to: url)
        return try RuntimeConfiguration.load(url)
    }
    func refused(_ object: [String: Any], _ fragment: String) throws -> Bool {
        // An empty fragment accepts any EngineError (Foundation's contains("") is false).
        do { _ = try load(object) } catch let error as EngineError { return fragment.isEmpty || error.description.contains(fragment) }
        return false
    }
    let minimal: [String: Any] = ["schema": "origami-config-v1", "profile": "exact"]
    guard try !load(minimal).prefillDoubleBuffer else { throw EngineError("prefill_double_buffer is not default-off") }
    guard
        try refused(
            ["schema": "origami-config-v1", "profile": "exact", "execution": ["prefill_double_buffer": true]], "prefill_double_buffer")
    else {
        throw EngineError("prefill_double_buffer was accepted without the whole schedule")
    }
    for context in [8192, 32768] {
        let source = CheckFixtures.file("config/exact-per-layer-\(context).json")
        var object = try JSONSerialization.jsonObject(with: Data(contentsOf: source)) as! [String: Any]
        var execution = object["execution"] as! [String: Any]
        execution["whole_execution_schedule"] = true
        object["execution"] = execution
        guard try !load(object).prefillDoubleBuffer else {
            throw EngineError("prefill_double_buffer is not default-off on the whole schedule")
        }
        execution["prefill_double_buffer"] = true
        object["execution"] = execution
        let enabled = try load(object)
        guard enabled.prefillDoubleBuffer && enabled.wholeExecutionSchedule else {
            throw EngineError("prefill_double_buffer flag was lost")
        }
        for (key, value, fragment) in [
            ("prefill_double_buffer", 1 as Any, "boolean"), ("prefill_double_buffer", "true" as Any, "boolean"),
            ("whole_execution_schedule", false as Any, ""),
        ] {
            var candidate = object
            var changed = execution
            changed[key] = value
            candidate["execution"] = changed
            guard try refused(candidate, fragment) else {
                throw EngineError("Invalid prefill_double_buffer dependency was accepted: " + key)
            }
        }
    }
}
