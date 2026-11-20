import Foundation
import OrigamiEngine

func wholeTokenPlanConfiguration() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("token-plan-" + UUID().uuidString + ".json")
    defer { try? FileManager.default.removeItem(at: url) }
    func load(_ object: [String: Any]) throws -> RuntimeConfiguration {
        try JSONSerialization.data(withJSONObject: object).write(to: url)
        return try RuntimeConfiguration.load(url)
    }
    guard try !load(["schema": "origami-config-v1", "profile": "exact"]).decodeTokenPlan else {
        throw EngineError("Token plan became default-on")
    }
    for context in [8192, 32768] {
        let source = CheckFixtures.file("config/exact-whole-\(context).json")
        var object = try JSONSerialization.jsonObject(with: Data(contentsOf: source)) as! [String: Any]
        var execution = object["execution"] as! [String: Any]
        execution["decode_early_demand"] = true
        execution["decode_token_plan"] = true
        object["execution"] = execution
        guard try load(object).decodeTokenPlan else { throw EngineError("Token plan flag was lost") }
        for (key, value) in [
            ("decode_token_plan", 1 as Any), ("decode_token_plan", "true" as Any),
            ("decode_early_demand", false), ("whole_execution_schedule", false), ("gdn_resident_state", false),
        ] {
            var candidate = object
            var changed = execution
            changed[key] = value
            candidate["execution"] = changed
            var denied = false
            do { _ = try load(candidate) } catch is EngineError { denied = true }
            guard denied else { throw EngineError("Token plan accepted invalid dependency: " + key) }
        }
    }
}
