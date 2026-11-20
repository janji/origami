import Foundation

/// Compile with Sources/origami/ObservationProvenance.swift; no model, server or environment mutation.
@main enum ObservationProvenanceChecks {
    struct Failure: Error { let message: String }
    static func require(_ ok: Bool, _ message: String) throws { if !ok { throw Failure(message: message) } }
    static func tags(
        _ settings: [String: Any] = [:], _ environment: [String: String] = [:],
        tuning: [String: Any]? = nil
    ) throws -> [String: String] {
        try ObservationProvenance.tags(
            settings: settings, environment: environment, startupTuning: tuning,
            tuningScope: "fixture startup tuning")
    }
    static func main() throws {
        let empty = try tags()
        for key in ["memory_limit_bytes", "expert_memory_bytes", "hot_copy_bytes"] {
            // Both formerly displayed as 30.1 GB, and even one-byte differences must retain their identity.
            let first = try tags([key: UInt64(30_100_000_000)])
            let second = try tags([key: UInt64(30_100_000_001)])
            try require(first["startup_settings_sha256"] != second["startup_settings_sha256"], "rounded budget collision: " + key)
        }
        let firstStore = try tags(["stores": [["path": "/fixture/a,b", "size_bytes": UInt64.max]]])
        let secondStore = try tags(["stores": [["path": "/fixture/a,b", "size_bytes": UInt64.max - 1]]])
        try require(firstStore["startup_settings_sha256"] != secondStore["startup_settings_sha256"], "UInt64 store budget rounded")
        try require(
            try tags(["models_dirs": ["a,b", "c"]])["startup_settings_sha256"]
                != tags(["models_dirs": ["a", "b,c"]])["startup_settings_sha256"], "joined path collision")
        try require(
            try tags(["a": 1, "b": ["x": 2, "y": 3]]) == tags(["b": ["y": 3, "x": 2], "a": 1]),
            "dictionary insertion order altered identity")
        for key in ObservationProvenance.executionEnvironmentKeys {
            let changed = try tags([:], [key: "1"])
            try require(
                changed["experimental_configuration_sha256"] != empty["experimental_configuration_sha256"], "execution flag omitted: " + key
            )
            try require(changed["startup_settings_sha256"] == empty["startup_settings_sha256"], "environment mixed into settings scope")
        }
        try require(ObservationProvenance.executionEnvironmentKeys.contains("ORIGAMI_FAMILY_PREFETCH_OBSERVATIONS"), "O12 flag missing")
        let prefetchOff = try tags([:], ["ORIGAMI_FAMILY_PREFETCH": "0"])
        let prefetchOn = try tags([:], ["ORIGAMI_FAMILY_PREFETCH": "1"])
        try require(
            prefetchOff["experimental_configuration_sha256"] != prefetchOn["experimental_configuration_sha256"],
            "prefetch on/off provenance collision")
        try require(
            try tags([:], ["ORIGAMI_FAMILY_SPIN": "unset"])["experimental_configuration_sha256"]
                != empty["experimental_configuration_sha256"], "absence confused with literal unset")
        let secretA = try tags(
            ["api_key": "fixture-a", "lan_key": "fixture-b"], ["ORIGAMI_API_KEY": "fixture-c", "HF_TOKEN": "fixture-d"])
        let secretB = try tags(["api_key": "fixture-changed"], ["ORIGAMI_UNKNOWN_CREDENTIAL": "fixture-e"])
        try require(secretA == empty && secretB == empty, "credential value or presence entered a digest")
        let tunedA = try tags(tuning: ["overrides": ["family_spin": true]])
        let tunedB = try tags(tuning: ["overrides": ["family_spin": false]])
        try require(tunedA["startup_tuning_sha256"] != tunedB["startup_tuning_sha256"], "startup tuning collision")
        try require(
            tunedA["experimental_configuration_sha256"] == tunedB["experimental_configuration_sha256"],
            "tuning mixed into environment scope")
        let resolved = try tags(tuning: ["overrides": ["family_spin": true], "resolved_family_knobs": ["family_spin": false]])
        try require(resolved["startup_tuning_sha256"] != tunedA["startup_tuning_sha256"], "resolved family settings omitted")
        try require(
            empty["startup_tuning_sha256"] == nil && empty["provenance_schema_version"] == "2", "missing tuning invented or schema missing")
        do {
            _ = try tags(["sampling_values": ["temperature": Double.nan]])
            throw Failure(message: "nonfinite setting accepted")
        } catch is ObservationProvenance.InvalidValue {}
        do {
            _ = try tags(tuning: ["invalid": Double.infinity])
            throw Failure(message: "nonfinite tuning accepted")
        } catch is ObservationProvenance.InvalidValue {}
        print(
            "Observation provenance: passed (exact budgets, arrays, all execution flags, tuning, redaction and deterministic serialization)"
        )
    }
}
