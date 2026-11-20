import CryptoKit
import Foundation

/// Hash only explicitly supplied non-secret startup inputs. These identities do not describe a request's
/// overrides, an automatic memory governor's decisions, or a later settings reload. Keep this Foundation-only
/// so model-free fixtures exercise the same canonical serialization used by the CLI.
enum ObservationProvenance {
    struct InvalidValue: Error, CustomStringConvertible {
        let description = "Startup provenance contains a value that is not valid finite JSON"
    }
    static let schema = 2
    /// Actual execution environment parsers, not an ORIGAMI_* prefix capture (which could ingest credentials).
    static let executionEnvironmentKeys = [
        "ORIGAMI_EXECUTION", "ORIGAMI_NO_BFLOAT_MATRICES", "ORIGAMI_SCRATCH_COMPACTION",
        "ORIGAMI_FAMILY_PREFETCH", "ORIGAMI_FAMILY_PREFETCH_DEPTH",
        "ORIGAMI_FAMILY_LAZY_STARTUP_WORKSPACE",
        "ORIGAMI_FAMILY_RESIDENT_Q4_PACK",
        "ORIGAMI_FAMILY_SPIN", "ORIGAMI_FAMILY_PREFILL_SLICES",
        "ORIGAMI_FAMILY_PREFETCH_OBSERVATIONS",
        "ORIGAMI_FAMILY_GROUPED_DECODE",
        "ORIGAMI_FAMILY_GROUPED_GATE_UP",
        "ORIGAMI_FAMILY_SAME_DRIVE_HOT_COPY", "ORIGAMI_FAMILY_PHASE_TIMING",
        "ORIGAMI_FAMILY_ROUTE_TRACE",
    ]

    static func hash(_ values: [String: Any]) throws -> String {
        let object: [String: Any] = ["schema": schema, "values": values]
        guard JSONSerialization.isValidJSONObject(object) else { throw InvalidValue() }
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func tags(
        settings: [String: Any], environment: [String: String],
        startupTuning: [String: Any]?, tuningScope: String
    ) throws -> [String: String] {
        // Defense in depth: callers provide typed settings and never these values, but neither the value nor
        // the presence of a secret should alter a public configuration digest.
        let safe = settings.filter { !["api_key", "lan_key"].contains($0.key) }
        let experiments: [String: Any] = Dictionary(
            uniqueKeysWithValues: executionEnvironmentKeys.map {
                ($0, environment[$0].map { $0 as Any } ?? NSNull())
            })
        var result = [
            "provenance_schema_version": String(schema),
            "startup_settings_sha256": try hash(safe),
            "startup_settings_scope": "typed non-secret startup settings; excludes later reloads and request overrides",
            "experimental_configuration_sha256": try hash(experiments),
            "experimental_configuration_scope": "explicit execution environment inputs; excludes automatic tuning and runtime decisions",
            "startup_tuning_scope": tuningScope,
        ]
        if let startupTuning { result["startup_tuning_sha256"] = try hash(startupTuning) }
        return result
    }
}
