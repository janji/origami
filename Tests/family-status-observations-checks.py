#!/usr/bin/env python3
"""Exercise FamilyStatus's actual bounded observation projection with synthetic notes.

Like front-proxy-metrics-checks.py, compile a production method unchanged with
Foundation-only fixtures. No model, GPU, listener, or network connection is used.
Run compilation only when the shared Swift build slot is free. --prepare-only
checks extraction/integration without invoking Swift or executing the fixtures.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import tempfile


HARNESS = r'''
import Foundation

struct Failure: Error { let message: String }
enum FamilyStatusFixture {
    /* PRODUCTION_PROJECTION */
    static func report(_ notes: [String: Any]) -> [String: Any] {
        var report: [String: Any] = ["model": ["family": "deepseek"],
            "telemetry": ["kind": "boundary_snapshot", "boundary": "request_end"]]
        /* PRODUCTION_INTEGRATION */
        return report
    }
}
func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw Failure(message: message) }
}
func wire(_ value: [String: Any]) throws -> Data {
    try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
}
func roundTrip(_ value: [String: Any]) throws -> [String: Any] {
    try JSONSerialization.jsonObject(with: wire(value)) as! [String: Any]
}
func same(_ actual: Any?, _ expected: [String: Any], _ message: String) throws {
    guard let actual = actual as? [String: Any] else { throw Failure(message: message + ": absent") }
    try require(try wire(actual) == wire(expected), message)
}
@main enum Main {
    static func main() throws {
        var checks: [[String: Any]] = []
        func passed(_ name: String) { checks.append(["scenario": name, "passed": true]) }

        let unsupported = try roundTrip(FamilyStatusFixture.report([
            "expert_hits": 999, "internal_only": ["must_not_escape": true]]))
        try require(unsupported["prefetch_observations"] == nil
            && unsupported["expert_hits"] == nil && unsupported["internal_only"] == nil,
            "unsupported observations or unrelated internal notes escaped")
        try same(unsupported["telemetry"], ["kind": "boundary_snapshot", "boundary": "request_end"],
            "existing boundary metadata changed")
        passed("unsupported_absent_and_unrelated_notes_excluded")

        let disabled: [String: Any] = ["experimental": true, "enabled": false,
            "requested": false, "admission_refused": false, "reserved_bytes": 0]
        let off = try roundTrip(FamilyStatusFixture.report(["prefetch_observations": disabled]))
        try same(off["prefetch_observations"], disabled, "disabled observer was hidden or rewritten")
        passed("disabled_dictionaries_preserved")

        let denied: [String: Any] = ["experimental": true, "enabled": false,
            "requested": true, "admission_refused": true, "reserved_bytes": 0]
        let refusal = try roundTrip(FamilyStatusFixture.report(["prefetch_observations": denied]))
        try same(refusal["prefetch_observations"], denied, "observer admission refusal disappeared")
        passed("admission_refusals_preserved")

        var rankCounts = [UInt64](repeating: 0, count: 40 * 7 * 24)
        rankCounts[168] = 63
        let enabled: [String: Any] = ["experimental": true, "enabled": true,
            "scope": "model_instance_cumulative", "instance_id": "fixture-instance",
            "reserved_bytes": UInt64(262144), "estimated_internal_bytes": UInt64(201216),
            "layers": 40, "experts": 384, "rank_buckets": 7, "overflow_rank": 6,
            "rank_counts": rankCounts, "layer_counts": [UInt64](repeating: 0, count: 680),
            "source_counts": [UInt64](repeating: 0, count: 400), "pending_prediction": false,
            "live_outcome_tickets": 3, "dropped_outcome_tickets": UInt64(0)]
        let live = FamilyStatusFixture.report(["prefetch_observations": enabled])
        try same(try roundTrip(live)["prefetch_observations"], enabled,
            "active structured population changed in JSON")
        passed("enabled_identity_capacity_and_population_roundtrip")

        let route: [String: Any] = ["enabled": true, "count": 1, "row_stride": 5,
            "rows": [Int32(615), 0, 2, 7, 12], "dropped": 0]
        try same(try roundTrip(FamilyStatusFixture.report(["route_trace": route]))["route_trace"], route,
            "bounded route trace did not reach status")
        passed("route_trace_roundtrip")

        var changed = enabled
        rankCounts[168] = 126
        changed["rank_counts"] = rankCounts
        let newer = FamilyStatusFixture.report(["prefetch_observations": changed])
        try same(live["prefetch_observations"], enabled, "escaped snapshot changed with later notes")
        try same(newer["prefetch_observations"], changed, "new boundary snapshot did not refresh")
        passed("published_values_survive_later_snapshot_changes")

        let malformed = FamilyStatusFixture.report([
            "prefetch_observations": "unexpected non-dictionary", "route_trace": NSNull()])
        try require(malformed["prefetch_observations"] == nil && malformed["route_trace"] == nil,
            "non-dictionary internal note was exposed")
        passed("non_dictionary_notes_not_exposed")

        print(String(decoding: try wire(["checks": checks]), as: UTF8.self))
    }
}
'''


def extract(source):
    start = source.index("    static func appendObservationNotes(")
    end = source.index("\n    }\n", start) + len("\n    }\n")
    method = source[start:end]
    report_start = source.index("    func report() -> [String: Any] {")
    report_end = source.index("\n    }\n", report_start) + len("\n    }\n")
    report = source[report_start:report_end]
    integration = "        Self.appendObservationNotes(notes, to: &report)"
    if report.count(integration) != 1:
        raise ValueError("Observation projection is not wired once into FamilyStatus.report")
    if not re.search(r"let snapshot = backend\.telemetrySnapshot(,\s*|\s+let )notes = snapshot\.notes", report):
        raise ValueError("Observation report no longer uses the published boundary snapshot")
    return method, integration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    source_path = root / "Sources/origami/FamilyStatus.swift"
    source = source_path.read_text()
    method, integration = extract(source)
    fixture_source = HARNESS.replace("    /* PRODUCTION_PROJECTION */", method).replace(
        "        /* PRODUCTION_INTEGRATION */", integration)
    if args.prepare_only:
        report = {"state": "prepared_not_compiled", "runtime_fixtures_executed": False,
                  "integration_extraction_passed": True}
    else:
        with tempfile.TemporaryDirectory(prefix="origami-family-status-observations-") as temporary:
            directory = Path(temporary)
            fixture = directory / "Fixture.swift"
            fixture.write_text(fixture_source)
            binary = directory / "fixture"
            subprocess.run(["swiftc", "-parse-as-library", str(fixture), "-o", str(binary)], check=True)
            result = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        report["state"] = "model_free_fixtures_passed"
        report["runtime_fixtures_executed"] = True
    report.update({
        "scope": "Actual bounded FamilyStatus projection and integration call; Foundation synthetic notes; no model/network/GPU",
        "model_inference_executed": False,
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "projection_sha256": hashlib.sha256(method.encode()).hexdigest(),
        "integration_sha256": hashlib.sha256(integration.encode()).hexdigest(),
        "generated_fixture_sha256": hashlib.sha256(fixture_source.encode()).hexdigest(),
        "status_paths": ["prefetch_observations", "resident_precision", "route_trace"],
    })
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
