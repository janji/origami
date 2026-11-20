import Foundation
import OrigamiEngine

/// `SlowReason.verdict` on each rule's synthetic request in `Tests/Fixtures/status/slow-reasons.json`, the file the
/// dashboard's checks run its `slowReason` against too, so `origami status` and the page say the same.
func slowReasonRules() throws {
    let cases =
        try JSONSerialization.jsonObject(with: Data(contentsOf: CheckFixtures.file("status/slow-reasons.json"))) as? [[String: Any]] ?? []
    guard cases.count >= 6 else { throw EngineError("slow reasons: fixture missing") }
    for item in cases {
        let got = SlowReason.verdict(request: item["request"] as? [String: Any] ?? [:], status: item["status"] as? [String: Any] ?? [:])
        guard got == item["verdict"] as? String else { throw EngineError("slow reasons: \(item["rule"] ?? "?"): \(got)") }
    }
}
