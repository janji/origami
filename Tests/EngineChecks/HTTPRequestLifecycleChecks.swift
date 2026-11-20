import Foundation
import OrigamiEngine

func httpRequestLifecycleChecks() throws {
    func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw EngineError("Request lifecycle: " + message) }
    }
    let metrics = HTTPRequestMetrics(scope: "fixture")
    let row = metrics.begin(requestID: "same-request-123")
    row.mark("queue_start")
    row.mark("queue_end")
    row.mark("service_start")
    row.fields(["completion_tokens": 7, "outcome": "cancelled", "partial_work": true])
    let group = DispatchGroup()
    for _ in 0..<8 {
        group.enter()
        DispatchQueue.global().async {
            metrics.finish(row, outcome: "completed")
            group.leave()
        }
    }
    guard group.wait(timeout: .now() + 5) == .success else { throw EngineError("Lifecycle finish race did not join") }
    let snapshot = metrics.snapshot
    let recent = snapshot["recent"] as? [[String: Any]] ?? []
    try require(
        snapshot["arrivals"] as? UInt64 == 1 && snapshot["finished"] as? UInt64 == 1 && recent.count == 1, "finish is not exactly once")
    try require(recent[0]["outcome"] as? String == "cancelled" && recent[0]["completion_tokens"] as? Int == 7, "partial failure lost")
    try require(recent[0]["queue_seconds"] as? Double != nil && recent[0]["service_seconds"] as? Double != nil, "stage intervals missing")
    try require(recent[0]["request_id"] as? String == "same-request-123", "correlation changed")
    for _ in 0..<60 {
        let next = metrics.begin()
        metrics.finish(next, outcome: "ingress_rejected")
    }
    try require((metrics.snapshot["recent"] as? [[String: Any]])?.count == 50, "history is not bounded")
    let invalid = HTTPRequestLifecycle(scope: "fixture", requestID: "bad\r\nheader")
    try require(!invalid.id.contains("\r") && !invalid.id.contains("\n"), "unsafe correlation accepted")
    let control = GenerationControl()
    try require(control.progress.generatedTokens == nil, "unknown token count must stay unknown")
    control.reportGeneratedTokens(0)
    control.reportGeneratedTokens(3)
    try require(control.progress.generatedTokens == 3 && control.progress.firstGeneratedAt != nil, "token count missing")
    let firstGenerated = control.progress.firstGeneratedAt
    control.reportGeneratedTokens(0)
    try require(
        control.progress.generatedTokens == 3 && control.progress.firstGeneratedAt == firstGenerated,
        "retry erased the request's output high-water mark or first-token time")
    try httpRequestMetricsPersistenceChecks()
}

/// The first sink call stays blocked until the test releases it. No sleeps or filesystem/model work.
private final class BlockedMetricSink: @unchecked Sendable {
    let entered = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var started = false, ids: [String] = []
    func record(_ row: [String: Any]) {
        lock.lock()
        let first = !started
        started = true
        lock.unlock()
        if first {
            entered.signal()
            release.wait()
        }
        lock.lock()
        ids.append(row["request_id"] as? String ?? "missing")
        lock.unlock()
    }
    var recorded: [String] {
        lock.lock()
        defer { lock.unlock() }
        return ids
    }
}

private func httpRequestMetricsPersistenceChecks() throws {
    func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw EngineError("Request persistence: " + message) }
    }
    func diagnostics(_ metrics: HTTPRequestMetrics) -> [String: Any] { metrics.snapshot["persistence"] as? [String: Any] ?? [:] }
    func finish(_ metrics: HTTPRequestMetrics, _ id: String, fields: [String: Any] = [:]) {
        let row = metrics.begin(requestID: id)
        row.fields(fields)
        metrics.finish(row, outcome: "completed")
    }
    let sink = BlockedMetricSink()
    let metrics = HTTPRequestMetrics(
        scope: "fixture", maximumPendingRecords: 2, maximumPendingBytes: 8_192,
        maximumRecordBytes: 2_048, record: { sink.record($0) })
    defer {
        sink.release.signal()
        _ = metrics.flush(timeout: 5)
    }
    // A blocked durable sink cannot block the simulated serial network queue or the next control task on it.
    let ingress = DispatchQueue(label: "fixture.http.network")
    let returned = DispatchSemaphore(value: 0)
    let first = metrics.begin(requestID: "first")
    ingress.async { metrics.finish(first, outcome: "completed") }
    ingress.async { returned.signal() }
    try require(sink.entered.wait(timeout: .now() + 5) == .success, "writer did not enter sink")
    try require(returned.wait(timeout: .now() + 5) == .success, "blocked sink stopped network/control progress")
    finish(metrics, "second")
    finish(metrics, "third")
    finish(metrics, "overflow")
    finish(metrics, "oversize", fields: ["fixture": String(repeating: "x", count: 4_096)])
    finish(metrics, "invalid", fields: ["fixture": Double.nan])
    let blocked = diagnostics(metrics)
    try require(blocked["queued_records"] as? Int == 2 && blocked["accepted_records"] as? UInt64 == 3, "record bound is not enforced")
    try require(
        blocked["dropped_queue_full"] as? UInt64 == 1 && blocked["dropped_oversize"] as? UInt64 == 1
            && blocked["dropped_invalid_json"] as? UInt64 == 1, "drop reasons are inaccurate")
    try require(blocked["sink_calls_returned"] as? UInt64 == 0 && !metrics.flush(timeout: 0), "blocked sink falsely reported completion")
    try require(
        metrics.snapshot["finished"] as? UInt64 == 6 && (metrics.snapshot["recent"] as? [[String: Any]])?.count == 6,
        "persistence drops lost live request observations")
    try require(JSONSerialization.isValidJSONObject(metrics.snapshot), "invalid producer field poisoned live status JSON")
    let invalidRow = (metrics.snapshot["recent"] as? [[String: Any]])?.last
    try require(
        invalidRow?["fixture"] is NSNull && invalidRow?["invalid_metric_fields"] as? [String] == ["fixture"],
        "invalid metric was not explicitly marked missing")
    sink.release.signal()
    try require(metrics.flush(timeout: 5), "accepted records did not drain")
    try require(sink.recorded == ["first", "second", "third"], "accepted FIFO records changed or dropped rows were sent")
    let drained = diagnostics(metrics)
    try require(
        drained["settled_records"] as? UInt64 == 3 && drained["sink_calls_returned"] as? UInt64 == 3
            && drained["queued_json_bytes"] as? Int == 0 && drained["active_json_bytes"] as? Int == 0, "drain diagnostics did not settle")
    finish(metrics, "later")
    try require(metrics.flush(timeout: 5) && sink.recorded.last == "later", "writer did not resume after an empty burst")

    // Byte capacity is independent of record count: one ~3KB row fits, a second does not, with ten count slots.
    let byteSink = BlockedMetricSink()
    let byteMetrics = HTTPRequestMetrics(
        scope: "fixture", maximumPendingRecords: 10, maximumPendingBytes: 4_096,
        maximumRecordBytes: 4_096, record: { byteSink.record($0) })
    defer {
        byteSink.release.signal()
        _ = byteMetrics.flush(timeout: 5)
    }
    finish(byteMetrics, "active")
    try require(byteSink.entered.wait(timeout: .now() + 5) == .success, "byte fixture writer did not start")
    finish(byteMetrics, "queued", fields: ["fixture": String(repeating: "x", count: 3_000)])
    finish(byteMetrics, "byte-overflow", fields: ["fixture": String(repeating: "x", count: 3_000)])
    let byteBound = diagnostics(byteMetrics)
    try require(
        byteBound["queued_records"] as? Int == 1 && byteBound["dropped_queue_full"] as? UInt64 == 1
            && (byteBound["queued_json_bytes"] as? Int ?? Int.max) <= 4_096, "queued JSON byte bound failed")
    byteSink.release.signal()
    try require(byteMetrics.flush(timeout: 5) && byteSink.recorded == ["active", "queued"], "byte-bounded FIFO did not drain")
}
