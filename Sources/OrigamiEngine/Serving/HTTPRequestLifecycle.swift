import Foundation

/// One HTTP connection/request, including failures before a body can be parsed.
/// Durations use a monotonic clock. Front and worker scopes are separate observations
/// of a proxied request and must never be added as distinct model requests.
public final class HTTPRequestLifecycle: @unchecked Sendable {
    public let id: String
    private let scope: String, received = Date(), began = DispatchTime.now().uptimeNanoseconds
    private let lock = NSLock()
    private var values: [String: Any] = [:], marks: [String: Double] = [:]
    private var finished = false
    public init(scope: String, requestID: String? = nil) {
        self.scope = scope
        // Only propagate a bounded opaque correlation ID, never arbitrary header text.
        if let requestID, requestID.utf8.count <= 64, !requestID.isEmpty,
            requestID.utf8.allSatisfy({
                (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0) || $0 == 45 || $0 == 95
            })
        {
            id = requestID
        } else {
            id = UUID().uuidString
        }
    }
    private var elapsed: Double { Double(DispatchTime.now().uptimeNanoseconds - began) / 1_000_000_000 }
    public func mark(_ name: String) {
        lock.lock()
        defer { lock.unlock() }
        guard !finished else { return }
        if marks[name] == nil { marks[name] = elapsed }
    }
    public func fields(_ fields: [String: Any]) {
        lock.lock()
        defer { lock.unlock() }
        guard !finished else { return }
        values.merge(fields) { $1 }
    }
    /// Exactly one caller obtains a row, even when disconnect races worker cleanup.
    public func finish(outcome: String) -> [String: Any]? {
        lock.lock()
        defer { lock.unlock() }
        guard !finished else { return nil }
        finished = true
        let total = elapsed
        var row = values
        row["schema_version"] = 2
        row["event"] = "http_request_finished"
        row["scope"] = scope
        row["request_id"] = id
        row["received_at"] = ISO8601DateFormatter().string(from: received)
        row["ended_at"] = ISO8601DateFormatter().string(from: Date())
        row["total_seconds"] = total
        row["marks_seconds"] = marks
        if row["outcome"] == nil { row["outcome"] = outcome }
        for name in ["queue", "load", "service"] {
            if let start = marks[name + "_start"] {
                row[name + "_seconds"] = max(0, (marks[name + "_end"] ?? total) - start)
            }
        }
        return row
    }
}

/// Bounded live diagnostics with an optional metrics-only persistence sink. A single writer drains a bounded FIFO;
/// request/network threads never call the sink. Dropped persistence records remain counted in live diagnostics.
public final class HTTPRequestMetrics: @unchecked Sendable {
    /// Default queue bytes plus one active row's conservative Foundation inflation/serialization overlap and bookkeeping.
    /// Charge this only when a persistence sink is configured. This bounds the handoff, not the sink's own storage.
    public static let persistenceMemoryAllowance: UInt64 = 512 * 1_024 + 128 * 1_024 * 66 + 64 * 1_024
    private let scope: String, record: (@Sendable ([String: Any]) -> Void)?
    private let lock = NSCondition()
    private let writer = DispatchQueue(label: "origami.http.metrics.writer", qos: .utility, autoreleaseFrequency: .workItem)
    private let writerKey = DispatchSpecificKey<UInt8>()
    private let maximumPendingRecords, maximumPendingBytes, maximumRecordBytes: Int
    private var arrivals: UInt64 = 0, completed: UInt64 = 0
    private var outcomes: [String: UInt64] = [:], recent: [[String: Any]] = []
    // Every field below is protected by lock. Data, rather than Foundation object graphs, makes the queued byte bound explicit.
    private var pending: [Data] = [], pendingBytes = 0, activeBytes = 0, peakPendingBytes = 0
    private var writerScheduled = false
    private var accepted: UInt64 = 0, settled: UInt64 = 0, sinkCalls: UInt64 = 0, decodeFailures: UInt64 = 0
    private var droppedFull: UInt64 = 0, droppedOversize: UInt64 = 0, droppedInvalid: UInt64 = 0

    public convenience init(scope: String, record: (@Sendable ([String: Any]) -> Void)? = nil) {
        self.init(
            scope: scope, maximumPendingRecords: 32, maximumPendingBytes: 512 * 1_024,
            maximumRecordBytes: 128 * 1_024, record: record)
    }
    /// Small limits let model-free fixtures exercise saturation without allocating a production-sized backlog.
    package init(
        scope: String, maximumPendingRecords: Int, maximumPendingBytes: Int, maximumRecordBytes: Int,
        record: (@Sendable ([String: Any]) -> Void)?
    ) {
        precondition(maximumPendingRecords > 0 && maximumPendingBytes > 0 && maximumRecordBytes > 0)
        self.scope = scope
        self.record = record
        self.maximumPendingRecords = maximumPendingRecords
        self.maximumPendingBytes = maximumPendingBytes
        self.maximumRecordBytes = maximumRecordBytes
        writer.setSpecific(key: writerKey, value: 1)
    }
    public func begin(requestID: String? = nil) -> HTTPRequestLifecycle {
        lock.lock()
        arrivals &+= 1
        lock.unlock()
        return HTTPRequestLifecycle(scope: scope, requestID: requestID)
    }
    public func finish(_ lifecycle: HTTPRequestLifecycle, outcome: String) {
        guard let row = lifecycle.finish(outcome: outcome) else { return }
        // Encoding is bounded by the request's existing metadata limits and performs no filesystem access. The
        // writer reconstructs the callback's JSON object off the network queue, one row at a time.
        // Foundation can raise an Objective-C exception for NaN instead of a catchable Swift error. Validate
        // first, and keep an explicitly incomplete but JSON-safe live observation if a producer supplied one.
        let valid = JSONSerialization.isValidJSONObject(row)
        let encoded: Data? = record == nil || !valid ? nil : (try? JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]))
        var live = row
        if !valid {
            let invalid = row.keys.filter { !JSONSerialization.isValidJSONObject([$0: row[$0]!]) }.sorted()
            for key in invalid { live[key] = NSNull() }
            live["invalid_metric_fields"] = invalid
        }
        lock.lock()
        completed &+= 1
        let name = row["outcome"] as? String ?? outcome
        outcomes[name, default: 0] &+= 1
        recent.append(live)
        if recent.count > 50 { recent.removeFirst(recent.count - 50) }
        var schedule = false
        if record != nil {
            if let encoded {
                if encoded.count > maximumRecordBytes {
                    droppedOversize &+= 1
                } else if pending.count >= maximumPendingRecords || encoded.count > maximumPendingBytes - pendingBytes {
                    droppedFull &+= 1
                } else {
                    pending.append(encoded)
                    pendingBytes += encoded.count
                    accepted &+= 1
                    peakPendingBytes = max(peakPendingBytes, pendingBytes)
                    if !writerScheduled {
                        writerScheduled = true
                        schedule = true
                    }
                }
            } else {
                droppedInvalid &+= 1
            }
        }
        lock.unlock()
        // At most one drain block is scheduled per nonempty burst, never one task per row.
        if schedule { writer.async { [self] in drain() } }
    }
    private func drain() {
        while true {
            lock.lock()
            guard !pending.isEmpty else {
                writerScheduled = false
                lock.broadcast()
                lock.unlock()
                return
            }
            let data = pending.removeFirst()
            pendingBytes -= data.count
            activeBytes = data.count
            lock.unlock()
            let decoded = autoreleasepool {
                guard let row = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return false }
                record?(row)
                return true
            }
            lock.lock()
            activeBytes = 0
            settled &+= 1
            if decoded { sinkCalls &+= 1 } else { decodeFailures &+= 1 }
            lock.broadcast()
            lock.unlock()
        }
    }
    /// Waits for sink calls accepted before this call to settle, including decode failures; consult diagnostics for
    /// drops/failures. Call only from shutdown/test code, never a network callback. A sink calling this returns false
    /// instead of deadlocking. A true result means calls returned, not that a Void-returning sink wrote to disk.
    @discardableResult public func flush(timeout: TimeInterval = 5) -> Bool {
        guard DispatchQueue.getSpecific(key: writerKey) == nil else { return false }
        lock.lock()
        defer { lock.unlock() }
        let target = accepted
        if settled >= target { return true }
        guard timeout.isFinite && timeout > 0 else { return false }
        let deadline = Date(timeIntervalSinceNow: timeout)
        while settled < target {
            if !lock.wait(until: deadline) { return settled >= target }
        }
        return true
    }
    public var snapshot: [String: Any] {
        lock.lock()
        defer { lock.unlock() }
        return [
            "scope": scope, "unit": "http_connections", "arrivals": arrivals, "finished": completed,
            "in_flight": arrivals - completed, "outcomes": outcomes, "recent": recent,
            // Legacy flag describes configuration only; persistence diagnostics do not claim a verified disk write.
            "durable": record != nil,
            "persistence": [
                "configured": record != nil, "policy": "bounded_fifo_drop_newest", "queued_records": pending.count,
                "queued_json_bytes": pendingBytes, "active_json_bytes": activeBytes, "peak_queued_json_bytes": peakPendingBytes,
                "maximum_queued_records": maximumPendingRecords, "maximum_queued_json_bytes": maximumPendingBytes,
                "maximum_record_json_bytes": maximumRecordBytes,
                "accepted_records": accepted, "settled_records": settled, "sink_calls_returned": sinkCalls,
                "decode_failures": decodeFailures, "dropped_queue_full": droppedFull,
                "dropped_oversize": droppedOversize, "dropped_invalid_json": droppedInvalid,
                "sink_write_confirmation": "unavailable_void_callback",
                "default_handoff_memory_allowance_bytes": record == nil ? 0 : Self.persistenceMemoryAllowance,
            ],
        ]
    }
}
