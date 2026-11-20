import Darwin
import Foundation
import OrigamiEngine

func httpMetricsShutdownChecks() -> [(String, () throws -> Void)] {
    [
        ("http_metrics_blocked_sink_keeps_health_and_shutdown_bounded", httpMetricsBlockedSinkKeepsHealth),
        ("http_metrics_shutdown_finalizes_ingress_before_flush", httpMetricsShutdownFinalizesIngress),
    ]
}

private func metricHTTPRequire(_ value: Bool, _ message: String) throws {
    guard value else { throw EngineError("HTTP metrics shutdown: " + message) }
}
private struct MetricsOnlyBackend: ServingBackend {
    func status() -> ServingStatus { ServingStatus(version: "metrics-fixture", ready: true) }
    func prepare(_ request: ServingRequest, cancellation: GenerationControl) throws -> any PreparedGeneration {
        throw EngineError("The metrics fixture must never execute inference")
    }
}
private final class HTTPMetricSink: @unchecked Sendable {
    let entered = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
    private let lock = NSLock(), blocks: Bool
    private var started = false, rows: [[String: Any]] = []
    init(blocks: Bool) { self.blocks = blocks }
    func record(_ row: [String: Any]) {
        lock.lock()
        let first = !started
        started = true
        lock.unlock()
        if first && blocks {
            entered.signal()
            release.wait()
        }
        lock.lock()
        rows.append(row)
        lock.unlock()
    }
    var recorded: [[String: Any]] {
        lock.lock()
        defer { lock.unlock() }
        return rows
    }
}
private func metricsServer(_ sink: HTTPMetricSink) throws -> LocalHTTPServer {
    var configuration = LocalHTTPConfiguration()
    configuration.port = 0
    configuration.recordRequest = { row, _ in sink.record(row) }
    let server = try LocalHTTPServer(
        configuration: configuration, backend: MetricsOnlyBackend(),
        ledger: MemoryLedger(ceiling: 1_000_000_000, transientMargin: 1_000_000))
    try server.start()
    return server
}
private func metricsConnect(_ port: UInt16) throws -> Int32 {
    let descriptor = socket(AF_INET, SOCK_STREAM, 0)
    guard descriptor >= 0 else { throw EngineError("metrics fixture socket failed") }
    var timeout = timeval(tv_sec: 2, tv_usec: 0)
    var one: Int32 = 1
    setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
    setsockopt(descriptor, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
    setsockopt(descriptor, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))
    var address = sockaddr_in()
    address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    address.sin_family = sa_family_t(AF_INET)
    address.sin_port = port.bigEndian
    address.sin_addr.s_addr = inet_addr("127.0.0.1")
    let result = withUnsafePointer(to: address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
    }
    guard result == 0 else {
        close(descriptor)
        throw EngineError("metrics fixture connect failed")
    }
    return descriptor
}
private func metricsSend(_ descriptor: Int32, _ text: String) throws {
    try Data(text.utf8).withUnsafeBytes { bytes in
        var at = 0
        while at < bytes.count {
            let count = write(descriptor, bytes.baseAddress! + at, bytes.count - at)
            if count < 0 && errno == EINTR { continue }
            guard count > 0 else { throw EngineError("metrics fixture send failed") }
            at += count
        }
    }
}
private func metricsGET(_ port: UInt16, path: String) throws -> Data {
    let descriptor = try metricsConnect(port)
    defer { close(descriptor) }
    try metricsSend(descriptor, "GET \(path) HTTP/1.1\r\nHost: 127.0.0.1:\(port)\r\n\r\n")
    var data = Data()
    var buffer = [UInt8](repeating: 0, count: 4096)
    while true {
        let count = read(descriptor, &buffer, buffer.count)
        if count < 0 && errno == EINTR { continue }
        guard count >= 0 else { throw EngineError("metrics fixture response timed out or failed") }
        if count == 0 { break }
        data.append(contentsOf: buffer[..<count])
    }
    guard String(decoding: data.prefix(32), as: UTF8.self).hasPrefix("HTTP/1.1 200 OK"),
        let split = data.range(of: Data("\r\n\r\n".utf8))
    else { throw EngineError("metrics fixture did not receive HTTP 200") }
    return data.subdata(in: split.upperBound..<data.endIndex)
}

/// The first finish originates on the actual serial Network.framework queue, before worker dispatch.
private func httpMetricsBlockedSinkKeepsHealth() throws {
    let sink = HTTPMetricSink(blocks: true)
    let server = try metricsServer(sink)
    defer {
        sink.release.signal()
        _ = server.stopAndJoin(timeout: 5)
    }
    let descriptor = try metricsConnect(server.boundPort!)
    try metricsSend(descriptor, "GET /health HTTP/1.1\r\nHost:")
    _ = shutdown(descriptor, SHUT_RDWR)
    close(descriptor)
    try metricHTTPRequire(sink.entered.wait(timeout: .now() + 5) == .success, "ingress finish never reached sink")
    _ = try metricsGET(server.boundPort!, path: "/health")
    let start = DispatchTime.now().uptimeNanoseconds
    let joined = server.stopAndJoin(timeout: 0.1)
    let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
    try metricHTTPRequire(!joined && elapsed < 1, "blocked writer exceeded the single shutdown deadline")
    try metricHTTPRequire(sink.recorded.isEmpty, "blocked sink incorrectly reported a returned call")
    sink.release.signal()
    try metricHTTPRequire(server.stopAndJoin(timeout: 5), "unblocked writer failed to drain on retry")
    let rows = sink.recorded
    try metricHTTPRequire(rows.count == 2 && rows.contains { $0["path"] as? String == "/health" }, "shutdown lost ingress/health rows")
    try metricHTTPRequire(Set(rows.compactMap { $0["request_id"] as? String }).count == 2, "shutdown duplicated lifecycle rows")
}

private func httpMetricsShutdownFinalizesIngress() throws {
    let sink = HTTPMetricSink(blocks: false)
    let server = try metricsServer(sink)
    defer { _ = server.stopAndJoin(timeout: 5) }
    let descriptor = try metricsConnect(server.boundPort!)
    defer { close(descriptor) }
    try metricsSend(descriptor, "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100\r\n\r\n")
    var arrivals = 0
    var ingressObserved = false
    for _ in 0..<10 {
        let data = try metricsGET(server.boundPort!, path: "/v1/status")
        let report = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let requests = report["requests"] as? [String: Any] ?? [:]
        let http = requests["http"] as? [String: Any] ?? [:]
        arrivals = (http["arrivals"] as? NSNumber)?.intValue ?? 0
        if ((http["in_flight"] as? NSNumber)?.intValue ?? 0) >= 2 {
            ingressObserved = true
            break
        }
    }
    try metricHTTPRequire(ingressObserved, "unfinished ingress was never admitted")
    try metricHTTPRequire(server.stopAndJoin(timeout: 5), "shutdown failed to finalize and flush admitted ingress")
    try metricHTTPRequire(sink.recorded.count == arrivals, "shutdown returned before all observed producers reached persistence")
}
