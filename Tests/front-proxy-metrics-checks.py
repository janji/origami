#!/usr/bin/env python3
"""Compile the actual ModelFront.pipe body with deterministic transport stubs.

The production lifecycle implementation is compiled unchanged. No listener, model,
GPU, server process, configuration mutation or real network connection is used.
Run only when the shared Swift build slot is free.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

HARNESS = r'''
import Darwin
import Foundation

struct ServingFailure {
    let status: Int, code: String
    init(_ message: String, status: Int, type: String = "failure", code: String? = nil) { self.status = status; self.code = code ?? type }
}
enum WireFormat { case openAI }
struct Failure: Error { let message: String }
// The relay's own behaviour (bound, stall, leave) is engine-checks' http_relay_*; here it copies the fixture's reads.
enum BoundedHTTPRelay {
    enum Outcome: Equatable { case finished, clientLeft, clientStalled, cancelled }
    static func relay(upstream: Int32, client: Int32, endOfInputLeaves: Bool, cancelled: () -> Bool, forwarded: () -> Void) -> Outcome {
        var buffer = [UInt8](repeating: 0, count: 64)
        while true {
            if cancelled() { return .cancelled }
            guard ProxyFixture.read(upstream, &buffer, buffer.count) > 0 else { return .finished }
            forwarded()
        }
    }
}
enum ProxyFixture {
    static let stoppedFailure = ServingFailure("stopped", status: 503, type: "server_unavailable")
    static var connectOK = true, requestSendOK = true, reads = 0
    static var responseStatus: Int?
    static func connect(port: UInt16) -> Int32 { connectOK ? 3 : -1 }
    static func send(_ fd: Int32, _ data: Data) -> Bool { fd != 3 || requestSendOK }
    static func reply(_ fd: Int32, _ failure: ServingFailure) { responseStatus = failure.status }
    static func close(_ fd: Int32) {}
    // A stream whose headers the front sent (`streaming`) is the dashboard control checks' scenario, not this one's.
    static func left(_ client: Int32, endOfInputLeaves: Bool = true) -> Bool { false }
    static func streamError(_ client: Int32, _ failure: ServingFailure, format: WireFormat, lifecycle: HTTPRequestLifecycle) {}
    static func poll(_ fds: UnsafeMutablePointer<pollfd>, _ count: nfds_t, _ timeout: Int32) -> Int32 {
        fds[0].revents = Int16(POLLIN); fds[1].revents = 0; return 1
    }
    static func read(_ fd: Int32, _ buffer: UnsafeMutableRawPointer, _ count: Int) -> Int {
        reads += 1
        guard reads == 1 else { return 0 }
        let bytes = Array("HTTP/1.1 200 OK\r\n\r\n".utf8)
        bytes.withUnsafeBytes { buffer.copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return bytes.count
    }
    /* PRODUCTION_PIPE */
    static func reset() { connectOK = true; requestSendOK = true; reads = 0; responseStatus = nil }
    static func run(_ name: String) throws -> [String: Any] {
        reset()
        if name == "connect_failure" { connectOK = false }
        if name == "request_send_failure" { requestSendOK = false }
        var calls = 0
        let lifecycle = HTTPRequestLifecycle(scope: "front")
        pipe(Data(), client: 4, port: 12345, cancelled: {
            calls += 1
            return name == "cancel_before_response" || (name == "cancel_after_response" && calls > 1)
        }, lifecycle: lifecycle)
        let row = lifecycle.finish(outcome: "closed")!
        let expected = name == "cancel_before_response" ? 503 : name.contains("failure") ? 502 : nil
        guard responseStatus == expected, row["http_status"] as? Int == expected else {
            throw Failure(message: "local wire status and recorded status disagree: " + name)
        }
        let outcome = name.hasPrefix("cancel_") ? "cancelled_proxy" : name.contains("failure") ? "upstream_unavailable" : "proxied_response"
        guard row["outcome"] as? String == outcome,
              let marks = row["marks_seconds"] as? [String: Double], marks["service_start"] != nil, marks["service_end"] != nil else {
            throw Failure(message: "outcome/service boundaries missing: " + name)
        }
        return ["scenario": name, "passed": true]
    }
}
@main enum Main {
    static func main() throws {
        let results = try ["connect_failure", "request_send_failure", "cancel_before_response", "cancel_after_response", "forwarded_response"].map(ProxyFixture.run)
        let encoded = try JSONSerialization.data(withJSONObject: ["checks": results], options: [.sortedKeys])
        print(String(decoding: encoded, as: UTF8.self))
    }
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    source_path = root / "Sources/origami/ModelFront.swift"
    lifecycle_path = root / "Sources/OrigamiEngine/Serving/HTTPRequestLifecycle.swift"
    source = source_path.read_text()
    start = source.index("    private static func pipe(")
    end = source.index("\n    }\n", start) + len("\n    }\n")
    pipe = source[start:end].replace("private static func pipe", "static func pipe", 1)
    with tempfile.TemporaryDirectory(prefix="origami-front-metrics-") as temporary:
        directory = Path(temporary)
        fixture = directory / "Fixture.swift"
        fixture.write_text(HARNESS.replace("    /* PRODUCTION_PIPE */", pipe))
        binary = directory / "fixture"
        subprocess.run(["swiftc", "-package-name", "Origami", str(lifecycle_path), str(fixture), "-o", str(binary)], check=True)
        result = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    report["production_pipe_sha256"] = hashlib.sha256(source[start:end].encode()).hexdigest()
    report["lifecycle_sha256"] = hashlib.sha256(lifecycle_path.read_bytes()).hexdigest()
    report["scope"] = "Production pipe body and lifecycle; deterministic transport stubs; no real network/model"
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
