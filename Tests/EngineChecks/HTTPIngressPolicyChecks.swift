import Darwin
import Foundation
import OrigamiEngine

func httpIngressPolicyChecks() -> [(String, () throws -> Void)] {
    [
        ("http_control_slots_reserved", httpControlSlotsReserved),
        ("http_ingress_absolute_deadline", httpIngressAbsoluteDeadline),
        ("http_front_json_budget", httpFrontJSONBudget),
        ("http_front_status_and_correlation", httpFrontStatusAndCorrelation),
        ("http_ingress_phase_deadlines_and_rate", httpIngressPhaseDeadlinesAndRate),
        ("http_ingress_peer_caps_and_header_bytes", httpIngressPeerCapsAndHeaderBytes),
        ("http_body_fairness_drops_late_upload", httpBodyFairnessDropsLateUpload),
        ("http_relay_bounded_writer", httpRelayBoundedWriter),
        ("http_auth_limiter_and_key_compare", httpAuthLimiterAndKeyCompare),
        ("http_forwarded_peer", httpForwardedPeer),
    ]
}

private func elapsed(since start: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9 }

private func writeAll(_ fd: Int32, _ text: String) {
    _ = Data(text.utf8).withUnsafeBytes { write(fd, $0.baseAddress, $0.count) }
}

/// The request line and headers get their own deadline; a body must keep a minimum rate; an honest body slower than
/// the window but above the rate, and one smaller than the rate, are served.
private func httpIngressPhaseDeadlinesAndRate() throws {
    func outcome(_ prefix: String, every: useconds_t, piece: Int, count: Int, _ receive: (Int32) throws -> BoundedHTTPRequest?) throws
        -> (ServingFailure?, Double, Int)
    {
        let pair = try socketPair()
        let done = DispatchSemaphore(value: 0)
        defer {
            close(pair[0])
            close(pair[1])
        }
        writeAll(pair[1], prefix)
        Thread.detachNewThread {
            defer { done.signal() }
            let bytes = [UInt8](repeating: 97, count: piece)
            for _ in 0..<count {
                usleep(every)
                if write(pair[1], bytes, piece) != piece { return }
            }
        }
        let began = DispatchTime.now().uptimeNanoseconds
        var failure: ServingFailure?
        var body = -1
        do { body = try receive(pair[0])?.body.count ?? -1 } catch let error as ServingFailure { failure = error }
        let took = elapsed(since: began)
        _ = shutdown(pair[1], SHUT_WR)
        _ = done.wait(timeout: .now() + 2)
        return (failure, took, body)
    }
    // A header that trickles stops at the header deadline, well before the whole request's.
    var (failure, took, _) = try outcome("GET /health HTTP/1.1\r\n", every: 20_000, piece: 1, count: 100) {
        try BoundedHTTPIngress.receive($0, timeout: 5, headerTimeout: 0.2)
    }
    try ingressRequire(failure?.status == 408 && took >= 0.18 && took < 1, "trickled headers not cut at their deadline: \(took)")
    // Nothing sent at all: the same.
    (failure, took, _) = try outcome("", every: 1_000, piece: 0, count: 0) {
        try BoundedHTTPIngress.receive($0, timeout: 5, headerTimeout: 0.2)
    }
    try ingressRequire(failure?.status == 408 && took < 1, "a silent connection was not cut at the header deadline: \(took)")
    // A body under the rate is cut after one window, long before the total deadline.
    let head = "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100000\r\n\r\n"
    var progressed = 0
    (failure, took, _) = try outcome(head, every: 20_000, piece: 10, count: 100) {
        try BoundedHTTPIngress.receive($0, timeout: 5, minimumRate: 4_096, rateWindow: 0.25, progress: { progressed += $0 })
    }
    try ingressRequire(
        failure?.status == 408 && failure?.code == "request_timeout" && took >= 0.24 && took < 1 && progressed > 0,
        "a trickled body was not cut by the rate floor: \(took) s, \(progressed) bytes")
    // A body above the rate over several windows is served in full (sleeps run long on a busy Mac: pieces are large).
    (failure, took, progressed) = try outcome(head, every: 1_000, piece: 5_000, count: 20) {
        try BoundedHTTPIngress.receive($0, timeout: 5, minimumRate: 4_096, rateWindow: 0.25)
    }
    try ingressRequire(failure == nil && progressed == 100_000, "an upload above the rate was refused: \(String(describing: failure))")
    // A small body smaller than the rate needs only its own bytes in a window.
    (failure, _, progressed) = try outcome(
        "POST /x HTTP/1.1\r\nHost: localhost\r\nContent-Length: 10\r\n\r\n", every: 1_000, piece: 1, count: 10
    ) { try BoundedHTTPIngress.receive($0, timeout: 5, minimumRate: 4_096, rateWindow: 2) }
    try ingressRequire(failure == nil && progressed == 10, "a small slow body was refused: \(String(describing: failure))")
}

/// Another device's unclassified connections are capped per address and in all; this Mac's are not; every connection's
/// header bytes are counted and stay within the stated maximum.
private func httpIngressPeerCapsAndHeaderBytes() throws {
    let budget = HTTPConnectionBudget(maximumRequests: 64, remoteIngress: 6, ingressPerPeer: 4)
    var ids: [Int] = []
    for _ in 0..<4 { ids.append(budget.accept(peer: "10.0.0.2")!) }
    try ingressRequire(budget.accept(peer: "10.0.0.2") == nil, "one address took more than its unclassified share")
    for _ in 0..<2 { ids.append(budget.accept(peer: "10.0.0.3")!) }
    try ingressRequire(budget.accept(peer: "10.0.0.4") == nil, "remote unclassified total not capped")
    for _ in 0..<20 { ids.append(budget.accept()!) }
    try ingressRequire(budget.classify(ids[0], control: false), "classification failed")
    ids.append(budget.accept(peer: "10.0.0.2")!)  // a classified one no longer counts as waiting
    let snapshot = budget.snapshot
    let headers = snapshot["header_reservation_bytes"] as? Int ?? -1
    try ingressRequire(
        headers == ids.count * budget.headerBytes && snapshot["ingress"] as? Int == ids.count - 1
            && headers + (snapshot["buffered_body_reservation_bytes"] as? Int ?? 0) <= snapshot["maximum_reserved_bytes"] as? Int ?? 0,
        "header bytes not counted: \(snapshot)")
    for id in ids { budget.release(id) }
    try ingressRequire(budget.accept(peer: "10.0.0.4") != nil, "released slots not reusable")
}

/// A large-body holder behind the pace that would finish its body in time loses its reservation to a request that
/// needs the room; one on pace never does, and holders without a drop hook (the model's server) are never chosen.
private func httpBodyFairnessDropsLateUpload() throws {
    final class Drops: @unchecked Sendable {
        var ids: [String] = []
    }
    let drops = Drops()
    let budget = HTTPConnectionBudget(maximumRequests: 8, maximumBodyBytes: 100, bodyDeadline: 2)
    let late = budget.accept(peer: "10.0.0.2", drop: { drops.ids.append("late") })!
    let steady = budget.accept(peer: "10.0.0.3", drop: { drops.ids.append("steady") })!
    let hookless = budget.accept()!
    try ingressRequire(budget.reserveBody(late, bytes: 40) && budget.reserveBody(steady, bytes: 40), "reservations refused")
    try ingressRequire(budget.reserveBody(hookless, bytes: 20), "hookless reservation refused")
    let newcomer = budget.accept()!
    try ingressRequire(!budget.reserveBody(newcomer, bytes: 40), "a fresh holder was dropped before its grace second")
    usleep(1_100_000)
    budget.received(late, bytes: 1)
    budget.received(steady, bytes: 39)  // on pace: 39 of 40 in 1.1 of 2 s
    try ingressRequire(budget.reserveBody(newcomer, bytes: 40), "a late upload kept its reservation from a waiting request")
    try ingressRequire(
        drops.ids == ["late"] && budget.evicted(late) && !budget.evicted(steady) && !budget.evicted(hookless),
        "wrong holder dropped: \(drops.ids)")
    let another = budget.accept()!
    try ingressRequire(!budget.reserveBody(another, bytes: 30), "an on-pace or hookless holder was dropped")
    try ingressRequire(drops.ids == ["late"], "an on-pace holder was dropped")
    for id in [late, steady, hookless, newcomer, another] { budget.release(id) }
}

/// One relay case's sockets (the model's side and the client's, each a connected pair) and the threads that use them.
/// `finish` shuts every socket down, which releases any call blocked on one (a close from another thread would not),
/// waits for the threads, then closes; every case runs it on every exit path, so a failure never leaves a thread behind.
private final class RelayRig: @unchecked Sendable {
    let upstream: [Int32], client: [Int32]
    private let threads = DispatchGroup()
    private let lock = NSLock()
    private var closed: Set<Int32> = []
    init(tcp: Bool = false) throws {
        upstream = try tcp ? tcpPair() : socketPair()
        client = try tcp ? tcpPair() : socketPair()
    }
    func spawn(_ body: @escaping @Sendable () -> Void) {
        threads.enter()
        Thread.detachNewThread {
            defer { self.threads.leave() }
            body()
        }
    }
    func close(_ fd: Int32) {
        if lock.withLock({ closed.insert(fd).inserted }) { _ = Darwin.close(fd) }
    }
    /// The relay on a thread of its own, with a hard limit: past it the check fails instead of hanging (`finish` then
    /// releases the relay).
    func relay(
        limit: TimeInterval, bound: Int, stall: TimeInterval, endOfInputLeaves: Bool = true,
        cancelled: @escaping @Sendable () -> Bool = { false }
    ) throws -> (
        BoundedHTTPRelay.Outcome, TimeInterval
    ) {
        final class Result: @unchecked Sendable { var outcome = BoundedHTTPRelay.Outcome.finished }
        let result = Result()
        let done = DispatchSemaphore(value: 0)
        let began = DispatchTime.now().uptimeNanoseconds
        spawn { [upstream, client] in
            result.outcome = BoundedHTTPRelay.relay(
                upstream: upstream[0], client: client[0], bound: bound, stall: stall, endOfInputLeaves: endOfInputLeaves,
                cancelled: cancelled)
            done.signal()
        }
        guard done.wait(timeout: .now() + limit) == .success else {
            throw EngineError("HTTP ingress: the relay was still in a call after \(limit) s (a send or read on its path blocked)")
        }
        return (result.outcome, elapsed(since: began))
    }
    func finish() throws {
        let open = lock.withLock { (upstream + client).filter { !closed.contains($0) } }
        for fd in open { _ = shutdown(fd, SHUT_RDWR) }
        guard threads.wait(timeout: .now() + 5) == .success else { throw EngineError("HTTP ingress: a relay check thread did not return") }
        for fd in open { close(fd) }
    }
}

/// The relay keeps reading the model's side while a client does not read, up to its bound; drops a client that takes
/// nothing for the stall time, also when it has room for less than what waits (no call blocks); passes a whole reply
/// to a client that reads; and notices a client that leaves (a reset; a full close once a byte is sent to it), while a
/// client that only shut its sending side after its request gets the whole reply.
private func httpRelayBoundedWriter() throws {
    func run(_ rig: RelayRig, _ body: (RelayRig) throws -> Void) throws {
        do { try body(rig) } catch {
            try? rig.finish()
            throw error
        }
        try rig.finish()
    }
    // A full reply, larger than the bound, to a client that reads.
    try run(try RelayRig()) { rig in
        let total = 3 << 20
        rig.spawn {
            let bytes = [UInt8](repeating: 120, count: 32 * 1_024)
            var sent = 0
            while sent < total {
                let count = write(rig.upstream[1], bytes, min(bytes.count, total - sent))
                guard count > 0 else { return }
                sent += count
            }
            _ = shutdown(rig.upstream[1], SHUT_WR)
        }
        final class Count: @unchecked Sendable { var bytes = 0 }
        let received = Count()
        let read = DispatchSemaphore(value: 0)
        rig.spawn {
            var buffer = [UInt8](repeating: 0, count: 65_536)
            while true {
                let count = Darwin.read(rig.client[1], &buffer, buffer.count)
                if count <= 0 { break }
                received.bytes += count
            }
            read.signal()
        }
        let (outcome, _) = try rig.relay(limit: 20, bound: 64 * 1_024, stall: 10)
        _ = shutdown(rig.client[0], SHUT_WR)  // the end of the reply: the reader sees EOF
        try ingressRequire(read.wait(timeout: .now() + 20) == .success, "reader did not finish")
        try ingressRequire(outcome == .finished && received.bytes == total, "relay lost bytes: \(outcome), \(received.bytes)")
    }
    // A client that never reads: the model's side keeps being read up to the bound, then the client is dropped.
    try run(try RelayRig()) { rig in
        final class Count: @unchecked Sendable { var bytes = 0, failed = false }
        let written = Count()
        let writer = DispatchSemaphore(value: 0)
        rig.spawn {
            defer { writer.signal() }
            let bytes = [UInt8](repeating: 121, count: 1_024)
            while true {
                let count = write(rig.upstream[1], bytes, bytes.count)
                if count <= 0 {
                    written.failed = true
                    return
                }
                written.bytes += count
            }
        }
        let (outcome, took) = try rig.relay(limit: 10, bound: 256 * 1_024, stall: 0.3)
        rig.close(rig.upstream[0])  // the front closes the model's connection: that cancels the request there
        try ingressRequire(writer.wait(timeout: .now() + 2) == .success && written.failed, "the model's side did not see the close")
        let drained = written.bytes
        try ingressRequire(outcome == .clientStalled && took >= 0.3 && took < 2, "stalled client not dropped in time: \(outcome) \(took)")
        try ingressRequire(drained >= 256 * 1_024, "the model's side was not read up to the bound: \(drained)")
    }
    // A client with room for less than what waits (it read a little, then stopped), on a local socket and on TCP (the
    // front's clients): a send must not wait for the rest. Darwin's send waits for room despite MSG_DONTWAIT.
    for tcp in [false, true] {
        try run(try RelayRig(tcp: tcp)) { rig in
            /// Writes until `fd` takes no more (TCP: none for 100 ms, as bytes move on to the far end).
            func fill(_ fd: Int32) -> Int {
                let flags = fcntl(fd, F_GETFL)
                _ = fcntl(fd, F_SETFL, flags | O_NONBLOCK)
                defer { _ = fcntl(fd, F_SETFL, flags) }
                let bytes = [UInt8](repeating: 122, count: 4_096)
                var total = 0
                while true {
                    let count = write(fd, bytes, bytes.count)
                    if count > 0 {
                        total += count
                        continue
                    }
                    var ready = pollfd(fd: fd, events: Int16(POLLOUT), revents: 0)
                    guard errno == EAGAIN, tcp, poll(&ready, 1, 100) == 1 else { return total }
                }
            }
            // Fixed TCP buffers: TCP would otherwise grow them as the far end reads, making room for everything.
            var size: Int32 = 64 * 1_024
            if tcp {
                _ = setsockopt(rig.client[0], SOL_SOCKET, SO_SNDBUF, &size, socklen_t(MemoryLayout<Int32>.size))
                _ = setsockopt(rig.client[1], SOL_SOCKET, SO_RCVBUF, &size, socklen_t(MemoryLayout<Int32>.size))
            }
            let full = fill(rig.client[0])
            var buffer = [UInt8](repeating: 0, count: full / 2)
            var taken = 0
            while taken < buffer.count {
                let count = buffer.withUnsafeMutableBytes { Darwin.read(rig.client[1], $0.baseAddress! + taken, $0.count - taken) }
                guard count > 0 else { break }
                taken += count
            }
            let waiting = fill(rig.upstream[1])
            _ = shutdown(rig.upstream[1], SHUT_WR)
            var ready = pollfd(fd: rig.client[0], events: Int16(POLLOUT), revents: 0)
            try ingressRequire(
                taken == full / 2 && waiting > taken && poll(&ready, 1, 2_000) == 1,
                "setup: room \(taken) of \(full), \(waiting) waiting (tcp \(tcp))")
            let (outcome, took) = try rig.relay(limit: 5, bound: 1 << 20, stall: 0.3)
            try ingressRequire(
                outcome == .clientStalled && took < 2, "partly full client not dropped in time (tcp \(tcp)): \(outcome) \(took)")
        }
    }
    // A client that leaves (a reset, as a killed client's socket gives), and a cancel.
    for leave in [true, false] {
        try run(try RelayRig(tcp: true)) { rig in
            if leave {
                var reset = linger(l_onoff: 1, l_linger: 0)
                _ = setsockopt(rig.client[1], SOL_SOCKET, SO_LINGER, &reset, socklen_t(MemoryLayout<linger>.size))
                rig.close(rig.client[1])
            }
            let (outcome, took) = try rig.relay(limit: 5, bound: 1 << 20, stall: 30, cancelled: { !leave })
            try ingressRequire(outcome == (leave ? .clientLeft : .cancelled) && took < 1, "leave/cancel not noticed: \(outcome)")
        }
    }
    // A stream (`endOfInputLeaves` false): a client that shuts its sending side after the request (as some HTTP clients do)
    // reads the whole reply, sent in two parts; one that closed entirely is dropped once the first part reaches it.
    for halfClose in [true, false] {
        try run(try RelayRig(tcp: true)) { rig in
            final class Count: @unchecked Sendable { var bytes = 0 }
            let received = Count()
            let read = DispatchSemaphore(value: 0)
            if halfClose {
                _ = shutdown(rig.client[1], SHUT_WR)
                rig.spawn {
                    var buffer = [UInt8](repeating: 0, count: 4_096)
                    while case let count = Darwin.read(rig.client[1], &buffer, buffer.count), count > 0 { received.bytes += count }
                    read.signal()
                }
            } else {
                rig.close(rig.client[1])
            }
            rig.spawn {
                writeAll(rig.upstream[1], "HTTP/1.1 200 OK\r\n\r\nfirst ")
                usleep(200_000)
                writeAll(rig.upstream[1], "second")
                usleep(200_000)
                writeAll(rig.upstream[1], " third")
                _ = shutdown(rig.upstream[1], SHUT_WR)
            }
            let (outcome, _) = try rig.relay(limit: 5, bound: 1 << 20, stall: 30, endOfInputLeaves: false)
            if halfClose {
                _ = shutdown(rig.client[0], SHUT_WR)
                try ingressRequire(read.wait(timeout: .now() + 5) == .success, "half-closed reader did not finish")
                try ingressRequire(
                    outcome == .finished && received.bytes == 37, "half-closed client lost its reply: \(outcome) \(received.bytes)")
            } else {
                try ingressRequire(outcome == .clientLeft, "a closed client was not noticed once written to: \(outcome)")
            }
        }
    }
    // A reply sent only once complete (the default): a client that closed, or half-closed, before any byte of it is gone at
    // once, so the model is not kept generating for nobody.
    for halfClose in [true, false] {
        try run(try RelayRig(tcp: true)) { rig in
            if halfClose { _ = shutdown(rig.client[1], SHUT_WR) } else { rig.close(rig.client[1]) }
            let (outcome, took) = try rig.relay(limit: 5, bound: 1 << 20, stall: 30)
            try ingressRequire(outcome == .clientLeft && took < 1, "an ended non-stream client was not dropped at once: \(outcome) \(took)")
        }
    }
}

/// Ten wrong keys from one address in a minute lock it out, not others; the key comparison accepts exactly the key, in
/// either header, and compares digests (the time does not depend on where a wrong key differs).
private func httpAuthLimiterAndKeyCompare() throws {
    let limiter = HTTPAuthLimiter(failures: 3, window: 0.4, maximumPeers: 2)
    for _ in 0..<3 {
        try ingressRequire(limiter.retryAfter("10.0.0.2") == nil, "locked out early")
        limiter.failed("10.0.0.2")
    }
    try ingressRequire((limiter.retryAfter("10.0.0.2") ?? 0) >= 1, "not locked out after the limit")
    try ingressRequire(limiter.retryAfter("10.0.0.3") == nil, "another address was locked out")
    limiter.failed("10.0.0.3")
    for peer in ["a", "b", "c"] { limiter.failed(peer) }  // past maximumPeers: they share one count
    try ingressRequire(limiter.retryAfter("d") != nil && limiter.retryAfter("10.0.0.3") == nil, "overflow peers not bounded")
    usleep(450_000)
    try ingressRequire(limiter.retryAfter("10.0.0.2") == nil, "lockout outlived its window")
    let key = "0123456789abcdef0123"
    try ingressRequire(
        LocalHTTPServer.carriesKey(key, headers: ["authorization": "Bearer " + key])
            && LocalHTTPServer.carriesKey(key, headers: ["x-api-key": key])
            && !LocalHTTPServer.carriesKey(key, headers: ["authorization": "Bearer " + key + "x"])
            && !LocalHTTPServer.carriesKey(key, headers: ["x-api-key": String(key.dropLast()) + "4"])
            && !LocalHTTPServer.carriesKey(key, headers: [:]), "key comparison wrong")
    // Equal work for a wrong key that differs at the first byte or the last: medians of interleaved timings agree.
    var first: [UInt64] = []
    var last: [UInt64] = []
    let early = ["x-api-key": "X" + String(key.dropFirst())]
    let late = ["x-api-key": String(key.dropLast()) + "X"]
    for _ in 0..<2_000 {
        for (headers, into) in [(early, 0), (late, 1)] {
            let start = DispatchTime.now().uptimeNanoseconds
            _ = LocalHTTPServer.carriesKey(key, headers: headers)
            let took = DispatchTime.now().uptimeNanoseconds - start
            if into == 0 { first.append(took) } else { last.append(took) }
        }
    }
    let a = Double(first.sorted()[1_000])
    let b = Double(last.sorted()[1_000])
    try ingressRequire(abs(a - b) < max(a, b) * 0.5 + 2_000, "key comparison time depends on the differing byte: \(a) vs \(b) ns")
}

/// The client a trusted proxy names: the last X-Forwarded-For entry, canonical; nothing without the header; a 400 for
/// a malformed one.
private func httpForwardedPeer() throws {
    func peer(_ value: String?) throws -> String? { try HTTPIngressPolicy.forwardedPeer(value.map { ["x-forwarded-for": $0] } ?? [:]) }
    try ingressRequire(try peer(nil) == nil, "a peer without the header")
    try ingressRequire(try peer("192.168.1.20") == "192.168.1.20", "plain IPv4")
    try ingressRequire(try peer("1.2.3.4, 10.0.0.7") == "10.0.0.7", "not the last entry (the one the proxy added)")
    try ingressRequire(try peer(" [FE80:0:0:0:0:0:0:1] ") == "fe80::1", "IPv6 not canonical")
    for bad in ["", "a, ", "10.0.0.7:5000", "unknown", "1.2.3.4.5", "10.0.0.7 ,", String(repeating: "1", count: 60)] {
        do {
            _ = try peer(bad)
            throw EngineError("HTTP ingress: malformed X-Forwarded-For accepted: \(bad)")
        } catch let failure as ServingFailure { try ingressRequire(failure.status == 400, "wrong status for \(bad)") }
    }
}

private func ingressRequire(_ ok: Bool, _ message: String) throws {
    if !ok { throw EngineError("HTTP ingress: " + message) }
}

private func httpControlSlotsReserved() throws {
    for count in [2, 4, 16] {
        let budget = HTTPConnectionBudget(maximumRequests: count, maximumBodyBytes: 32)
        var requests: [Int] = []
        for _ in 0..<count {
            guard let id = budget.accept() else { throw EngineError("missing request admission") }
            try ingressRequire(budget.classify(id, control: false), "normal request refused early")
            requests.append(id)
        }
        let extra = budget.accept()!
        try ingressRequire(!budget.classify(extra, control: false), "inference consumed reserved control capacity")
        try ingressRequire(budget.classify(extra, control: true), "health could not use its reserve")
        let second = budget.accept()!
        try ingressRequire(budget.classify(second, control: true) && budget.accept() == nil, "total connection bound changed")
        try ingressRequire(budget.reserveBody(requests[0], bytes: 24), "first body refused")
        try ingressRequire(!budget.reserveBody(requests[1], bytes: 16), "aggregate body bound bypassed")
        budget.release(requests[0])
        requests.removeFirst()
        try ingressRequire(budget.reserveBody(requests[0], bytes: 16), "released body capacity leaked")
        for id in requests + [extra, second] { budget.release(id) }
        try ingressRequire(budget.snapshot["connections"] as? Int == 0, "admissions leaked")
    }
    try ingressRequire(HTTPIngressPolicy.isControl(method: "GET", target: "/v1/status?sample=1"), "status query not control")
    try ingressRequire(
        !HTTPIngressPolicy.isControl(method: "POST", target: "/v1/status")
            && !HTTPIngressPolicy.isControl(method: "GET", target: "/v1/prompts"), "expensive/mutating route entered reserve")
}

private func socketPair() throws -> [Int32] {
    var pair = [Int32](repeating: -1, count: 2)
    guard socketpair(AF_UNIX, SOCK_STREAM, 0, &pair) == 0 else { throw EngineError("socketpair") }
    var one: Int32 = 1
    for fd in pair { _ = setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size)) }
    return pair
}

/// A connected TCP pair on loopback (an ephemeral port), without SIGPIPE.
private func tcpPair() throws -> [Int32] {
    var address = sockaddr_in()
    address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    address.sin_family = sa_family_t(AF_INET)
    address.sin_addr.s_addr = inet_addr("127.0.0.1")
    var length = socklen_t(MemoryLayout<sockaddr_in>.size)
    let listener = socket(AF_INET, SOCK_STREAM, 0)
    let near = socket(AF_INET, SOCK_STREAM, 0)
    defer { close(listener) }
    let ready = withUnsafeMutablePointer(to: &address) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            bind(listener, $0, length) == 0 && getsockname(listener, $0, &length) == 0 && listen(listener, 1) == 0
                && connect(near, $0, length) == 0
        }
    }
    let far = ready ? accept(listener, nil, nil) : -1
    guard far >= 0 else {
        close(near)
        throw EngineError("tcp pair")
    }
    var one: Int32 = 1
    for fd in [far, near] { _ = setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size)) }
    return [far, near]
}

private func httpIngressAbsoluteDeadline() throws {
    // Both an unfinished header and a body that keeps making progress must time out as a whole request.
    for prefix in ["G", "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100\r\n\r\n"] {
        let pair = try socketPair()
        let done = DispatchSemaphore(value: 0)
        defer {
            close(pair[0])
            close(pair[1])
        }
        let initial = Data(prefix.utf8)
        _ = initial.withUnsafeBytes { write(pair[1], $0.baseAddress, $0.count) }
        Thread.detachNewThread {
            defer { done.signal() }
            var byte: UInt8 = 32
            for _ in 0..<30 {
                usleep(25_000)
                if write(pair[1], &byte, 1) != 1 { return }
            }
        }
        let began = DispatchTime.now().uptimeNanoseconds
        do {
            _ = try BoundedHTTPIngress.receive(pair[0], timeout: 0.12)
            throw EngineError("trickled request bypassed total deadline")
        } catch let failure as ServingFailure { try ingressRequire(failure.status == 408, "wrong timeout response") }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - began) / 1e9
        try ingressRequire(elapsed >= 0.10 && elapsed < 0.6, "absolute timeout did not bound trickling: \(elapsed)")
        // Stop the writer from its own side: on macOS, writes after the reader's shutdown stall instead of failing.
        _ = shutdown(pair[1], SHUT_WR)
        try ingressRequire(done.wait(timeout: .now() + 1) == .success, "trickle writer did not finish")
    }
    // Reject from headers before the announced body arrives.
    let pair = try socketPair()
    defer {
        close(pair[0])
        close(pair[1])
    }
    let header = Data("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100\r\n\r\n".utf8)
    _ = header.withUnsafeBytes { write(pair[1], $0.baseAddress, $0.count) }
    do {
        _ = try BoundedHTTPIngress.receive(pair[0], timeout: 1, headers: { _, _, _ in throw ServingFailure("unauthorized", status: 401) })
        throw EngineError("header authentication waited for body")
    } catch let failure as ServingFailure { try ingressRequire(failure.status == 401, "early authentication not preserved") }
}

private func httpFrontJSONBudget() throws {
    let tooLarge = Data(("{\"model\":\"\\u0061\",\"origami_tune\":{},\"many\":[" + String(repeating: "0,", count: 140_000) + "0]}").utf8)
    do {
        try HTTPIngressPolicy.checkJSON(tooLarge)
        throw EngineError("oversized front structure accepted")
    } catch let failure as ServingFailure { try ingressRequire(failure.status == 413, "wrong structure rejection") }
    let deep = Data((String(repeating: "[", count: 65) + "0" + String(repeating: "]", count: 65)).utf8)
    do {
        try HTTPIngressPolicy.checkJSON(deep)
        throw EngineError("deep nesting accepted")
    } catch is ServingFailure {}
    // Large string content (e.g. image data) is governed by body reservation, not structural expansion.
    try HTTPIngressPolicy.checkJSON(Data(("{\"image\":\"" + String(repeating: "a", count: 300_000) + "\"}").utf8))
    try HTTPIngressPolicy.checkJSON(Data("{\"text\":\"\\\" [ { \\\\ \\\\u0061\"}".utf8))
}

private func httpFrontStatusAndCorrelation() throws {
    let unavailable = HTTPIngressPolicy.observedStatus(["error": ["message": "busy"]], httpStatus: 503)
    try ingressRequire(
        unavailable["status"] as? String == "unavailable" && unavailable["upstream_http_status"] as? Int == 503,
        "upstream failure disguised as ready")
    try ingressRequire(
        HTTPIngressPolicy.observedStatus(["status": "ready"], httpStatus: 200)["status"] as? String == "ready", "ready state changed")
    var decoder = BoundedHTTPRequestDecoder()
    let body = "{\"model\":\"same\"}"
    let raw = Data(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nX-Origami-Request-ID: supplied\r\nContent-Length: \(body.utf8.count)\r\n\r\n\(body)"
            .utf8)
    let request = try decoder.append(raw)!
    let forwarded = BoundedHTTPIngress.forwarded(request, requestID: "owned-id")
    var next = BoundedHTTPRequestDecoder()
    let received = try next.append(forwarded)!
    try ingressRequire(
        received.headers["x-origami-request-id"] == "owned-id" && received.body == request.body && received.target == request.target,
        "forward changed body or retained supplied ID")
    let pair = try socketPair()
    defer {
        close(pair[0])
        close(pair[1])
    }
    _ = raw.withUnsafeBytes { write(pair[1], $0.baseAddress, $0.count) }
    try ingressRequire(try BoundedHTTPIngress.receive(pair[0], timeout: 1)?.body == request.body, "normal ingress failed")
}
