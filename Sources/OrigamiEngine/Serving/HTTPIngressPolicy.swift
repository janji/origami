import Darwin
import Foundation

/// Shared front/worker routing rule. Only small, read-only observations use the reserved control slots.
public enum HTTPIngressPolicy {
    public static func observedStatus(_ object: [String: Any], httpStatus: Int) -> [String: Any] {
        var state = object
        state["upstream_http_status"] = httpStatus
        if !(200..<300).contains(httpStatus) || state["status"] == nil {
            state["status"] = "unavailable"
            state["failure"] = "The model's status endpoint returned HTTP \(httpStatus). Reconnecting…"
        }
        return state
    }
    public static func isControl(method: String, target: String) -> Bool {
        guard method == "GET" || method == "HEAD" else { return false }
        let path = String(target.split(separator: "?", maxSplits: 1).first ?? "")
        return ["/health", "/v1/health", "/v1/status", "/v1/dashboard", "/api/ps"].contains(path)
    }

    /// Must run before any front-side JSON deserializer, including model selection and tuning inspection.
    public static func checkJSON(_ body: Data, maximumStructureBytes: Int = 256 * 1_024) throws {
        guard try ServingRequest.checkNesting(body) <= maximumStructureBytes else {
            throw ServingFailure("Request JSON is too large outside its strings", status: 413)
        }
    }

    /// The client a trusted proxy on this Mac names: the last `X-Forwarded-For` entry (the one that proxy added; repeated
    /// fields arrive joined by commas), as a canonical IPv4 or IPv6 address. Nil without the header; a 400 when the
    /// entry is not an address. Only for a connection from loopback with `trusted_proxy loopback` set.
    public static func forwardedPeer(_ headers: [String: String]) throws -> String? {
        guard let field = headers["x-forwarded-for"] else { return nil }
        var entry = (field.split(separator: ",", omittingEmptySubsequences: false).last ?? "").trimmingCharacters(in: .whitespaces)
        if entry.hasPrefix("[") && entry.hasSuffix("]") && entry.count > 2 { entry = String(entry.dropFirst().dropLast()) }
        var v4 = in_addr()
        var v6 = in6_addr()
        var text = [CChar](repeating: 0, count: Int(INET6_ADDRSTRLEN))
        if entry.utf8.count <= 45, inet_pton(AF_INET, entry, &v4) == 1, inet_ntop(AF_INET, &v4, &text, socklen_t(text.count)) != nil {
            return String(decoding: text.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        }
        if entry.utf8.count <= 45, inet_pton(AF_INET6, entry, &v6) == 1, inet_ntop(AF_INET6, &v6, &text, socklen_t(text.count)) != nil {
            return String(decoding: text.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        }
        throw ServingFailure("X-Forwarded-For must end with the client's address")
    }
}

/// Accepted but unclassified connections are bounded too. Classification never allows inference to occupy the
/// reserved control slots. Body reservations cover buffered request bytes, independent of parser inflation limits.
/// Every open connection also holds `headerBytes` (its header buffer and one receive chunk), counted in the snapshot.
/// Connections from another device (`accept(peer:)` with its address) are capped while unclassified: `ingressPerPeer`
/// from one address and `remoteIngress` in all, so one peer cannot take every slot; this Mac's own are not.
/// A connection accepted with `drop` loses its body reservation to another request that needs the room when it is
/// behind the pace that would bring its announced body in within `bodyDeadline` (it would be cut off anyway): `drop`
/// then runs, under the lock, so never after `release`, and `evicted` says so.
public final class HTTPConnectionBudget: @unchecked Sendable {
    private enum Kind: Equatable { case ingress, request, control }
    private struct Entry {
        var kind = Kind.ingress
        let peer: String?
        let drop: (() -> Void)?
        var body = 0, received = 0, since: UInt64 = 0, evicted = false
    }
    private let lock = NSLock()
    public let maximumRequests, reservedControl: Int
    private let maximumBodyBytes: Int
    public let headerBytes, remoteIngress, ingressPerPeer: Int
    public let bodyDeadline: TimeInterval
    private var next = 0, entries: [Int: Entry] = [:]
    public init(
        maximumRequests: Int, reservedControl: Int = 2, maximumBodyBytes: Int = 64 * 1_024 * 1_024, headerBytes: Int = 48 * 1_024,
        remoteIngress: Int = 32, ingressPerPeer: Int = 8, bodyDeadline: TimeInterval = 30
    ) {
        precondition(maximumRequests > 0 && reservedControl > 0 && maximumBodyBytes >= 0 && headerBytes >= 0)
        precondition(remoteIngress > 0 && ingressPerPeer > 0 && bodyDeadline > 0)
        self.maximumRequests = maximumRequests
        self.reservedControl = reservedControl
        self.maximumBodyBytes = maximumBodyBytes
        self.headerBytes = headerBytes
        self.remoteIngress = remoteIngress
        self.ingressPerPeer = ingressPerPeer
        self.bodyDeadline = bodyDeadline
    }
    /// An admission id, or nil when every slot is taken or `peer` (another device's address; nil for this Mac) is at
    /// an unclassified cap.
    public func accept(peer: String? = nil, drop: (() -> Void)? = nil) -> Int? {
        lock.withLock {
            guard entries.count < maximumRequests + reservedControl else { return nil }
            if let peer {
                let waiting = entries.values.filter { $0.kind == .ingress && $0.peer != nil }
                guard waiting.count < remoteIngress, waiting.filter({ $0.peer == peer }).count < ingressPerPeer else { return nil }
            }
            let id = next
            next += 1
            entries[id] = Entry(peer: peer, drop: drop)
            return id
        }
    }
    public func classify(_ id: Int, control: Bool) -> Bool {
        lock.withLock {
            guard let kind = entries[id]?.kind else { return false }
            guard kind == .ingress else { return control || kind == .request }
            let count = { (kind: Kind) in self.entries.values.filter { $0.kind == kind }.count }
            // Health and status past their reserve take a free request slot: an open dashboard must not turn a health check into a 503.
            let desired: Kind = control && count(.control) < reservedControl ? .control : .request
            guard desired == .control || count(.request) < maximumRequests else { return false }
            entries[id]?.kind = desired
            return true
        }
    }
    public func reserveBody(_ id: Int, bytes: Int) -> Bool {
        lock.withLock {
            guard entries[id] != nil, bytes >= 0, bytes <= maximumBodyBytes else { return false }
            let now = DispatchTime.now().uptimeNanoseconds
            let held = entries.reduce(0) { $0 + ($1.key == id ? 0 : $1.value.body) }
            if bytes > maximumBodyBytes - held {
                // Behind pace: at its average rate so far, its body would not be in by the deadline. Held a second at least.
                let deadline = UInt64(bodyDeadline * 1e9)
                let late = entries.filter { key, entry in
                    let elapsed = now - min(now, entry.since)
                    return key != id && entry.drop != nil && entry.body > 0 && elapsed >= 1_000_000_000
                        && UInt64(entry.received) * deadline < UInt64(entry.body) * elapsed
                }.sorted { Double($0.value.received) / Double($0.value.body) < Double($1.value.received) / Double($1.value.body) }
                var freed = 0
                var victims: [Int] = []
                for (key, entry) in late where bytes > maximumBodyBytes - (held - freed) {
                    victims.append(key)
                    freed += entry.body
                }
                guard bytes <= maximumBodyBytes - (held - freed) else { return false }
                for key in victims {
                    entries[key]?.body = 0
                    entries[key]?.evicted = true
                    entries[key]?.drop?()
                }
            }
            entries[id]?.body = bytes
            entries[id]?.received = 0
            entries[id]?.since = now
            return true
        }
    }
    /// Body bytes `id` received since its reservation (`BoundedHTTPIngress.receive`'s `progress`).
    public func received(_ id: Int, bytes: Int) {
        lock.withLock { entries[id]?.received += bytes }
    }
    /// Whether `id` lost its body reservation to another request.
    public func evicted(_ id: Int) -> Bool { lock.withLock { entries[id]?.evicted ?? false } }
    public func release(_ id: Int) {
        lock.withLock { _ = entries.removeValue(forKey: id) }
    }
    public var snapshot: [String: Any] {
        lock.withLock {
            [
                "connections": entries.count, "requests": entries.values.filter { $0.kind == .request }.count,
                "control": entries.values.filter { $0.kind == .control }.count,
                "ingress": entries.values.filter { $0.kind == .ingress }.count,
                "buffered_body_reservation_bytes": entries.values.reduce(0) { $0 + $1.body },
                "header_reservation_bytes": entries.count * headerBytes,
                "maximum_reserved_bytes": maximumBodyBytes + (maximumRequests + reservedControl) * headerBytes,
                "maximum_requests": maximumRequests, "reserved_control": reservedControl,
            ]
        }
    }
}

/// Failed key attempts per peer: after `failures` within `window` seconds of its first, the peer is refused (429) until
/// that window ends. At most `maximumPeers` are remembered; past that, new peers share one count.
public final class HTTPAuthLimiter: @unchecked Sendable {
    private let lock = NSLock()
    private let failures, maximumPeers: Int
    private let window: UInt64
    private var peers: [String: (count: Int, since: UInt64)] = [:]
    public init(failures: Int = 10, window: TimeInterval = 60, maximumPeers: Int = 1_024) {
        precondition(failures > 0 && window > 0 && maximumPeers > 0)
        self.failures = failures
        self.window = UInt64(window * 1e9)
        self.maximumPeers = maximumPeers
    }
    /// Seconds until `peer` may try a key again, or nil when it may now.
    public func retryAfter(_ peer: String) -> Int? {
        lock.withLock {
            let now = DispatchTime.now().uptimeNanoseconds
            guard let entry = peers[key(peer)], now - entry.since < window, entry.count >= failures else { return nil }
            return Int((entry.since + window - now + 999_999_999) / 1_000_000_000)
        }
    }
    public func failed(_ peer: String) {
        lock.withLock {
            let now = DispatchTime.now().uptimeNanoseconds
            if peers[peer] == nil && peers.count >= maximumPeers { peers = peers.filter { now - $0.value.since < window } }
            let name = key(peer)
            if let entry = peers[name], now - entry.since < window {
                peers[name] = (entry.count + 1, entry.since)
            } else {
                peers[name] = (1, now)
            }
        }
    }
    private func key(_ peer: String) -> String { peers[peer] != nil || peers.count < maximumPeers ? peer : "" }
}

/// A response copied from the model's server to a client through a bounded buffer: the upstream is read whenever
/// fewer than `bound` bytes wait, so the model never waits on a client that keeps reading, and a client that takes no
/// byte for `stall` seconds while bytes wait for it is dropped (the caller then closes the upstream, which cancels the
/// request there). The client is nonblocking while it runs: on Darwin `MSG_DONTWAIT` does not stop `send` waiting for
/// room (the kernel checks only the socket's own flag), so a send larger than the room left would block.
/// An end of input from the client (a close, or `shutdown(SHUT_WR)`, which look the same until a byte is written to it)
/// counts as leaving unless `endOfInputLeaves` is false: a stream, which writes all the time, then still answers a client
/// that half-closed after its request, and one that closed is known by a reset, a socket error or a failed send. A
/// reply written only once complete keeps the rule, so a client that gave up never keeps the model generating.
public enum BoundedHTTPRelay {
    public enum Outcome: Equatable { case finished, clientLeft, clientStalled, cancelled }

    /// A client whose request was read: true when it left (a reset or error), sent more (one request per connection),
    /// or, with `endOfInputLeaves`, ended its input; false when it sent nothing.
    public static func left(_ client: Int32, endOfInputLeaves: Bool = true) -> Bool {
        var ready = pollfd(fd: client, events: Int16(POLLIN), revents: 0)
        guard poll(&ready, 1, 0) > 0 else { return false }
        if ready.revents & Int16(POLLERR | POLLNVAL) != 0 { return true }
        var byte: UInt8 = 0
        let count = recv(client, &byte, 1, MSG_PEEK | MSG_DONTWAIT)
        return count > 0 || (count == 0 && endOfInputLeaves) || (count < 0 && errno != EAGAIN && errno != EINTR)
    }

    public static func relay(
        upstream: Int32, client: Int32, bound: Int = 1 << 20, stall: TimeInterval = 30, endOfInputLeaves: Bool = true,
        cancelled: () -> Bool = { false }, forwarded: () -> Void = {}
    ) -> Outcome {
        precondition(bound > 0 && stall > 0)
        let flags = fcntl(client, F_GETFL)
        guard flags >= 0, fcntl(client, F_SETFL, flags | O_NONBLOCK) == 0 else { return .clientLeft }
        defer { _ = fcntl(client, F_SETFL, flags) }
        var pending = [UInt8]()
        var head = 0
        var open = true
        var inputEnded = false  // the client ended its input: it is no longer polled (its POLLHUP stays), only written to
        var chunk = [UInt8](repeating: 0, count: 64 * 1_024)
        var waiting = DispatchTime.now().uptimeNanoseconds
        let limit = UInt64(stall * 1e9)
        while true {
            if head == pending.count {
                guard open else { return .finished }
                waiting = DispatchTime.now().uptimeNanoseconds
            }
            let room = open && pending.count - head < bound
            var fds = [
                pollfd(fd: upstream, events: room ? Int16(POLLIN) : 0, revents: 0),
                pollfd(fd: inputEnded ? -1 : client, events: Int16(POLLIN) | (head < pending.count ? Int16(POLLOUT) : 0), revents: 0),
            ]
            let left = limit - min(limit, DispatchTime.now().uptimeNanoseconds - waiting)
            // Without POLLOUT to wait for (Darwin reports none beside POLLHUP), a half-closed client is tried every 10 ms.
            let most: UInt64 = inputEnded && head < pending.count ? 10 : 500
            if poll(&fds, 2, Int32(min(most, max(1, left / 1_000_000)))) < 0 && errno != EINTR { return .clientLeft }
            if cancelled() { return .cancelled }
            // The client sent more, ended its input, or failed (POLLOUT alone is room to write).
            if fds[1].revents & Int16(POLLIN | POLLHUP | POLLERR | POLLNVAL) != 0 {
                if Self.left(client, endOfInputLeaves: endOfInputLeaves) { return .clientLeft }
                // Only a confirmed end of input stops the polling: a wake-up with nothing to read leaves it on.
                var byte: UInt8 = 0
                inputEnded = !endOfInputLeaves && recv(client, &byte, 1, MSG_PEEK | MSG_DONTWAIT) == 0
            }
            if inputEnded || fds[1].revents & Int16(POLLOUT) != 0, head < pending.count {
                let sent = pending.withUnsafeBytes { send(client, $0.baseAddress! + head, pending.count - head, 0) }
                if sent > 0 {
                    head += sent
                    waiting = DispatchTime.now().uptimeNanoseconds
                    if head == pending.count || head >= bound {
                        pending.removeFirst(head)
                        head = 0
                    }
                } else if sent < 0 && errno != EAGAIN && errno != EINTR {
                    return .clientLeft
                }
            }
            if room, fds[0].revents != 0 {
                let count = read(upstream, &chunk, min(chunk.count, bound - (pending.count - head)))
                if count > 0 {
                    pending += chunk[0..<count]
                    forwarded()
                } else if count == 0 || (errno != EINTR && errno != EAGAIN) {
                    open = false
                }
            }
            if head < pending.count, DispatchTime.now().uptimeNanoseconds - waiting >= limit { return .clientStalled }
        }
    }
}

/// Blocking socket ingress with monotonic, absolute deadlines: the request line and headers within `headerTimeout`, the
/// whole request within `timeout`, both from the start; progress never resets them. While the body arrives, every
/// `rateWindow` seconds must bring `minimumRate` bytes (or the rest of the body), so a trickled upload is cut off
/// early. poll plus nonblocking recv avoids extending a deadline by a fresh SO_RCVTIMEO after every trickled byte.
/// `progress` gets the body bytes as they arrive.
public enum BoundedHTTPIngress {
    public static func receive(
        _ fd: Int32, timeout: TimeInterval = 30, headerTimeout: TimeInterval = 10, minimumRate: Int = 16 * 1_024,
        rateWindow: TimeInterval = 5, maximumHeaderBytes: Int = 32 * 1_024, maximumBodyBytes: Int = 24 * 1_024 * 1_024,
        lan: Bool = false, headers: (String, [String: String], Int) throws -> Void = { _, _, _ in }, progress: (Int) -> Void = { _ in }
    ) throws -> BoundedHTTPRequest? {
        precondition(timeout.isFinite && timeout >= 0 && headerTimeout.isFinite && headerTimeout >= 0)
        precondition(rateWindow.isFinite && rateWindow > 0 && minimumRate >= 0)
        func nanoseconds(_ seconds: TimeInterval) -> UInt64 { UInt64(min(seconds * 1e9, Double(UInt64.max / 2))) }
        let start = DispatchTime.now().uptimeNanoseconds
        let total = nanoseconds(timeout)
        let window = nanoseconds(rateWindow)
        var deadline = nanoseconds(min(headerTimeout, timeout))
        var decoder = BoundedHTTPRequestDecoder(maximumHeaderBytes: maximumHeaderBytes, maximumBodyBytes: maximumBodyBytes, lan: lan)
        var buffer = [UInt8](repeating: 0, count: 16 * 1_024)
        var checked = false
        // The body's rate window: when it started, what it brought, what it must bring, and what is still to come.
        var windowStart: UInt64 = 0
        var windowBytes = 0
        var windowNeed = 0
        var remaining = 0
        func expired() -> ServingFailure {
            ServingFailure(
                checked ? "HTTP request deadline exceeded" : "HTTP headers deadline exceeded", status: 408, code: "request_timeout")
        }
        while true {
            let now = DispatchTime.now().uptimeNanoseconds
            guard now - start < deadline else { throw expired() }
            if checked && now - windowStart >= window {
                guard windowBytes >= windowNeed else {
                    throw ServingFailure("HTTP request body arrived too slowly", status: 408, code: "request_timeout")
                }
                windowStart = now
                windowBytes = 0
                windowNeed = min(minimumRate, remaining)
            }
            var wait = deadline - (now - start)
            if checked { wait = min(wait, windowStart + window - now) }
            let millis = Int32(min(UInt64(Int32.max), max(1, (wait + 999_999) / 1_000_000)))
            var ready = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)
            let result = poll(&ready, 1, millis)
            if result < 0 {
                if errno == EINTR { continue }
                return nil
            }
            if result == 0 { continue }
            let count = recv(fd, &buffer, buffer.count, MSG_DONTWAIT)
            if count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) { continue }
            guard count > 0 else { return nil }
            guard DispatchTime.now().uptimeNanoseconds - start < deadline else { throw expired() }
            let request = try decoder.append(Data(buffer[0..<count]))
            if checked {
                windowBytes += count
                remaining -= count
                progress(count)
            } else if let announced = decoder.announced {
                try headers(announced.target, announced.headers, decoder.announcedBodyBytes ?? 0)
                checked = true
                deadline = total
                let arrived = request?.body.count ?? decoder.receivedBodyBytes ?? 0
                remaining = (decoder.announcedBodyBytes ?? 0) - arrived
                progress(arrived)
                windowStart = DispatchTime.now().uptimeNanoseconds
                windowBytes = 0
                windowNeed = min(minimumRate, remaining)
            }
            if let request { return request }
        }
    }

    /// The front owns the correlation ID; remove any supplied value so one request has one unambiguous ID.
    public static func forwarded(_ request: BoundedHTTPRequest, requestID: String) -> Data {
        precondition(
            !requestID.isEmpty && requestID.utf8.count <= 64
                && requestID.utf8.allSatisfy {
                    (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0) || $0 == 45 || $0 == 95
                })
        var lines = ["\(request.method) \(request.target) HTTP/1.1"]
        lines += request.headers.filter { $0.key != "x-origami-request-id" }.sorted { $0.key < $1.key }.map { "\($0.key): \($0.value)" }
        lines.append("X-Origami-Request-ID: \(requestID)")
        return (lines.joined(separator: "\r\n") + "\r\n\r\n").data(using: .isoLatin1)! + request.body
    }
}
