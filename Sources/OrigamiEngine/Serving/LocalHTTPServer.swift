import CryptoKit
import Foundation
import Network

/// Single HTTP/1.1 request per connection, with no transfer coding or pipelining.
/// Limits are applied before JSON parsing or response generation allocates state.
public struct BoundedHTTPRequest: Sendable {
    public let method, target: String
    public let headers: [String: String]
    public let body: Data
}

/// Incremental HTTP/1.1 request parser for one connection: fed raw receive chunks, it returns the request once its
/// headers and exactly `Content-Length` body bytes are in. Refuses chunked bodies, `Expect`, repeats of the fields it
/// reads as one value, a `Host` that is not this Mac's (`HostGuard`, DNS rebinding), and any byte past the body (no pipelining).
public struct BoundedHTTPRequestDecoder: Sendable {
    private let maximumHeaderBytes, maximumBodyBytes: Int
    private var buffer = Data()
    private var headerBytes: Int?
    private var contentLength = 0
    private var method = "", target = ""
    private var headers: [String: String] = [:]
    private var completed = false
    /// `listen lan`: this Mac's network names and addresses are also its Host.
    private let lan: Bool
    /// The body length once the headers are in.
    public var announcedBodyBytes: Int? { headerBytes == nil ? nil : contentLength }
    /// The body bytes buffered so far, once the headers are in.
    public var receivedBodyBytes: Int? { headerBytes.map { buffer.count - $0 } }
    public var announcedMethod: String? { headerBytes == nil ? nil : method }
    /// The request target and header fields once the headers are in.
    public var announced: (target: String, headers: [String: String])? { headerBytes == nil ? nil : (target, headers) }
    public init(maximumHeaderBytes: Int = 32 * 1_024, maximumBodyBytes: Int = 1_024 * 1_024, lan: Bool = false) {
        precondition(maximumHeaderBytes > 0 && maximumBodyBytes >= 0)
        self.maximumHeaderBytes = maximumHeaderBytes
        self.maximumBodyBytes = maximumBodyBytes
        self.lan = lan
    }
    public mutating func append(_ data: Data) throws -> BoundedHTTPRequest? {
        guard !completed else { throw ServingFailure("HTTP pipelining is not supported") }
        guard data.count <= maximumHeaderBytes + maximumBodyBytes - buffer.count else {
            throw ServingFailure("HTTP request exceeds its byte budget", status: 413)
        }
        buffer.append(data)
        if headerBytes == nil {
            guard let delimiter = buffer.range(of: Data("\r\n\r\n".utf8)) else {
                guard buffer.count <= maximumHeaderBytes else { throw ServingFailure("HTTP headers too large", status: 431) }
                return nil
            }
            let count = delimiter.upperBound
            // ISO-8859-1 maps every byte to one scalar: values may carry obs-text (bytes 0x80 and up, RFC 9110 §5.5).
            guard count <= maximumHeaderBytes,
                let text = String(data: buffer.prefix(delimiter.lowerBound), encoding: .isoLatin1),
                text.unicodeScalars.allSatisfy({ $0.value == 9 || $0.value == 10 || $0.value == 13 || ($0.value >= 32 && $0.value != 127) })
            else {
                throw ServingFailure("Invalid or oversized HTTP headers", status: 431)
            }
            let lines = text.components(separatedBy: "\r\n")
            let first = lines[0].split(separator: " ", omittingEmptySubsequences: false)
            guard first.count == 3, first[2] == "HTTP/1.1", !first[0].isEmpty,
                first[0].allSatisfy({ $0.isASCII && $0.isUppercase }), first[1].hasPrefix("/"),
                first[1].allSatisfy({ $0.isASCII && !$0.isWhitespace })
            else { throw ServingFailure("Invalid HTTP request line") }
            method = String(first[0])
            target = String(first[1])
            let tokenCharacters = Set("!#$%&'*+-.^_`|~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
            // Repeated fields combine as comma lists (RFC 9110, §5.3), as proxies send them. The fields this server
            // reads as one value (framing, Host, Origin, the keep name) must stay unambiguous.
            let singletons: Set<String> = ["host", "content-length", "transfer-encoding", "content-type", "origin", "x-origami-keep"]
            for line in lines.dropFirst() {
                guard !line.isEmpty, let colon = line.firstIndex(of: ":"), colon != line.startIndex else {
                    throw ServingFailure("Invalid HTTP header")
                }
                let name = String(line[..<colon]).lowercased()
                guard name.allSatisfy({ tokenCharacters.contains($0) }) else { throw ServingFailure("Invalid HTTP header") }
                let value = String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
                guard !value.contains("\r"), !value.contains("\n") else { throw ServingFailure("Invalid HTTP header value") }
                if let previous = headers[name] {
                    guard !singletons.contains(name) else { throw ServingFailure("Duplicate HTTP header") }
                    headers[name] = previous + ", " + value
                } else {
                    headers[name] = value
                }
            }
            guard let host = headers["host"], !host.isEmpty else { throw ServingFailure("HTTP/1.1 requires Host") }
            // DNS rebinding: a page from another origin reaches this port under its own host name.
            guard HostGuard.allows(host, lan: lan) else {
                throw ServingFailure(
                    lan
                        ? "Host must be this Mac's name or address, 127.0.0.1, localhost or [::1]"
                        : "Host must be 127.0.0.1, localhost or [::1]")
            }
            guard headers["transfer-encoding"] == nil else { throw ServingFailure("Transfer-Encoding is not supported") }
            guard headers["expect"] == nil else { throw ServingFailure("Expect is not supported", status: 417) }
            if let length = headers["content-length"] {
                guard !length.isEmpty, length.utf8.allSatisfy({ (48...57).contains($0) }), let size = Int(length) else {
                    throw ServingFailure("Invalid Content-Length")
                }
                guard size <= maximumBodyBytes else { throw ServingFailure("HTTP body too large", status: 413) }
                contentLength = size
            } else if method == "POST" {
                throw ServingFailure("POST requires Content-Length", status: 411)
            }
            if method == "GET" && contentLength != 0 { throw ServingFailure("GET body is not supported") }
            headerBytes = count
        }
        guard let headerBytes else { return nil }
        let expected = headerBytes + contentLength
        guard buffer.count <= expected else { throw ServingFailure("HTTP pipelining or surplus body bytes are not supported") }
        guard buffer.count == expected else { return nil }
        completed = true
        let request = BoundedHTTPRequest(method: method, target: target, headers: headers, body: Data(buffer.suffix(contentLength)))
        buffer = Data()
        return request
    }
}

/// The server's port, limits and CLI-supplied callbacks. The byte limits size the server's ledger reservation
/// (`admittedBytes`, minus the large-body share that image bodies reserve per request).
public struct LocalHTTPConfiguration: Sendable {
    public var port: UInt16 = 11439
    public var model = "qwen3.8-flash-next"
    /// The served model's family, for Ollama's model details (`/api/show`, `/api/tags`) and the quality levels it offers.
    public var family = ModelFamily.qwen
    /// Open request connections, which wait in the queue for the one generation slot; health and status have two more.
    public var maximumConnections = 64
    /// Requests the buffer budget below holds while they are received and parsed (JSON structure inflates about 64
    /// times). A parsed request waiting its turn keeps about 4 times its bytes of it (`bodyPoolBytes`).
    public var bufferedRequests = 4
    public var maximumHeaderBytes = 32 * 1_024
    /// Images arrive inline as base64 data URLs, so a body may be large.
    public var maximumBodyBytes = 24 * 1_024 * 1_024
    /// JSON outside strings, which the parser inflates most (about 64 times).
    public var maximumStructureBytes = 256 * 1_024
    /// Buffers are budgeted as `bufferedRequests` bodies of this size inflated 64 times (the limit before
    /// images); bodies above the structure bound share what the structure bound leaves of that budget.
    public var budgetBodyBytes = 1_024 * 1_024
    public var maximumOutputBytes = 4 * 1_024 * 1_024
    public var requestTimeout: TimeInterval = 30
    public var writeTimeout: TimeInterval = 30
    /// How long a request waits for the one generation slot (or for the model to load) before a 503.
    /// Above one full 32K request (about 20 minutes); a client that leaves also leaves the queue.
    public var queueTimeout: TimeInterval = 3600
    /// A streamed response sends an SSE comment after this many seconds of silence (queue, load, prefill).
    public var keepAliveInterval: TimeInterval = 10
    /// Values for fields a request omits (reasoning effort).
    public var defaults = ServingDefaults()
    /// The key every request must carry (`Authorization: Bearer <key>` or `x-api-key: <key>`), or nil for none
    /// (`api_key`). Only the dashboard page itself loads without it: it holds no data and sends the key on its requests.
    public var apiKey: String? = nil
    /// `listen lan`: the front this server runs behind also listens on the network, so this Mac's network names are a
    /// Host here too (the front checks the key of other devices; this server stays on 127.0.0.1).
    public var lan = false
    /// Answers an in-chat command (supplied by the CLI); nil turns commands off. Returns the reply text.
    public var commands: (@Sendable (ChatCommand, ServingRequest, GenerationControl) throws -> String)? = nil
    /// `origami quality <level>` (or `origami fast on|off`) for a system prompt (by `ServingRequest.systemKey`): the level
    /// of requests that choose none; nil keeps the server's default.
    public var stickyQuality: (@Sendable (String) -> QualityDial?)? = nil
    /// Model, Mac, settings, memory, caches and stores for /v1/status (supplied by the CLI).
    public var statusReport: (@Sendable () -> [String: Any])? = nil
    /// The memory plan other settings would give, for GET /v1/plan?context=&memory_limit= (supplied by the CLI).
    public var planReport: (@Sendable ([String: String]) throws -> [String: Any])? = nil
    /// Saves settings from the dashboard (POST /v1/config, a JSON object of key to text, as `origami config set`
    /// takes them; supplied by the CLI). They apply when Origami restarts.
    public var configWriter: (@Sendable ([String: String]) throws -> [String: Any])? = nil
    /// The hot copy from the dashboard (POST /v1/hot-copy, `{"build": true}`, `{"drive": "<volume>"|"auto"}` or
    /// `{"size": "40GB"|"auto"|"off"}`): saved and applied while serving; answers its status (supplied by the CLI).
    public var hotCopy: (@Sendable ([String: Any]) throws -> [String: Any])? = nil
    /// More POST endpoints for this server's own page, by path (`/v1/memory/learned`, which forgets the learned
    /// memory budget), each taking the JSON object sent (supplied by the CLI).
    public var actions: [String: @Sendable ([String: Any]) throws -> [String: Any]] = [:]
    /// The saved prompts for the dashboard (`/v1/prompts`, supplied by the CLI): method, the entry's id (nil for POST),
    /// the JSON body, for POST the request that saves its text as a chat's system prompt, and the request's control.
    public var prompts: (@Sendable (String, String?, [String: Any], ServingRequest?, GenerationControl) throws -> [String: Any])? = nil
    /// Server events worth a log line (long queue waits).
    public var log: (@Sendable (String) -> Void)? = nil
    /// Receives each finished request's record and its text units (system prompt, tools) by hash, when request
    /// records are on (`request_log`).
    public var recordRequest: (@Sendable ([String: Any], [String: String]) -> Void)? = nil
    /// Request logs are metrics-only by default; collecting prompt/tool text requires an explicit caller opt-in.
    public var recordRequestTextUnits = false
    /// Non-secret build/model/configuration identities, attached even when parsing or admission fails.
    public var observationTags: [String: String] = [:]
    /// Fields added to each finished request's record in `/v1/status` (whether the hot copy served reads), asked once as
    /// it finishes, with the seconds it ran (supplied by the CLI).
    public var requestTags: (@Sendable (_ seconds: TimeInterval) -> [String: Any])? = nil
    public init() {}
    public mutating func applyContextLimits(_ context: Int) {
        if context == ConversationGeometry.maximumFileContext {
            bufferedRequests = 2
            budgetBodyBytes = 4 * 1024 * 1024
            maximumStructureBytes = 1024 * 1024
        }
    }
    package var structureBytes: Int { min(maximumStructureBytes, budgetBodyBytes, maximumBodyBytes) }
    /// The allowance bodies above the structure bound (images, at four times their size) share.
    package var largeBodyBytes: UInt64 { UInt64(bufferedRequests + 1) * UInt64((budgetBodyBytes - structureBytes) * 64) }
    /// The part of the reservation every request's body shares: its structure bound inflated 64 times while it is
    /// received and parsed, then 4 times its bytes and headers while it waits (`HTTPConnectionBudget.reserveBody`).
    package var bodyPoolBytes: Int { (bufferedRequests + 1) * (structureBytes * 64 + maximumHeaderBytes * 8) }
    package var persistenceBytes: UInt64 { recordRequest == nil ? 0 : HTTPRequestMetrics.persistenceMemoryAllowance }
    package var admittedBytes: UInt64 {
        UInt64(bufferedRequests + 1) * UInt64(budgetBodyBytes * 64 + maximumHeaderBytes * 8)
            + UInt64(maximumOutputBytes) * 16 + 2 * 1024 * 1024
            + UInt64(2 * (maximumHeaderBytes * 8 + 16 * 1_024))
            + persistenceBytes
    }
}

/// Private loopback server. One backend prepare/run pair owns the generation
/// slot until actual completion, even if its client has already disconnected;
/// later requests wait their turn (bounded by the connection limit).
/// Input, open connections and output are bounded; synchronous writes provide
/// backpressure without accumulating token callbacks or pending socket writes.
public final class LocalHTTPServer: @unchecked Sendable {
    /// When this server was made: Ollama's `modified_at` for the model it serves.
    private let made = ISO8601DateFormatter().string(from: Date())
    private final class SendCompletion: @unchecked Sendable {
        let semaphore = DispatchSemaphore(value: 0)
        let lock = NSLock()
        var error: NWError?
        func finish(_ value: NWError?) {
            lock.lock()
            error = value
            lock.unlock()
            semaphore.signal()
        }
        func check() throws {
            lock.lock()
            defer { lock.unlock() }
            if let error { throw error }
        }
    }
    private final class Client: @unchecked Sendable {
        let id = UUID(), connection: NWConnection
        let lifecycle: HTTPRequestLifecycle
        let admission: Int
        let control = GenerationControl()
        var decoder: BoundedHTTPRequestDecoder  // network queue only
        var dispatched = false  // network queue only
        var requestTimer: DispatchWorkItem?  // network queue only
        var waitingShare = 0  // its body pool share once parsed; set on the network queue before dispatch
        var largeBody: MemoryLedger.Reservation?  // under lock
        let lock = NSLock()
        var closed = false
        var workerRegistered = false  // under lock; cleanup waits for actual backend unwind
        var keepAlive: DispatchSourceTimer?  // under lock
        /// Serializes writes from the request worker and the keepalive timer; recursive so a frame batch can be built
        /// and sent under it.
        let sendLock = NSRecursiveLock()
        var lastSend = Date()  // under sendLock
        init(_ connection: NWConnection, configuration: LocalHTTPConfiguration, admission: Int, lifecycle: HTTPRequestLifecycle) {
            self.connection = connection
            self.admission = admission
            self.lifecycle = lifecycle
            decoder = BoundedHTTPRequestDecoder(
                maximumHeaderBytes: configuration.maximumHeaderBytes,
                maximumBodyBytes: configuration.maximumBodyBytes, lan: configuration.lan)
        }
        func close() {
            lock.lock()
            guard !closed else {
                lock.unlock()
                return
            }
            closed = true
            let timer = keepAlive
            keepAlive = nil
            lock.unlock()
            timer?.cancel()
            control.cancel()
            connection.cancel()
        }
    }
    private let configuration: LocalHTTPConfiguration
    /// The writer can outlive server shutdown when a filesystem call is blocked. Its captured lease follows
    /// that lifetime, so queued/active record memory is never made available while still retained.
    private final class PersistenceLease: @unchecked Sendable {
        let ledger: MemoryLedger, reservation: MemoryLedger.Reservation
        init(_ ledger: MemoryLedger, bytes: UInt64) throws {
            self.ledger = ledger
            reservation = try ledger.reserve(bytes: bytes, category: "http-record-handoff")
        }
        deinit { try? ledger.release(reservation) }
    }
    private let backend: any ServingBackend
    private let ledger: MemoryLedger
    private let reservation: MemoryLedger.Reservation
    private let connectionBudget: HTTPConnectionBudget
    private let httpMetrics: HTTPRequestMetrics
    private let workerInstance = UUID().uuidString
    private let network = DispatchQueue(label: "origami.http.network", qos: .userInitiated)
    private let networkKey = DispatchSpecificKey<Bool>()
    private let workers = DispatchQueue(
        label: "origami.http.work", qos: .userInitiated, attributes: .concurrent, autoreleaseFrequency: .workItem)
    private let work = DispatchGroup()
    private let lock = NSLock()
    private var listener: NWListener?
    private var clients: [UUID: Client] = [:]
    private var active: UUID?
    private var activeSince: Date?
    /// The request holding the slot (for /v1/status) and the last 50 finished ones.
    private var current: (control: GenerationControl, started: Date, firstPiece: Date?, pieces: Int, effort: String, mode: String)?
    private var progressLogged = Date.distantPast, progressTimer: DispatchSourceTimer?
    private var recent: [[String: Any]] = []
    private var completedRequests: UInt64 = 0
    /// Requests waiting for the slot: arrival, and whether a person (not an agent's tool turn) sent it.
    private var waiting: [UUID: (since: Date, person: Bool)] = [:]
    private let turn = NSCondition()
    private var started = false, stopped = false
    private var actualPort: UInt16?

    public init(
        configuration: LocalHTTPConfiguration = LocalHTTPConfiguration(), backend: any ServingBackend,
        ledger: MemoryLedger
    ) throws {
        guard (1...256).contains(configuration.maximumConnections), (1...16).contains(configuration.bufferedRequests),
            (1_024...65_536).contains(configuration.maximumHeaderBytes),
            (1_024...32 * 1_024 * 1_024).contains(configuration.maximumBodyBytes), configuration.maximumStructureBytes >= 1_024,
            (1_024...32 * 1_024 * 1_024).contains(configuration.maximumOutputBytes),
            configuration.requestTimeout.isFinite, configuration.requestTimeout > 0,
            configuration.writeTimeout.isFinite, configuration.writeTimeout > 0,
            !configuration.model.isEmpty, configuration.model.utf8.count <= 256
        else { throw EngineError("Invalid HTTP limits") }
        self.configuration = configuration
        self.backend = backend
        self.ledger = ledger
        connectionBudget = HTTPConnectionBudget(
            maximumRequests: configuration.maximumConnections, maximumBodyBytes: configuration.bodyPoolBytes)
        if let record = configuration.recordRequest {
            let lease = try PersistenceLease(ledger, bytes: configuration.persistenceBytes)
            httpMetrics = HTTPRequestMetrics(scope: "worker", record: { row in withExtendedLifetime(lease) { record(row, [:]) } })
        } else {
            httpMetrics = HTTPRequestMetrics(scope: "worker")
        }
        // Foundation object inflation, UTF-16 strings, parser buffers, serialized
        // response copies and Network.framework's single outstanding send.
        // Two extra ingress/control slots hold only headers plus one receive chunk before classification.
        reservation = try ledger.reserve(
            bytes: configuration.admittedBytes - configuration.largeBodyBytes - configuration.persistenceBytes,
            category: "http-bounded-buffers")
        network.setSpecific(key: networkKey, value: true)
    }
    deinit {
        stop()
        try? ledger.release(reservation)
    }
    public var boundPort: UInt16? {
        lock.lock()
        defer { lock.unlock() }
        return actualPort
    }
    public var isGenerating: Bool {
        lock.lock()
        defer { lock.unlock() }
        return active != nil
    }
    public var isStopped: Bool {
        lock.lock()
        defer { lock.unlock() }
        return stopped
    }

    /// Returns only once the loopback socket is listening, or throws. A server
    /// instance cannot be restarted; construct a new one after stop.
    public func start(timeout: TimeInterval = 5) throws {
        guard backend.status().contextTier > 0 else { throw EngineError("Invalid backend context tier") }
        lock.lock()
        guard !started, !stopped else {
            lock.unlock()
            throw EngineError("HTTP server already started or stopped")
        }
        started = true
        lock.unlock()
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(
            host: NWEndpoint.Host("127.0.0.1"), port: NWEndpoint.Port(rawValue: configuration.port)!)
        let listener: NWListener
        do { listener = try NWListener(using: parameters) } catch {
            stop()
            throw error
        }
        lock.lock()
        self.listener = listener
        lock.unlock()
        let readiness = SendCompletion()
        listener.stateUpdateHandler = { [weak self, weak listener] state in
            guard let self else { return }
            switch state {
            case .ready:
                self.lock.lock()
                self.actualPort = listener?.port?.rawValue
                self.lock.unlock()
                readiness.finish(nil)
            case .failed(let error):
                readiness.finish(error)
                self.stop()
            case .cancelled: readiness.semaphore.signal()
            default: break
            }
        }
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else {
                connection.cancel()
                return
            }
            self.accept(connection)
        }
        listener.start(queue: network)
        if readiness.semaphore.wait(timeout: .now() + timeout) == .timedOut {
            stop()
            throw EngineError("HTTP listen timed out")
        }
        try readiness.check()
        guard boundPort != nil, !isStopped else { throw EngineError("HTTP listener stopped before readiness") }
        if configuration.log != nil {
            let timer = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
            timer.schedule(deadline: .now() + 10, repeating: 10)
            timer.setEventHandler { [weak self] in self?.logProgress() }
            lock.lock()
            progressTimer = timer
            lock.unlock()
            timer.resume()
        }
    }
    public func stop() {
        lock.lock()
        guard !stopped else {
            lock.unlock()
            return
        }
        stopped = true
        let listener = self.listener
        let connections = Array(clients.values)
        let progress = progressTimer
        self.listener = nil
        progressTimer = nil
        lock.unlock()
        progress?.cancel()
        listener?.cancel()
        for client in connections { client.close() }
        turn.lock()
        turn.broadcast()
        turn.unlock()
    }
    /// Never call from backend prepare/run, or a network callback. One deadline covers worker
    /// unwind, ingress finalization and the metrics sink. False means one of these is still pending; its owners
    /// retain their reservations. True confirms sink calls returned, not that a Void-returning sink persisted them.
    @discardableResult public func stopAndJoin(timeout: TimeInterval = 30) -> Bool {
        stop()
        guard DispatchQueue.getSpecific(key: networkKey) == nil, timeout.isFinite, timeout >= 0 else { return false }
        let deadline = DispatchTime.now() + timeout
        guard work.wait(timeout: deadline) == .success else { return false }
        let ingress = DispatchSemaphore(value: 0)
        network.async { [self] in
            // NWConnection cancellation callbacks may arrive after this barrier. Finish every remaining ingress
            // lifecycle here; later callbacks are idempotent. accept() starts no lifecycle after stop.
            lock.lock()
            let remaining = Array(clients.values)
            lock.unlock()
            for client in remaining { close(client) }
            ingress.signal()
        }
        guard ingress.wait(timeout: deadline) == .success else { return false }
        let now = DispatchTime.now().uptimeNanoseconds
        let left = deadline.uptimeNanoseconds > now ? Double(deadline.uptimeNanoseconds - now) / 1e9 : 0
        let flushed = flushDiagnostics(timeout: left)
        // Every request has ended: the backend's own stop work (a family engine saves its live conversation, bounded).
        (backend as? ServingBackendStopping)?.serverStopped()
        return flushed
    }
    /// Drains the currently accepted metrics watermark without stopping producers. Shutdown should use stopAndJoin.
    @discardableResult public func flushDiagnostics(timeout: TimeInterval = 5) -> Bool {
        guard DispatchQueue.getSpecific(key: networkKey) == nil else { return false }
        return httpMetrics.flush(timeout: timeout)
    }
    private func accept(_ connection: NWConnection) {
        // A listener cancellation may deliver one last connection after the shutdown barrier.
        guard !isStopped else {
            connection.cancel()
            return
        }
        let lifecycle = httpMetrics.begin()
        lifecycle.fields(configuration.observationTags.merging(["worker_instance": workerInstance]) { $1 })
        let admission = connectionBudget.accept()
        lock.lock()
        guard !stopped, let admission else {
            let stopped = self.stopped
            lock.unlock()
            if let admission { connectionBudget.release(admission) }
            // Hard admission cap: do not start another connection, receive buffer or timeout outside the budget.
            // Requests classified within the bounded ingress slots still receive an ordinary HTTP 503.
            lifecycle.fields(["transport_action": "cancel_before_http_response"])
            httpMetrics.finish(lifecycle, outcome: stopped ? "server_stopped" : "connection_rejected")
            connection.cancel()
            return
        }
        let client = Client(connection, configuration: configuration, admission: admission, lifecycle: lifecycle)
        clients[client.id] = client
        lock.unlock()
        connection.stateUpdateHandler = { [weak self, weak client] state in
            guard let self, let client else { return }
            switch state {
            case .failed, .cancelled: self.close(client)
            default: break
            }
        }
        let timer = DispatchWorkItem { [weak self, weak client] in
            guard let self, let client, !client.dispatched else { return }
            client.lifecycle.fields(["outcome": "ingress_timeout", "http_status": 408])
            self.close(client)
        }
        client.requestTimer = timer
        network.asyncAfter(deadline: .now() + configuration.requestTimeout, execute: timer)
        connection.start(queue: network)
        receive(client)
    }
    private func receive(_ client: Client) {
        client.connection.receive(minimumIncompleteLength: 1, maximumLength: 16 * 1_024) {
            [weak self, weak client] data, _, complete, error in
            guard let self, let client else { return }
            if error != nil {
                self.close(client)
                return
            }
            if let data, !data.isEmpty {
                do {
                    let request = try client.decoder.append(data)
                    if let announced = client.decoder.announced, let method = client.decoder.announcedMethod {
                        try self.requireKey(target: announced.target, headers: announced.headers)
                        let control =
                            HTTPIngressPolicy.isControl(method: method, target: announced.target) && client.decoder.announcedBodyBytes == 0
                        guard self.connectionBudget.classify(client.admission, control: control) else {
                            throw ServingFailure("Too many open requests; retry in a moment", status: 503, type: "server_busy")
                        }
                        if !control {
                            let body = min(client.decoder.announcedBodyBytes ?? 0, self.configuration.structureBytes)
                            let header = self.configuration.maximumHeaderBytes
                            client.waitingShare = (body + header) * 4
                            guard self.connectionBudget.reserveBody(client.admission, bytes: body * 64 + header * 8) else {
                                throw ServingFailure("Request buffers are busy; retry in a moment", status: 503, type: "server_busy")
                            }
                        }
                        client.lifecycle.fields(["method": method, "path": String(announced.target.split(separator: "?").first ?? "")])
                        if let parent = announced.headers["x-origami-request-id"], parent.utf8.count <= 64 {
                            client.lifecycle.fields(["parent_request_id": parent])
                        }
                    }
                    if let request {
                        if complete {
                            self.close(client)
                            return
                        }
                        client.dispatched = true
                        client.requestTimer?.cancel()
                        client.requestTimer = nil
                        self.launch(client) { try self.handle(request, client: client) }
                        if !complete { self.watchDisconnect(client) }
                        return
                    }
                    if let (target, headers) = client.decoder.announced { try self.requireKey(target: target, headers: headers) }
                    try self.admitLargeBody(client)
                } catch {
                    client.dispatched = true
                    client.requestTimer?.cancel()
                    client.requestTimer = nil
                    let failure = self.failure(error)
                    client.lifecycle.fields(["outcome": "ingress_rejected", "http_status": failure.status])
                    self.launch(client) { try self.sendFailure(failure, client: client) }
                    return
                }
            }
            if complete { self.close(client) } else { self.receive(client) }
        }
    }
    private func watchDisconnect(_ client: Client) {
        client.connection.receive(minimumIncompleteLength: 1, maximumLength: 1) { [weak self, weak client] data, _, complete, error in
            guard let self, let client else { return }
            // Half-closed requests and pipelining are deliberately unsupported.
            // Observing EOF cancels prefill too, before its first output send.
            if error != nil || complete || !(data?.isEmpty ?? true) {
                self.close(client)
                return
            }
            self.watchDisconnect(client)
        }
    }
    private func launch(_ client: Client, operation: @escaping @Sendable () throws -> Void) {
        // Register while holding the same lock used by stopAndJoin's stop:
        // no new worker may appear after stop has taken its client snapshot.
        lock.lock()
        guard !stopped else {
            lock.unlock()
            close(client)
            return
        }
        client.lock.lock()
        client.workerRegistered = true
        client.lock.unlock()
        work.enter()
        lock.unlock()
        workers.async { [self, client] in
            defer {
                finishLifecycle(client)
                close(client)
                releaseBody(client)
                connectionBudget.release(client.admission)
                work.leave()
            }
            do {
                try client.control.check()
                try operation()
            } catch { /* Per-route handlers send errors before this close. */  }
        }
    }
    private func close(_ client: Client) {
        client.close()
        client.lock.lock()
        let hasWorker = client.workerRegistered
        client.lock.unlock()
        if !hasWorker {
            finishLifecycle(client)
            releaseBody(client)
            connectionBudget.release(client.admission)
        }
        lock.lock()
        clients.removeValue(forKey: client.id)
        lock.unlock()
    }
    /// A disconnected worker may still hold the decoded request body. Retire its reservation only after unwind.
    private func releaseBody(_ client: Client) {
        client.lock.lock()
        let body = client.largeBody
        client.largeBody = nil
        client.lock.unlock()
        if let body { try? ledger.release(body) }
    }
    private func finishLifecycle(_ client: Client) {
        let p = client.control.progress
        var fields: [String: Any] = ["phase_at_finish": p.phase]
        if p.promptTokens > 0 {
            fields["prompt_tokens"] = p.promptTokens
            fields["completed_prompt_tokens"] = p.completedPromptTokens
        }
        if let restored = p.restoredTokens { fields["cached_tokens"] = restored }
        if let count = p.generatedTokens { fields["completion_tokens"] = count }
        if !p.stageSeconds.isEmpty { fields["partial_stage_seconds"] = p.stageSeconds }
        if !p.phaseCounters.isEmpty { fields["partial_phase_counters"] = p.phaseCounters }
        fields.merge(attemptFields(p)) { $1 }
        client.lifecycle.fields(fields)
        httpMetrics.finish(client.lifecycle, outcome: client.control.isCancelled ? "cancelled" : "completed")
    }
    private func attemptFields(_ progress: GenerationControl.Progress) -> [String: Any] {
        guard !progress.attempts.isEmpty else { return [:] }
        return [
            "generation_attempts": progress.attempts.map(\.wire), "omitted_generation_attempts": progress.omittedAttempts,
            "preparation_stage_seconds": progress.preparationStageSeconds,
            "stage_counter_scope": "last_attempt; generation_attempts includes retries; preparation is once per request",
        ]
    }
    /// Whether `headers` carry `key` (`Authorization: Bearer <key>` or `x-api-key: <key>`), compared in the same time
    /// for every wrong key of a length.
    public static func carriesKey(_ key: String, headers: [String: String]) -> Bool {
        let bearer = headers["authorization"].flatMap { value -> String? in
            let parts = value.split(separator: " ", maxSplits: 1)
            return parts.count == 2 && parts[0].lowercased() == "bearer" ? String(parts[1]).trimmingCharacters(in: .whitespaces) : nil
        }
        let expected = SHA256.hash(data: Data(key.utf8))
        let sent = [bearer, headers["x-api-key"]].compactMap { $0 }.map { SHA256.hash(data: Data($0.utf8)) }
        return sent.contains { digest in zip(digest, expected).reduce(UInt8(0)) { $0 | ($1.0 ^ $1.1) } == 0 }
    }
    /// Ollama's `details` and `capabilities` for a model of `family`. Every family thinks (its template's switch,
    /// `encodeChat`) and calls tools; `vision` only with a vision tower (Qwen's checkpoints; the others read text only).
    public static func ollamaDescription(_ family: ModelFamily, quantization: String, vision: Bool) -> (
        details: [String: Any], capabilities: [String]
    ) {
        let name = family == .qwen ? "qwen4_exp" : family.rawValue
        return (
            [
                "format": family == .qwen ? "mlx" : "gguf", "family": name, "families": [name], "parameter_size": "",
                "quantization_level": quantization,
            ],
            ["completion", "tools", "thinking"] + (vision ? ["vision"] : [])
        )
    }
    /// The request fields a harness may offer, in OpenRouter's `supported_parameters` names (`include_reasoning` and
    /// `reasoning` are OpenRouter's; `reasoning_effort` OpenAI's).
    public static let supportedParameters = [
        "reasoning", "reasoning_effort", "include_reasoning", "tools", "tool_choice", "max_tokens",
        "max_completion_tokens", "temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "stop", "seed",
    ]
    /// A `/v1/models` entry: OpenAI's fields, then what harnesses read to offer their settings (a thinking-effort menu,
    /// images, the context): OpenRouter's `supported_parameters`, the efforts, `context_length` and Ollama's
    /// `capabilities`. `origami` holds the server's own details.
    public static func modelEntry(_ id: String, family: ModelFamily, vision: Bool, context: Int, origami: [String: Any]) -> [String: Any] {
        [
            "id": id, "object": "model", "created": 0, "owned_by": "origami", "context_length": context,
            "reasoning_efforts": ServingDefaults.efforts(for: family),
            "supported_parameters": supportedParameters,
            "capabilities": ollamaDescription(family, quantization: "", vision: vision).capabilities, "origami": origami,
        ]
    }
    /// The dashboard page: for a browser at `/`, and always at `/dashboard`.
    public static func isDashboard(path: String, headers: [String: String]) -> Bool {
        path == "/dashboard" || (path == "/" && (headers["accept"] ?? "").contains("text/html"))
    }
    /// The page's icon, manifest and home-screen icons (`DashboardPage.asset`), fetched without the API key like the page.
    public static func isAsset(path: String) -> Bool { DashboardPage.asset(path) != nil }
    /// Refuses a request without the configured API key (401), checked as soon as its headers are in, before its body
    /// is buffered. The comparison takes the same time for every wrong key of a length.
    private func requireKey(target: String, headers: [String: String]) throws {
        let path = String(target.split(separator: "?").first ?? "")
        guard let key = configuration.apiKey, !Self.isDashboard(path: path, headers: headers), !Self.isAsset(path: path) else { return }
        guard Self.carriesKey(key, headers: headers) else {
            throw ServingFailure(
                "This server needs its API key: send Authorization: Bearer <key> or x-api-key: <key>",
                status: 401, type: "authentication_error", code: "invalid_api_key")
        }
    }
    /// A body above the structure bound (an image) takes its share of the large-body allowance before it is buffered.
    private func admitLargeBody(_ client: Client) throws {
        guard let length = client.decoder.announcedBodyBytes, length > configuration.structureBytes else { return }
        client.lock.lock()
        defer { client.lock.unlock() }
        guard client.largeBody == nil, !client.closed else { return }
        do { client.largeBody = try ledger.reserve(bytes: UInt64(length) * 4, category: "http-large-body") } catch {
            throw ServingFailure("Another large request (an image) is being received; retry in a moment", status: 503, type: "engine_busy")
        }
    }
    /// Waits for the generation slot, first come first served, a person's message ahead of an agent's
    /// tool-result turn. A client that disconnects while waiting leaves the queue. `onWait` runs once, outside the
    /// locks, the first time the request has to wait (a streamed request starts its stream then).
    private func acquire(_ client: Client, person: Bool, onWait: (() throws -> Void)? = nil) throws {
        client.lifecycle.mark("queue_start")
        defer { client.lifecycle.mark("queue_end") }
        // Parsed: the parser's inflation is gone, so a waiting request keeps only its own share of the body pool.
        _ = connectionBudget.reserveBody(client.admission, bytes: client.waitingShare)
        let arrived = Date()
        let deadline = Date(timeIntervalSinceNow: configuration.queueTimeout)
        var nextLog = arrived.addingTimeInterval(30)
        var onWait = onWait
        lock.lock()
        waiting[client.id] = (arrived, person)
        lock.unlock()
        defer {
            lock.lock()
            waiting[client.id] = nil
            lock.unlock()
        }
        turn.lock()
        defer { turn.unlock() }
        while true {
            lock.lock()
            if stopped || client.control.isCancelled {
                lock.unlock()
                throw IOOperationCancelled()
            }
            let ahead = waiting.contains {
                $0.key != client.id && ($0.value.person && !person || $0.value.person == person && $0.value.since < arrived)
            }
            if active == nil && !ahead {
                active = client.id
                activeSince = Date()
                lock.unlock()
                return
            }
            let queued = waiting.count
            lock.unlock()
            if let first = onWait {
                onWait = nil
                turn.unlock()
                defer { turn.lock() }
                try first()
                continue
            }
            let now = Date()
            if now >= nextLog {
                configuration.log?(
                    String(
                        format: "a request has waited %.0f s for the one generation slot (%d waiting)", now.timeIntervalSince(arrived),
                        queued))
                nextLog = now.addingTimeInterval(300)
            }
            guard now < deadline else {
                throw ServingFailure("Timed out waiting for the previous request to finish", status: 503, type: "server_busy")
            }
            _ = turn.wait(until: min(deadline, Date(timeIntervalSinceNow: 0.25)))
        }
    }
    private func release(_ client: Client) {
        lock.lock()
        if active == client.id {
            active = nil
            activeSince = nil
        }
        lock.unlock()
        turn.lock()
        turn.broadcast()
        turn.unlock()
    }
    /// What a client may send and what it gets when it sends nothing (/health, /v1/models).
    private var capabilities: [String: Any] {
        [
            "defaults": [
                "reasoning_effort": configuration.defaults.reasoningEffort,
                "max_thinking_tokens": configuration.defaults.maxThinkingTokens.map { $0 as Any } ?? NSNull(),
                "sampling": configuration.defaults.sampling.wire, "sampling_recommended": configuration.defaults.recommendedSampling.wire,
                "sampling_override": configuration.defaults.samplingOverride,
            ],
            "reasoning_efforts": ServingDefaults.efforts(for: configuration.family),
            "endpoints": [
                "/v1/chat/completions", "/v1/messages", "/v1/messages/count_tokens", "/v1/responses", "/api/chat", "/api/generate",
                "/api/tags", "/api/show", "/api/ps",
                "/api/version", "/v1/models", "/v1/status", "/v1/plan", "/v1/config", "/v1/hot-copy", "/v1/prompts", "/dashboard",
                "/health",
            ],
            "supported": [
                "reasoning_effort", "think", "reasoning.effort", "reasoning.enabled", "reasoning.exclude", "thinking.type",
                "chat_template_kwargs.enable_thinking", "chat_template_kwargs.reasoning_effort", "max_thinking_tokens",
                "thinking_budget_tokens", "thinking_budget", "thinking_token_budget", "reasoning.max_tokens", "thinking.budget_tokens",
                "include_reasoning", "max_tokens", "max_completion_tokens", "max_output_tokens", "origami_mode", "prompt_cache_policy",
                "status_messages", "return_progress", "stop", "seed", "tools", "tool_choice",
                "temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "stream_options.include_usage",
                "options.num_ctx",
            ],
        ]
    }
    /// The running request and the last 50 finished ones, for /v1/status.
    private var requestsWire: [String: Any] {
        lock.lock()
        defer { lock.unlock() }
        var wire: [String: Any] = ["recent": recent, "completed": completedRequests, "http": httpMetrics.snapshot]
        if let current {
            let progress = current.control.progress
            let now = Date()
            var running: [String: Any] = [
                "phase": progress.phase, "prompt_tokens": progress.promptTokens,
                "completed_prompt_tokens": progress.completedPromptTokens, "reused_tokens": progress.restoredTokens ?? 0,
                "output_pieces": current.pieces,
                "elapsed_s": now.timeIntervalSince(current.started), "reasoning_effort": current.effort, "mode": current.mode,
            ]
            if let activity = progress.activity { running["activity"] = activity }
            if !progress.phaseCounters.isEmpty { running["phase_counters"] = progress.phaseCounters }
            if !progress.attempts.isEmpty { running["completed_generation_attempts"] = progress.attempts.count }
            if let count = progress.generatedTokens {
                running["completion_tokens"] = count
                if count > 1, let first = progress.firstGeneratedAt {
                    running["decode_tps"] = Double(count - 1) / max(0.001, now.timeIntervalSince(first))
                }
            }
            if let first = current.firstPiece {
                running["ttft_s"] = first.timeIntervalSince(current.started)
            }
            wire["current"] = running
        }
        return wire
    }
    /// What a streamed request hears each keepalive tick while it waits or reads its prompt: "Origami ·"
    /// status lines as reasoning (`status_messages`) and `prompt_progress` frames (`return_progress`).
    private final class StreamTicker: @unchecked Sendable {  // touched only by its timer's serial handler
        let started = Date()
        var lastLine = "", lastProcessed = -1, announcedRestore = false
        func line(_ progress: GenerationControl.Progress, loading: Bool, queued: Bool) -> String? {
            let marker = ChatCommand.marker
            let line: String
            if loading {
                line = "\(marker) loading the model"
            } else if queued || progress.phase == "queued" {
                line = "\(marker) waiting for the request before this one"
            } else if progress.phase != "prefill" {
                return nil
            } else if let restored = progress.restoredTokens, restored > 0, !announcedRestore {
                announcedRestore = true
                line = "\(marker) reusing \(restored) saved tokens of \(progress.promptTokens)"
            } else {
                let percent = 100 * progress.completedPromptTokens / max(1, progress.promptTokens)
                line = "\(marker) reading the prompt: \(progress.completedPromptTokens) of \(progress.promptTokens) tokens (\(percent)%)"
            }
            guard line != lastLine else { return nil }
            lastLine = line
            return line
        }
    }
    private func isQueued(_ client: Client) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return waiting[client.id] != nil
    }
    /// A server log line every 20 s while a request reads its prompt and every 30 s while it answers.
    private func logProgress() {
        lock.lock()
        guard let running = current, Date().timeIntervalSince(progressLogged) >= (running.firstPiece == nil ? 20 : 30) else {
            lock.unlock()
            return
        }
        progressLogged = Date()
        lock.unlock()
        let progress = running.control.progress
        let elapsed = Date().timeIntervalSince(running.started)
        if let count = progress.generatedTokens, let first = progress.firstGeneratedAt {
            configuration.log?(
                String(
                    format: "request: %d tokens generated, %.1f tok/s", count,
                    Double(max(0, count - 1)) / max(0.001, Date().timeIntervalSince(first))))
        } else if running.firstPiece != nil {
            configuration.log?("request: \(running.pieces) output pieces delivered (token count unavailable)")
        } else if progress.phase == "prefill" {
            configuration.log?(
                String(
                    format: "request: prompt %d of %d tokens read (%d restored), %.0f s", progress.completedPromptTokens,
                    progress.promptTokens,
                    progress.restoredTokens ?? 0, elapsed))
        }
    }

    /// The server log's line for a finished request: /v1/status forgets its rows when the model is unloaded.
    package static func summary(_ row: [String: Any]) -> String {
        func number(_ key: String) -> Double? { (row[key] as? NSNumber)?.doubleValue }
        var text = String(
            format: "request: %@ in %.0f s: %.0f prompt tokens (%.0f restored)", row["finish"] as? String ?? "unknown",
            number("seconds") ?? 0, number("prompt_tokens") ?? 0, number("cached_tokens") ?? 0)
        if let first = number("ttft_s") { text += String(format: ", first word after %.1f s", first) }
        if let count = number("completion_tokens") { text += String(format: ", %.0f tokens written", count) }
        if let speed = number("decode_tps") { text += String(format: " at %.1f tok/s", speed) }
        return text
    }

    /// Records a finished (or failed) request in the ring buffer.
    private func finishRequest(
        _ client: Client, _ stats: GenerationStatistics?, error: String? = nil, record: [String: Any] = [:], units: [String: String] = [:]
    ) {
        // A cancel while the prompt is read is logged as the failure it raises; one while answering says how far it got.
        if error == "cancelled", case let progress = client.control.progress, progress.phase == "decode" {
            configuration.log?(
                "request: cancelled by the client after \(progress.generatedTokens.map(String.init) ?? "an unknown number of") tokens")
        }
        let started = lock.withLock { current?.started }
        let tags = configuration.requestTags?(started.map { Date().timeIntervalSince($0) } ?? 0) ?? [:]
        lock.lock()
        guard let finished = current, finished.control === client.control else {
            lock.unlock()
            return
        }
        current = nil
        let end = Date()
        var row: [String: Any] = [
            "time": ISO8601DateFormatter().string(from: finished.started), "seconds": end.timeIntervalSince(finished.started),
            "reasoning_effort": finished.effort, "mode": finished.mode, "finish": error ?? stats?.finishReason ?? "unknown",
        ]
        row["request_id"] = client.lifecycle.id
        row["outcome"] = error ?? "completed"
        row["partial_work"] = error != nil
        let partial = client.control.progress
        row.merge(attemptFields(partial)) { $1 }
        if let count = partial.generatedTokens { row["completion_tokens"] = count }
        row["prompt_tokens"] = partial.promptTokens
        row["cached_tokens"] = partial.restoredTokens ?? 0
        if !partial.stageSeconds.isEmpty { row["stage_seconds"] = partial.stageSeconds }
        if !partial.phaseCounters.isEmpty { row["phase_counters"] = partial.phaseCounters }
        if let stats {
            row["prompt_tokens"] = stats.promptTokens
            row["cached_tokens"] = stats.cachedTokens
            row["completion_tokens"] = stats.completionTokens
            if let from = stats.exactFrom { row["exact_from"] = from }
            if let retries = stats.gpuRetries { row["gpu_retries"] = retries }
            if let tokens = stats.reasoningTokens { row["reasoning_tokens"] = tokens }
            if let reason = stats.reasoningClosedBy { row["reasoning_closed_by"] = reason }
            if !stats.stageSeconds.isEmpty { row["stage_seconds"] = stats.stageSeconds }
            if !stats.phaseCounters.isEmpty { row["phase_counters"] = stats.phaseCounters }
        }
        if let first = finished.firstPiece {
            let ttft = first.timeIntervalSince(finished.started)
            row["ttft_s"] = ttft
            if let stats, ttft > 0 { row["prefill_tps"] = Double(stats.promptTokens - stats.cachedTokens) / ttft }
            if let stats, stats.completionTokens > 1 {
                row["decode_tps"] = Double(stats.completionTokens - 1) / max(0.001, end.timeIntervalSince(first))
            }
        }
        row.merge(tags) { $1 }
        recent.append(row)
        completedRequests &+= 1
        if recent.count > 50 { recent.removeFirst(recent.count - 50) }
        lock.unlock()
        configuration.log?(Self.summary(row))
        client.lifecycle.mark("service_end")
        client.lifecycle.fields(row.merging(record) { $1 })
        if configuration.recordRequestTextUnits, !units.isEmpty {
            configuration.recordRequest?(["event": "request_text_units", "request_id": client.lifecycle.id], units)
        }
    }
    /// Queue numbers for /health.
    private var queueWire: [String: Any] {
        lock.lock()
        defer { lock.unlock() }
        let now = Date()
        return [
            "busy": active != nil, "queued": waiting.count,
            "longest_wait_s": Int(waiting.values.map { now.timeIntervalSince($0.since) }.max() ?? 0),
            "busy_for_s": Int(activeSince.map { now.timeIntervalSince($0) } ?? 0),
        ]
    }
    /// The backend's status for the served model, whose quality levels it lists.
    private func servedStatus() -> ServingStatus {
        var status = backend.status()
        status.family = configuration.family
        if let away = ModelDrive.away?.note {
            status.degraded = away
            status.driveAway = true
        }
        return status
    }
    /// Requests that arrive while the model loads wait for it, like a queued request.
    private func ready(_ client: Client) throws -> ServingStatus {
        let deadline = Date(timeIntervalSinceNow: configuration.queueTimeout)
        while true {
            let status = servedStatus()
            if status.ready { return status }
            if let failure = status.failure {
                throw ServingFailure("The model failed to load: \(failure)", status: 503, type: "engine_unavailable")
            }
            try client.control.check()
            guard Date() < deadline, !isStopped else {
                throw ServingFailure("The model is still loading", status: 503, type: "engine_unavailable")
            }
            Thread.sleep(forTimeInterval: 0.2)
        }
    }
    private func send(_ data: Data, client: Client) throws {
        try client.control.check()
        client.sendLock.lock()
        defer {
            client.lastSend = Date()
            client.sendLock.unlock()
        }
        let completion = SendCompletion()
        client.connection.send(
            content: data,
            completion: .contentProcessed { [self] error in
                // Keep the buffer reservation alive until Network releases its send,
                // including when our bounded wait has already timed out.
                withExtendedLifetime(self) { completion.finish(error) }
            })
        if completion.semaphore.wait(timeout: .now() + configuration.writeTimeout) == .timedOut {
            close(client)
            throw ServingFailure("Client write timed out", status: 408, type: "client_timeout")
        }
        // No cancellation check after a send that went out: with `Connection: close` the client may close as soon as it
        // has the reply, and a delivered reply would be recorded as cancelled. The next send checks first.
        try completion.check()
    }
    private func headers(
        status: Int, type: String, length: Int?, metadata: ServingStatus? = nil, requestMode: QualityDial? = nil,
        cache: String = "no-store",
        extra: [String] = []
    ) -> Data {
        let phrase: String
        switch status {
        case 200: phrase = "OK"
        case 400: phrase = "Bad Request"
        case 401: phrase = "Unauthorized"
        case 403: phrase = "Forbidden"
        case 404: phrase = "Not Found"
        case 405: phrase = "Method Not Allowed"
        case 408: phrase = "Request Timeout"
        case 409: phrase = "Conflict"
        case 411: phrase = "Length Required"
        case 413: phrase = "Content Too Large"
        case 415: phrase = "Unsupported Media Type"
        case 417: phrase = "Expectation Failed"
        case 431: phrase = "Request Header Fields Too Large"
        case 503: phrase = "Service Unavailable"
        case 507: phrase = "Insufficient Storage"
        default: phrase = "Internal Server Error"
        }
        var lines = [
            "HTTP/1.1 \(status) \(phrase)", "Content-Type: \(type)", "Connection: close", "Cache-Control: \(cache)",
            "X-Content-Type-Options: nosniff",
        ]
        if let length { lines.append("Content-Length: \(length)") }
        if status == 401 { lines.append("WWW-Authenticate: Bearer") }
        if let metadata {
            // Reject control characters so a malformed backend cannot inject response headers.
            func safe(_ value: String) -> String {
                value.unicodeScalars.filter { $0.value >= 32 && $0.value < 127 }.map(String.init).joined()
            }
            lines += ["X-Origami-Version: \(safe(metadata.version))", "X-Origami-Context: \(metadata.contextTier)"]
            if let note = metadata.degraded { lines.append("X-Origami-Note: \(safe(note))") }
        }
        // The quality level of this completion (`origami_mode`; a non-exact one is refused before headers where not allowed).
        if let requestMode { lines.append("X-Origami-Mode: \(requestMode.name(configuration.family))") }
        lines += extra
        return Data((lines.joined(separator: "\r\n") + "\r\n\r\n").utf8)
    }
    /// `head`: a HEAD request gets the headers (with the body's length) and no body.
    private func response(
        _ object: [String: Any], status: Int = 200, client: Client, metadata: ServingStatus? = nil,
        requestMode: QualityDial? = nil, head: Bool = false
    ) throws {
        let data = try ServingWire.json(object)
        guard data.count <= configuration.maximumOutputBytes * 6 else { throw ServingFailure("Serialized response too large", status: 507) }
        try send(
            headers(
                status: status, type: "application/json; charset=utf-8", length: data.count, metadata: metadata,
                requestMode: requestMode), client: client)
        client.lifecycle.fields(["http_status": status])
        if !head { try send(data, client: client) }
    }
    /// Builds and sends frames under the client's send lock, so a keepalive tick's frames never cross a reply's.
    private func send(_ client: Client, frames: () throws -> [Data]) throws {
        client.sendLock.lock()
        defer { client.sendLock.unlock() }
        for frame in try frames() { try send(frame, client: client) }
    }
    /// Sends a comment line (ignored by every SSE client) whenever the stream has been silent for
    /// `keepAliveInterval`, so clients and proxies with idle timers keep waiting through a long prefill.
    private func startKeepAlive(_ client: Client, tick: (@Sendable () -> [Data])? = nil, frame: Data? = ServingWire.keepAlive) {
        let timer = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
        let interval = configuration.keepAliveInterval
        timer.schedule(deadline: .now() + interval / 2, repeating: interval / 2)
        timer.setEventHandler { [weak self, weak client] in
            guard let self, let client else { return }
            try? self.send(client) { tick?() ?? [] }
            client.sendLock.lock()
            let silent = Date().timeIntervalSince(client.lastSend)
            client.sendLock.unlock()
            if silent >= interval, let frame { try? self.send(frame, client: client) }
        }
        // Resumed before the closed check: libdispatch aborts on releasing a source that was never resumed.
        timer.resume()
        client.lock.lock()
        guard !client.closed else {
            client.lock.unlock()
            timer.cancel()
            return
        }
        client.keepAlive = timer
        client.lock.unlock()
    }
    private func failure(_ error: Error) -> ServingFailure { ServingFailure.from(error) }
    private func sendFailure(_ error: ServingFailure, client: Client) throws {
        try response(["error": error.wire], status: error.status, client: client)
    }
    private func handle(_ http: BoundedHTTPRequest, client: Client) throws {
        client.lifecycle.mark("parsed")
        var streamingStarted = false
        let path = String(http.target.split(separator: "?").first ?? "")
        let formats: [String: WireFormat] = [
            "/v1/chat/completions": .openAI, "/v1/messages": .anthropic, "/v1/responses": .responses,
            "/api/chat": .ollama, "/api/generate": .ollama, "/v1/messages/count_tokens": .anthropic,
        ]
        let format = formats[path] ?? .openAI
        let generate = path == "/api/generate"
        let id =
            ["anthropic": "msg_", "responses": "resp_"][format.rawValue, default: "chatcmpl-"]
            + UUID().uuidString.replacingOccurrences(of: "-", with: "")
        let created = Int(Date().timeIntervalSince1970)
        var wire = WireWriter(format: format, id: id, created: created, model: configuration.model, includeUsage: false, generate: generate)
        do {
            // A page on another web origin reaches this loopback port but can never read a reply (no CORS headers):
            // refuse its requests, whose side effects would still happen. Requests without an Origin (apps, curl) and
            // from non-web origins (browser extensions) stay allowed; this server's own page, also behind a proxy that
            // rewrites Host and Origin together, matches its Host. "null" is an opaque origin any page can take.
            let host = (http.headers["host"] ?? "").lowercased()
            if let origin = http.headers["origin"]?.lowercased(),
                origin == "null" || origin.hasPrefix("http://") || origin.hasPrefix("https://"),
                origin != "http://" + host, origin != "https://" + host
            {
                throw ServingFailure("Requests from web pages on other origins are refused", status: 403)
            }
            try requireKey(target: http.target, headers: http.headers)
            // Bounded before any wire parses the body: JSON structure inflates about 64 times in the parser.
            guard try ServingRequest.checkNesting(http.body) <= configuration.structureBytes else {
                throw ServingFailure("Request JSON is too large outside its strings", status: 413)
            }
            if try handleControl(http, path: path, client: client) { return }
            guard formats[path] != nil else {
                throw ServingFailure("Unknown endpoint \(path); use \(formats.keys.sorted().joined(separator: ", "))", status: 404)
            }
            guard http.method == "POST" else { throw ServingFailure("Use POST for completions", status: 405) }
            // Ollama clients often send no Content-Type; the others must say JSON.
            guard
                format == .ollama
                    || http.headers["content-type"]?.split(separator: ";").first?.trimmingCharacters(in: .whitespaces).lowercased()
                        == "application/json"
            else {
                throw ServingFailure("Content-Type must be application/json", status: 415)
            }
            // An empty `/api/generate` prompt asks Ollama to load the model (`ollama run` sends one first); it is loaded.
            if generate, let body = (try? JSONSerialization.jsonObject(with: http.body)) as? [String: Any],
                (body["prompt"] as? String ?? "").isEmpty
            {
                try response(
                    [
                        "model": body["model"] as? String ?? configuration.model, "created_at": ISO8601DateFormatter().string(from: Date()),
                        "response": "", "done": true, "done_reason": "load",
                    ], client: client)
                return
            }
            let vision = backend.status().vision
            // Anthropic's count: the request as `/v1/messages` would read it (its prompt tokens), without a reply.
            if path == "/v1/messages/count_tokens" {
                guard let counting = backend as? ServingPromptCounting,
                    var body = (try? JSONSerialization.jsonObject(with: http.body)) as? [String: Any]
                else { throw ServingFailure("Counting tokens needs a JSON body and a loaded model", status: 400) }
                body["max_tokens"] = body["max_tokens"] ?? 1
                let request = try ServingRequest.parse(
                    WireTranslation.chatBody(JSONSerialization.data(withJSONObject: body), format: .anthropic, vision: vision),
                    model: configuration.model, contextTier: backend.status().contextTier, defaults: configuration.defaults,
                    maximumStructureBytes: configuration.structureBytes)
                let tokens: Int
                do { tokens = try counting.promptTokens(request) } catch { throw ServingFailure.promptEncoding(error) }
                try response(["input_tokens": tokens], client: client)
                return
            }
            var parsed = try ServingRequest.parse(
                WireTranslation.chatBody(http.body, format: format, generate: generate, vision: vision), model: configuration.model,
                contextTier: backend.status().contextTier, defaults: configuration.defaults,
                maximumStructureBytes: configuration.structureBytes)
            wire = WireWriter(
                format: format, id: id, created: created, model: parsed.model, includeUsage: parsed.includeUsage, generate: generate,
                request: parsed)
            if let keep = http.headers["x-origami-keep"] {
                guard ChatCommand.isName(keep) else { throw ServingFailure("X-Origami-Keep " + ChatCommand.nameRule) }
                parsed.keep = keep
            }
            if let command = parsed.command, let handler = configuration.commands {
                try answer(command, handler: handler, parsed: parsed, wire: wire, client: client, streamingStarted: &streamingStarted)
                return
            }
            if !parsed.explicitMode, let key = parsed.systemKey, let sticky = configuration.stickyQuality?(key),
                sticky.isExact || backend.status().fastOptIn
            {
                parsed.quality = sticky
            }
            // A level this model does not offer runs as its nearest offered one, which usage and the record then name.
            parsed.quality = parsed.quality.offered(by: configuration.family)
            let request = parsed
            let writer = wire
            defer {
                client.lock.lock()
                let timer = client.keepAlive
                client.keepAlive = nil
                client.lock.unlock()
                timer?.cancel()
            }
            let tick = streamTick(request: request, writer: writer, client: client, id: id, created: created)
            func startStream(_ status: ServingStatus) throws {
                try send(
                    headers(
                        status: 200, type: writer.contentType, length: nil, metadata: status,
                        requestMode: request.quality), client: client)
                streamingStarted = true
                client.lifecycle.fields(["http_status": 200])
                for frame in try writer.open() { try send(frame, client: client) }
                startKeepAlive(client, tick: tick, frame: writer.keepAlive)
            }
            // A streamed request that has to wait (busy slot, model still loading) starts its stream as soon as it
            // waits, so keepalives reach the client; a later refusal arrives as an SSE error event.
            let arrival = backend.status()
            if request.stream && !arrival.ready { try startStream(arrival) }
            try acquire(
                client, person: !request.agentTurn, onWait: request.stream && !streamingStarted ? { try startStream(arrival) } : nil)
            defer { release(client) }
            client.lifecycle.mark("load_start")
            let status = try ready(client)
            client.lifecycle.mark("load_end")
            client.lifecycle.mark("service_start")
            lock.lock()
            current = (client.control, Date(), nil, 0, request.reasoningEffort, request.quality.name(configuration.family))
            lock.unlock()
            var produced: GenerationStatistics?
            var delivered = false
            var interrupted = false
            var record: [String: Any] = [:]
            var units: [String: String] = [:]
            if configuration.recordRequest != nil { (record, units) = requestRecord(http, format: format, request: request) }
            // Keep generation statistics even if final socket delivery fails.
            defer {
                finishRequest(
                    client, produced,
                    error: !delivered ? (client.control.isCancelled ? "cancelled" : interrupted ? "gpu_interrupted" : "error") : nil,
                    record: record, units: units)
            }
            let prepared = try backend.prepare(request, cancellation: client.control)
            try client.control.check()
            let stream = ServingStream(request: request, family: configuration.family, maximumBytes: configuration.maximumOutputBytes)
            if request.stream && !streamingStarted { try startStream(status) }
            // A request macOS kept interrupting on the GPU (after `GPURetry`) is "gpu_interrupted" in the recent requests.
            let stats: GenerationStatistics
            do {
                stats = try prepared.run { [self, client] piece in
                    // Per-token Foundation temporaries (JSON frames) go here, not in the request's pool.
                    try autoreleasepool {
                        try client.control.check()
                        lock.lock()
                        var first = false
                        if current?.control === client.control {
                            current!.pieces += 1
                            if current!.firstPiece == nil {
                                current!.firstPiece = Date()
                                first = true
                            }
                        }
                        lock.unlock()
                        if first { client.control.report { $0.phase = "decode" } }
                        let deltas = try stream.push(piece)
                        if request.stream { try send(client) { try deltas.flatMap { try writer.frames($0) } } }
                        return !stream.finishedSingleCall
                    }
                }
            } catch {
                interrupted = GPURetry.isInterruption(error)
                throw error
            }
            produced = stats
            try client.control.check()
            var usage = try ServingWire.usage(stats, family: configuration.family)
            var details: [String: Any] = ["reasoning_effort": request.reasoningEffort]
            if let tokens = stats.reasoningTokens { details["reasoning_tokens"] = tokens }
            if let closedBy = stats.reasoningClosedBy { details["reasoning_closed_by"] = closedBy }
            guard stats.completionTokens <= request.maximumTokens,
                stats.promptTokens <= request.contextLimit - stats.completionTokens,
                ["stop", "length", "tool_calls"].contains(stats.finishReason)
            else { throw EngineError("Backend violated generation contract") }
            let end = try stream.finish(reason: stats.finishReason)
            if let empty = stream.emptyAnswer { details["empty_answer"] = empty }
            usage["completion_tokens_details"] = details
            if !record.isEmpty {
                record["tool_calls"] = stream.calls.count
                if let empty = stream.emptyAnswer { record["empty_answer"] = empty }
            }
            if request.stream {
                try send(client) {
                    try end.deltas.flatMap { try writer.frames($0) }
                        + writer.close(reason: end.reason, usage: usage, stopSequence: stats.stopSequence)
                }
            } else {
                try response(
                    writer.body(message: stream.message, reason: end.reason, usage: usage, stopSequence: stats.stopSequence),
                    client: client,
                    metadata: status,
                    requestMode: stats.quality)
            }
            delivered = true
        } catch {
            if client.control.isCancelled {
                client.lifecycle.fields(["outcome": "cancelled"])
                throw error
            }
            let error = failure(error)
            client.lifecycle.fields(["outcome": "rejected_or_failed", "error_status": error.status])
            if streamingStarted {
                try send(client) { try wire.errorFrame(error) }
            } else {
                try response(wire.error(error), status: error.status, client: client)
            }
        }
    }

    /// The dashboard, health, status, saved-prompt, settings and model endpoints: true when `path` was one of them.
    private func handleControl(_ http: BoundedHTTPRequest, path: String, client: Client) throws -> Bool {
        // The dashboard and its files (icons, manifest), which do not change while it runs. It loads nothing from elsewhere.
        let asset = DashboardPage.asset(path)
        if asset != nil || Self.isDashboard(path: path, headers: http.headers) {
            guard http.method == "GET" || http.method == "HEAD" else { throw ServingFailure("Use GET for the dashboard", status: 405) }
            let (type, page) = asset ?? ("text/html; charset=utf-8", Data(DashboardPage.html.utf8))
            try send(
                headers(
                    status: 200, type: type, length: page.count, cache: asset == nil ? "no-store" : "max-age=86400",
                    extra: [
                        "Content-Security-Policy: \(DashboardPage.csp)", "Referrer-Policy: no-referrer",
                    ]), client: client)
            client.lifecycle.fields(["http_status": 200])
            if http.method == "GET" { try send(page, client: client) }
            return true
        }
        if ["/health", "/", "/v1/health"].contains(path) {
            guard http.method == "GET" || http.method == "HEAD" else { throw ServingFailure("Use GET for health", status: 405) }
            let status = servedStatus()
            try response(
                status.wire.merging(queueWire) { $1 }.merging(capabilities) { $1 }.merging(["model": configuration.model]) { $1 },
                status: status.failure == nil ? 200 : 503, client: client, metadata: status, head: http.method == "HEAD")
            return true
        }
        if path == "/v1/status" {
            guard http.method == "GET" else { throw ServingFailure("Use GET for status", status: 405) }
            let status = servedStatus()
            var report = status.wire.merging(capabilities) { $1 }
            report["model_id"] = configuration.model
            report["queue"] = queueWire
            report["requests"] = requestsWire
            for (key, value) in configuration.statusReport?() ?? [:] { report[key] = value }
            try response(report, client: client, metadata: status)
            return true
        }
        if path == "/v1/prompts" || path.hasPrefix("/v1/prompts/") {
            guard let prompts = configuration.prompts else { throw ServingFailure("Saved prompts are not served here", status: 404) }
            let id = path == "/v1/prompts" ? nil : String(path.dropFirst("/v1/prompts/".count))
            guard ["GET", "PATCH", "DELETE"].contains(http.method) && id != nil || http.method == "POST" && id == nil else {
                throw ServingFailure("Use GET, PATCH or DELETE /v1/prompts/<id>, or POST /v1/prompts", status: 405)
            }
            // A body forces a browser's cross-site preflight as JSON (other origins are refused above).
            guard ["GET", "DELETE"].contains(http.method) || (http.headers["content-type"] ?? "").hasPrefix("application/json") else {
                throw ServingFailure("Content-Type must be application/json", status: 415)
            }
            guard let body = http.body.isEmpty ? [:] : (try? JSONSerialization.jsonObject(with: http.body)) as? [String: Any] else {
                throw ServingFailure("Send a JSON object")
            }
            var save: ServingRequest?
            if http.method == "POST" {
                guard let text = body["text"] as? String, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    throw ServingFailure("POST {\"text\": \"<system prompt>\", \"name\": \"<name>\"}")
                }
                let chat = try JSONSerialization.data(withJSONObject: [
                    "messages": [["role": "system", "content": text], ["role": "user", "content": "Hi"]]
                ])
                save = try ServingRequest.parse(
                    chat, model: configuration.model, contextTier: backend.status().contextTier, defaults: configuration.defaults,
                    maximumStructureBytes: configuration.structureBytes)
            }
            // Saving prefills and deleting removes state a running request may be restoring, so both take the slot.
            if save != nil || http.method == "DELETE" {
                try acquire(client, person: true)
                defer { release(client) }
                _ = try ready(client)
                try response(try prompts(http.method, id, body, save, client.control), client: client)
            } else {
                try response(try prompts(http.method, id, body, nil, client.control), client: client)
            }
            return true
        }
        if path == "/v1/config" {
            guard http.method == "POST" else { throw ServingFailure("Use POST to save settings", status: 405) }
            guard let configWriter = configuration.configWriter else {
                throw ServingFailure("Settings cannot be saved from this server", status: 404)
            }
            // Only this server's own page may save settings: a JSON body forces a browser's cross-site preflight, and
            // an Origin, when sent, must be this host.
            guard (http.headers["content-type"] ?? "").hasPrefix("application/json"),
                HostGuard.ownPage(origin: http.headers["origin"], host: http.headers["host"] ?? "")
            else {
                throw ServingFailure("Settings are saved only from this server's own dashboard", status: 403)
            }
            // Values are text, as `origami config set` takes them; a number or boolean would be stringified wrongly.
            guard let object = (try? JSONSerialization.jsonObject(with: http.body)) as? [String: Any], !object.isEmpty,
                let settings = object as? [String: String]
            else {
                throw ServingFailure("POST a JSON object of settings with text values, such as {\"context\": \"32k\"}")
            }
            try response(try configWriter(settings), client: client)
            return true
        }
        if path == "/v1/hot-copy" {
            guard http.method == "POST" else { throw ServingFailure("Use POST to change the hot copy", status: 405) }
            guard let hotCopy = configuration.hotCopy else {
                throw ServingFailure("This model has no hot copy to change here", status: 404)
            }
            guard (http.headers["content-type"] ?? "").hasPrefix("application/json"),
                HostGuard.ownPage(origin: http.headers["origin"], host: http.headers["host"] ?? "")
            else {
                throw ServingFailure("The hot copy is changed only from this server's own dashboard", status: 403)
            }
            guard let object = (try? JSONSerialization.jsonObject(with: http.body)) as? [String: Any] else {
                throw ServingFailure("POST {\"build\": true}, {\"drive\": \"/Volumes/<drive>\"} or {\"size\": \"40GB\"}")
            }
            try response(try hotCopy(object), client: client)
            return true
        }
        if let action = configuration.actions[path] {
            guard http.method == "POST" else { throw ServingFailure("Use POST for \(path)", status: 405) }
            guard (http.headers["content-type"] ?? "").hasPrefix("application/json"),
                HostGuard.ownPage(origin: http.headers["origin"], host: http.headers["host"] ?? "")
            else {
                throw ServingFailure("This is changed only from this server's own dashboard", status: 403)
            }
            try response(try action((try? JSONSerialization.jsonObject(with: http.body)) as? [String: Any] ?? [:]), client: client)
            return true
        }
        if path == "/v1/plan" {
            guard http.method == "GET" else { throw ServingFailure("Use GET for plans", status: 405) }
            guard let planReport = configuration.planReport else {
                throw ServingFailure("No plan previews on this server", status: 404)
            }
            let items = formQueryItems(http.target)
            try response(
                try planReport(Dictionary(items.compactMap { item in item.value.map { (item.name, $0) } }) { $1 }), client: client)
            return true
        }
        if path == "/v1/models" {
            guard http.method == "GET" else { throw ServingFailure("Use GET for models", status: 405) }
            let status = servedStatus()
            let ids = modelIDs(status)
            let wire = status.wire.merging(capabilities) { $1 }
            try response(
                [
                    "object": "list",
                    "data": ids.map {
                        Self.modelEntry(
                            $0, family: configuration.family, vision: status.vision, context: status.contextTier, origami: wire)
                    },
                ],
                client: client, metadata: status)
            return true
        }
        if path.hasPrefix("/v1/models/") {
            guard http.method == "GET" else { throw ServingFailure("Use GET for models", status: 405) }
            let status = servedStatus()
            let id = String(path.dropFirst("/v1/models/".count)).removingPercentEncoding ?? ""
            guard modelIDs(status).contains(id) else {
                throw ServingFailure("No model named \(id)", status: 404, code: "model_not_found")
            }
            try response(
                Self.modelEntry(
                    id, family: configuration.family, vision: status.vision, context: status.contextTier,
                    origami: status.wire.merging(capabilities) { $1 }), client: client, metadata: status)
            return true
        }
        if ["/api/tags", "/api/version", "/api/show", "/api/ps"].contains(path) {
            try ollamaInfo(path, client: client)
            return true
        }
        return false
    }

    /// A chat command (`save`, `forget` and the others) answered as a one-message reply.
    private func answer(
        _ command: ChatCommand, handler: @Sendable (ChatCommand, ServingRequest, GenerationControl) throws -> String,
        parsed: ServingRequest, wire: WireWriter, client: Client,
        streamingStarted: inout Bool
    ) throws {
        // `save` prefills and `forget` deletes saved state a running request may be restoring, so both take the
        // slot; the others answer at once, also while the model loads.
        var text: String
        if ["save", "forget"].contains(command.verb) {
            try acquire(client, person: true)
            defer { release(client) }
            _ = try ready(client)
            text = try handler(command, parsed, client.control)
        } else {
            text = try handler(command, parsed, client.control)
        }
        text = ChatCommand.marker + " " + text
        let usage: [String: Any] = [
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "prompt_tokens_details": ["cached_tokens": 0],
        ]
        if parsed.stream {
            try send(headers(status: 200, type: wire.contentType, length: nil), client: client)
            client.lifecycle.fields(["http_status": 200])
            streamingStarted = true
            for frame in try wire.open() + wire.frames(["content": text]) + wire.close(reason: "stop", usage: usage) {
                try send(frame, client: client)
            }
        } else {
            try response(wire.body(message: ["role": "assistant", "content": text], reason: "stop", usage: usage), client: client)
        }
    }

    /// The stream's ticker: status lines and prompt progress frames between tokens (nil when neither is asked for).
    private func streamTick(request: ServingRequest, writer: WireWriter, client: Client, id: String, created: Int)
        -> (@Sendable () -> [Data])?
    {
        let wantsLines = (request.statusMessages ?? configuration.defaults.statusMessages) && !request.excludeReasoning
        let wantsFrames = request.returnProgress
        let ticker = StreamTicker()
        // Runs under the client's send lock (startKeepAlive), so its frames never cross the reply's.
        return !wantsLines && !wantsFrames
            ? nil
            : { @Sendable [self, client] in
                let progress = client.control.progress
                var frames: [Data] = []
                if wantsLines, let line = ticker.line(progress, loading: !self.backend.status().ready, queued: self.isQueued(client)),
                    let lines = try? writer.frames(["reasoning_content": line + "\n"])
                {
                    frames += lines
                }
                if wantsFrames, writer.format == .openAI, progress.phase == "prefill",
                    progress.completedPromptTokens != ticker.lastProcessed
                {
                    ticker.lastProcessed = progress.completedPromptTokens
                    var frame: [String: Any] = [
                        "id": id, "object": "chat.completion.chunk", "created": created, "model": request.model,
                        "choices": [["index": 0, "delta": [String: Any](), "finish_reason": NSNull()]],
                    ]
                    frame["prompt_progress"] = [
                        "total": progress.promptTokens, "cache": progress.restoredTokens ?? 0,
                        "processed": progress.completedPromptTokens, "time_ms": Int(Date().timeIntervalSince(ticker.started) * 1000),
                    ]
                    if let data = try? ServingWire.sse(frame) { frames.append(data) }
                }
                return frames
            }
    }

    /// The request's entry for `recordRequest`.
    private func requestRecord(_ http: BoundedHTTPRequest, format: WireFormat, request: ServingRequest)
        -> (record: [String: Any], units: [String: String])
    {
        var record: [String: Any] = [:]
        var units: [String: String] = [:]
        // What kind of request this was; texts only by hash, the system prompt and tools kept once each.
        record = [
            "wire": format.rawValue,
            "client": String((http.headers["user-agent"] ?? "unknown").split(separator: "/").first ?? "unknown").lowercased(),
            "stream": request.stream, "messages": request.messages.count, "tools": request.tools.count,
            "agent_turn": request.agentTurn,
            "requested_context_tokens": request.contextLimit, "requested_maximum_tokens": request.maximumTokens,
        ]
        if let system = request.messages.first(where: { $0["role"] == .string("system") }),
            case .string(let text)? = system["content"]
        {
            let hash = RequestRecorder.hash(text)
            record["system_sha256"] = hash
            if configuration.recordRequestTextUnits { units[hash] = text }
        }
        if !request.tools.isEmpty, let data = try? JSONEncoder().encode(request.tools) {
            let text = String(decoding: data, as: UTF8.self)
            let hash = RequestRecorder.hash(text)
            record["tools_sha256"] = hash
            if configuration.recordRequestTextUnits { units[hash] = text }
        }
        if let first = request.messages.first(where: { $0["role"] == .string("user") }), let data = try? JSONEncoder().encode(first) {
            record["session"] = String(RequestRecorder.hash(String(decoding: data, as: UTF8.self)).prefix(12))
        }
        return (record, units)
    }
    /// The served model id and, where non-exact levels are allowed, one `<id>-<level>` per level the model offers (clients
    /// that cannot send `origami_mode` pick a level by model).
    private func modelIDs(_ status: ServingStatus) -> [String] {
        [configuration.model]
            + (status.fastOptIn ? QualityDial.offered(status.family).dropFirst().map { configuration.model + "-" + $0 } : [])
    }
    /// Ollama's model listing endpoints, for clients that discover models that way.
    private func ollamaInfo(_ path: String, client: Client) throws {
        let status = servedStatus()
        let report = configuration.statusReport?() ?? [:]
        let model = report["model"] as? [String: Any]
        // The checkpoint's scheme (affine-4bit-g64, nvfp4-experts, ...) and size, from the status report when there is one.
        let (details, abilities) = Self.ollamaDescription(
            configuration.family, quantization: model?["quantization"] as? String ?? "unknown", vision: status.vision)
        let names = modelIDs(status)
        // `ollama list` prints the first 12 characters of the digest (and crashes on a shorter one).
        func digest(_ name: String) -> String { SHA256.hash(data: Data(name.utf8)).map { String(format: "%02x", $0) }.joined() }
        let size = (model?["bytes"] as? NSNumber)?.uint64Value ?? 0
        let models = names.map {
            [
                "name": $0, "model": $0, "modified_at": made, "size": size, "digest": digest($0), "details": details,
                "capabilities": abilities,
            ] as [String: Any]
        }
        switch path {
        case "/api/version": try response(["version": status.version], client: client)
        case "/api/show":
            let architecture = details["family"] as? String ?? ""
            try response(
                [
                    "modelfile": "", "parameters": "", "template": "", "details": details,
                    "model_info": ["general.architecture": architecture, architecture + ".context_length": status.contextTier],
                    "capabilities": abilities,
                ], client: client)
        case "/api/ps":
            // In memory (all of it on the GPU, as Ollama counts unified memory), with the context, until the server stops.
            let memory = report["memory"] as? [String: Any]
            let resident = (memory?["reserved_bytes"] as? NSNumber)?.uint64Value ?? ledger.snapshot().reserved
            // One loaded model: its `-<level>` names are not models of their own.
            try response(
                [
                    "models": status.ready
                        ? models.prefix(1).map {
                            $0.merging([
                                "size": resident, "size_vram": resident, "context_length": status.contextTier,
                                "expires_at": "2318-01-01T00:00:00Z",
                            ]) { $1 }
                        } : []
                ], client: client)
        default: try response(["models": models], client: client)
        }
    }
}

/// A request target's query, read as a web form encodes it (the dashboard's `URLSearchParams`): `+` is a space and a
/// literal plus arrives as %2B; otherwise a folder such as "Application Support" would arrive as "Application+Support".
public func formQueryItems(_ target: String) -> [URLQueryItem] {
    URLComponents(string: target.replacingOccurrences(of: "+", with: "%20"))?.queryItems ?? []
}
