import Foundation
import Synchronization

/// Protected by the backend lock. Admission belongs to a request, never to the
/// persistent sequence, whose identity can survive many prepared requests.
package struct ServingAdmission {
    private var active: UUID?
    package init() {}
    package var isIdle: Bool { active == nil }
    package mutating func begin() throws -> UUID {
        guard isIdle else { throw ServingFailure("Engine is busy", status: 503, type: "engine_busy") }
        let token = UUID()
        active = token
        return token
    }
    package mutating func retire(_ token: UUID, cleanup: () -> Void) {
        guard active == token else { return }
        cleanup()
        active = nil
    }
}

/// A backend's lock, and background work's turns at it (the hot copy's records): a request is counted as arriving
/// from before it takes the lock until its `prepare` returns, and background work takes a turn only while no request
/// arrives and the backend says it is idle (`idle`, read under `lock`). Without that count a builder taking the lock
/// back to back starved `prepare` (NSLock is not fair), so a reply waited, and shared the model's drive, while the
/// copy grew.
package final class ServingGate: @unchecked Sendable {
    package let lock = NSLock()
    private let arriving = Atomic<Int>(0)
    package init() {}
    /// `body` (which takes `lock` and may admit the request) as an arriving request.
    package func arrive<T>(_ body: () throws -> T) rethrows -> T {
        arriving.add(1, ordering: .sequentiallyConsistent)
        defer { arriving.subtract(1, ordering: .sequentiallyConsistent) }
        return try body()
    }
    /// No request arrives and `idle` holds (read under `lock`).
    package func idle(_ idle: () -> Bool) -> Bool { lock.withLock { quiet(idle) } }
    /// `work` under `lock` while no request arrives and `idle` holds; false (nothing done) otherwise.
    package func whileIdle(_ idle: () -> Bool, _ work: () throws -> Void) rethrows -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard quiet(idle) else { return false }
        try work()
        return true
    }
    private func quiet(_ idle: () -> Bool) -> Bool { arriving.load(ordering: .sequentiallyConsistent) == 0 && idle() }
}
