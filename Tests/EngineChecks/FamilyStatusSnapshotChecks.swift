import Dispatch
import Foundation
import Metal
import OrigamiEngine

/// Status readers race a mutable fake cache while a real PreparedGeneration holds the backend. No request
/// work or Metal model is needed: a live metadata read would be counted, even when its values happen to match.
func familyStatusBoundarySnapshots() throws {
    final class Stub: FamilyModel, @unchecked Sendable {
        let counterLock = NSLock()
        var active = false, unsafeReads = 0, notesReads = 0, value = 0
        var sequence = [0]
        var position = 0, routing = QualityDial.exact
        var contextLimit: Int { 8192 }
        var shouldContinue: () -> Bool = { true }
        var routedExperts: Int { 8 }
        var vocabulary: Int { 261 }
        var expertUses: [Int: Int] { [:] }
        var expertMisses: [Int: Int] { [:] }
        var timing: FamilyRunner.Timing { .init() }
        var memory: (fixed: UInt64, experts: UInt64, slots: Int, totalExperts: Int, maximumContext: Int) {
            counterLock.withLock {
                if active { unsafeReads += 1 }
                return (64, UInt64(value), value, 512, 8192)
            }
        }
        func statusNotes() -> [String: Any] {
            counterLock.withLock {
                notesReads += 1
                if active { unsafeReads += 1 }
                return ["expert_hits": value, "expert_prefetch_issued_by_rank": sequence]
            }
        }
        func mutate(_ next: Int) {
            counterLock.withLock {
                value = next
                sequence[0] = next
            }
        }
        func stateParts(at position: Int) -> [(buffer: MTLBuffer, bytes: Int)] { [] }
        func restored(_ tokens: [Int]) { position = tokens.count }
        func reset() throws { position = 0 }
        func forward(_ tokens: [Int]) throws -> [Float] {
            position += tokens.count
            return [Float](repeating: 0, count: 261)
        }
    }
    final class Results: @unchecked Sendable {
        let lock = NSLock()
        var failures = 0
        func fail() { lock.withLock { failures += 1 } }
    }
    func require(_ value: Bool, _ reason: String) throws { if !value { throw EngineError("Family status snapshot: " + reason) } }
    let stub = Stub()
    let backend = FamilyServingBackend(
        model: stub,
        tokenizer: try tinyFamilyTokenizer("{% for m in messages %}{{ m.content }}{% endfor %}"),
        stops: [259], version: "check", fastAllowed: false)
    try require(backend.telemetrySnapshot.boundary == "load", "missing initial publication")
    let request = try ServingRequest.parse(
        JSONSerialization.data(withJSONObject: [
            "model": "family", "max_tokens": 1, "messages": [["role": "user", "content": "hi"]],
        ]), contextTier: 8192)
    var prepared: (any PreparedGeneration)? = try backend.prepare(request, cancellation: GenerationControl())
    let old = backend.telemetrySnapshot
    let readsBefore = stub.counterLock.withLock { stub.notesReads }
    try require(old.boundary == "request_start", "missing prepared boundary")
    stub.counterLock.withLock { stub.active = true }
    let result = Results()
    withExtendedLifetime(prepared) {
        DispatchQueue.concurrentPerform(iterations: 5) { lane in
            for i in 1...1_000 {
                if lane == 0 {
                    stub.mutate(i)
                    continue
                }
                let snapshot = backend.telemetrySnapshot
                if snapshot.boundary != "request_start" || snapshot.notes["expert_hits"] as? Int != 0
                    || snapshot.notes["expert_prefetch_issued_by_rank"] as? [Int] != [0]
                    || snapshot.memory.experts != 0
                {
                    result.fail()
                }
            }
        }
        _ = backend.trimExperts()  // a busy trim is only a deferred request, never a live snapshot read
    }
    try require(
        result.failures == 0 && stub.counterLock.withLock { stub.unsafeReads == 0 && stub.notesReads == readsBefore },
        "a status reader or busy trim sampled mutable model metadata")
    stub.counterLock.withLock { stub.active = false }
    autoreleasepool { prepared = nil }  // abandoned request uses the same joined end publication
    let ended = backend.telemetrySnapshot
    try require(
        ended.boundary == "request_end" && ended.notes["expert_hits"] as? Int == 1_000
            && ended.notes["expert_prefetch_issued_by_rank"] as? [Int] == [1_000] && ended.memory.experts == 1_000,
        "request completion did not publish the new values")
    try require(old.notes["expert_prefetch_issued_by_rank"] as? [Int] == [0], "published array mutated after escape")
    stub.mutate(2_000)
    try require(backend.telemetrySnapshot.memory.experts == 1_000, "idle getter unexpectedly inspected live model")
    _ = backend.trimExperts()
    try require(
        backend.telemetrySnapshot.boundary == "idle_trim" && backend.telemetrySnapshot.memory.experts == 2_000,
        "idle trim did not refresh telemetry")

    // A stalled copy syscall/publication holds the engine lock. Status must still return the
    // previously published immutable values without waiting for that I/O or reading the model.
    let copying = DispatchSemaphore(value: 0)
    let releaseCopy = DispatchSemaphore(value: 0)
    let copyEnded = DispatchSemaphore(value: 0)
    let statusReturned = DispatchSemaphore(value: 0)
    DispatchQueue.global().async {
        let admitted = backend.withIdleHotCopy {
            copying.signal()
            _ = releaseCopy.wait(timeout: .now() + 10)
        }
        if !admitted { result.fail() }
        copyEnded.signal()
    }
    guard copying.wait(timeout: .now() + 2) == .success else {
        releaseCopy.signal()
        _ = copyEnded.wait(timeout: .now() + 2)
        throw EngineError("Family status snapshot: copy fixture never acquired idle ownership")
    }
    DispatchQueue.global().async {
        let snapshot = backend.telemetrySnapshot
        if snapshot.boundary != "idle_trim" || snapshot.memory.experts != 2_000 { result.fail() }
        statusReturned.signal()
    }
    let responsive = statusReturned.wait(timeout: .now() + 1) == .success
    releaseCopy.signal()
    let joined = copyEnded.wait(timeout: .now() + 2) == .success
    let readerJoined = responsive || statusReturned.wait(timeout: .now() + 2) == .success
    try require(joined && readerJoined && result.failures == 0, "copy/status fixture did not join cleanly")
    try require(responsive, "status waited for hot-copy I/O under the engine lock")

    // Build now can end a completed tuner's idle hold, but never an active request.
    let tuned = try ServingRequest.parse(
        JSONSerialization.data(withJSONObject: [
            "model": "family", "max_tokens": 1, "messages": [["role": "user", "content": "hi"]],
            "origami_mode": "exact", "origami_tune": [String: Any](),
        ]), contextTier: 8192)
    prepared = try backend.prepare(tuned, cancellation: GenerationControl())
    withExtendedLifetime(prepared) {
        if backend.releaseIdleHotCopyHold() || backend.withIdleHotCopy({}) { result.fail() }
    }
    autoreleasepool { prepared = nil }
    try require(!backend.idle && !backend.withIdleHotCopy({}), "ordinary maintenance lost the tuning hold")
    try require(
        backend.releaseIdleHotCopyHold() && backend.idle && backend.withIdleHotCopy({}),
        "explicit idle build could not end the completed tuning hold")
    prepared = try backend.prepare(request, cancellation: GenerationControl())
    withExtendedLifetime(prepared) {
        if backend.releaseIdleHotCopyHold() || backend.withIdleHotCopy({}) { result.fail() }
    }
    autoreleasepool { prepared = nil }
    try require(result.failures == 0 && backend.idle, "explicit build interrupted a live request")
}
