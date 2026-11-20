import Foundation
import Metal
import OrigamiEngine
import Synchronization

/// A request arriving while the hot copy's keeper builds is admitted at once, not after the build: the keeper finishes
/// the record in hand, then reads and appends nothing while the request runs, and completes the copy after it. The
/// keeper takes the backend lock for one record at a time, back to back; NSLock is not fair, so before `ServingGate`
/// counted arriving requests it starved `prepare` for tens of seconds while the copy grew. `admit` starts a request as
/// the backend's `prepare` does and returns how it ends.
private func keeperYields(
    _ name: String, idle: @escaping () -> Bool, withIdle: @escaping (_ work: () throws -> Void) throws -> Bool,
    admit: () throws -> () throws -> Void
) throws {
    func require(_ ok: Bool, _ message: @autoclosure () -> String) throws {
        if !ok { throw EngineError("\(name) keeper yield: " + message()) }
    }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-keeper-yield-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    final class Slow: HotCopySource, @unchecked Sendable {
        let reads = Atomic<Int>(0)
        func hotCopyRecordBytes(_ key: Int) -> Int? { (0..<200).contains(key) ? 16384 : nil }
        func readHotCopyRecord(_ key: Int, into: UnsafeMutableRawPointer) throws {
            usleep(3000)  // a record read from a model's drive
            memset(into, Int32(key & 0xFF), 16384)
            reads.add(1, ordering: .sequentiallyConsistent)
        }
    }
    let source = Slow()
    let folder = root.appendingPathComponent("hot")
    let keys = Array(0..<200)
    let keeper = HotCopyKeeper(
        source: source, identity: "fixture", copy: nil, counts: ([0: 1], [:]), choose: { _, _ in keys },
        place: { _ in .folder(folder, drive: "fixture", budget: 200 * 16384, free: .max) },
        attach: { _ in (try? withIdle {}) ?? false }, idle: idle, withIdle: withIdle, event: { _ in })
    defer { keeper.stop(waiting: 3) }
    func reads() -> Int { source.reads.load(ordering: .sequentiallyConsistent) }
    func appended() -> UInt64 {
        let files = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        return files.filter { $0.lastPathComponent.hasPrefix("experts") }.reduce(0) {
            $0 + UInt64((try? $1.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
    }
    func waitFor(_ what: String, _ condition: () -> Bool) throws {
        let end = Date().addingTimeInterval(20)
        while !condition() {
            guard Date() < end else { throw EngineError("\(name) keeper yield: \(what) (\(keeper.report), \(reads()) reads)") }
            Thread.sleep(forTimeInterval: 0.005)
        }
    }
    keeper.trigger()
    try waitFor("building") { reads() >= 20 }
    let arrived = Date()
    let end = try admit()
    let waited = Date().timeIntervalSince(arrived)
    try require(waited < 0.5, String(format: "the request waited %.2f s for the keeper", waited))
    let (readsAtStart, bytesAtStart) = (reads(), appended())
    try require(readsAtStart < 200, "the copy finished before the request was admitted")
    try waitFor("paused") { keeper.report["state"] as? String == "paused" }
    Thread.sleep(forTimeInterval: 0.5)
    try require(
        reads() == readsAtStart && appended() == bytesAtStart,
        "the keeper read \(reads() - readsAtStart) records and appended \(appended() - bytesAtStart) bytes during a request")
    try end()
    try waitFor("ready after the request") { keeper.report["state"] as? String == "ready" }
    try require(reads() == 200 && keeper.records == 200, "resumed build: \(reads()) reads, \(keeper.records) records")
}

/// Qwen's backend gate and admission, as `NativeServingBackend` admits a request.
func hotCopyKeeperYieldsToArrivingRequest() throws {
    final class Backend: @unchecked Sendable {
        let gate = ServingGate()
        var admission = ServingAdmission()
    }
    let backend = Backend()
    let gate = backend.gate
    try keeperYields(
        "Qwen", idle: { gate.idle { backend.admission.isIdle } },
        withIdle: { work in try gate.whileIdle({ backend.admission.isIdle }, work) }
    ) {
        let request = try gate.arrive { try gate.lock.withLock { try backend.admission.begin() } }
        return { gate.lock.withLock { backend.admission.retire(request) {} } }
    }
}

/// GLM's and DeepSeek's backend, a real `FamilyServingBackend` over a stub model: its `prepare` and the keeper's hooks
/// as `origami serve` wires them.
func familyHotCopyKeeperYieldsToArrivingRequest() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    final class Stub: FamilyModel {
        var position = 0, contextLimit = 8192, routing = QualityDial.exact
        var shouldContinue: () -> Bool = { true }
        var routedExperts: Int { 6 }
        var vocabulary: Int { 261 }
        var expertUses: [Int: Int] { [:] }
        var expertMisses: [Int: Int] { [:] }
        var timing: FamilyRunner.Timing { .init() }
        func statusNotes() -> [String: Any] { [:] }
        let state: MTLBuffer
        init(_ device: MTLDevice) { state = device.makeBuffer(length: 16, options: .storageModeShared)! }
        func forward(_ tokens: [Int]) throws -> [Float] {
            position += tokens.count
            var logits = [Float](repeating: 0, count: 261)
            logits[72] = 100
            return logits
        }
        func reset() throws { position = 0 }
        func stateParts(at position: Int) -> [(buffer: MTLBuffer, bytes: Int)] { [(state, 16)] }
        func restored(_ tokens: [Int]) { position = tokens.count }
    }
    let backend = FamilyServingBackend(
        model: Stub(device), tokenizer: try tinyFamilyTokenizer("{% for m in messages %}{{ m.content }}{% endfor %}"), stops: [259],
        version: "check", fastAllowed: false)
    let request = try ServingRequest.parse(
        JSONSerialization.data(withJSONObject: [
            "model": "family", "max_tokens": 2, "temperature": 0, "reasoning_effort": "none",
            "messages": [["role": "user", "content": "Hi"]],
        ]), contextTier: 8192)
    try keeperYields("Family", idle: { backend.idle }, withIdle: { work in try backend.withIdleHotCopy(work) }) {
        // The request holds the engine until its prepared generation is released, as the server's does.
        var prepared: (any PreparedGeneration)? = try backend.prepare(request, cancellation: GenerationControl())
        return {
            _ = try prepared?.run { _ in true }
            prepared = nil
        }
    }
}
