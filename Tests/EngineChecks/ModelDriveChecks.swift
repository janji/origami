import Darwin
import Foundation
import Metal
import OrigamiEngine

func modelDriveChecks() -> [(String, () throws -> Void)] {
    [
        ("model_drive_returns_family", { for family in [ModelFamily.glm, .deepseek] { try familyDriveReturns(family) } }),
        ("model_drive_returns_qwen", qwenDriveReturns),
        ("model_drive_swap_never_mixes", modelDriveSwapNeverMixes),
    ]
}

private func require(_ condition: Bool, _ what: @autoclosure () -> String) throws {
    if !condition { throw EngineError("model drive: " + what()) }
}
private func waitFor(_ what: @autoclosure () -> String, seconds: Double = 10, _ done: () -> Bool) throws {
    let until = Date().addingTimeInterval(seconds)
    while !done() {
        guard Date() < until else { throw EngineError("model drive: \(what())") }
        usleep(2_000)
    }
}
private func withRetry<T>(_ seconds: Double, _ body: () throws -> T) rethrows -> T {
    let saved = ModelDrive.retrySeconds
    ModelDrive.retrySeconds = seconds
    defer { ModelDrive.retrySeconds = saved }
    return try body()
}

/// Puts a different file at `url` in place: the same inode, size and modification time, other bytes in the sample a
/// reopen compares. Returns what undoes it.
private func swapContent(_ url: URL) throws -> () throws -> Void {
    var info = stat()
    guard stat(url.path, &info) == 0 else { throw EngineError("model drive: no \(url.path)") }
    let (offset, length) = ModelDrive.sample(size: UInt64(info.st_size))
    let handle = try FileHandle(forUpdating: url)
    try handle.seek(toOffset: UInt64(offset))
    let original = try handle.read(upToCount: length) ?? Data()
    try handle.seek(toOffset: UInt64(offset))
    try handle.write(contentsOf: Data(original.map { ~$0 }))
    try handle.close()
    func restoreTimes() throws {
        var times = [info.st_atimespec, info.st_mtimespec]
        guard utimensat(AT_FDCWD, url.path, &times, 0) == 0 else { throw EngineError("model drive: utimensat") }
    }
    try restoreTimes()
    return {
        let handle = try FileHandle(forUpdating: url)
        try handle.seek(toOffset: UInt64(offset))
        try handle.write(contentsOf: original)
        try handle.close()
        try restoreTimes()
    }
}

/// GLM and DeepSeek on the synthetic GGUF: the drive leaves after a request; the next request fails at once with the
/// drive error (status degraded); a different file at the path is refused and the drive stays away, saying so; with the
/// model's own file back, the next request gives the bits a run that never lost the drive gives, with the same model
/// (no reload) and the old descriptors closed.
private func familyDriveReturns(_ family: ModelFamily) throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let fixture = try SyntheticFixture(family, device: device)
    defer { fixture.remove() }
    let first: [String: Any] = ["messages": [["role": "user", "content": fixture.opening]], "max_tokens": 6]
    let second: [String: Any] = ["messages": [["role": "user", "content": fixture.other]], "max_tokens": 6]
    let clean = try autoreleasepool { () throws -> (reply: [Int], state: String) in
        let server = try fixture.server()
        _ = try server.send(first)
        let stats = try server.send(second)
        return (stats.outputTokens ?? [], sha256(server.model))
    }
    let drive = fixture.weights.drive
    try withRetry(0.05) {
        try autoreleasepool {
            let server = try fixture.server()
            _ = try server.send(first)
            let open = drive.openDescriptors
            drive.unplug()
            func failsAtOnce(_ what: String) throws {
                let started = Date()
                do {
                    _ = try server.send(second)
                    throw EngineError("model drive: a request read a gone drive (\(what))")
                } catch  where ServingFailure.from(error).code == "drive_stalled" && ServingFailure.from(error).description == drive.note {}
                try require(Date().timeIntervalSince(started) < 3, "\(what): the failure took \(Date().timeIntervalSince(started)) s")
                try require(ModelDrive.away === drive && drive.note != nil, "\(what): the drive is not shown away")
            }
            try failsAtOnce("gone")
            let undo = try swapContent(fixture.weights.gguf.url(fixture.weights.gguf.files[0]))
            drive.plug()
            try waitFor("a different file was not refused") { drive.note?.contains("A different file") == true }
            try require(drive.isAway && drive.note!.contains("content"), "the refusal: \(drive.note ?? "none")")
            try failsAtOnce("a different file")
            try undo()
            try waitFor("the drive did not come back") { !drive.isAway }
            try require(ModelDrive.away == nil, "status still degraded")
            let after = try server.send(second)
            try require(
                after.outputTokens ?? [] == clean.reply && sha256(server.model) == clean.state,
                "the request after the return differs from a clean run")
            try require(drive.openDescriptors == open, "old descriptors left open: \(drive.openDescriptors) of \(open)")
        }
    }
}

/// Qwen's reads (`PackStore`) on a span checkpoint with a hot copy holding one record: while the drive is gone the
/// copy serves its record and every other record fails at once with the drive error; a different file is refused;
/// once the file is back every record reads its exact bytes from the same store.
private func qwenDriveReturns() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-drive-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let model = try qwenSpanModel(root.appendingPathComponent("model"))
    let copy = try HotCopy.write(source: try PackStore(model: model), keys: [1], folder: root.appendingPathComponent("hot"), identity: "q")
    let store = try PackStore(model: model, log: { _ in })
    store.attachHotCopy(copy)
    func read(_ expert: Int) throws -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: PackStore.expertBytes)
        try bytes.withUnsafeMutableBytes { try store.readInternalExpert(ExpertKey(layer: 0, expert: expert), into: $0) }
        return bytes
    }
    func failsAtOnce(_ what: String) throws {
        let started = Date()
        do {
            _ = try read(2)
            throw EngineError("model drive: Qwen read a gone drive (\(what))")
        } catch  where ServingFailure.from(error).code == "drive_stalled" && ServingFailure.from(error).description == store.drive.note {}
        try require(Date().timeIntervalSince(started) < 1, "\(what): Qwen's failure took \(Date().timeIntervalSince(started)) s")
        try require(ModelDrive.away === store.drive, "\(what): Qwen's drive is not shown away")
    }
    try withRetry(0.05) {
        for expert in 0..<4 { try require(try read(expert) == qwenRecord(expert), "Qwen record \(expert)") }
        store.drive.unplug()
        try failsAtOnce("gone")
        try require(try read(1) == qwenRecord(1), "the hot copy stopped serving while the drive was gone")
        try failsAtOnce("after a hot read")
        let undo = try swapContent(model.directory.appendingPathComponent("model.bin"))
        store.drive.plug()
        try waitFor("Qwen: a different file was not refused") { store.drive.note?.contains("A different file") == true }
        try failsAtOnce("a different file")
        try undo()
        try waitFor("Qwen: the drive did not come back") { !store.drive.isAway }
        for expert in 0..<4 { try require(try read(expert) == qwenRecord(expert), "Qwen record \(expert) after the return") }
        try require(store.drive.openDescriptors == 1, "Qwen: old descriptors left open: \(store.drive.openDescriptors)")
    }
}

/// Six readers hammer a file while its drive leaves and returns 100 times, each time with other bytes at the path
/// (accepted for this check): every read that returns holds one descriptor's bytes, never a mix; reads in flight at a
/// swap read again on the new descriptor; nothing else fails; the old descriptors are all closed at the end.
private func modelDriveSwapNeverMixes() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-drive-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let url = root.appendingPathComponent("model.bin")
    // Records long enough that readers are still in their read when the swap lands.
    let record = 1 << 20
    let records = 4
    func write(_ value: UInt8) throws {
        // A new file renamed over the path: the old descriptor keeps reading the old bytes.
        let next = root.appendingPathComponent("next")
        try Data(repeating: value, count: record * records).write(to: next)
        guard rename(next.path, url.path) == 0 else { throw EngineError("model drive: rename") }
    }
    try write(1)
    let drive = ModelDrive(log: { _ in })
    drive.acceptAny = true
    let file = try ExactFile(url: url, drive: drive)
    final class Tally: @unchecked Sendable {
        let lock = NSLock()
        var stop = false, reads = 0, mixed = 0, other: [String] = []
    }
    let tally = Tally()
    // One reader per swap is held after its read on the old descriptor ended, until the swap is done: it then sees its
    // descriptor gone and must read the whole range again on the new one.
    final class Hold: @unchecked Sendable {
        let condition = NSCondition()
        /// Round `armed` holds the next reader; rounds up to `released` go on.
        var armed = 0, held = 0, released = 0
        func enter() {
            condition.lock()
            defer { condition.unlock() }
            guard armed > held else { return }
            held = armed
            let round = armed
            while released < round { condition.wait() }
        }
    }
    let hold = Hold()
    drive.readHook = { hold.enter() }
    let readers = DispatchGroup()
    try withRetry(0) {
        for n in 0..<6 {
            readers.enter()
            Thread.detachNewThread {
                defer { readers.leave() }
                var buffer = [UInt8](repeating: 0, count: record)
                var at = n
                while !tally.lock.withLock({ tally.stop }) {
                    at = (at &* 7 &+ 3) % records
                    do {
                        try buffer.withUnsafeMutableBytes { try file.read(offset: Int64(at * record), into: $0) }
                        let mixed = buffer.contains { $0 != buffer[0] }
                        tally.lock.withLock {
                            tally.reads += 1
                            if mixed { tally.mixed += 1 }
                        }
                    } catch let error as EngineError where error.code == "drive_away" {
                    } catch {
                        tally.lock.withLock { tally.other.append("\(error)") }
                    }
                }
            }
        }
        defer {
            tally.lock.withLock { tally.stop = true }
            _ = readers.wait(timeout: .now() + 10)
        }
        for swap in 0..<100 {
            let generation = drive.generation
            hold.condition.withLock { hold.armed = swap + 1 }
            try waitFor("no reader was held at swap \(swap)") { hold.condition.withLock { hold.held == swap + 1 } }
            drive.unplug()
            try write(UInt8(swap + 2))
            drive.plug()
            try waitFor(
                "swap \(swap) did not happen: generation \(drive.generation) of \(generation + 1), away \(drive.isAway), retries \(drive.retries), reads \(tally.lock.withLock { tally.reads }), \(drive.note ?? "")"
            ) { drive.generation == generation + 1 && !drive.isAway }
            hold.condition.withLock {
                hold.released = swap + 1
                hold.condition.broadcast()
            }
        }
        try waitFor("reads after the last swap") { tally.lock.withLock { tally.reads } > 0 }
    }
    drive.readHook = nil
    var last = [UInt8](repeating: 0, count: record)
    try last.withUnsafeMutableBytes { try file.read(offset: 0, into: $0) }
    try tally.lock.withLock {
        try require(tally.mixed == 0, "\(tally.mixed) of \(tally.reads) reads mixed two descriptors' bytes")
        try require(tally.other.isEmpty, "reads failed otherwise: \(tally.other.prefix(3))")
        try require(tally.reads > 100, "only \(tally.reads) reads")
    }
    try require(last.allSatisfy { $0 == 101 }, "the last swap's bytes are not read")
    // A held reader already on its one retry fails instead (once only), so not quite every swap counts one.
    try require(drive.retries >= 50, "reads in flight at a swap were not read again (\(drive.retries) retries)")
    try require(drive.openDescriptors == 1, "old descriptors left open: \(drive.openDescriptors)")
}
