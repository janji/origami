import Darwin
import Foundation
import Metal
import OrigamiEngine

private func phaseRequire(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    if try !condition() { throw EngineError("Family phase upgrade: " + message) }
}

/// File I/O and shared host buffers only. Cancellation before work preserves an existing state; cancellation
/// between 64 MiB chunks or before publication removes staging files; a cancelled restore leaves the disk intact.
func familyStateChunkCancellation() throws {
    guard let device = MTLCreateSystemDefaultDevice(),
        let buffer = device.makeBuffer(length: (64 << 20) + 32, options: .storageModeShared)
    else {
        throw FixtureUnavailable(description: "no Metal shared buffer")
    }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-state-cancel-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let parts = [(buffer: buffer, bytes: buffer.length)]
    let tokens = [1, 4, 9]
    let logits: [Float] = [0, -0.0, 1.5]
    let staging = root.appendingPathComponent(".state.bin.staging")
    let stagingHeader = root.appendingPathComponent(".state.json.staging")
    memset(buffer.contents(), 0x37, buffer.length)
    var checks = 0
    var observedBytes = -1
    do {
        try FamilyStateFile.write(parts, logits: logits, tokens: tokens, epoch: "fixture", to: root) {
            checks += 1
            if checks == 4 {  // initial check, first chunk, first write, then next chunk
                observedBytes = (try FileManager.default.attributesOfItem(atPath: staging.path)[.size] as? NSNumber)?.intValue ?? -1
                throw IOOperationCancelled()
            }
        }
        throw EngineError("Family phase upgrade: interrupted save completed")
    } catch is IOOperationCancelled {}
    try phaseRequire(observedBytes == 64 << 20, "cancel did not occur between payload chunks")
    try phaseRequire(
        !FileManager.default.fileExists(atPath: staging.path)
            && !FileManager.default.fileExists(atPath: FamilyStateFile.headerURL(root).path), "partial save was published or retained")

    try FamilyStateFile.write(parts, logits: logits, tokens: tokens, epoch: "fixture", to: root)
    let header = try FamilyStateFile.header(root, epoch: "fixture", prompt: tokens)
    do {
        try FamilyStateFile.write(parts, logits: nil, tokens: [2], epoch: "new", to: root) { throw IOOperationCancelled() }
        throw EngineError("Family phase upgrade: pre-cancelled save completed")
    } catch is IOOperationCancelled {}
    try phaseRequire(try FamilyStateFile.header(root, epoch: "fixture", prompt: tokens) == header, "pre-cancel removed a valid state")

    memset(buffer.contents(), 0xA5, buffer.length)
    checks = 0
    do {
        _ = try FamilyStateFile.read(root, header: header, into: parts) {
            checks += 1
            if checks == 3 { throw IOOperationCancelled() }
        }
        throw EngineError("Family phase upgrade: interrupted restore completed")
    } catch is IOOperationCancelled {}
    let bytes = buffer.contents().assumingMemoryBound(to: UInt8.self)
    try phaseRequire(bytes[0] == 0x37 && bytes[64 << 20] == 0xA5, "restore did not stop at its next chunk")
    let restored = try FamilyStateFile.read(root, header: header, into: parts)
    try phaseRequire(
        restored?.map(\.bitPattern) == logits.map(\.bitPattern)
            && bytes[buffer.length - 1] == 0x37, "cancelled restore damaged the disk state")

    do {
        try FamilyStateFile.write([], logits: [3], tokens: [2], epoch: "new", to: root) {
            if FileManager.default.fileExists(atPath: stagingHeader.path) { throw IOOperationCancelled() }
        }
        throw EngineError("Family phase upgrade: publication-cancelled save completed")
    } catch is IOOperationCancelled {}
    try phaseRequire(
        !FileManager.default.fileExists(atPath: staging.path)
            && !FileManager.default.fileExists(atPath: stagingHeader.path)
            && FileManager.default.fileExists(atPath: FamilyStateFile.headerURL(root).path),
        "cancel before publication lost the prior header or retained staging")
    try phaseRequire(
        try FamilyStateFile.header(root, epoch: "fixture", prompt: tokens) == header,
        "cancel before publication replaced the prior header")
    try phaseRequire(
        try FamilyStateFile.read(root, header: header, into: parts)?.map(\.bitPattern) == logits.map(\.bitPattern),
        "cancel before publication replaced the prior payload")
}

/// A failed replacement must preserve both the cache index and its payload, not only the lower-level file header.
/// An immutable existing header exercises a real unlink failure before payload replacement.
func familyStateReplacementPreservesPrior() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-state-replace-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let ledger = try MemoryLedger(ceiling: 1 << 30, transientMargin: 1, availableFloor: 1)
    let cache = try PromptCache(
        root: root, epoch: "fixture", entries: 1, systemEntries: 0,
        entryBytes: PromptCache.entryDiskBytes(context: 8192) * 2,
        ledger: ledger, capacity: 8192, minimumFreeBytes: 1)
    try cache.writeState(tokens: [1], bytes: 4) { folder in
        try FamilyStateFile.write([], logits: [0.5], tokens: [1], epoch: "fixture", to: folder)
    }
    let folder = cache.stateFolder(0)
    let header = try FamilyStateFile.header(folder, epoch: "fixture", prompt: [1])
    let oldID = cache.entries[0].id
    for cancelled in [false, true] {
        do {
            try cache.writeState(tokens: [2], bytes: 4) { destination in
                // PATCH is a separate HTTP worker; it must not report success and then lose the pin/name when this
                // save publishes its previously captured metadata. GET may still read the complete old entry.
                do {
                    _ = try cache.update(oldID, name: "lost-name", keep: true)
                    throw EngineError("Family phase upgrade: in-flight replacement accepted metadata edit")
                } catch let error as ServingFailure where error.status == 409 {}
                try phaseRequire(try cache.saved(oldID).tokens == [1], "staging hid the previous readable entry")
                try FamilyStateFile.write([], logits: [9], tokens: [2], epoch: "fixture", to: destination) {
                    if FileManager.default.fileExists(atPath: destination.appendingPathComponent(".state.json.staging").path) {
                        if cancelled { throw IOOperationCancelled() }
                        throw EngineError("Injected staging failure", code: "storage_unavailable")
                    }
                }
            }
            throw EngineError("Family phase upgrade: failed replacement completed")
        } catch is IOOperationCancelled {
            try phaseRequire(cancelled, "unexpected cancellation")
        } catch let error as EngineError where error.code == "storage_unavailable" {
            try phaseRequire(!cancelled, "unexpected staging error")
        }
        try phaseRequire(
            cache.contains([Int64(1)][...]) && !cache.contains([Int64(2)][...]), "replacement failure lost or changed the index")
        try phaseRequire(try FamilyStateFile.read(folder, header: header, into: []) == [0.5], "replacement failure lost prior bytes")
        try phaseRequire(try cache.update(oldID, name: "available", keep: false).name == "available", "failed staging kept metadata locked")
    }
    do {
        try cache.writeState(tokens: [2], bytes: UInt64.max) { _ in throw EngineError("Space refusal called writer") }
        throw EngineError("Family phase upgrade: impossible staging allowance succeeded")
    } catch let error as EngineError {
        try phaseRequire(
            error.description.contains("disk allowance overflow") || error.description.contains("Too little free disk"),
            "unexpected space refusal: \(error)")
    }
    try phaseRequire(cache.contains([Int64(1)][...]), "space refusal evicted the replacement's prior state")

    let headerURL = FamilyStateFile.headerURL(folder)
    guard chflags(headerURL.path, UInt32(UF_IMMUTABLE)) == 0 else { throw EngineError("Cannot mark fixture header immutable: \(errno)") }
    defer { _ = chflags(headerURL.path, 0) }
    do {
        try cache.writeState(tokens: [2], bytes: 4) { destination in
            try FamilyStateFile.write([], logits: [9], tokens: [2], epoch: "fixture", to: destination)
        }
        throw EngineError("Family phase upgrade: immutable header was replaced")
    } catch let error as EngineError where error.code == "storage_unavailable" {}
    try phaseRequire(
        try FamilyStateFile.header(folder, epoch: "fixture", prompt: [1]) == header,
        "failed unlink replaced an immutable header")
    try phaseRequire(
        try FamilyStateFile.read(folder, header: header, into: []) == [0.5]
            && cache.contains([Int64(1)][...]), "failed unlink replaced prior payload/index")
}

/// Tiny malformed JSON headers must fail before allocation or writes into caller state. No huge buffers/files are
/// created: declared huge logits are deliberately paired with the same twelve-byte valid payload.
func familyStateRejectsMalformedSizes() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-state-sizes-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    try FamilyStateFile.write([], logits: [1, 2, 3], tokens: [1], epoch: "fixture", to: root)
    let good = try FamilyStateFile.header(root, epoch: "fixture", prompt: [1])
    let original = try JSONSerialization.jsonObject(with: Data(contentsOf: FamilyStateFile.headerURL(root))) as! [String: Any]
    func header(_ logits: Int, parts: [Int] = []) throws -> FamilyStateFile.Header {
        var object = original
        object["logits"] = logits
        object["parts"] = parts
        return try JSONDecoder().decode(FamilyStateFile.Header.self, from: JSONSerialization.data(withJSONObject: object))
    }
    func reject(_ body: () throws -> Void) throws {
        do { try body() } catch let error as EngineError where error.code == "prompt_cache_mismatch" { return }
        throw EngineError("Family phase upgrade: malformed checkpoint accepted")
    }
    for count in [-1, Int.max, Int.max / 4, 4, 0] {
        try reject { _ = try FamilyStateFile.read(root, header: header(count), into: []) }
    }
    try reject { _ = try FamilyStateFile.read(root, header: good, into: [], expectedLogits: 4) }
    try phaseRequire(
        try FamilyStateFile.read(root, header: good, into: [], expectedLogits: 3) == [1, 2, 3], "valid vocabulary/payload refused")
    guard let device = MTLCreateSystemDefaultDevice(), let buffer = device.makeBuffer(length: 16, options: .storageModeShared) else {
        throw FixtureUnavailable(description: "no Metal shared buffer")
    }
    memset(buffer.contents(), 0xA5, 16)
    try reject { _ = try FamilyStateFile.read(root, header: header(3, parts: [-1]), into: [(buffer, -1)]) }
    try reject { _ = try FamilyStateFile.read(root, header: header(3, parts: [17]), into: [(buffer, 17)]) }
    let file = try FileHandle(forWritingTo: FamilyStateFile.dataURL(root))
    try file.truncate(atOffset: 0)
    try file.close()
    try reject { _ = try FamilyStateFile.read(root, header: header(0, parts: [16]), into: [(buffer, 16)]) }
    try phaseRequire(
        (0..<16).allSatisfy { buffer.contents().assumingMemoryBound(to: UInt8.self)[$0] == 0xA5 }, "invalid payload wrote into model state")
    try FamilyStateFile.write([], logits: nil, tokens: [1], epoch: "fixture", to: root)
    let prefix = try FamilyStateFile.header(root, epoch: "fixture", prompt: [1, 2])
    try phaseRequire(
        try FamilyStateFile.read(root, header: prefix, into: [], expectedLogits: 3) == nil, "valid prefix without logits refused")
}

/// Stale family staging must be reclaimed under the same store lock used by the writer, before checking free
/// space. The fake capacity changes only when the real orphan files disappear; no large disk allocation is used.
func familyStateOrphanRecovery() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-state-orphan-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let ledger = try MemoryLedger(ceiling: 1 << 30, transientMargin: 1, availableFloor: 1)
    let cache = try PromptCache(
        root: root, epoch: "fixture", entries: 1, systemEntries: 0,
        entryBytes: PromptCache.entryDiskBytes(context: 8192) * 2,
        ledger: ledger, capacity: 8192, minimumFreeBytes: 1)
    try cache.writeState(tokens: [1], bytes: 4) {
        try FamilyStateFile.write([], logits: [0.5], tokens: [1], epoch: "fixture", to: $0)
    }
    let folder = cache.stateFolder(0)
    let staging = folder.appendingPathComponent(".state.bin.staging")
    let stagingHeader = folder.appendingPathComponent(".state.json.staging")
    let unrelated = folder.appendingPathComponent("keep-me")
    try Data(repeating: 1, count: 32).write(to: staging)
    try Data("incomplete".utf8).write(to: stagingHeader)
    try Data("unrelated".utf8).write(to: unrelated)
    cache.freeBytesForChecks = {
        FileManager.default.fileExists(atPath: staging.path) || FileManager.default.fileExists(atPath: stagingHeader.path) ? 0 : 1000
    }
    try cache.writeState(tokens: [2], bytes: 4) { destination in
        try phaseRequire(cache.contains([Int64(1)][...]), "orphan recovery evicted the old entry")
        try FamilyStateFile.write([], logits: [9], tokens: [2], epoch: "fixture", to: destination)
    }
    let header = try FamilyStateFile.header(folder, epoch: "fixture", prompt: [2])
    try phaseRequire(
        try FamilyStateFile.read(folder, header: header, into: []) == [9]
            && cache.contains([Int64(2)][...]), "replacement after orphan recovery failed")
    try phaseRequire(try Data(contentsOf: unrelated) == Data("unrelated".utf8), "orphan cleanup removed an unrelated file")

    let outside = root.appendingPathComponent("outside-payload")
    try Data("do not truncate".utf8).write(to: outside)
    try FileManager.default.createSymbolicLink(at: staging, withDestinationURL: outside)
    do {
        try cache.writeState(tokens: [3], bytes: 4) {
            try FamilyStateFile.write([], logits: [10], tokens: [3], epoch: "fixture", to: $0)
        }
        throw EngineError("Family phase upgrade: symlink staging was followed")
    } catch let error as EngineError where error.code == "storage_unavailable" {}
    try phaseRequire(
        try Data(contentsOf: outside) == Data("do not truncate".utf8)
            && cache.contains([Int64(2)][...]), "unexpected staging object changed data or index")
}
