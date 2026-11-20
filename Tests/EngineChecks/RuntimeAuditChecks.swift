import Darwin
import Foundation
import Metal
import OrigamiEngine

private func auditRequire(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
    if try !value() { throw EngineError("Runtime audit regression: " + message) }
}

/// The identities Origami 2.x gave a model (Qwen: config, index and shard sizes; GGUF: part names and sizes), which name
/// the copies it made, are recomputed exactly: the values here come from 2.4.3's formula, computed outside Swift.
func earlierVersionIdentities() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-earlier-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: root.appendingPathComponent("config.json"))
    try Data("abcde".utf8).write(to: root.appendingPathComponent("model.safetensors"))
    try Data("abcde".utf8).write(to: root.appendingPathComponent("m-00001-of-00002.gguf"))
    try Data("abc".utf8).write(to: root.appendingPathComponent("m-00002-of-00002.gguf"))
    try auditRequire(
        ModelSource.earlierFingerprint(directory: root) == "821e50ad228ff65541c91ab5940a210a46ed34c70375349693384edf1ccac12b",
        "a 2.x Qwen fingerprint is not recognised")
    try auditRequire(
        FamilySupport.earlierKey(folder: root, files: ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]) == "80a54b4b87b1dae2",
        "a 2.x GGUF key is not recognised")
}

/// Small fake payloads suffice: the fingerprint-only path must bind files before any tensor headers are parsed.
func modelIncarnationsSeparateSameShapeWeights() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-identity-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let first = root.appendingPathComponent("normal")
    let second = root.appendingPathComponent("other")
    for folder in [first, second] { try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true) }
    var config = ModelContract.geometry.mapValues { $0 as Any }
    config["model_type"] = "qwen4_exp"
    config["quantization"] = ["bits": 4, "group_size": 32]
    let configData = try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
    let file = "model.safetensors"
    let original = Data(repeating: 0x17, count: 64)
    for folder in [first, second] {
        try configData.write(to: folder.appendingPathComponent("config.json"))
        try original.write(to: folder.appendingPathComponent(file))
    }
    let before = try ModelSource.fingerprint(directory: first)
    let familyBefore = FamilySupport.key(folder: first, files: [file])
    try auditRequire(before == ModelSource.fingerprint(directory: first), "unchanged files changed identity")
    try auditRequire(before != ModelSource.fingerprint(directory: second), "different local files aliased")
    try auditRequire(familyBefore != FamilySupport.key(folder: second, files: [file]), "family files aliased")
    let alias = root.appendingPathComponent("alias")
    try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: first)
    try auditRequire(before == ModelSource.fingerprint(directory: alias), "symlink to the same checkpoint changed identity")

    // In-place replacement preserves byte count and restores mtime. ctime/file incarnation must invalidate reuse.
    let source = first.appendingPathComponent(file)
    let oldDate = try FileManager.default.attributesOfItem(atPath: source.path)[.modificationDate]!
    let writer = try FileHandle(forWritingTo: source)
    try writer.write(contentsOf: Data(repeating: 0x29, count: 64))
    try writer.synchronize()
    try writer.close()
    try FileManager.default.setAttributes([.modificationDate: oldDate], ofItemAtPath: source.path)
    try auditRequire(before != ModelSource.fingerprint(directory: first), "same-size rewrite reused Qwen identity")
    try auditRequire(familyBefore != FamilySupport.key(folder: first, files: [file]), "same-size rewrite reused family identity")
    let changed = try ModelSource.fingerprint(directory: first)
    let replacement = first.appendingPathComponent("replacement")
    try original.write(to: replacement)
    guard rename(replacement.path, source.path) == 0 else { throw EngineError("Fixture replacement failed") }
    try FileManager.default.setAttributes([.modificationDate: oldDate], ofItemAtPath: source.path)
    try auditRequire(changed != ModelSource.fingerprint(directory: first), "replacement file reused identity")
}

/// Tiny fake expert bytes and shared buffers only: no inference or GPU command execution.
func familyHotFallbackCountsActualSource() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-hot-fallback-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: root)), device: device)
    let budget = UInt64(4 * (212_992 + 114_688))
    let maker = try FamilyExperts(weights: weights, budget: budget)
    let copy = try HotCopy.write(source: maker, keys: [4, 5, 6, 7], folder: root.appendingPathComponent("hot"), identity: "fixture")
    let control = try FamilyExperts(weights: weights, budget: budget)
    control.attach(copy)
    _ = try control.fetch(layer: 1, experts: [0, 1, 2, 3], decode: true)
    try auditRequire(control.counters.hotReads > 0, "fixture did not exercise hot reads")
    let writer = try FileHandle(forWritingTo: copy.payload)
    try writer.truncate(atOffset: 0)
    try writer.close()
    for ahead in [false, true] {
        let pool = try FamilyExperts(weights: weights, budget: budget)
        pool.attach(copy)
        if ahead {
            pool.readAhead(layer: 1, experts: [0, 1, 2, 3])
            pool.waitForReads()
        }
        let slots = try pool.fetch(layer: 1, experts: [0, 1, 2, 3], decode: !ahead)
        let per = GGUFQuantize.bytes(8, count: 256 * 256)
        for (expert, slot) in slots.enumerated() {
            for part in 0..<3 {
                let bytes = UnsafeRawBufferPointer(start: slot.buffer.contents() + slot.offsets[part], count: per)
                try auditRequire(bytes.allSatisfy { $0 == UInt8(50 + expert * 10 + part) }, "fallback corrupted a record")
            }
        }
        try auditRequire(pool.counters.hotReads == 0 && pool.counters.hotBytes == 0, "fallback counted successful hot bytes")
        try auditRequire(pool.counters.readBytes == 4 * 3 * per, "fallback lost logical bytes")
    }
}
