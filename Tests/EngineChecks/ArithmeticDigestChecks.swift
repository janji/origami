import CryptoKit
import Foundation
import OrigamiEngine

/// `ArithmeticDigest` against the source tree: each engine's checked-in value must be the digest of its sources now.
func arithmeticDigestChecks() -> [(String, () throws -> Void)] {
    [("arithmetic_digest_current", arithmeticDigestCurrent)]
}

/// `scripts/arithmetic-digest.py`'s digest of `engine` for the tree at `root`: every .swift, .metal, .c and .h file under
/// an included path, not under an excluded one and not given to other engines alone (`only`), in byte order of its path,
/// as path, NUL, hex SHA-256 of its bytes, newline.
func arithmeticDigest(root: URL, engine: String) throws -> String {
    var files = Set<String>()
    for entry in ArithmeticDigest.included {
        let url = root.appendingPathComponent(entry)
        var directory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &directory) else { continue }
        guard directory.boolValue else {
            files.insert(entry)
            continue
        }
        let base = url.resolvingSymlinksInPath().path
        for case let file as URL in FileManager.default.enumerator(at: url, includingPropertiesForKeys: nil) ?? .init()
        where [".swift", ".metal", ".c", ".h"].contains(where: file.lastPathComponent.hasSuffix) {
            files.insert(entry + file.resolvingSymlinksInPath().path.dropFirst(base.count + 1))
        }
    }
    func under(_ file: String, _ prefix: String) -> Bool { file == prefix || file.hasPrefix(prefix) }
    var total = SHA256()
    for file in files.sorted() where !ArithmeticDigest.excluded.contains(where: { under(file, $0) }) {
        guard ArithmeticDigest.only.first(where: { under(file, $0.0) }).map({ $0.1.contains(engine) }) ?? true else { continue }
        let bytes = try Data(contentsOf: root.appendingPathComponent(file))
        let line = file + "\0" + SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() + "\n"
        total.update(data: Data(line.utf8))
    }
    return total.finalize().map { String(format: "%02x", $0) }.joined()
}

/// Recomputed from the sources (`#filePath`; skipped where they are not there), each engine's digest is the checked-in
/// one. On a copy of the covered sources plus a Serving file and a command-line file: changing either of those leaves
/// every digest as it was; a family-only file moves only the family digest, a Qwen kernel only Qwen's two, the GGUF
/// parser only the GGUF engines' (not Qwen with MLX experts), and a shared file all three.
private func arithmeticDigestCurrent() throws {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let script = root.appendingPathComponent("scripts/arithmetic-digest.py")
    guard FileManager.default.fileExists(atPath: script.path),
        FileManager.default.fileExists(atPath: root.appendingPathComponent("Sources/OrigamiEngine/Qwen").path)
    else { throw FixtureUnavailable(description: "the sources are not at \(root.path)") }
    func digests(_ root: URL) throws -> [String: String] {
        try Dictionary(uniqueKeysWithValues: ArithmeticDigest.all.map { ($0.name, try arithmeticDigest(root: root, engine: $0.name)) })
    }
    let current = try digests(root)
    for engine in ArithmeticDigest.all where current[engine.name] != engine.value {
        throw EngineError(
            "ArithmeticDigest.\(engine.name) is stale (\(engine.value.prefix(12)), the sources give \(current[engine.name]!.prefix(12))): "
                + "run python3 -B scripts/arithmetic-digest.py, and rebuild")
    }
    guard Set(current.values).count == current.count else { throw EngineError("Two engines have the same digest") }
    let copy = FileManager.default.temporaryDirectory.appendingPathComponent("origami-digest-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: copy) }
    let outside = ["Sources/OrigamiEngine/Serving/LocalHTTPServer.swift", "Sources/origami/Commands.swift"]
    for entry in ArithmeticDigest.included + outside {
        let target = copy.appendingPathComponent(entry)
        try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: root.appendingPathComponent(entry), to: target)
    }
    func append(_ file: String) throws {
        let handle = try FileHandle(forWritingTo: copy.appendingPathComponent(file))
        try handle.seekToEnd()
        try handle.write(contentsOf: Data("\n// changed\n".utf8))
        try handle.close()
    }
    guard try digests(copy) == current else { throw EngineError("A copy of the sources gives other digests") }
    for file in outside { try append(file) }
    guard try digests(copy) == current else { throw EngineError("Changing Serving or the command-line tool changed a digest") }
    var expected = current
    let cases: [(String, [String])] = [
        ("Sources/OrigamiEngine/Family/FamilyKernels.swift", ["family"]),
        ("Sources/OrigamiEngine/Qwen/Kernels/ReferenceKernels.swift", ["qwen", "qwen_gguf"]),
        ("Sources/OrigamiEngine/Formats/GGUF.swift", ["qwen_gguf", "family"]),
        ("Sources/OrigamiEngine/Core/BF16.swift", ["qwen", "qwen_gguf", "family"]),
    ]
    for (file, moved) in cases {
        try append(file)
        let now = try digests(copy)
        for engine in now.keys where (now[engine] != expected[engine]) != moved.contains(engine) {
            throw EngineError(
                "Changing \(file) \(moved.contains(engine) ? "left" : "changed") the \(engine) digest"
                    + (moved.contains(engine) ? " as it was" : ""))
        }
        expected = now
    }
}
