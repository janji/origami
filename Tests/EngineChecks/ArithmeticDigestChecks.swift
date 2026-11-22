import CryptoKit
import Foundation
import OrigamiEngine

/// `ArithmeticDigest` against the source tree: the checked-in value must be the digest of the sources it covers now.
func arithmeticDigestChecks() -> [(String, () throws -> Void)] {
    [("arithmetic_digest_current", arithmeticDigestCurrent)]
}

/// `scripts/arithmetic-digest.py`'s digest of the tree at `root`: every .swift, .metal, .c and .h file under an included
/// path and not under an excluded one, in byte order of its path, as path, NUL, hex SHA-256 of its bytes, newline.
func arithmeticDigest(root: URL) throws -> String {
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
    var total = SHA256()
    for file in files.sorted() where !ArithmeticDigest.excluded.contains(where: { file == $0 || file.hasPrefix($0) }) {
        let bytes = try Data(contentsOf: root.appendingPathComponent(file))
        let line = file + "\0" + SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() + "\n"
        total.update(data: Data(line.utf8))
    }
    return total.finalize().map { String(format: "%02x", $0) }.joined()
}

/// Recomputed from the sources (`#filePath`; skipped where they are not there), the digest is the checked-in one. On a
/// copy of the covered sources plus a Serving file and a command-line file, changing either of those leaves the digest
/// as it was, and changing a covered kernel file changes it.
private func arithmeticDigestCurrent() throws {
    let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    let script = root.appendingPathComponent("scripts/arithmetic-digest.py")
    guard FileManager.default.fileExists(atPath: script.path),
        FileManager.default.fileExists(atPath: root.appendingPathComponent("Sources/OrigamiEngine/Qwen").path)
    else { throw FixtureUnavailable(description: "the sources are not at \(root.path)") }
    let current = try arithmeticDigest(root: root)
    guard current == ArithmeticDigest.value else {
        throw EngineError(
            "ArithmeticDigest.value is stale (\(ArithmeticDigest.value.prefix(12)), the sources give \(current.prefix(12))): run "
                + "python3 -B scripts/arithmetic-digest.py, and rebuild")
    }
    let copy = FileManager.default.temporaryDirectory.appendingPathComponent("origami-digest-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: copy) }
    let kernel = "Sources/OrigamiEngine/Qwen/Kernels/ReferenceKernels.swift"
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
    guard try arithmeticDigest(root: copy) == current else { throw EngineError("A copy of the sources gives another digest") }
    for file in outside { try append(file) }
    guard try arithmeticDigest(root: copy) == current else {
        throw EngineError("Changing Serving or the command-line tool changed the digest")
    }
    try append(kernel)
    guard try arithmeticDigest(root: copy) != current else { throw EngineError("Changing a kernel file left the digest as it was") }
}
