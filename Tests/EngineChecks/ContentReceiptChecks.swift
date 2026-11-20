import CryptoKit
import Darwin
import Foundation
import OrigamiEngine

private func receiptRequire(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    if try !condition() { throw EngineError("Content receipt: " + message) }
}

private func withReceiptFixture(_ body: (URL, URL, VerifiedContentReceipt.Manifest) throws -> Void) throws {
    let home = FileManager.default.temporaryDirectory.appendingPathComponent("origami-content-check-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: home) }
    let root = home.appendingPathComponent("weights")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let payloads: [(String, Data)] = [
        ("first.bin", Data("first-shard".utf8)), ("second.bin", Data("second-shard".utf8)), ("empty.bin", Data()),
    ]
    let files = try payloads.map { name, data in
        try data.write(to: root.appendingPathComponent(name))
        return VerifiedContentReceipt.File(
            path: name, size: UInt64(data.count),
            sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
    }
    try body(root, home.appendingPathComponent("receipt.json"), .init(files: files))
}

private func receiptRejects(_ message: String, _ body: () throws -> Void) throws {
    do { try body() } catch { return }
    throw EngineError("Content receipt accepted " + message)
}

/// Portable identity is based on actual verified ordered bytes, not root/stat declarations or JSON formatting.
func contentReceiptRoundTrip() throws {
    try withReceiptFixture { root, output, manifest in
        let receipt = try VerifiedContentReceipt.verify(manifest, root: root, writingTo: output, chunkBytes: 4)
        try VerifiedContentReceipt.check(VerifiedContentReceipt.readReceipt(output))
        let reversed = VerifiedContentReceipt.Manifest(files: manifest.files.reversed())
        try receiptRequire(receipt.contentIdentity != VerifiedContentReceipt.identity(reversed), "shard order did not distinguish identity")
        let copy = root.deletingLastPathComponent().appendingPathComponent("copied")
        try FileManager.default.copyItem(at: root, to: copy)
        let imported = try VerifiedContentReceipt.verify(
            VerifiedContentReceipt.readReceipt(output).manifest,
            root: copy, writingTo: output.deletingLastPathComponent().appendingPathComponent("imported.json"))
        try receiptRequire(
            imported.contentIdentity == receipt.contentIdentity && imported.root != receipt.root, "portable import changed identity")
        try VerifiedContentReceipt.check(imported)
        let attributes = try FileManager.default.attributesOfItem(atPath: output.path)
        try receiptRequire((attributes[.posixPermissions] as? NSNumber)?.intValue == 0o600, "receipt permissions are not owner-only")
        let contents = try Data(contentsOf: output)
        try receiptRequire(!contents.isEmpty, "empty published receipt")
    }
}

func contentReceiptRejectsChangedIncarnations() throws {
    try withReceiptFixture { root, output, manifest in
        let receipt = try VerifiedContentReceipt.verify(manifest, root: root, writingTo: output)
        let file = root.appendingPathComponent("first.bin")
        let old = try FileManager.default.attributesOfItem(atPath: file.path)[.modificationDate]!
        let handle = try FileHandle(forWritingTo: file)
        try handle.write(contentsOf: Data("FIRST-shard".utf8))
        try handle.synchronize()
        try handle.close()
        try FileManager.default.setAttributes([.modificationDate: old], ofItemAtPath: file.path)
        try receiptRejects("a same-size rewrite with restored mtime") { try VerifiedContentReceipt.check(receipt) }
        try receiptRejects("declared digest without matching bytes") {
            try VerifiedContentReceipt.verify(manifest, root: root, writingTo: output)
        }
    }
    try withReceiptFixture { root, output, manifest in
        let receipt = try VerifiedContentReceipt.verify(manifest, root: root, writingTo: output)
        let file = root.appendingPathComponent("first.bin")
        let replacement = root.appendingPathComponent("replacement")
        let old = try FileManager.default.attributesOfItem(atPath: file.path)[.modificationDate]!
        try Data("first-shard".utf8).write(to: replacement)
        guard rename(replacement.path, file.path) == 0 else { throw EngineError("Receipt replacement fixture failed") }
        try FileManager.default.setAttributes([.modificationDate: old], ofItemAtPath: file.path)
        try receiptRejects("a replacement inode with identical bytes and restored mtime") { try VerifiedContentReceipt.check(receipt) }
        let fresh = try VerifiedContentReceipt.verify(manifest, root: root, writingTo: output)
        try receiptRequire(fresh.contentIdentity == receipt.contentIdentity, "reverified same content changed identity")
    }
    try withReceiptFixture { root, output, manifest in
        let alias = root.appendingPathComponent("alias.bin")
        let original = root.appendingPathComponent("first.bin")
        let same = root.appendingPathComponent("same.bin")
        try FileManager.default.copyItem(at: original, to: same)
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: original)
        let expected = manifest.files[0]
        let linked = VerifiedContentReceipt.Manifest(files: [.init(path: "alias.bin", size: expected.size, sha256: expected.sha256)])
        let receipt = try VerifiedContentReceipt.verify(linked, root: root, writingTo: output)
        try FileManager.default.removeItem(at: alias)
        try FileManager.default.createSymbolicLink(at: alias, withDestinationURL: same)
        try receiptRejects("a retargeted symlink") { try VerifiedContentReceipt.check(receipt) }
    }
}

func contentReceiptRejectsRacingMutation() throws {
    try withReceiptFixture { root, output, manifest in
        try VerifiedContentReceipt.verify(manifest, root: root, writingTo: output)
        let previous = try Data(contentsOf: output)
        var changed = false
        try receiptRejects("mutation during a streamed hash") {
            try VerifiedContentReceipt.verify(
                manifest, root: root, writingTo: output, chunkBytes: 4,
                progress: { index, _ in
                    guard index == 0, !changed else { return }
                    changed = true
                    let file = root.appendingPathComponent("first.bin")
                    let old = try FileManager.default.attributesOfItem(atPath: file.path)[.modificationDate]!
                    let handle = try FileHandle(forWritingTo: file)
                    try handle.write(contentsOf: Data("FIRST-shard".utf8))
                    try handle.close()
                    try FileManager.default.setAttributes([.modificationDate: old], ofItemAtPath: file.path)
                })
        }
        try receiptRequire(changed && Data(contentsOf: output) == previous, "failed verification replaced the old receipt")
    }
    try withReceiptFixture { root, output, manifest in
        var changed = false
        try receiptRejects("an earlier shard changing while a later shard is read") {
            try VerifiedContentReceipt.verify(
                manifest, root: root, writingTo: output, chunkBytes: 4,
                progress: { index, _ in
                    guard index == 1, !changed else { return }
                    changed = true
                    try Data("FIRST-shard".utf8).write(to: root.appendingPathComponent("first.bin"))
                })
        }
        try receiptRequire(changed && !FileManager.default.fileExists(atPath: output.path), "cross-shard mutation published a receipt")
    }
}

func contentReceiptInterruptedPublication() throws {
    enum Interrupted: Error { case stop }
    try withReceiptFixture { root, output, manifest in
        try VerifiedContentReceipt.verify(manifest, root: root, writingTo: output)
        let previous = try Data(contentsOf: output)
        var interrupted = false
        try receiptRejects("interrupted hashing") {
            try VerifiedContentReceipt.verify(
                manifest, root: root, writingTo: output, chunkBytes: 4,
                progress: { _, _ in
                    interrupted = true
                    throw Interrupted.stop
                })
        }
        try receiptRequire(interrupted && Data(contentsOf: output) == previous, "interrupted hashing damaged receipt")
        var staged = false
        try receiptRejects("interruption after the staging write") {
            try VerifiedContentReceipt.verify(
                manifest, root: root, writingTo: output,
                checkpoint: {
                    let names = try FileManager.default.contentsOfDirectory(atPath: output.deletingLastPathComponent().path)
                    if names.contains(where: { $0.hasPrefix(".origami-content-") }) {
                        staged = true
                        throw Interrupted.stop
                    }
                })
        }
        try receiptRequire(staged && Data(contentsOf: output) == previous, "interrupted publication damaged receipt")
        let names = try FileManager.default.contentsOfDirectory(atPath: output.deletingLastPathComponent().path)
        try receiptRequire(!names.contains(where: { $0.hasPrefix(".origami-content-") }), "handled interruption leaked staging")
        try VerifiedContentReceipt.check(VerifiedContentReceipt.readReceipt(output))
    }
}

func contentReceiptRejectsMalformedInput() throws {
    try withReceiptFixture { root, output, manifest in
        let first = manifest.files[0]
        for name in ["", "/first.bin", "../first.bin", "./first.bin", "x//first.bin", "first.bin/", "nul\0.bin"] {
            try receiptRejects("unsafe relative path") {
                _ = try VerifiedContentReceipt.identity(.init(files: [.init(path: name, size: first.size, sha256: first.sha256)]))
            }
        }
        for digest in ["", "ab", String(repeating: "g", count: 64), first.sha256.uppercased()] {
            try receiptRejects("malformed digest") {
                _ = try VerifiedContentReceipt.identity(.init(files: [.init(path: first.path, size: first.size, sha256: digest)]))
            }
        }
        try receiptRejects("duplicate file paths") { _ = try VerifiedContentReceipt.identity(.init(files: [first, first])) }
        try receiptRejects("empty manifest") { _ = try VerifiedContentReceipt.identity(.init(files: [])) }
        try receiptRejects("overflowing total byte count") {
            _ = try VerifiedContentReceipt.identity(
                .init(files: [.init(path: "a", size: .max, sha256: first.sha256), .init(path: "b", size: 1, sha256: first.sha256)]))
        }
        let malformed = root.appendingPathComponent("bad.json")
        try Data("{\"schemaVersion\":2,\"algorithm\":\"sha256\",\"files\":[]}".utf8).write(to: malformed)
        try receiptRejects("unknown manifest schema") { _ = try VerifiedContentReceipt.readManifest(malformed) }
        let arbitrary = Data("keep this file".utf8)
        try arbitrary.write(to: output)
        try receiptRejects("overwriting an unrelated file") { try VerifiedContentReceipt.verify(manifest, root: root, writingTo: output) }
        try receiptRequire(Data(contentsOf: output) == arbitrary, "unrelated output was overwritten")
        let payload = root.appendingPathComponent(first.path)
        let before = try Data(contentsOf: payload)
        try receiptRejects("overwriting a payload") { try VerifiedContentReceipt.verify(manifest, root: root, writingTo: payload) }
        try receiptRequire(Data(contentsOf: payload) == before, "payload was overwritten")
        let link = root.deletingLastPathComponent().appendingPathComponent("hard-link")
        try FileManager.default.linkItem(at: payload, to: link)
        try receiptRejects("overwriting a payload hard link") { try VerifiedContentReceipt.verify(manifest, root: root, writingTo: link) }
        try receiptRequire(Data(contentsOf: link) == before, "payload hard link was overwritten")
    }
}
