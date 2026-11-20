import CryptoKit
import Foundation

/// Opt-in request records (`request_log`). One JSON line per request in `requests-<day>.jsonl` (wire, client family, sizes, cache
/// share, timings, how it finished), and each system prompt and tool list once under `units/<sha256>.txt`, so the
/// prompt a harness really sends can be studied without logging every message. Files are readable by the owner only;
/// day files older than 30 days go, and units stop being written beyond 200 MB.
public final class RequestRecorder: @unchecked Sendable {
    private let folder: URL, units: URL
    private let lock = NSLock()
    private var pruned = Date.distantPast
    /// Bytes under `units/`: listed once, then counted as units are written.
    private var unitBytes: Int?

    public init(folder: URL) throws {
        self.folder = folder
        units = folder.appendingPathComponent("units", isDirectory: true)
        try FileManager.default.createDirectory(at: units, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        try? FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: folder.path)
    }

    /// sha256 (hex) of a text unit.
    public static func hash(_ text: String) -> String { SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined() }

    public func record(_ row: [String: Any], units texts: [String: String]) {
        lock.lock()
        defer { lock.unlock() }
        let fm = FileManager.default
        if Date().timeIntervalSince(pruned) > 3600 {
            prune()
            pruned = Date()
        }
        var unitBytes =
            self.unitBytes
            ?? ((try? fm.contentsOfDirectory(at: units, includingPropertiesForKeys: [.fileSizeKey])) ?? [])
            .reduce(0) { $0 + ((try? $1.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) }
        for (hash, text) in texts where unitBytes < 200_000_000 {
            let url = units.appendingPathComponent(hash + ".txt")
            if !fm.fileExists(atPath: url.path),
                fm.createFile(atPath: url.path, contents: Data(text.utf8), attributes: [.posixPermissions: 0o600])
            {
                unitBytes += text.utf8.count
            }
        }
        self.unitBytes = unitBytes
        guard JSONSerialization.isValidJSONObject(row), var line = try? JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
        else { return }
        line.append(0x0a)
        let day = ISO8601DateFormatter.string(from: Date(), timeZone: .current, formatOptions: [.withFullDate])
        let url = folder.appendingPathComponent("requests-\(day).jsonl")
        if !fm.fileExists(atPath: url.path) { fm.createFile(atPath: url.path, contents: nil, attributes: [.posixPermissions: 0o600]) }
        guard let handle = try? FileHandle(forWritingTo: url) else { return }
        defer { try? handle.close() }
        _ = try? handle.seekToEnd()
        try? handle.write(contentsOf: line)
    }

    private func prune() {
        let cutoff = Date().addingTimeInterval(-30 * 86_400)
        for url in (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.contentModificationDateKey]))
            ?? []
        where url.lastPathComponent.hasPrefix("requests-") {
            if let modified = try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate, modified < cutoff {
                try? FileManager.default.removeItem(at: url)
            }
        }
    }
}
