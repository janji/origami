import CryptoKit
import Darwin
import Foundation
import OrigamiEngine

/// A move survives browser and process restarts. Recovery never guesses which copy to delete.
final class MoveJournal {
    let url: URL
    private var record: [String: Any]
    private init(url: URL, record: [String: Any]) {
        self.url = url
        self.record = record
    }
    static func begin(_ move: Storage.Move, settings: Settings) throws -> MoveJournal {
        let id = UUID().uuidString
        let journal = MoveJournal(
            url: settings.dataDir.appendingPathComponent("moves/" + id + ".json"),
            record: [
                "id": id, "name": move.item.name, "phase": "copying", "created": Date().timeIntervalSince1970,
                "renames": move.renames, "kind": move.item.move ?? "", "model": move.item.model ?? "", "folder": move.folder,
                "pairs": move.pairs.map { part -> [String: Any] in
                    var info = stat()
                    _ = lstat(part.from, &info)
                    return ["from": part.from, "to": part.to, "device": Int64(info.st_dev), "inode": UInt64(info.st_ino)]
                },
            ])
        try journal.phase("copying")
        return journal
    }
    func phase(_ value: String) throws {
        record["phase"] = value
        try Settings.write(record.mapValues { Optional($0) }, to: url)
    }
    func finish() throws { try FileManager.default.removeItem(at: url) }
    /// Records the moved model's identity before the move (its own and each related model's, by name) and where it goes.
    func rebinding(to path: String, model: String?, own: String?, related: [String: String]) throws {
        record["rebind"] = ["path": path, "model": model ?? path, "own": own ?? NSNull(), "related": related] as [String: Any]
        try Settings.write(record.mapValues { Optional($0) }, to: url)
    }
    /// Once committed: moves what is keyed by the identity recorded to the moved files' (idempotent, `ModelRefresh.rebind`).
    func rebind(settings: Settings) throws { try Self.rebind(record, settings: settings) }
    private static func rebind(_ record: [String: Any], settings: Settings) throws {
        guard let rebind = record["rebind"] as? [String: Any], let path = rebind["path"] as? String else { return }
        var probe = settings
        probe.model = rebind["model"] as? String ?? path
        let own = rebind["own"] as? String
        // Never skipped without a word: what is keyed on the identity before the move waits, journal kept, for a retry.
        if own != nil, Refresh.identities(URL(fileURLWithPath: path), settings: probe).own == nil {
            throw ServingFailure(
                "\(probe.model ?? path)'s files at \(path) cannot be read now, so its hot copy, counts, tuning and saved prompts still "
                    + "belong to it before its move: once they are readable, run `origami move recover` (or the dashboard's Recover)",
                status: 409)
        }
        _ = try Refresh.rebind(
            URL(fileURLWithPath: path), before: own, relatedBefore: rebind["related"] as? [String: String] ?? [:], settings: probe)
    }
    /// The identities pending journals still have to carry over (each moved model's before its move, and its related
    /// ones'), full and as their 12-character prefix, by model: what is keyed on them belongs to that model, not to no one.
    static func waiting(_ settings: Settings) -> [String: String] {
        var owners: [String: String] = [:]
        for record in pending(settings) {
            guard let rebind = record["rebind"] as? [String: Any], let model = rebind["model"] as? String else { continue }
            for id in [rebind["own"] as? String].compactMap({ $0 }) + (rebind["related"] as? [String: String] ?? [:]).values {
                owners[id] = model
                owners[String(id.prefix(12))] = model
            }
        }
        return owners
    }
    static func pending(_ settings: Settings) -> [[String: Any]] {
        let folder = settings.dataDir.appendingPathComponent("moves")
        return ((try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "json" }.compactMap { try? Settings.readConfig($0) }
    }
    static func recover(_ id: String, settings: Settings) throws -> String {
        guard UUID(uuidString: id) != nil else { throw ServingFailure("Unknown move") }
        let url = settings.dataDir.appendingPathComponent("moves/" + id + ".json")
        let record = try Settings.readConfig(url)
        guard let pairs = record["pairs"] as? [[String: Any]], !pairs.isEmpty else { throw ServingFailure("Move record is incomplete") }
        let entries = try ModelRegistry.entries(settings)
        let config = try Settings.readConfig(settings.configFile)
        let folder = record["folder"] as? String ?? ""
        let committed =
            record["phase"] as? String == "committed"
            || pairs.contains { pair in
                entries.contains { $0["path"] as? String == pair["to"] as? String } || config["model"] as? String == pair["to"] as? String
            } || (record["kind"] as? String == "prompts" && config["context_dir"] as? String == folder)
        // A hot-copy profile or store entry may have committed just before the journal write was interrupted.
        let owner = entries.first { $0["name"] as? String == record["model"] as? String }
        let hotCommitted = record["kind"] as? String == "hot" && ((owner?["hot_copy_dir"] ?? config["hot_copy_dir"]) as? String) == folder
        let packCommitted = record["kind"] as? String == "pack" && settings.stores.contains { $0.path == folder }
        // A setting that already names the destination folder commits nothing until every part is there.
        let placed = pairs.allSatisfy { ($0["to"] as? String).map { FileManager.default.fileExists(atPath: $0) } ?? false }
        let isCommitted = committed || (hotCommitted || packCommitted) && placed
        let renames = record["renames"] as? Bool == true
        let fm = FileManager.default
        for pair in pairs.reversed() {
            guard let from = pair["from"] as? String, let to = pair["to"] as? String, from != to, from.hasPrefix("/"), to.hasPrefix("/")
            else { throw ServingFailure("Move paths are invalid") }
            let source = fm.fileExists(atPath: from)
            let destination = fm.fileExists(atPath: to)
            if isCommitted {
                guard destination else { throw ServingFailure("The destination is unavailable. Reconnect its disk before recovering.") }
                if source {
                    guard try equal(from, to) else {
                        throw ServingFailure("The copies differ. Both were kept; inspect them before deleting either.")
                    }
                    try fm.removeItem(atPath: from)
                }
            } else if renames, destination, !source {
                var info = stat()
                guard lstat(to, &info) == 0, Int64(info.st_dev) == (pair["device"] as? NSNumber)?.int64Value,
                    UInt64(info.st_ino) == (pair["inode"] as? NSNumber)?.uint64Value
                else { throw ServingFailure("The destination changed since this move. Both locations were kept.") }
                guard Storage.renameNew(to, from) == 0 else { throw ServingFailure("Could not restore " + from) }
            } else if !renames, destination {
                guard source, try equal(from, to) else { throw ServingFailure("Cannot verify the original. Both locations were kept.") }
                try fm.removeItem(atPath: to)
            } else if !source {
                throw ServingFailure("The original disk is unavailable. Reconnect it before recovering.")
            }
            let staging = (to as NSString).deletingLastPathComponent + "/.origami-moving-" + (to as NSString).lastPathComponent
            if fm.fileExists(atPath: staging), fm.fileExists(atPath: from) { try fm.removeItem(atPath: staging) }
        }
        if isCommitted { try rebind(record, settings: settings) }
        try fm.removeItem(at: url)
        return isCommitted ? "Move recovered; verified originals removed" : "Original locations restored"
    }
    private static func equal(_ a: String, _ b: String) throws -> Bool {
        let fm = FileManager.default
        var da: ObjCBool = false
        var db: ObjCBool = false
        guard fm.fileExists(atPath: a, isDirectory: &da), fm.fileExists(atPath: b, isDirectory: &db), da.boolValue == db.boolValue else {
            return false
        }
        if da.boolValue {
            let left = try fm.contentsOfDirectory(atPath: a).sorted()
            let right = try fm.contentsOfDirectory(atPath: b).sorted()
            guard left == right else { return false }
            for name in left { if try !equal(a + "/" + name, b + "/" + name) { return false } }
            return true
        }
        func digest(_ path: String) throws -> SHA256.Digest {
            let before = try fm.attributesOfItem(atPath: path)
            let file = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
            defer { try? file.close() }
            var hash = SHA256()
            while let bytes = try file.read(upToCount: 8 << 20), !bytes.isEmpty { hash.update(data: bytes) }
            let after = try fm.attributesOfItem(atPath: path)
            guard before[.size] as? NSNumber == after[.size] as? NSNumber,
                before[.modificationDate] as? Date == after[.modificationDate] as? Date
            else { throw ServingFailure("A file changed during verification; nothing was removed") }
            return hash.finalize()
        }
        return try digest(a) == digest(b)
    }
}
