import CryptoKit
import Darwin
import Foundation

/// `origami refresh` (ROADMAP R15): rewrites every weight file of a model in place, so a drive that reads heavily used
/// blocks slowly stores the same bytes on fresh blocks, then moves what was keyed by the model's old local identity
/// (`LocalModelIdentity`: the rewrite changes each file's change time) to the new one.
///
/// Safe to interrupt: every write repeats the bytes just read from that position, so the file holds the same bytes at
/// every moment. Each file is read and written back (SHA-256 of what was read), flushed with `F_FULLFSYNC`, read again
/// (SHA-256 again) and its modification time restored; the two digests must match. The journal
/// (`<data_dir>/refresh.json`) records the model, its old identity, each file's original stat and, once done, its
/// digest and stat after; a rerun skips the files done and resumes. Only when every file is proven unchanged does the
/// journal take the new identity, and only then does `rebind` run, in the same run or a later one that finds every
/// file as this refresh left it. A file found otherwise (replaced, edited, missing) ends the refresh with nothing
/// re-bound: its copies stay orphaned, as for any changed model.
package enum ModelRefresh {
    /// Where artefacts keyed by a model's identity live: the data folder, the prompt cache's root (`context_dir`), and
    /// every folder that holds per-model folders (the stores, the home store, `hot_copy_dir`).
    package struct Places: Sendable {
        package var data: URL, prompts: URL, folders: [URL]
        package init(data: URL, prompts: URL, folders: [URL]) {
            self.data = data
            self.prompts = prompts
            self.folders = folders
        }
    }

    /// A file's stat, as far as the local identity binds it.
    struct Stamp: Codable, Equatable {
        let device: Int64, inode: UInt64, size: Int64
        let modified: [Int64], changed: [Int64], born: [Int64]
        init?(_ path: String) {
            var info = stat()
            guard stat(path, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG else { return nil }
            device = Int64(info.st_dev)
            inode = UInt64(info.st_ino)
            size = Int64(info.st_size)
            modified = [Int64(info.st_mtimespec.tv_sec), Int64(info.st_mtimespec.tv_nsec)]
            changed = [Int64(info.st_ctimespec.tv_sec), Int64(info.st_ctimespec.tv_nsec)]
            born = [Int64(info.st_birthtimespec.tv_sec), Int64(info.st_birthtimespec.tv_nsec)]
        }
        /// The same file object: what a rewrite of its own bytes keeps (it changes the modification and change times).
        func sameFile(_ other: Stamp) -> Bool {
            device == other.device && inode == other.inode && size == other.size && born == other.born
        }
    }
    struct Entry: Codable {
        let path: String
        let original: Stamp
        var started = false
        /// Read back differently from what was read (`rewrite`): its bytes are no longer known to be the model's.
        var mismatch: Bool?
        var sha256: String?
        var after: Stamp?
    }
    struct Journal: Codable {
        var version = 1
        let model: String
        let family: Bool
        let old: String
        var new: String?
        var files: [Entry]
        /// Each related model's identity before and after (`Related`), by name.
        var related: [String: String]?
        var relatedNew: [String: String]?
    }
    package static func journalURL(data: URL) -> URL { data.appendingPathComponent("refresh.json") }
    /// The model an unfinished refresh is for (its path), when there is one: its files' identity is in flux, so nothing
    /// may build or re-key copies of it until a rerun finishes.
    package static func unfinished(data: URL) -> String? {
        (try? Data(contentsOf: journalURL(data: data))).flatMap { try? JSONDecoder().decode(Journal.self, from: $0) }?.model
    }

    /// Fault injection and interruption points for the checks.
    package struct Hooks {
        /// Called on each chunk before it is written back (a check may change the bytes).
        package var write: ((String, Int64, UnsafeMutableRawBufferPointer) -> Void)?
        package var afterFile: ((Int) throws -> Void)?
        package var beforeRebind: (() throws -> Void)?
        package init() {}
    }

    /// Another model whose identity binds some of the files rewritten (a low-bit variant of the model, or the base of
    /// the variant refreshed, and its other variants): what was keyed by it is moved to its new identity the same way.
    package struct Related {
        package let name: String
        package let identity: () throws -> String
        package let epochs: (String) -> [String]
        package init(name: String, identity: @escaping () throws -> String, epochs: @escaping (String) -> [String] = { _ in [] }) {
            self.name = name
            self.identity = identity
            self.epochs = epochs
        }
    }

    package struct Outcome {
        package let old: String, new: String
        /// The files rewritten in this run (a rerun skips the ones done before).
        package let rewritten: [String]
        /// What `rebind` moved, dropped or could not reach, one line each.
        package let rebound: [String]
    }

    /// Runs, or resumes, a refresh of the model at `model` (a Qwen folder, `family` false, or a GGUF) whose weight
    /// files are `files`; `identity` computes its fingerprint (Qwen) or key (GGUF) from the files as they are now.
    /// `epochs` gives a GGUF key's saved-prompt epochs (see `rebind`). `related`: the other models these files' identity
    /// binds, re-bound too. `progress` gets the bytes swept and to sweep.
    package static func run(
        model: URL, family: Bool, files: [URL], identity: () throws -> String, places: Places, epochs: (String) -> [String] = { _ in [] },
        related: [Related] = [], progress: (Int64, Int64) -> Void = { _, _ in }, hooks: Hooks = Hooks()
    ) throws -> Outcome {
        /// Each related model's identity now; one that cannot be computed is left out (nothing of it is moved).
        func relatedIdentities() -> [String: String] {
            Dictionary(related.compactMap { item in (try? item.identity()).map { (item.name, $0) } }) { first, _ in first }
        }
        let locks = try holdPromptLocks(places.prompts)
        defer { for fd in locks { Darwin.close(fd) } }
        let url = journalURL(data: places.data)
        var journal: Journal
        if let data = try? Data(contentsOf: url) {
            guard let found = try? JSONDecoder().decode(Journal.self, from: data) else {
                throw EngineError(
                    "\(url.path) cannot be read; delete it to start over (copies of the files already refreshed stay orphaned)")
            }
            guard found.model == model.path else {
                throw EngineError(
                    "A refresh of \(found.model) is unfinished; run `origami refresh --model \(found.model)` first (or delete \(url.path) to give it up)"
                )
            }
            // A low-bit variant shares its base's path but not its files (or identity): resumed only for the same files.
            guard Set(found.files.map(\.path)) == Set(files.map { $0.standardizedFileURL.resolvingSymlinksInPath().path }) else {
                throw EngineError(
                    "A refresh of other files of \(found.model) is unfinished (\(found.files.map { URL(fileURLWithPath: $0.path).lastPathComponent }.joined(separator: ", "))); run `origami refresh` again for the same model first (or delete \(url.path) to give it up)"
                )
            }
            journal = found
        } else {
            let old = try identity()
            var entries: [Entry] = []
            for file in files {
                let path = file.standardizedFileURL.resolvingSymlinksInPath().path
                guard let stamp = Stamp(path) else { throw EngineError("\(path) is not a readable file") }
                // Opened for writing before any journal exists: a read-only drive, a 0444 file or a privacy refusal (which
                // `access` does not see) stops here with nothing changed, instead of leaving the model locked out.
                let fd = Darwin.open(path, O_RDWR | O_NOFOLLOW | O_CLOEXEC)
                guard fd >= 0 else {
                    throw EngineError(
                        "Cannot open \(path) for writing (\(String(cString: strerror(errno)))); nothing was changed. Make the files writable (or allow Origami in System Settings > Privacy & Security), then run `origami refresh` again"
                    )
                }
                Darwin.close(fd)
                if !entries.contains(where: { $0.path == path }) { entries.append(Entry(path: path, original: stamp)) }
            }
            journal = Journal(model: model.path, family: family, old: old, files: entries, related: relatedIdentities())
            try FileManager.default.createDirectory(at: places.data, withIntermediateDirectories: true)
            try save(journal, to: url)
        }
        func abandon(_ why: String) throws -> Never {
            try? FileManager.default.removeItem(at: url)
            throw EngineError(
                "\(why); nothing was re-bound, so copies, usage counts and saved prompts made for the earlier files are not used. Run `origami refresh` again to refresh the files as they are now"
            )
        }
        var rewritten: [String] = []
        if journal.new == nil {
            let total = journal.files.filter { $0.after == nil }.reduce(Int64(0)) { $0 + 2 * $1.original.size }
            var swept: Int64 = 0
            for index in journal.files.indices {
                let entry = journal.files[index]
                guard let now = Stamp(entry.path) else { try abandon("\(entry.path) is gone") }
                if let after = entry.after {
                    guard now == after else { try abandon("\(entry.path) changed after it was refreshed") }
                    continue
                }
                // A file that read back differently is never carried over: rehashing it would take the changed bytes as the model's.
                guard entry.mismatch != true else {
                    throw EngineError(
                        "\(entry.path) read back differently when an earlier refresh wrote it, so its bytes may no longer be the model's; nothing was re-bound. Check the drive (Disk Utility's First Aid), download that file again, then delete \(url.path); copies, usage counts and saved prompts made for the earlier files are then not used"
                    )
                }
                guard entry.started ? now.sameFile(entry.original) : now == entry.original else {
                    try abandon("\(entry.path) was replaced or edited since the refresh began")
                }
                journal.files[index].started = true
                try save(journal, to: url)
                let base = swept
                let digest: String
                do {
                    digest = try rewrite(entry.path, modified: entry.original.modified, hook: hooks.write) { done in
                        progress(base + done, total)
                    }
                } catch let error as EngineError where error.code == "refresh_read_back_mismatch" {
                    journal.files[index].mismatch = true
                    try save(journal, to: url)
                    throw error
                }
                swept += 2 * entry.original.size
                journal.files[index].sha256 = digest
                journal.files[index].after = Stamp(entry.path)
                try save(journal, to: url)
                rewritten.append(entry.path)
                try hooks.afterFile?(index)
            }
            for entry in journal.files where Stamp(entry.path) != entry.after {
                try abandon("\(entry.path) changed after it was refreshed")
            }
            journal.new = try identity()
            journal.relatedNew = relatedIdentities()
            try save(journal, to: url)
        } else {
            for entry in journal.files where Stamp(entry.path) != entry.after {
                try abandon("\(entry.path) changed after it was refreshed")
            }
            guard try identity() == journal.new else { try abandon("the model's files changed after they were refreshed") }
        }
        try hooks.beforeRebind?()
        let new = journal.new!
        var rebound =
            journal.old == new
            ? [] : try rebind(family: journal.family, old: journal.old, new: new, places: places, epochs: epochs, verified: true)
        for item in related {
            guard let old = journal.related?[item.name], let now = journal.relatedNew?[item.name], old != now else { continue }
            rebound += try rebind(family: journal.family, old: old, new: now, places: places, epochs: item.epochs, verified: true)
        }
        try FileManager.default.removeItem(at: url)
        return Outcome(old: journal.old, new: new, rewritten: rewritten, rebound: rebound)
    }

    private static func save(_ journal: Journal, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(journal).write(to: url, options: .atomic)
    }

    /// Rewrites the file at `path` in place, 8 chunks of 16 MiB at a time, uncached; returns the SHA-256 of its bytes
    /// once the written bytes read back the same, with its modification time put back to `modified`.
    static func rewrite(
        _ path: String, modified: [Int64], hook: ((String, Int64, UnsafeMutableRawBufferPointer) -> Void)?, progress: (Int64) -> Void
    ) throws -> String {
        let chunk = 16 << 20
        let wide = 8
        let fd = Darwin.open(path, O_RDWR | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else { throw EngineError("Cannot open \(path) for writing: \(String(cString: strerror(errno)))") }
        defer { Darwin.close(fd) }
        _ = fcntl(fd, F_NOCACHE, 1)
        var info = stat()
        guard fstat(fd, &info) == 0 else { throw EngineError("Cannot stat \(path)") }
        let size = Int64(info.st_size)
        let buffers = (0..<wide).map { _ in UnsafeMutableRawBufferPointer.allocate(byteCount: chunk, alignment: 16384) }
        defer { for buffer in buffers { buffer.deallocate() } }
        nonisolated(unsafe) let shared = buffers
        func sweep(write: Bool, done: Int64) throws -> String {
            var hash = SHA256()
            var base: Int64 = 0
            while base < size {
                let offsets = Array(stride(from: base, to: min(base + Int64(chunk * wide), size), by: chunk))
                let lengths = offsets.map { Int(min(Int64(chunk), size - $0)) }
                let failed = Atomic()
                DispatchQueue.concurrentPerform(iterations: offsets.count) { i in
                    if !Self.full({ pread($0, $1, $2, $3) }, fd, shared[i], lengths[i], offsets[i]) { failed.set() }
                }
                guard !failed.value else { throw EngineError("Reading \(path) failed: \(String(cString: strerror(errno)))") }
                for i in offsets.indices { hash.update(bufferPointer: UnsafeRawBufferPointer(rebasing: shared[i][..<lengths[i]])) }
                if write {
                    if let hook {
                        for i in offsets.indices {
                            hook(path, offsets[i], UnsafeMutableRawBufferPointer(rebasing: shared[i][..<lengths[i]]))
                        }
                    }
                    DispatchQueue.concurrentPerform(iterations: offsets.count) { i in
                        if !Self.full({ pwrite($0, UnsafeRawPointer($1), $2, $3) }, fd, shared[i], lengths[i], offsets[i]) { failed.set() }
                    }
                    guard !failed.value else { throw EngineError("Writing \(path) failed: \(String(cString: strerror(errno)))") }
                }
                base += Int64(lengths.reduce(0, +))
                progress(done + base)
            }
            return hash.finalize().map { String(format: "%02x", $0) }.joined()
        }
        let before = try sweep(write: true, done: 0)
        guard fsync(fd) == 0, fcntl(fd, F_FULLFSYNC) != -1 else { throw EngineError("Flushing \(path) failed") }
        let after = try sweep(write: false, done: size)
        var times = [timespec(tv_sec: 0, tv_nsec: Int(UTIME_OMIT)), timespec(tv_sec: Int(modified[0]), tv_nsec: Int(modified[1]))]
        // The read-back is judged first: a modification time that cannot be put back must not hide a mismatch from the journal.
        let restored = utimensat(AT_FDCWD, path, &times, 0) == 0
        guard before == after else {
            throw EngineError(
                "\(path) read back differently after it was written (SHA-256 \(before) before, \(after) after); stopped, nothing was re-bound. Every write repeated the bytes read from the same place, so the drive changed them: check it (Disk Utility's First Aid) and download that file again; a later `origami refresh` refuses until then",
                code: "refresh_read_back_mismatch")
        }
        guard restored else { throw EngineError("Cannot restore the modification time of \(path)") }
        return before
    }
    private static func full(
        _ call: (Int32, UnsafeMutableRawPointer, Int, off_t) -> Int, _ fd: Int32, _ buffer: UnsafeMutableRawBufferPointer, _ length: Int,
        _ offset: Int64
    ) -> Bool {
        var done = 0
        while done < length {
            let n = call(fd, buffer.baseAddress! + done, length - done, off_t(offset) + off_t(done))
            if n < 0 && errno == EINTR { continue }
            guard n > 0 else { return false }
            done += n
        }
        return true
    }
    private final class Atomic: @unchecked Sendable {
        private let lock = NSLock()
        private var flag = false
        func set() { lock.withLock { flag = true } }
        var value: Bool { lock.withLock { flag } }
    }

    /// The prompt cache's own lock: the `.lock` (flock) a server holds on its epoch folder in `context_dir` (and, for
    /// Qwen, in each `<fingerprint12>-<context>` folder). Held by another process: refused, naming it; else held here
    /// until the refresh ends, so a server started meanwhile neither reads nor writes saved prompts.
    static func holdPromptLocks(_ root: URL) throws -> [Int32] {
        func children(_ folder: URL) -> [URL] {
            (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
        }
        func epoch(_ name: String) -> Bool { name.count == 16 && name.allSatisfy(\.isHexDigit) }
        let folders = children(root).flatMap { folder -> [URL] in
            epoch(folder.lastPathComponent)
                ? [folder]
                : PromptCache.modelFolder(folder.lastPathComponent) != nil ? children(folder).filter { epoch($0.lastPathComponent) } : []
        }
        var held: [Int32] = []
        for folder in folders {
            let lock = folder.appendingPathComponent(".lock")
            let fd = Darwin.open(lock.path, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o600)
            guard fd >= 0 else { continue }
            if flock(fd, LOCK_EX | LOCK_NB) != 0 {
                let busy = errno == EWOULDBLOCK
                Darwin.close(fd)
                guard !busy else {
                    for fd in held { Darwin.close(fd) }
                    throw EngineError(
                        "A running Origami server holds the prompt cache lock \(lock.path); stop it (and any other server using this context_dir), then run `origami refresh` again"
                    )
                }
                continue  // a drive without locks
            }
            held.append(fd)
        }
        return held
    }

    /// Moves everything keyed by a model's identity `old` to `new` (a Qwen fingerprint, or a GGUF key when `family`):
    /// per artefact atomically (a folder rename, a JSON file rewritten beside and renamed over), and idempotently, so
    /// an interrupted re-bind finishes when run again. Only for bytes proven unchanged: `run` calls it after every file
    /// read back the same, and ROADMAP R12 (adopting copies of an earlier version) is meant to reuse it.
    ///
    /// - Family (`K` → `N`): `<data>/family/K` (usage counts, a hot copy kept there) and `<data>/support/K`, and
    ///   `family-K` in every folder (a hot copy in `hot_copy_dir`, packs in stores) are renamed, and the key inside each
    ///   copy's `index.json` and each pack's `pack.json` replaced. The tuning files' `K[..12]` become `N[..12]`, and
    ///   `models.json`'s `K` becomes `N`. Saved prompts: `epochs(K)[i]` names an epoch of `<prompts>/<epoch16>` whose
    ///   `index.json` and each `entry-*/state.json` carry it: they are rewritten to `epochs(N)[i]` and the folder
    ///   renamed (a family state's own digest covers its tokens and bytes, not the epoch).
    /// - Qwen (`O` → `P`): each `<name>-O[..12]` folder (packs, further copies, the hot copy) becomes `<name>-P[..12]`,
    ///   with `O` replaced in its store headers (`model_fingerprint`) and `hot/index.json`; `routing-O[..12].json`
    ///   becomes `routing-P[..12].json`; `calibration.json`'s keys `O[..16]`, the tuning files' `O[..12]` and
    ///   `models.json`'s `O` are replaced. Saved prompts in `<prompts>/O[..12]-<context>`: each image embeds the
    ///   fingerprint in its identity and the epoch under its own digests, so they cannot be re-stamped. When `verified`
    ///   (a refresh read every file back unchanged, `run`), the folder becomes `<prompts>/P[..12]-<context>` and records
    ///   `O` (`earlierFingerprints`): the server finds the saves by it and reads them under the identity they were written
    ///   with (`PromptCanary.adopt`). Otherwise (a move or Locate, which compare less) they are dropped.
    package static func rebind(
        family: Bool, old: String, new: String, places: Places, epochs: (String) -> [String] = { _ in [] }, verified: Bool = false
    ) throws -> [String] {
        var lines: [String] = []
        let manager = FileManager.default
        func exists(_ url: URL) -> Bool { manager.fileExists(atPath: url.path) }
        /// Renames `from` to `to` when only `from` exists; the folder that now holds the artefact, if any.
        func move(_ from: URL, _ to: URL) throws -> URL? {
            switch (exists(from), exists(to)) {
            case (true, false):
                try manager.moveItem(at: from, to: to)
                lines.append("moved \(from.path) to \(to.lastPathComponent)")
                return to
            case (true, true):
                lines.append("kept \(from.path): \(to.path) already exists")
                return nil
            case (false, true): return to
            case (false, false): return nil
            }
        }
        /// `old` replaced by `new` in each JSON file of `folder` and of its subfolders one level down.
        func restampTree(_ folder: URL, _ old: String, _ new: String) throws {
            let inner = ((try? manager.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? [])
            for url in [folder] + inner.filter({ $0.hasDirectoryPath }) {
                for file in (try? manager.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)) ?? []
                where file.pathExtension == "json" && !file.lastPathComponent.hasPrefix(".") {
                    try restamp(file, old, new)
                }
            }
        }
        let data = places.data
        let tuning = ["tuning.json", "tuning-proposal.json", "tuning-history.jsonl"].map { data.appendingPathComponent($0) }
        if family {
            if let folder = try move(data.appendingPathComponent("family/\(old)"), data.appendingPathComponent("family/\(new)")) {
                try restampTree(folder, old, new)
            }
            _ = try move(data.appendingPathComponent("support/\(old)"), data.appendingPathComponent("support/\(new)"))
            for root in places.folders {
                if let folder = try move(root.appendingPathComponent("family-\(old)"), root.appendingPathComponent("family-\(new)")) {
                    try restampTree(folder, old, new)
                }
            }
            for (from, to) in zip(epochs(old), epochs(new)) where from != to {
                let source = places.prompts.appendingPathComponent(String(from.prefix(16)))
                let target = places.prompts.appendingPathComponent(String(to.prefix(16)))
                // Re-stamped before the rename, so a stop in between is finished by running again.
                if exists(source) && !exists(target) { try restampTree(source, from, to) }
                if let folder = try move(source, target) { try restampTree(folder, from, to) }
            }
        } else {
            let (old12, new12) = (String(old.prefix(12)), String(new.prefix(12)))
            for root in places.folders {
                for folder in (try? manager.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
                where folder.lastPathComponent.hasSuffix("-" + old12) {
                    let name = String(folder.lastPathComponent.dropLast(old12.count)) + new12
                    if let moved = try move(folder, root.appendingPathComponent(name)) { try restampTree(moved, old, new) }
                }
            }
            if let file = try move(
                data.appendingPathComponent("routing-\(old12).json"), data.appendingPathComponent("routing-\(new12).json"))
            {
                try restamp(file, old, new)
            }
            try restamp(data.appendingPathComponent("calibration.json"), String(old.prefix(16)), String(new.prefix(16)))
            for folder in (try? manager.contentsOfDirectory(at: places.prompts, includingPropertiesForKeys: nil)) ?? []
            where PromptCache.modelFolder(folder.lastPathComponent).map({ $0 == old12 }) == true {
                guard verified else {
                    try manager.removeItem(at: folder)
                    lines.append("dropped the saved prompts in \(folder.path): they cannot be re-stamped")
                    continue
                }
                let name = new12 + folder.lastPathComponent.dropFirst(old12.count)
                if let moved = try move(folder, places.prompts.appendingPathComponent(name, isDirectory: true)) {
                    try noteEarlier(moved, old)
                }
            }
        }
        for file in tuning { try restamp(file, String(old.prefix(12)), String(new.prefix(12))) }
        try restamp(data.appendingPathComponent("models.json"), old, new)
        return lines
    }

    /// The fingerprints the model of a Qwen `<fingerprint12>-<context>` prompt folder had before refreshes that read its
    /// bytes back unchanged, oldest first (`rebind`): its saves may still be keyed on them (`PromptCanary.adopt`).
    package static func earlierFingerprints(_ folder: URL) -> [String] {
        (try? Data(contentsOf: folder.appendingPathComponent(earlierFile))).flatMap { try? JSONDecoder().decode([String].self, from: $0) }
            ?? []
    }
    static let earlierFile = "earlier-fingerprints.json"
    /// Adds `fingerprint` to the folder's `earlierFingerprints` (once), written beside and renamed over.
    static func noteEarlier(_ folder: URL, _ fingerprint: String) throws {
        var list = earlierFingerprints(folder)
        guard !list.contains(fingerprint) else { return }
        list.append(fingerprint)
        let url = folder.appendingPathComponent(earlierFile)
        try JSONEncoder().encode(list).write(to: url, options: .atomic)
    }

    /// Replaces every `old` in the text file at `url` by `new` (a file without it is left alone): written beside and
    /// renamed over, with the file's permissions. Identities are hex digests, so they appear nowhere by accident.
    static func restamp(_ url: URL, _ old: String, _ new: String) throws {
        guard old != new, let data = try? Data(contentsOf: url), let text = String(data: data, encoding: .utf8), text.contains(old) else {
            return
        }
        let mode = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.posixPermissions] as? NSNumber
        let staging = url.deletingLastPathComponent().appendingPathComponent(".\(url.lastPathComponent).refresh")
        try Data(text.replacingOccurrences(of: old, with: new).utf8).write(to: staging)
        if let mode { try FileManager.default.setAttributes([.posixPermissions: mode], ofItemAtPath: staging.path) }
        guard Darwin.rename(staging.path, url.path) == 0 else {
            try? FileManager.default.removeItem(at: staging)
            throw EngineError("Cannot replace \(url.path)")
        }
    }
}
