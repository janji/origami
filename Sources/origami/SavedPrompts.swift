import Darwin
import Foundation
import OrigamiEngine

/// The saved prompts in `context_dir`, one row per build's folder, as `origami cache` lists them and the dashboard's
/// storage map shows them: Qwen's `<fingerprint12>-<context>/<epoch16>` (and 1.0.0's entries in `<fingerprint12>-<context>`
/// itself), the family engines' `<epoch16>`, and a GLM or DeepSeek context left on disk. Each is in use (a running
/// server holds its `.lock`), usable by this build, or unusable, with the reason; `prune` removes the unusable ones and
/// never a folder a server holds.
enum SavedPrompts {
    /// The models whose saved prompts may be read: registered or configured Qwen fingerprint12s and GGUF keys, with their
    /// names (`known` false: some model's identity could not be read, so no folder is taken for another model file's).
    struct Owners {
        var qwen: [String: String] = [:], family: [String: String] = [:]
        var qwenKnown = false, familyKnown = false
        /// The last name of each identity no model has now (`ModelRegistry.former`).
        var former: [String: [String: String]] = [:]
        /// How a build of no registered model names its model: its last name, else its identity.
        func gone(_ id: String, known: Bool) -> String {
            former[id].map(ModelRegistry.formerName) ?? (known ? "from a model no longer registered · " : "a model not identified · ") + id
        }
    }
    enum State: Equatable {
        case inUse, usable
        case unusable(String)
        var text: String {
            switch self {
            case .inUse: "in use"
            case .usable: "usable by this build"
            case .unusable(let reason): "unusable: " + reason
            }
        }
    }
    struct Build {
        /// Qwen's `<fingerprint12>-<context>`, a family epoch or `family-context`: the folder the build is listed under.
        let group: URL
        /// What removing it deletes.
        let items: [URL]
        let label: String, model: String
        /// Its model's last name and what became of it (`Owners.gone`), when no registered model has it.
        let owner: String?
        let context: Int?
        let used: Date
        let bytes: UInt64
        let conversations: Int
        /// The names of the conversations and prompts it keeps by name or pin (`PromptCache` never drops them itself).
        let kept: [String]
        /// Its index is there but cannot be read: it may hold named or pinned prompts.
        let unreadable: Bool
        let state: State
        var holdsKept: Bool { !kept.isEmpty || unreadable }
        /// Names the build on the page: its one folder, else the folder it is listed under.
        var path: String { items.count == 1 ? items[0].path : group.path }
        var json: [String: Any] {
            var row: [String: Any] = [
                "path": path, "label": label, "model": model, "bytes": bytes,
                "used": used == .distantPast ? NSNull() : used.timeIntervalSince1970, "conversations": conversations,
                "kept": kept, "unreadable": unreadable,
                "state": state == .inUse ? "in_use" : state == .usable ? "usable" : "unusable",
            ]
            if let context { row["context"] = context }
            if let owner { row["owner"] = owner }
            if case .unusable(let reason) = state { row["reason"] = reason }
            return row
        }
    }

    /// Why the running build cannot read the saved prompts in `folder` (one build's: a Qwen
    /// `<fingerprint12>-<context>/<epoch16>`, a family `<epoch16>`), nil when it can or may: `PromptCanary` decides (the
    /// same arithmetic, and a canary this build reproduced or has yet to be checked against); the words name the folder's
    /// build from its marker, and one without a marker was made by a build before markers.
    static func whyUnreadable(_ folder: URL) -> String? {
        guard PromptCanary.whyUnreadable(folder) != nil else { return nil }
        guard let marker = FamilySetup.epoch(folder) else { return "made by an earlier Origami build" }
        return "made by another Origami build" + ((marker["version"] as? String).map { " (version \($0))" } ?? "")
    }

    static func builds(_ root: URL, owners: Owners) -> [Build] {
        let manager = FileManager.default
        func children(_ url: URL) -> [URL] {
            ((try? manager.contentsOfDirectory(at: url, includingPropertiesForKeys: [.isDirectoryKey])) ?? [])
                .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true }.sorted { $0.path < $1.path }
        }
        func epoch(_ url: URL) -> Bool { url.lastPathComponent.range(of: "^[0-9a-f]{16}$", options: .regularExpression) != nil }
        /// When its index was last written: its last use.
        func used(_ index: URL) -> Date {
            (try? index.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
        }
        func make(
            _ group: URL, _ items: [URL], label: String, model: String, owner: String? = nil, context: Int?, index: URL?, state: State
        ) -> Build {
            // An index that is missing holds nothing; one that cannot be read may hold named or pinned prompts.
            let read = index.map { try? Settings.readConfig($0) } ?? [:]
            let entries = read?["entries"] as? [String: [String: Any]] ?? [:]
            let kept = entries.values.compactMap { $0["name"] as? String ?? ($0["pinned"] as? Bool == true ? "a pinned prompt" : nil) }
            return Build(
                group: group, items: items, label: label, model: model, owner: owner, context: context,
                used: index.map(used) ?? .distantPast,
                bytes: items.reduce(0) { $0 + Storage.size($1.path).allocated }, conversations: entries.count, kept: kept.sorted(),
                unreadable: read == nil, state: state)
        }
        var builds: [Build] = []
        for group in children(root) {
            let name = group.lastPathComponent
            if let fingerprint = PromptCache.modelFolder(name) {
                let id = String(fingerprint)
                let context = Int(name.dropFirst(13))
                let model = owners.qwen[id] ?? id
                let orphan = owners.qwenKnown && owners.qwen[id] == nil
                let owner = owners.qwen[id] == nil ? owners.gone(id, known: owners.qwenKnown) : nil
                for build in children(group) where epoch(build) {
                    let state: State =
                        PromptCache.inUse(build)
                        ? .inUse : orphan ? .unusable("a model file no model uses") : whyUnreadable(build).map(State.unusable) ?? .usable
                    builds.append(
                        make(
                            group, [build], label: name + "/" + build.lastPathComponent, model: model, owner: owner, context: context,
                            index: build.appendingPathComponent("index.json"), state: state))
                }
                // 1.0.0 kept its entries in `<fingerprint12>-<context>` itself; no server reads them now.
                let legacy = ((try? manager.contentsOfDirectory(at: group, includingPropertiesForKeys: nil)) ?? []).filter {
                    $0.lastPathComponent.hasPrefix("entry-") || $0.lastPathComponent == "index.json"
                        || $0.lastPathComponent.hasPrefix("frontiers-")
                }
                if !legacy.isEmpty {
                    builds.append(
                        make(
                            group, legacy.sorted { $0.path < $1.path }, label: name + " (1.0.0)", model: model, owner: owner,
                            context: context,
                            index: group.appendingPathComponent("index.json"), state: .unusable("made by Origami 1.0.0")))
                }
            } else if epoch(group) {
                let marker = FamilySetup.epoch(group)
                let key = marker?["family_key"] as? String ?? ""
                let model = owners.family[key] ?? marker?["model"] as? String ?? "GLM or DeepSeek"
                let state: State =
                    PromptCache.inUse(group)
                    ? .inUse
                    : marker != nil && owners.familyKnown && owners.family[key] == nil
                        ? .unusable("a model file no model uses") : whyUnreadable(group).map(State.unusable) ?? .usable
                let label =
                    name
                    + (marker.map {
                        " (\($0["model"] as? String ?? "?"), \($0["context"] as? Int ?? 0) context, version \($0["version"] as? String ?? "?"))"
                    } ?? "")
                builds.append(
                    make(
                        group, [group], label: label, model: model,
                        owner: owners.family[key] == nil && !key.isEmpty ? owners.gone(key, known: owners.familyKnown) : nil,
                        context: marker?["context"] as? Int,
                        index: group.appendingPathComponent("index.json"), state: state))
            } else if name == "family-context" {
                // Its files are unlinked as soon as they are mapped: what is left is a process's that ended before.
                let left = ((try? manager.contentsOfDirectory(at: group, includingPropertiesForKeys: nil)) ?? []).filter { file in
                    let parts = file.lastPathComponent.split(separator: "-")
                    guard parts.count > 2, parts[0] == "state", let pid = Int32(parts[1]) else { return false }
                    return kill(pid, 0) != 0 && errno == ESRCH
                }
                if !left.isEmpty {
                    builds.append(
                        make(
                            group, left, label: "family-context", model: "GLM or DeepSeek", context: nil, index: nil,
                            state: .unusable("a GLM or DeepSeek context left on disk")))
                }
            }
        }
        return builds
    }

    /// Removes one build's files unless a server holds them. An epoch folder is locked first (its `.lock`, as a server
    /// holds it) and renamed aside, so a server starting meanwhile finds it held, or makes a fresh one, and never reads
    /// a folder being deleted. False: a server holds it.
    static func remove(_ build: Build) throws -> Bool {
        let manager = FileManager.default
        for item in build.items {
            guard item.lastPathComponent.range(of: "^[0-9a-f]{16}$", options: .regularExpression) != nil else {
                try manager.removeItem(at: item)
                continue
            }
            let fd = Darwin.open(item.appendingPathComponent(".lock").path, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o600)
            defer { if fd >= 0 { Darwin.close(fd) } }
            if fd >= 0, flock(fd, LOCK_EX | LOCK_NB) != 0 { return false }
            let aside = item.deletingLastPathComponent().appendingPathComponent(".origami-removing-" + item.lastPathComponent)
            try? manager.removeItem(at: aside)  // what an interrupted removal left
            guard rename(item.path, aside.path) == 0 else {
                throw CLIError("Cannot remove \(item.path): \(String(cString: strerror(errno)))")
            }
            do { try manager.removeItem(at: aside) } catch {
                _ = rename(aside.path, item.path)
                throw CLIError("Cannot remove \(item.path): \(error.localizedDescription)")
            }
        }
        // An emptied `<fingerprint12>-<context>` stays: a server may be making its next epoch folder in it.
        return true
    }

    /// `origami cache prune` and the dashboard's Remove unusable: removes the unusable builds (`all`: every build no
    /// server holds), keeping one with named or pinned conversations, or an index that cannot be read, unless `yes`
    /// says it may go. Returns the lines saying what went and what stayed, and why, and the bytes freed.
    static func prune(_ builds: [Build], all: Bool, yes: (Build) -> Bool, command: String) -> (lines: [String], freed: UInt64, removed: Int)
    {
        var lines: [String] = []
        var freed: UInt64 = 0
        var removed = 0
        for build in builds {
            let size = OrigamiPlan.gigabytes(build.bytes)
            guard build.state != .inUse, all || build.state != .usable else {
                lines.append("Keeps \(build.label)  \(size): \(build.state == .inUse ? "a running server holds it" : build.state.text)")
                continue
            }
            if build.holdsKept && !yes(build) {
                let why =
                    (build.kept.isEmpty ? [] : ["it holds " + build.kept.joined(separator: ", ")])
                    + (build.unreadable ? ["its index could not be read, so it may hold named or pinned prompts"] : [])
                lines.append("Keeps \(build.label)  \(size): \(why.joined(separator: "; ")); \(command) removes it too")
                continue
            }
            do {
                guard try remove(build) else {
                    lines.append("Keeps \(build.label)  \(size): a running server holds it")
                    continue
                }
                freed += build.bytes
                removed += 1
                lines.append(
                    "Removes \(build.label)  \(size)"
                        + (build.unreadable ? "  (its index could not be read, so any named or pinned prompts in it go too)" : ""))
            } catch {
                lines.append("Keeps \(build.label)  \(size): \(error.localizedDescription)")
            }
        }
        return (lines, freed, removed)
    }
}
