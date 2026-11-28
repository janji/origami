import Foundation
import OrigamiEngine

/// `origami refresh` (ROADMAP R15): rewrites the model's weight files in place (`ModelRefresh.run`) and moves its copies,
/// usage counts and saved prompts to the identity the files have afterwards. Refused while a server answers on the port
/// or holds the prompt cache's lock; a server does not start the model while its refresh is unfinished.
/// `--copy` rewrites the model's hot copies instead (`copies`): nothing is keyed by their bytes, so nothing moves.
enum Refresh {
    static func run(flags: [String: [String]], yes: Bool, copy: Bool = false) throws {
        let settings = try Settings.load(flags: flags)
        guard let url = Models.locate(settings.model, settings: settings) else {
            throw CLIError(
                settings.model.map { "No model to refresh at \($0): check the name (`origami model list`) or the path" }
                    ?? "No model to refresh: pass --model <name>, or set one with `origami model default <name>`")
        }
        try Commands.requireNoServer(settings, copy ? "refresh --copy" : "refresh")
        if copy { return try copies(url, settings: settings, yes: yes) }
        let (family, files, identity, epochs, related) = try binding(url, settings: settings)
        let bytes = files.reduce(UInt64(0)) {
            $0 + (((try? FileManager.default.attributesOfItem(atPath: $1.path))?[.size] as? NSNumber)?.uint64Value ?? 0)
        }
        let resuming = ModelRefresh.unfinished(data: settings.dataDir) == url.path
        if !resuming {
            print(
                "This rewrites the \(files.count) weight file\(files.count == 1 ? "" : "s") of \(url.path) (\(OrigamiPlan.gigabytes(bytes))) in place:"
                    + " each byte is read and written back to the same place, then read again and compared (SHA-256), so the drive stores"
                    + " them on fresh blocks. It is safe to stop; run it again to continue. Its copies, usage counts and saved prompts"
                    + " are moved to the files' new identity afterwards.")
            guard Terminal.confirm("Refresh \(url.lastPathComponent)?", yes: yes) else { throw CLIError("Nothing was changed") }
        } else {
            print("Continuing the unfinished refresh of \(url.path)")
        }
        let places = ModelRefresh.Places(
            data: settings.dataDir, prompts: settings.promptCacheRoot, folders: settings.storeFolders)
        let offline = settings.storeFolders.filter { !FileManager.default.fileExists(atPath: $0.path) }
        let started = Date()
        var shown = Date.distantPast
        let outcome = try ModelRefresh.run(
            model: url, family: family, files: files, identity: identity, places: places, epochs: epochs, related: related,
            progress: { done, total in
                guard Date().timeIntervalSince(shown) >= 5 || done == total else { return }
                shown = Date()
                let seconds = Date().timeIntervalSince(started)
                let rate = seconds > 0 ? Double(done) / seconds : 0
                let left = rate > 0 ? Double(total - done) / rate : 0
                print(
                    "refresh: \(OrigamiPlan.gigabytes(UInt64(done))) of \(OrigamiPlan.gigabytes(UInt64(total))) swept, "
                        + String(format: "%.0f MB/s", rate / 1e6)
                        + (done < total ? ", about \(Int((left / 60).rounded(.up))) min left" : ""))
            })
        // The drive history's count for these files starts again (`origami doctor`, the dashboard's Drive line).
        DriveReadLog.rewritten(family ? url.deletingLastPathComponent() : url, data: settings.dataDir)
        forgetSpeed(url, settings: settings)
        // What stayed keyed by the old identity (a copy on a drive not connected) is named as the model before its refresh.
        if let entry = ModelRegistry.entry(settings.model, settings: settings) {
            ModelRegistry.retire(
                entry, replacedBy: entry["name"] as? String, ids: [family ? outcome.old : String(outcome.old.prefix(12))],
                settings: settings)
        }
        print("Rewrote \(outcome.rewritten.count) file\(outcome.rewritten.count == 1 ? "" : "s"); every one read back the same bytes.")
        for line in outcome.rebound { print("  " + line) }
        for folder in offline {
            print("  not connected: \(folder.path); copies there keep the old identity and are listed as orphans once it is back")
        }
        print("Identity \(outcome.old.prefix(12)) → \(outcome.new.prefix(12)). Start the server as usual.")
    }

    /// The drive's saved speed was measured on the files as they read before the rewrite, often slowly (the reason to
    /// refresh), and slow by less than `DriveSpeedReview` drops: the next start measures the rewritten files.
    static func forgetSpeed(_ url: URL, settings: Settings) {
        if let key = DriveSpeeds.key(url) { DriveSpeeds.forget([key], in: settings.driveSpeedsFile) }
    }

    /// `origami refresh --copy`: rewrites each hot copy of the model at `url` in place (`HotCopy.rewriteInPlace`), for a
    /// drive that reads the copy slowly (ROADMAP R15: the internal drive cannot be replaced). The copy keeps its
    /// records, and the model its identity, counts and saved prompts.
    static func copies(_ url: URL, settings: Settings, yes: Bool) throws {
        let identity: String
        let first: URL?
        if Models.isGGUF(url.path) {
            let key = FamilySupport.key(try ExpertVariant.source(url, settings: settings))
            identity = LocalModelIdentity.familyRepresentation(source: key)
            first = FamilySetup.hotFolder(settings, key: key)
        } else {
            let model = try ModelSource(directory: url)
            identity = Stores.hotCopyIdentity(model)
            first = Stores.hotCopyFolder(model: model, settings: settings, found: Stores.scan(model: model, settings: settings))
        }
        let folders = [first].compactMap { $0 } + settings.hotCopyMore.compactMap { HotCopy.copyFolder(settings, place: $0.folder) }
        let found = folders.compactMap { OrigamiEngine.HotCopy.open(folder: $0, identity: identity) }
        guard !found.isEmpty else { throw CLIError("\(url.lastPathComponent) has no hot copy to refresh") }
        let bytes = found.reduce(Int64(0)) { $0 + $1.bytes }
        print(
            "This rewrites the hot cop\(found.count == 1 ? "y" : "ies") of \(url.lastPathComponent) (\(OrigamiPlan.gigabytes(UInt64(bytes)))) in place:"
                + " each byte is read and written back to the same place, then read again and compared (SHA-256), so the drive stores"
                + " them on fresh blocks. It is safe to stop and run again. The model's files are not touched.")
        guard Terminal.confirm("Refresh the hot copy of \(url.lastPathComponent)?", yes: yes) else { throw CLIError("Nothing was changed") }
        for copy in found {
            var shown = Date()
            try copy.rewriteInPlace { done in
                guard Date().timeIntervalSince(shown) >= 5 else { return }
                shown = Date()
                print("refresh: \(OrigamiPlan.gigabytes(UInt64(done))) of \(OrigamiPlan.gigabytes(UInt64(2 * copy.bytes))) swept")
            }
            DriveReadLog.rewritten(copy.folder, data: settings.dataDir)
            forgetSpeed(copy.folder, settings: settings)
            print("Rewrote \(copy.payload.path); it read back the same bytes.")
        }
    }

    /// What the identity of the model at `url` binds (`ModelRefresh`): whether it is a GGUF, its weight files, its identity
    /// as they are now, a GGUF key's saved-prompt epochs, and the other models these files' identity binds. Refresh,
    /// a dashboard move and Locate re-bind with it.
    static func binding(_ url: URL, settings: Settings) throws -> (
        family: Bool, files: [URL], identity: () throws -> String, epochs: (String) -> [String], related: [ModelRefresh.Related]
    ) {
        let family = Models.isGGUF(url.path)
        let files: [URL]
        let identity: () throws -> String
        var epochs: (String) -> [String] = { _ in [] }
        // A low-bit variant's identity binds its base's files and its experts file: refreshing either re-keys the base
        // and every variant of it (`ModelRefresh.Related`).
        var related: [ModelRefresh.Related] = []
        let entries = (try? ModelRegistry.entries(settings)) ?? []
        let own = ModelRegistry.entry(settings.model, settings: settings)
        let ownName = own?["name"] as? String
        if family {
            let parts = try GGUFSource.parts(url)
            // A variant's experts file is read as heavily as the base's experts: it is rewritten too.
            let experts = own.map(ExpertVariant.expertFiles) ?? []
            // Unreadable, the variant would be refreshed as its base and its counts left under a key nothing has.
            if own?["family"] != nil, let file = own?["experts"] as? String, experts.isEmpty {
                throw CLIError("The low-bit experts file \(file) cannot be read: connect its drive, then run `origami refresh` again")
            }
            files = parts.files.map { parts.folder.appendingPathComponent($0) } + experts.map { URL(fileURLWithPath: $0) }
            identity = { FamilySupport.key(folder: parts.folder, files: parts.files, experts: experts) }
            // A GGUF whose geometry cannot be read has no saved prompts to carry; its copies and counts still are.
            let context = (try? ModelGeometry(gguf: try GGUFSource(url)).context).map { min(settings.context, $0) }
            // The saved prompts of the current settings; those of other contexts or builds stay and are removed as other
            // builds' are.
            /// The epochs of the model `name` (nil: the base) reads, as `serve` reads it.
            func epochsOf(_ name: String?) -> (String) -> [String] {
                var probe = settings
                probe.model = name ?? url.path
                guard let context, let gguf = try? ExpertVariant.source(url, settings: probe) else { return { _ in [] } }
                // Under this build's digest, and 3.1.8.3's while it keys the same bits (`PromptCanary.adopt` takes those later).
                let digests = [ArithmeticDigest.family.value] + [ArithmeticDigest.family.before].compactMap { $0 }
                return { key in
                    digests.compactMap {
                        try? Commands.familyEpoch(
                            gguf, identity: LocalModelIdentity.familyRepresentation(source: key), resident: "", context: context,
                            arithmetic: $0)
                    }
                }
            }
            epochs = epochsOf(experts.isEmpty ? nil : ownName)
            if !experts.isEmpty {
                related.append(
                    .init(
                        name: url.path, identity: { FamilySupport.key(folder: parts.folder, files: parts.files) }, epochs: epochsOf(nil)))
            }
            for entry in entries {
                let variant = ExpertVariant.expertFiles(entry)
                guard let name = entry["name"] as? String, name != ownName, entry["path"] as? String == url.path, !variant.isEmpty
                else { continue }
                related.append(
                    .init(
                        name: name, identity: { FamilySupport.key(folder: parts.folder, files: parts.files, experts: variant) },
                        epochs: epochsOf(name)))
            }
        } else {
            let model = try ModelSource(directory: url)
            files =
                model.shardSizes.keys.sorted().map { model.directory.appendingPathComponent($0) }
                + (model.ggufExperts?.files.map { URL(fileURLWithPath: $0.path) } ?? [])
            identity = { try ModelSource.fingerprint(directory: url) }
            // A Qwen variant's folder links its base's shards: the base and each variant of it.
            func baseOf(_ entry: [String: Any]) -> String? {
                guard let base = entry["base"] as? String, let path = entry["path"] as? String, ExpertVariant.owns(path, settings: settings)
                else { return nil }
                return Models.locate(base, settings: settings)?.standardizedFileURL.path
            }
            let base = own.flatMap(baseOf) ?? url.standardizedFileURL.path
            let group = [base] + entries.compactMap { baseOf($0) == base ? $0["path"] as? String : nil }
            for path in Set(group) where URL(fileURLWithPath: path).standardizedFileURL.path != url.standardizedFileURL.path {
                related.append(.init(name: path, identity: { try ModelSource.fingerprint(directory: URL(fileURLWithPath: path)) }))
            }
        }
        return (family, files, identity, epochs, related)
    }

    /// Moves what is keyed by the model's identity `before` (each related model's in `relatedBefore`, by name) to the
    /// identity of the files at `url` now: after a move or a Locate, whose bytes are the same (a rename, or a copy checked
    /// by SHA-256). Each model's identity that cannot be computed is left as it is. What was moved, one line each.
    static func rebind(
        _ url: URL, before: String?, relatedBefore: [String: String], settings: Settings
    ) throws -> [String] {
        // A model whose identity was not known before, or cannot be computed now (not a model Origami opens), has none to move.
        guard before != nil || !relatedBefore.isEmpty, let bound = try? binding(url, settings: settings) else { return [] }
        let (family, identity, epochs, related) = (bound.family, bound.identity, bound.epochs, bound.related)
        let places = ModelRefresh.Places(data: settings.dataDir, prompts: settings.promptCacheRoot, folders: settings.storeFolders)
        var lines: [String] = []
        if let before, let now = try? identity(), now != before {
            lines += try ModelRefresh.rebind(family: family, old: before, new: now, places: places, epochs: epochs)
        }
        for item in related {
            guard let old = relatedBefore[item.name], let now = try? item.identity(), now != old else { continue }
            lines += try ModelRefresh.rebind(family: family, old: old, new: now, places: places, epochs: item.epochs)
        }
        return lines
    }
    /// The identity of the model at `url` and of each related one (by name), as `rebind` takes them; nil when unknown.
    static func identities(_ url: URL, settings: Settings) -> (own: String?, related: [String: String]) {
        guard let bound = try? binding(url, settings: settings) else { return (nil, [:]) }
        return (
            try? bound.identity(),
            Dictionary(bound.related.compactMap { item in (try? item.identity()).map { (item.name, $0) } }) { first, _ in first }
        )
    }
}
