import CryptoKit
import Darwin
import Foundation
import OrigamiEngine

/// Where Origami's files are, and moving or removing them from the dashboard (`GET v1/storage`, the `move` and `remove`
/// jobs). The map is read from what Origami already knows, never by searching drives: the model registry and the
/// configured model, the stores and the hot copy's folder (Qwen's packs, further copies and hot copy, the family engines' packs and hot
/// copies), the data folder (the family hot copies and support files, downloads and conversions), the saved prompts'
/// folder, and the folders those settings named before (`former`), whose leftovers nothing reads. Each item has a role,
/// its model (by fingerprint or GGUF key; none: an orphan, unless a model found on this Mac has it) and what may be done with it: a model, a pack or the saved
/// prompts are moved (copied, read back and compared, then the source deleted), a hot copy is rebuilt where the hot
/// copy's folder is set (cheaper than a copy: it is made from counts at the next start), and what nothing uses is removed.
enum Storage {
    struct Item: Sendable {
        let path: String, role: String
        var model: String?
        var bytes: UInt64 = 0, allocated: UInt64 = 0
        var note: String?
        /// Nothing reads it (an orphan, an interrupted build or move, a copy in a folder no longer used): it may be removed.
        var unused = false
        /// Unused itself, kept only for what it holds or is inside: it does not stop a move of what holds it.
        var kept = false
        /// Its drive answered; an item on one that did not is listed from the registry alone, without actions.
        var readable = true
        /// What a move updates: `model` (the registry), `setting` (the `model` setting), `pack` (the stores) or `prompts`
        /// (`context_dir`); nil: not moved.
        var move: String?
        var deleteBlocked: String?
        /// The model an earlier Origami version (2.x) made it for, under that version's identity: this one never reads it.
        var earlier: String?
        /// Every name its model is registered under (one folder or fingerprint, two names): a move stops any of them.
        var users: Set<String> = []
        /// The paths it is made of, each with its name in a destination folder (a split GGUF's parts, a family pack's
        /// `family-<key>/pack-<type>`); empty: the item's own path under its own name.
        var parts: [Part] = []
        var pieces: [Part] { parts.isEmpty ? [Part(from: path, to: (path as NSString).lastPathComponent)] : parts }
        var name: String { role == "model" ? model ?? (path as NSString).lastPathComponent : (path as NSString).lastPathComponent }
    }
    struct Part: Sendable, Codable { let from: String, to: String }

    /// A checked move: each part's source and target, the bytes a copy reads, and whether every part only changes its
    /// name on its own drive (a rename: nothing copied, no room needed).
    struct Move: Sendable { let item: Item, folder: String, pairs: [Part], bytes: UInt64, renames: Bool, device: dev_t }
    /// The models on this Mac not added (the front's last complete `Models.found`, with each one's `fingerprint` or
    /// `key`), and why that list may be incomplete (no search finished yet, a drive did not answer): a pack, copy or
    /// hot copy of a model not registered is removable only once its model is known to be nowhere on this Mac.
    typealias Found = (models: [[String: Any]], unknown: String?)

    // MARK: the map

    static func items(_ settings: Settings, found: Found) -> [Item] {
        // A registry that cannot be read (a file being replaced by hand) would make every pack an orphan: nothing goes then.
        let entries = try? ModelRegistry.entries(settings)
        var items: [Item] = []
        var fingerprints: [String: String] = [:]
        var keys: [String: String] = [:]
        var keysKnown = true
        var fingerprintsKnown = true
        var registered = Set<String>()
        var sources = Set<String>()
        var parents = [settings.modelsDir.path]
        // The files models read (resolved where their drive answers): what holds or is inside one is never removed.
        var needed: [String] = []
        // Each name's identities (its folder, fingerprint, GGUF key), and the GGUFs a conversion reads with its reader.
        var identities: [String: Set<String>] = [:]
        var borrowed: [(path: String, reader: String)] = []
        var modelPaths: [String: String] = [:]
        // Each model's 2.x identity (a Qwen fingerprint12, a GGUF key), which names the copies that version made.
        var earlierModels: [String: String] = [:]
        /// `listed` false: a GLM or DeepSeek low-bit variant, whose path is its base's GGUF: counted (its identities, its
        /// experts file) but no item of its own, so nothing done to it moves, deletes or unlists the base's files.
        func add(
            _ name: String, path: String, family: Bool, fingerprint: String?, move: String, experts: [String] = [], listed: Bool = true
        ) {
            if let fingerprint {
                fingerprints[String(fingerprint.prefix(12))] = name
                identities[name, default: []].insert("f" + fingerprint.prefix(12))
            }
            identities[name, default: []].insert("p" + (reachable(path) ? resolved(path) : path))
            if listed { registered.insert(path) }
            parents.append((path as NSString).deletingLastPathComponent)
            var item = Item(path: path, role: "model", model: name, move: move)
            guard reachable(path) else {
                if family { keysKnown = false }
                needed.append(path)
                item.readable = false
                item.move = nil
                item.note = "its drive does not answer"
                if listed { items.append(item) }
                return
            }
            guard FileManager.default.fileExists(atPath: path) else {
                if family { keysKnown = false }
                item.readable = false
                item.move = nil
                item.note = "Model location is unavailable"
                if listed { items.append(item) }
                return
            }
            needed.append(resolved(path))
            modelPaths[name] = path
            if Models.isGGUF(path), let parts = try? GGUFSource.parts(URL(fileURLWithPath: path)) {
                if family && !listed && experts.isEmpty {
                    // A variant whose experts file cannot be read: its key is unknown, so nothing is taken for an orphan.
                    keysKnown = false
                } else if family {
                    let key = FamilySupport.key(folder: parts.folder, files: parts.files, experts: experts)
                    keys[key] = name
                    identities[name, default: []].insert("k" + key)
                    earlierModels[FamilySupport.earlierKey(folder: parts.folder, files: parts.files)] = name
                }
                if !isFolder(path) { item.parts = parts.files.map { Part(from: parts.folder.appendingPathComponent($0).path, to: $0) } }
            } else if family {
                keysKnown = false
            } else if let fingerprint = ModelSource.earlierFingerprint(directory: URL(fileURLWithPath: path)) {
                earlierModels[String(fingerprint.prefix(12))] = name
            }
            needed += item.parts.map { resolved($0.from) } + experts.map { reachable($0) ? resolved($0) : $0 }
            // A Qwen conversion of a GGUF reads its routed experts from that GGUF, wherever it is.
            if let data = try? Data(contentsOf: URL(fileURLWithPath: path + "/" + GGUFExperts.file)),
                let files = ((try? JSONSerialization.jsonObject(with: data)) as? [String: Any])?["files"] as? [[String: Any]]
            {
                let read = files.compactMap { ($0["path"] as? String).map { reachable($0) ? resolved($0) : $0 } }
                needed += read
                borrowed += read.map { ($0, name) }
            }
            if listed { items.append(item) }
        }
        for entry in entries ?? [] {
            guard let name = entry["name"] as? String, let path = entry["path"] as? String else { continue }
            let manifest = entry["manifest"] as? [String: Any]
            let variant = entry["family"] != nil && entry["experts"] != nil
            add(
                name, path: path, family: entry["family"] != nil, fingerprint: manifest?["fingerprint"] as? String, move: "model",
                experts: ExpertVariant.expertFiles(entry), listed: !variant)
            // A Qwen variant's folder (`model add --base`) is links to its base's files: moved, it would copy the base.
            if entry["base"] != nil, ExpertVariant.owns(path, settings: settings), items.last?.path == path,
                items[items.count - 1].readable
            {
                items[items.count - 1].move = nil
                items[items.count - 1].note = "a low-bit variant's links to its base's files, which a move would copy whole"
            }
            if !variant, let from = (entry["converted_from"] ?? manifest?["converted_from"]) as? String {
                sources.insert(from)
                if items[items.count - 1].readable { items[items.count - 1].note = "converted from " + from }
            }
        }
        // The configured model when it is a path or a download not registered: its copies are not orphans. It is named as
        // the front lists it (`servedID`), so a move of its files stops it while it runs.
        if ModelRegistry.entry(settings.model, settings: settings) == nil {
            let spec = settings.model.map(Settings.expand)
            if let spec, Models.isGGUF(spec) {
                if !registered.contains(spec) {
                    add(
                        reachable(spec) ? ModelFront.servedID(settings) : Models.displayName(URL(fileURLWithPath: spec)), path: spec,
                        family: true, fingerprint: nil, move: "setting")
                }
            } else if let url = Models.locate(settings.model, settings: settings) {
                if !registered.contains(url.path) {
                    let fingerprint = (try? ModelSource(directory: url))?.fingerprint
                    if fingerprint == nil { fingerprintsKnown = false }
                    add(ModelFront.servedID(settings), path: url.path, family: false, fingerprint: fingerprint, move: "setting")
                }
                // A path with nothing there is gone; a drive that does not answer, or a repo id not found, may still hold it.
            } else if !(spec.map { $0.hasPrefix("/") && reachable($0) } ?? false) {
                fingerprintsKnown = false
            }
        }
        let profileFolders = (entries ?? []).flatMap { entry in
            [entry["hot_copy_dir"] as? String].compactMap { $0 }
                + ((entry["hot_copy_more"] as? String).flatMap { try? Settings.copyPlaces($0) } ?? []).map(\.folder.path)
        }
        let stores = unique(settings.storeFolders.map(\.path) + profileFolders)
        let hot = settings.hotCopyDir?.path
        // Every hot copy place's folder (`hot_copy_dir`, `hot_copy_more`): a family copy there is in use.
        let hotPlaces = Set([hot].compactMap { $0 } + settings.hotCopyMore.map(\.folder.path))
        let former = former(settings)
        let manager = FileManager.default
        // A model not registered may be one found on this Mac (by a Qwen fingerprint12 or a GGUF key): its files stay.
        var owners: [String: String] = [:]
        for model in found.models {
            guard let name = model["name"] as? String else { continue }
            if let fingerprint = model["fingerprint"] as? String { owners[String(fingerprint.prefix(12))] = name }
            if let key = model["key"] as? String { owners[key] = name }
        }
        // A move whose carry-over waits for its model's files to be readable: what is keyed on its identity before is its.
        let waiting = MoveJournal.waiting(settings)
        func orphan(_ model: String?, _ id: String) -> String? {
            if model == nil || model == "?", let name = waiting[id] {
                return "belongs to \(name) before its move: recover with `origami move recover` once its files are readable"
            }
            guard model == nil else { return nil }
            if let name = owners[id] { return "for \(name), found on this Mac, not added" }
            return found.unknown.map { "its model is not registered and may be on this Mac: " + $0 } ?? "its model is not registered"
        }
        /// Neither registered nor configured nor found on this Mac, by a complete search, nor waiting for a move's carry-over.
        func gone(_ model: String?, _ id: String) -> Bool {
            model == nil && owners[id] == nil && waiting[id] == nil && found.unknown == nil
        }
        /// A copy a 2.x Origami made for `name` (its identity then): unused, whatever the model search found.
        func earlier(_ item: Item, _ name: String?) -> Item {
            guard let name else { return item }
            var item = item
            item.earlier = name
            item.model = nil
            item.unused = true
            item.move = nil
            return item
        }
        let elsewhere = "in a folder Origami no longer uses"
        let retired = "smaller experts an earlier Origami made; nothing reads them now"
        // A family hot copy moving to another drive: the old one is read until the new one (`experts.next` meanwhile) is ready.
        let replaced = "read until the new hot copy is ready"
        func moving(_ key: String) -> Bool {
            manager.fileExists(atPath: FamilySetup.hotFolder(settings, key: key).appendingPathComponent("experts.next").path)
        }
        // A former folder named another way (a link, /tmp for /private/tmp) is a current one: its packs are in use.
        let currentReal = Set((stores + [settings.promptCacheRoot.path]).filter(reachable).map(resolved))
        // Where an earlier version's setup wrote Qwen's hot mirror (`Stores.mirrorHome`): `hot_copy_dir`, else a store on
        // the model's drive. A mirror elsewhere is a further copy on another drive.
        let modelDisk = settings.hotCopyDir == nil ? Models.locate(settings.model, settings: settings).map { Stores.volume($0).disk } : nil
        func oldHotHome(_ store: String) -> Bool {
            settings.hotCopy != 0
                && (settings.hotCopyDir.map { resolved($0.path) == resolved(store) }
                    ?? (modelDisk != nil && Stores.volume(URL(fileURLWithPath: store)).disk == modelDisk))
        }
        for folder in unique(stores + former) where reachable(folder) {
            let current = stores.contains(folder)
            if !current, currentReal.contains(resolved(folder)) { continue }
            // Each Qwen payload is independently movable: full, reduced-width and partial expert copies.
            for sub in Commands.origamiFolders(in: URL(fileURLWithPath: folder, isDirectory: true)) {
                let names = Set(children(sub.path))
                let id = String(sub.lastPathComponent.suffix(12))
                let model = fingerprints[id] ?? (fingerprintsKnown ? nil : "?")
                var tracked = Set<String>()
                for header in ["pack.json", "pack-3bit.json", "pack-2bit.json", "mirror.json"] where names.contains(header) {
                    let path = sub.appendingPathComponent(header).path
                    guard let object = try? Settings.readConfig(URL(fileURLWithPath: path)),
                        let payload = object["expert_file"] as? String
                    else { continue }
                    let files = [header, payload] + ((object["ple_file"] as? String).map { [$0] } ?? [])
                    guard files.allSatisfy({ !$0.isEmpty && !$0.contains("/") && !$0.hasPrefix(".") && names.contains($0) }) else {
                        continue
                    }
                    tracked.formUnion(files)
                    var item = Item(path: path, role: header == "mirror.json" ? "hot copy" : "pack", model: model)
                    item.parts = files.map { Part(from: sub.appendingPathComponent($0).path, to: sub.lastPathComponent + "/" + $0) }
                    item.unused = gone(model, id) || !current
                    item.note =
                        !current && !gone(model, id) ? elsewhere : orphan(model, id) ?? header.replacingOccurrences(of: ".json", with: "")
                    item.move = model != nil && model != "?" ? (item.role == "hot copy" ? "hot" : "pack") : nil
                    // A 3- or 2-bit pack (`expert_bits`, since removed): nothing reads it (`Stores.retiredExperts`).
                    if header.hasPrefix("pack-") {
                        item.unused = true
                        item.note = retired
                        item.move = nil
                    }
                    // The offline hot mirror an earlier version's setup wrote where the hot copy now goes: not read; the
                    // keeper removes it when it builds the hot copy (`Stores.retireLegacyMirror`).
                    if header == "mirror.json", current, !gone(model, id), oldHotHome(folder) {
                        item.unused = true
                        item.note = "offline mirror of an earlier version; the hot copy built while serving replaces it"
                        item.move = nil
                    }
                    items.append(earlier(item, fingerprints[id] == nil ? earlierModels[id] : nil))
                }
                // Qwen's hot copy (`hot/index.json`), built while serving: moved with `hot_copy_dir`.
                if names.contains("hot"), manager.fileExists(atPath: sub.appendingPathComponent("hot/index.json").path) {
                    tracked.insert("hot")
                    var item = Item(path: sub.appendingPathComponent("hot").path, role: "hot copy", model: model)
                    item.parts = [Part(from: item.path, to: sub.lastPathComponent + "/hot")]
                    item.unused = gone(model, id) || !current
                    item.note = !current && !gone(model, id) ? elsewhere : orphan(model, id)
                    item.move = model != nil && model != "?" ? "hot" : nil
                    items.append(earlier(item, fingerprints[id] == nil ? earlierModels[id] : nil))
                }
                // A further copy in the hot-copy format (`copy/index.json`), written by `origami setup` on another drive.
                let further = PackStore.furtherCopyFolder
                if names.contains(further), manager.fileExists(atPath: sub.appendingPathComponent(further + "/index.json").path) {
                    tracked.insert(further)
                    var item = Item(path: sub.appendingPathComponent(further).path, role: "hot copy", model: model)
                    item.parts = [Part(from: item.path, to: sub.lastPathComponent + "/" + further)]
                    item.unused = gone(model, id) || !current
                    item.note = !current && !gone(model, id) ? elsewhere : orphan(model, id) ?? "further copy"
                    items.append(earlier(item, fingerprints[id] == nil ? earlierModels[id] : nil))
                }
                let leftovers = names.subtracting(tracked).filter {
                    $0.hasSuffix(".partial") || $0.hasSuffix(".next") || $0.hasSuffix(".tmp")
                }
                for name in leftovers {
                    items.append(
                        Item(
                            path: sub.appendingPathComponent(name).path, role: "staging", model: model, note: "unfinished copy",
                            unused: true))
                }
                if tracked.isEmpty && !names.isEmpty && leftovers.isEmpty {
                    items.append(
                        Item(
                            path: sub.path, role: "support", model: model, note: "Unrecognized or incomplete store; inspect before removing"
                        ))
                }
            }
            // GLM and DeepSeek: `family-<key>/pack-<type>` in a store, the hot copy in `family-<key>/hot` in hot_copy_dir.
            for child in children(folder) where child.range(of: "^family-[0-9a-f]{16}$", options: .regularExpression) != nil {
                let key = String(child.dropFirst(7))
                let base = folder + "/" + child
                let model = keysKnown || keys[key] != nil ? keys[key] : "?"
                for pack in children(base)
                where pack.hasPrefix("pack-") && manager.fileExists(atPath: base + "/" + pack + "/pack.json") {
                    // Converted experts (`expert_bits`, since removed): nothing reads them (`Stores.retiredExperts`).
                    var item = Item(path: base + "/" + pack, role: "pack", model: model, unused: true)
                    item.note = pack.dropFirst(5) + " experts: " + retired
                    item.parts = [Part(from: item.path, to: child + "/" + pack)]
                    items.append(earlier(item, keys[key] == nil ? earlierModels[key] : nil))
                }
                for hotName in children(base) where hotName == "hot" || hotName.hasPrefix("hot-") {
                    var item = Item(
                        path: base + "/" + hotName, role: "hot copy", model: model,
                        unused: gone(model, key) || !stores.contains(folder) && !hotPlaces.contains(folder) && !moving(key)
                            || hotName != "hot")
                    item.note =
                        hotName != "hot"
                        ? retired
                        : !hotPlaces.contains(folder) && !gone(model, key) ? (moving(key) ? replaced : elsewhere) : orphan(model, key)
                    item.parts = [Part(from: item.path, to: child + "/" + hotName)]
                    item.move = "hot"
                    items.append(earlier(item, keys[key] == nil ? earlierModels[key] : nil))
                }
            }
            if !current { items += prompts(folder, unused: true) }
        }
        // The family engines' counts and default hot copy, and their tokenizer and template, by GGUF key.
        let data = settings.dataDir.path
        for key in children(data + "/family") where isFolder(data + "/family/" + key) {
            let model = keysKnown || keys[key] != nil ? keys[key] : "?"
            if model == nil || keys[key] == nil && earlierModels[key] != nil {
                items.append(
                    earlier(
                        Item(path: data + "/family/" + key, role: "hot copy", note: orphan(nil, key), unused: gone(nil, key)),
                        earlierModels[key]))
            } else {
                for name in children(data + "/family/" + key) where name == "hot" || name.hasPrefix("hot-") {
                    var item = Item(path: data + "/family/" + key + "/" + name, role: "hot copy", model: model, move: "hot")
                    if name != "hot" {
                        item.unused = true
                        item.note = retired
                        item.move = nil
                    }
                    item.parts = [Part(from: item.path, to: "family-" + key + "/" + name)]
                    items.append(item)
                }
            }
        }
        for key in children(data + "/support") where isFolder(data + "/support/" + key) {
            let model = keysKnown || keys[key] != nil ? keys[key] : "?"
            items.append(
                earlier(
                    Item(
                        path: data + "/support/" + key, role: "support", model: model,
                        note: orphan(model, key) ?? "tokenizer and template", unused: gone(model, key)),
                    keys[key] == nil ? earlierModels[key] : nil))
        }
        // Downloads and conversions Origami wrote (by their marker files) that no model is registered for.
        for parent in unique(parents) where reachable(parent) {
            for child in children(parent) where !child.hasPrefix(".") {
                let path = parent + "/" + child
                // A conversion stopped part way: Qwen's (a folder) keeps its finished shards, GLM's (a file) starts again.
                if child.hasSuffix(".partial") {
                    items.append(
                        Item(
                            path: path, role: "conversion",
                            note: isFolder(path)
                                ? "unfinished conversion (resume with the same `origami model add`, or delete)"
                                : "unfinished conversion (start again with the same `origami model add`, or delete)", unused: true))
                    continue
                }
                guard !registered.contains(path), isFolder(path) else { continue }
                let names = children(path)
                if names.contains("origami-model.json") {
                    items.append(Item(path: path, role: "conversion", note: "a conversion no model is registered for", unused: true))
                    // A folder with `.part` files is an unfinished download only in Origami's own models folder.
                } else if names.contains(".origami-download.json")
                    || parent == settings.modelsDir.path && names.contains(where: { $0.hasSuffix(".part") })
                {
                    let note =
                        sources.contains(path)
                        ? "the download a model was converted from; the model runs from its conversion"
                        : names.contains(".origami-download.json") ? "a download no model is registered for" : "an unfinished download"
                    items.append(Item(path: path, role: "download", note: note, unused: true))
                }
            }
        }
        if reachable(settings.modelsDir.path), manager.fileExists(atPath: Models.ggufSupportFolder(settings).path) {
            items.append(Item(path: Models.ggufSupportFolder(settings).path, role: "support", note: "kept for GGUF conversions"))
        }
        if reachable(settings.promptCacheRoot.path) {
            // A Qwen `<fingerprint12>-<context>` folder no registered or configured model has: unreachable by any build.
            let known =
                fingerprintsKnown && entries != nil
                && entries!.allSatisfy { $0["family"] != nil || ($0["manifest"] as? [String: Any])?["fingerprint"] != nil }
            // A family epoch whose marker names another build, or a GGUF no model has (`FamilySetup.epochMarker`).
            let current = { (marker: [String: Any]) in
                marker["build"] as? String == FamilySetup.build && (!keysKnown || keys[marker["family_key"] as? String ?? ""] != nil)
            }
            for item in prompts(
                settings.promptCacheRoot.path, unused: false, unreachable: { known && fingerprints[String($0)] == nil }, current: current)
            {
                let id = PromptCache.modelFolder((item.path as NSString).lastPathComponent).map(String.init)
                items.append(item.unused ? earlier(item, id.flatMap { earlierModels[$0] }) : item)
            }
        }
        // A move that never finished (the server stopped mid-copy) leaves its staging copy.
        for path in pending(settings) where manager.fileExists(atPath: path) {
            let now = liveLock.withLock { live.contains(path) }
            items.append(
                Item(
                    path: path, role: "staging", note: now ? "a move under way" : "an interrupted move; the original is where it was",
                    unused: !now))
        }
        for index in items.indices where items[index].readable {
            for part in items[index].pieces {
                let (bytes, allocated) = size(part.from)
                items[index].bytes += bytes
                items[index].allocated += allocated
            }
            if let name = items[index].earlier {
                items[index].note =
                    "made by an earlier Origami version for \(name); not used by this version; safe to delete (frees "
                    + OrigamiPlan.gigabytes(items[index].allocated) + ")"
            }
        }
        // What is in use (another item's files, Origami's folders) is never removed through an item holding it or named
        // another way: a folder something was moved into, a former folder that is a current one by another spelling.
        let usedPieces = items.filter { !$0.unused && $0.readable }.flatMap(\.pieces).map { resolved($0.from) }
        let own = ownFolders(settings)
        for index in items.indices where items[index].unused {
            let paths = items[index].pieces.map { resolved($0.from) }
            let note =
                entries == nil
                ? "kept while Origami cannot read its list of models"
                : paths.contains(where: { path in needed.contains { overlaps(path, $0) } })
                    ? "a model reads files in it"
                    : paths.contains(where: { path in (usedPieces + own).contains { $0 == path || $0.hasPrefix(path + "/") } })
                        ? "it holds files Origami uses" : nil
            if let note {
                items[index].unused = false
                items[index].kept = true
                items[index].note = note
            }
        }
        for index in items.indices {
            guard let model = items[index].model, let own = identities[model] else { continue }
            items[index].users = Set(identities.filter { !$0.value.isDisjoint(with: own) }.keys)
            // A conversion reads its experts from that GGUF by its path: moved, the conversion would read nothing.
            if items[index].move != nil,
                let reader = borrowed.first(where: { pair in
                    !items[index].users.contains(pair.reader) && items[index].pieces.contains { overlaps(resolved($0.from), pair.path) }
                })?.reader
            {
                items[index].move = nil
                items[index].deleteBlocked = "\(reader) reads its experts from it"
                items[index].note = items[index].deleteBlocked
            }
        }
        for index in items.indices {
            let item = items[index]
            if ["pack", "hot copy"].contains(item.role), item.model == "?" || item.model == nil && !item.unused {
                items[index].deleteBlocked = "Reconnect the model disk to verify ownership"
            }
            for piece in item.pieces.map({ resolved($0.from) }) {
                if own.contains(where: { $0 == piece || $0.hasPrefix(piece + "/") }) {
                    items[index].deleteBlocked = "Contains an Origami data folder"
                }
                if let dependency = borrowed.first(where: { overlaps(piece, $0.path) && !item.users.contains($0.reader) }) {
                    items[index].deleteBlocked = "Used by " + dependency.reader
                }
                if items.contains(where: { other in
                    other.path != item.path && !other.unused
                        && other.pieces.contains { resolved($0.from) == piece || resolved($0.from).hasPrefix(piece + "/") }
                }) {
                    items[index].deleteBlocked = "Contains files belonging to another item"
                }
            }
            if item.role == "hot copy", items[index].move == nil, item.model != nil, item.model != "?" { items[index].move = "hot" }
        }
        var seen = Set<String>()
        return items.filter { seen.insert($0.readable ? resolved($0.path) : $0.path).inserted }
    }

    /// The saved prompts and long contexts Origami keeps in `folder` (Qwen's `<fingerprint12>-<context>`, the family
    /// engines' `<epoch16>` and `family-context`), as one item; anything else there is the user's. A Qwen folder of a
    /// fingerprint no model has (`unreachable`), and a family epoch whose marker is not `current`, is an item of its own,
    /// unused, unless a running server holds it; a family epoch without a marker stays in the shared item.
    private static func prompts(
        _ folder: String, unused: Bool, unreachable: (Substring) -> Bool = { _ in false }, current: ([String: Any]) -> Bool = { _ in true }
    ) -> [Item] {
        var names = children(folder).filter {
            $0 == "family-context" || $0.range(of: "^([0-9a-f]{12}-[0-9]+|[0-9a-f]{16})$", options: .regularExpression) != nil
        }
        let stale =
            unused
            ? []
            : names.filter {
                PromptCache.modelFolder($0).map(unreachable) == true && !PromptCache.inUse(URL(fileURLWithPath: folder + "/" + $0))
            }
        let marked: [(String, [String: Any])] =
            unused
            ? []
            : names.compactMap { name in
                FamilySetup.epoch(URL(fileURLWithPath: folder + "/" + name)).map { (name, $0) }
            }.filter { !current($0.1) && !PromptCache.inUse(URL(fileURLWithPath: folder + "/" + $0.0)) }
        names.removeAll { name in stale.contains(name) || marked.contains { $0.0 == name } }
        let old =
            stale.sorted().map {
                Item(
                    path: folder + "/" + $0, role: "prompts",
                    note: "saved prompts from an earlier Origami version or another model file; safe to delete", unused: true)
            }
            + marked.sorted { $0.0 < $1.0 }.map { name, marker in
                Item(
                    path: folder + "/" + name, role: "prompts",
                    note:
                        "saved prompts of \(marker["model"] as? String ?? "another model") from version \(marker["version"] as? String ?? "?"); safe to delete",
                    unused: true)
            }
        guard !names.isEmpty else { return old }
        return old + [
            Item(
                path: folder, role: "prompts", note: unused ? "in a folder Origami no longer uses" : "saved prompts and long contexts",
                unused: unused, move: unused ? nil : "prompts", parts: names.sorted().map { Part(from: folder + "/" + $0, to: $0) })
        ]
    }

    /// Whether the running model (`running`, a listed id) reads it: its own files and copies, and the saved prompts.
    static func inUse(_ item: Item, running: String?) -> Bool {
        guard let running, !item.unused else { return false }
        return item.model == running || item.users.contains(running) || item.role == "prompts"
    }

    /// `GET v1/storage`: the drives holding Origami's items and the folders it writes to, each with its size, free space
    /// and the bytes Origami's items take there by role, and its items.
    static func report(_ settings: Settings, running: String?, found: Found, sources: [[String: Any]]? = nil) -> [String: Any] {
        let items = items(settings, found: found)
        let mounted = volumes()
        noteVolumes(settings, volumes: mounted)
        let folders = settings.storeFolders.map(\.path) + [settings.dataDir.path, settings.promptCacheRoot.path]
        var drives: [String: [String: Any]] = [:]
        var order: [String] = []
        for path in unique(items.map(\.path) + folders) {
            let name = Stores.driveName(path)
            guard drives[name] == nil else { continue }
            var drive: [String: Any] = [
                "name": name, "path": name == "internal drive" ? FileManager.default.homeDirectoryForCurrentUser.path : "/Volumes/" + name,
                "items": [Any](), "roles": [String: UInt64](),
            ]
            if let volume = Folders.space(path) {
                if volume.mounted {
                    drive["free_bytes"] = volume.free
                    drive["total_bytes"] = volume.total
                } else {
                    drive["not_connected"] = true
                }
            } else {
                drive["permission"] = Folders.permission(path)
            }
            drives[name] = drive
            order.append(name)
        }
        for item in items {
            let name = Stores.driveName(item.path)
            var actions: [String] = []
            if item.readable, item.move != nil { actions.append("move") }
            // A hot copy of no registered model (found on this Mac, or not yet known) has nothing to rebuild it.
            if item.readable, item.role == "hot copy", !item.unused, item.model != nil { actions.append("rebuild") }
            if item.unused, item.deleteBlocked == nil { actions.append("remove") }
            if item.readable, item.deleteBlocked == nil, ["model", "pack", "hot copy", "prompts"].contains(item.role) {
                actions.append("delete")
            }
            if item.role == "model" {
                actions.append("forget")
                actions.append("relink")
            }
            var entry: [String: Any] = [
                "path": item.path, "name": item.name, "role": item.role, "bytes": item.bytes, "allocated_bytes": item.allocated,
                "in_use": inUse(item, running: running), "unused": item.unused, "actions": actions,
                "readable": item.readable, "blocked": item.deleteBlocked ?? NSNull(),
                "state": !item.readable
                    ? "unavailable"
                    : item.role == "staging" ? "incomplete" : item.unused ? "unused" : state(item, running: running, sources: sources),
            ]
            // "?": a family item whose model's drive did not answer, so it is neither matched nor an orphan.
            if let model = item.model, model != "?" { entry["model"] = model }
            if let note = item.note { entry["note"] = note }
            guard var drive = drives[name] else { continue }
            var roles = drive["roles"] as? [String: UInt64] ?? [:]
            roles[item.role, default: 0] += item.allocated
            drive["roles"] = roles
            drive["items"] = (drive["items"] as? [Any] ?? []) + [entry]
            drives[name] = drive
        }
        return [
            "observed_at": Date().timeIntervalSince1970, "revision": DashboardState.revision(settings),
            "operations": MoveJournal.pending(settings), "drives": order.compactMap { drives[$0] }, "running": running ?? NSNull(),
            // Folders on a drive renamed or mounted under another name: shown with `origami relink`, never relinked here.
            "relinks": relinks(settings, volumes: mounted).map { ["setting": $0.setting, "from": $0.from, "to": $0.to, "how": $0.how] },
            "stores": settings.stores.map(\.path), "hot_copy_dir": settings.hotCopyDir?.path ?? NSNull(),
            "context_dir": settings.promptCacheRoot.path,
        ]
    }

    // Hold every artifact of a loaded model during file operations, but mark only its actual sources active.
    private static func state(_ item: Item, running: String?, sources: [[String: Any]]?) -> String {
        guard inUse(item, running: running) else { return "retained" }
        guard ["hot copy", "pack"].contains(item.role) else { return "active" }
        guard let sources else { return "checking" }
        let active = sources.contains { source in
            guard let path = source["path"] as? String, let role = source["role"] as? String,
                !["off", "not_built", "not_needed", "not_possible", "not_connected", "not_read", "failed"].contains(
                    source["state"] as? String ?? "")
            else { return false }
            if item.path == path { return item.role == "hot copy" ? role == "hot" : ["pack", "more"].contains(role) }
            guard (item.path as NSString).deletingLastPathComponent == path else { return false }
            if item.role == "hot copy" {
                return ["hot", "more"].contains(role) && ["mirror.json", PackStore.furtherCopyFolder].contains(item.name)
            }
            return ["pack", "more"].contains(role) && item.name == "pack.json"
        }
        return active ? "active" : "retained"
    }

    // MARK: moves

    /// Checks moving `item` into `folder` before anything is written: the folder is one the folder browser shows, not
    /// inside the item, without an item of that name, and (unless every part is only renamed on its drive) with the room
    /// `Models.conversionNeeds` counts: the bytes and 10 GB kept free. A pack does not go to the hot copy's folder, which
    /// holds the hot copy alone.
    static func plan(_ item: Item, to asked: String, settings: Settings, items: [Item]) throws -> Move {
        guard item.move != nil, item.readable else {
            throw ServingFailure(
                "\(item.name) is not moved: "
                    + (item.role == "hot copy"
                        ? "a hot copy is rebuilt where the hot copy's folder is set"
                        : item.unused ? "nothing uses it" : item.note ?? "it stays where it is"),
                status: 403)
        }
        guard let folder = Folders.timed(3, on: asked, { Folders.real(asked) }) ?? nil, isFolder(folder) else {
            throw ServingFailure("No folder at \(asked)", status: 404)
        }
        guard Folders.root(of: folder, Folders.roots(settings)) != nil else {
            throw ServingFailure("Items move into a drive, your home folder or Origami's folders only", status: 403)
        }
        if item.move == "pack", folder == settings.hotCopyDir.map({ resolved($0.path) }) {
            throw ServingFailure("The hot copy's folder holds the hot copy alone; choose another folder for a pack")
        }
        let pairs = item.pieces.map { Part(from: $0.from, to: folder + "/" + $0.to) }
        // What it holds of Origami's (a store, the data or prompts folder, another model) would leave where it is named.
        let held =
            items.filter { $0.path != item.path && !$0.unused && !$0.kept }
            .flatMap { other in other.pieces.map { other.readable ? resolved($0.from) : $0.from } }
        let others = held + ownFolders(settings)
        for piece in item.pieces.map({ resolved($0.from) }) {
            if let held = others.first(where: { $0 == piece || $0.hasPrefix(piece + "/") }) {
                throw ServingFailure("\(item.name) holds \(held), which Origami uses; move that out first", status: 409)
            }
        }
        // A copy follows links (a Hugging Face snapshot's): never into another item's files, which it would copy whole.
        for piece in item.pieces {
            if let target = links(piece.from).first(where: { target in held.contains { overlaps(target, $0) } }) {
                throw ServingFailure("\(item.name) links to \(target), which belongs to another item; it is not moved", status: 409)
            }
        }
        for pair in pairs {
            // Resolved: a copy into its own (linked) folder would copy itself again and again.
            let from = resolved(pair.from)
            if folder == from || folder.hasPrefix(from + "/") { throw ServingFailure("\(item.name) cannot move into itself") }
            if pair.from == pair.to { throw ServingFailure("\(item.name) is already in \(folder)") }
            if FileManager.default.fileExists(atPath: pair.to) {
                throw ServingFailure("\(folder) already holds \((pair.to as NSString).lastPathComponent)", status: 409)
            }
        }
        guard access(folder, W_OK) == 0 else { throw ServingFailure("The destination is read-only", status: 403) }
        for pair in pairs {
            let staging = (pair.to as NSString).deletingLastPathComponent + "/.origami-moving-" + (pair.to as NSString).lastPathComponent
            guard !FileManager.default.fileExists(atPath: staging) else {
                throw ServingFailure("An interrupted copy is already at the destination. Recover it first.", status: 409)
            }
        }
        var here = stat()
        _ = stat(folder, &here)
        // A link is copied, not renamed: moved, a relative one (a Hugging Face snapshot's) would point nowhere.
        let renames = pairs.allSatisfy {
            var info = stat()
            return stat($0.from, &info) == 0 && info.st_dev == here.st_dev && !linked($0.from)
        }
        let bytes = item.pieces.reduce(UInt64(0)) { $0 + size($1.from).bytes }
        if !renames {
            let needs = Models.conversionNeeds(bytes)
            let free = Folders.space(folder)?.free ?? 0
            guard free >= needs else {
                throw ServingFailure(
                    "Moving \(item.name) needs \(OrigamiPlan.gigabytes(needs)) free there, 10 GB to spare included (\(OrigamiPlan.gigabytes(free)) now)"
                )
            }
        }
        return Move(item: item, folder: folder, pairs: pairs, bytes: bytes, renames: renames, device: here.st_dev)
    }

    /// Moves the parts, then updates what names their place, then deletes the sources. On its own drive each part is
    /// renamed; across drives it is copied to `.origami-moving-<name>` beside its target, read back past the cache and
    /// compared by SHA-256 with what was read from the source, and renamed into place only once every part is. Any
    /// failure or cancel before the update removes the copies and leaves the source as it was.
    static func execute(_ move: Move, settings: Settings, step: @escaping (String, Double?) -> Void, cancelled: @escaping () -> Bool) throws
        -> String
    {
        let manager = FileManager.default
        // A keeper swaps a hot copy's records in place under its folder's `.refresh.lock`: held here, the copy moved is
        // one published state (a swap in progress finishes first; later ones fail until the move is done).
        var locks: [Int32] = []
        defer {
            for lock in locks {
                _ = flock(lock, LOCK_UN)
                close(lock)
            }
        }
        if move.item.role == "hot copy" {
            for pair in move.pairs {
                let lock = open(pair.from + "/.refresh.lock", O_WRONLY)
                if lock >= 0 {
                    _ = flock(lock, LOCK_EX)
                    locks.append(lock)
                }
            }
        }
        for pair in move.pairs {
            try manager.createDirectory(atPath: (pair.to as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
        }
        let journal = try MoveJournal.begin(move, settings: settings)
        // A model's identity binds its path: what is keyed on it (copies, counts, tuning, saved prompts) is re-bound
        // between "committed" and the journal's end, so a stop in between is finished by Recover.
        if ["model", "setting"].contains(move.item.move) {
            var probe = settings
            probe.model = move.item.move == "model" ? move.item.model : move.item.path
            let before = Refresh.identities(URL(fileURLWithPath: move.item.path), settings: probe)
            try journal.rebinding(
                to: move.folder + "/" + (move.item.path as NSString).lastPathComponent, model: probe.model, own: before.own,
                related: before.related)
        }
        var placed: [Part] = []
        /// Puts back what was placed; the journal goes only once every part is back where it was.
        func undo() {
            var clean = true
            for pair in placed.reversed() {
                if move.renames {
                    if Storage.renameNew(pair.to, pair.from) != 0 { clean = false }
                } else if (try? manager.removeItem(atPath: pair.to)) == nil {
                    clean = false
                }
            }
            if clean { try? journal.finish() }
        }
        if move.renames {
            for pair in move.pairs {
                if cancelled() {
                    undo()
                    throw CLIError("Cancelled")
                }
                guard Storage.renameNew(pair.from, pair.to) == 0 else {
                    let reason = String(cString: strerror(errno))
                    undo()
                    throw CLIError("Cannot move \(pair.from): \(reason)")
                }
                placed.append(pair)
            }
            do {
                if cancelled() { throw CLIError("Cancelled") }
                try update(move, settings: settings)
            } catch {
                undo()
                throw error
            }
            try journal.phase("committed")
            for pair in move.pairs where pair.to.dropFirst(move.folder.count + 1).contains("/") {
                rmdir((pair.from as NSString).deletingLastPathComponent)
            }
            let unbound = carry(journal, settings: settings)
            if unbound == nil { try journal.finish() }
            return message(move, settings: settings) + (unbound ?? "")
        }
        let stagings = move.pairs.map {
            ($0.to as NSString).deletingLastPathComponent + "/.origami-moving-" + ($0.to as NSString).lastPathComponent
        }
        try setPending(settings, adding: stagings)
        liveLock.withLock { live.formUnion(stagings) }
        defer {
            liveLock.withLock { live.subtract(stagings) }
            // A copy on a drive that went away stays listed, to be removed once the drive is back.
            let gone = stagings.filter { staging in
                var info = stat()
                return (try? manager.removeItem(atPath: staging)) != nil
                    || lstat(staging, &info) != 0 && stat(move.folder, &info) == 0 && info.st_dev == move.device
            }
            try? setPending(settings, removing: gone)
        }
        let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: 8 << 20, alignment: 16384)
        defer { buffer.deallocate() }
        var done: UInt64 = 0
        var shown = Date.distantPast
        let total = max(1, move.bytes) * 2
        let counted: (UInt64) -> Void = { bytes in
            done += bytes
            if Date().timeIntervalSince(shown) > 0.25 {
                shown = Date()
                step("Copying and checking", min(0.99, Double(done) / Double(total)))
            }
        }
        // What each source file was before it was copied: one written meanwhile keeps the original.
        let before = move.pairs.map { stamps($0.from) }
        do {
            for (pair, staging) in zip(move.pairs, stagings) {
                try? manager.removeItem(atPath: staging)
                try copy(pair.from, staging, buffer: buffer, counted: counted, cancelled: cancelled, depth: 0)
            }
            pauseForChecks()
            for (pair, then) in zip(move.pairs, before) {
                let now = stamps(pair.from)
                if let changed = Set(then.keys).union(now.keys).sorted().first(where: { then[$0] != now[$0] }) {
                    throw CLIError("\(changed) changed while it was copied: the original is kept and the copy removed; try again")
                }
            }
            for (pair, staging) in zip(move.pairs, stagings) {
                guard Storage.renameNew(staging, pair.to) == 0 else {
                    throw CLIError("Cannot put the copy in place at \(pair.to): \(String(cString: strerror(errno)))")
                }
                placed.append(pair)
            }
            // The renames into place reach the drive before anything names them (and before the source goes).
            for folder in Set(move.pairs.map { ($0.to as NSString).deletingLastPathComponent }) {
                let fd = open(folder, O_RDONLY)
                if fd >= 0 {
                    if fcntl(fd, F_FULLFSYNC) == -1 { fsync(fd) }
                    close(fd)
                }
            }
            try journal.phase("verified")
            if cancelled() { throw CLIError("Cancelled") }
            try update(move, settings: settings)
        } catch {
            undo()
            throw error
        }
        try journal.phase("committed")
        let unbound = carry(journal, settings: settings)
        step("Deleting the original", nil)
        var kept: [String] = []
        for (pair, then) in zip(move.pairs, before) {
            // Written since the comparison above: kept rather than lost.
            if stamps(pair.from) != then {
                kept.append(pair.from + " (it changed after it was copied)")
                continue
            }
            do { try manager.removeItem(atPath: pair.from) } catch { kept.append(pair.from) }
            // A family pack's `family-<key>` folder goes once empty.
            if pair.to.dropFirst(move.folder.count + 1).contains("/") { rmdir((pair.from as NSString).deletingLastPathComponent) }
        }
        if kept.isEmpty, unbound == nil { try journal.finish() }
        return message(move, settings: settings) + (unbound ?? "")
            + (kept.isEmpty
                ? "" : " The original was not deleted; delete it by hand once it is not needed: " + kept.joined(separator: ", "))
    }
    /// Carries what is keyed on a moved model over to its new identity. When that cannot be done now, the journal stays
    /// (Recover retries it) and the reason goes with the move's outcome, which the dashboard's job log and the command print.
    private static func carry(_ journal: MoveJournal, settings: Settings) -> String? {
        do {
            try journal.rebind(settings: settings)
            return nil
        } catch { return " Not carried over yet: \(error)." }
    }

    /// What names the item's new place: the registry entry, the `model` setting, the stores (a pack's folder joins them)
    /// or `context_dir`.
    private static func update(_ move: Move, settings: Settings) throws {
        let item = move.item
        let moved = move.folder + "/" + (item.path as NSString).lastPathComponent
        switch item.move {
        case "model":
            guard
                try ModelRegistry.entries(settings).contains(where: {
                    $0["name"] as? String == item.model || $0["path"] as? String == item.path
                })
            else { throw CLIError("\(item.name) is no longer registered") }
            let followsPath = settings.model.map(Settings.expand) == item.path
            if followsPath { try Settings.write(["model": moved], to: settings.configFile) }
            // Every name registered for that path follows it: one left behind would name a folder about to be deleted.
            do {
                try ModelRegistry.update(settings) { entries in
                    for i in entries.indices where entries[i]["name"] as? String == item.model || entries[i]["path"] as? String == item.path
                    {
                        entries[i]["path"] = moved
                    }
                }
            } catch {
                if followsPath { try Settings.write(["model": settings.model], to: settings.configFile) }
                throw error
            }
        case "setting":
            try Settings.write(["model": moved], to: settings.configFile)
        case "pack":
            guard !settings.stores.contains(where: { $0.path == move.folder }), move.folder != settings.homeStore.path else { return }
            remember(settings)
            let listed = (try Settings.readConfig(settings.configFile)["stores"] as? [[String: Any]]) ?? []
            try Settings.write(["stores": listed + [["path": move.folder]]], to: settings.configFile)
        case "hot":
            // The place that holds it follows it; a copy in a folder no longer used changes no setting.
            var target = try owner(item, settings: settings)
            guard let index = place(item, settings: target) else { return }
            remember(settings)
            target.model = item.model
            var changes: [String: Any?] = ["hot_copy_dir": move.folder]
            if index >= 0 {
                var more = target.hotCopyMore
                more[index].folder = URL(fileURLWithPath: move.folder, isDirectory: true)
                changes = ["hot_copy_more": Settings.copyPlacesText(more)]
            }
            try Commands.modelSettings(&changes, target: target)
            if !changes.isEmpty { try Settings.write(changes, to: settings.configFile) }
        case "prompts":
            remember(settings)
            try Settings.write(["context_dir": move.folder], to: settings.configFile)
        default: break
        }
    }

    private static func message(_ move: Move, settings: Settings) -> String {
        "Moved \(move.item.name) to \(move.folder)."
            + (move.item.move == "pack" && !settings.stores.contains(where: { $0.path == move.folder })
                && move.folder != settings.homeStore.path
                ? " That folder is now one of the stores."
                : move.item.move == "prompts" ? " Saved prompts and long contexts are kept there now." : "")
    }

    /// Per file under `path` (links followed, as `copy` reads them): its size, modification time and inode.
    private static func stamps(_ path: String, depth: Int = 0) -> [String: [Int64]] {
        var info = stat()
        guard stat(path, &info) == 0 else { return [path: []] }
        guard info.st_mode & S_IFMT == S_IFDIR else {
            return [
                path: [
                    Int64(info.st_size), Int64(info.st_mtimespec.tv_sec), Int64(info.st_mtimespec.tv_nsec),
                    Int64(bitPattern: UInt64(info.st_ino)),
                ]
            ]
        }
        guard depth < 8 else { return [:] }
        return children(path).reduce(into: [:]) { all, name in all.merge(stamps(path + "/" + name, depth: depth + 1)) { $1 } }
    }
    /// Checks only (`ORIGAMI_TEST_MOVE_PAUSE`): while the file it names exists, a cross-volume move stops after its copy,
    /// before the originals are compared, and says so with `<file>.copied` (30 s at most).
    private static func pauseForChecks() {
        guard let marker = ProcessInfo.processInfo.environment["ORIGAMI_TEST_MOVE_PAUSE"], FileManager.default.fileExists(atPath: marker)
        else { return }
        FileManager.default.createFile(atPath: marker + ".copied", contents: nil)
        let until = Date().addingTimeInterval(30)
        while FileManager.default.fileExists(atPath: marker) && Date() < until { usleep(20_000) }
    }

    /// Copies `from` (a file, or a folder and everything in it, links followed: a Hugging Face snapshot's links become
    /// its files) to `to`, then checks each file: its size, and the SHA-256 of the bytes read from the source against
    /// that of the copy read back with the cache bypassed (`F_NOCACHE`), so what is compared is what the drive holds.
    private static func copy(
        _ from: String, _ to: String, buffer: UnsafeMutableRawBufferPointer, counted: (UInt64) -> Void, cancelled: () -> Bool, depth: Int
    ) throws {
        var info = stat()
        guard stat(from, &info) == 0 else { throw CLIError("Cannot read \(from): \(String(cString: strerror(errno)))") }
        if info.st_mode & S_IFMT == S_IFDIR {
            guard depth < 8 else { throw CLIError("\(from) is nested too deeply to move") }
            guard mkdir(to, 0o755) == 0 else { throw CLIError("Cannot write \(to): \(String(cString: strerror(errno)))") }
            let names: [String]
            do { names = try FileManager.default.contentsOfDirectory(atPath: from) } catch {
                throw CLIError("Cannot read \(from): \(error.localizedDescription)")
            }
            for name in names.sorted() {
                try copy(from + "/" + name, to + "/" + name, buffer: buffer, counted: counted, cancelled: cancelled, depth: depth + 1)
            }
            return
        }
        guard info.st_mode & S_IFMT == S_IFREG else { throw CLIError("\(from) is not a file or a folder") }
        func pass(_ path: String, write output: Int32?) throws -> (bytes: Int64, digest: SHA256.Digest) {
            let input = open(path, O_RDONLY)
            guard input >= 0 else { throw CLIError("Cannot read \(path): \(String(cString: strerror(errno)))") }
            defer { close(input) }
            if output == nil { _ = fcntl(input, F_NOCACHE, 1) }
            var hash = SHA256()
            var bytes: Int64 = 0
            while true {
                if cancelled() { throw CLIError("Cancelled") }
                let count = read(input, buffer.baseAddress, buffer.count)
                if count < 0 {
                    if errno == EINTR { continue }
                    throw CLIError("Cannot read \(path): \(String(cString: strerror(errno)))")
                }
                if count == 0 { break }
                let chunk = UnsafeRawBufferPointer(rebasing: buffer[0..<count])
                hash.update(bufferPointer: chunk)
                if let output {
                    var at = 0
                    while at < count {
                        let wrote = write(output, chunk.baseAddress! + at, count - at)
                        if wrote < 0 {
                            if errno == EINTR { continue }
                            throw CLIError("Cannot write \(to): \(String(cString: strerror(errno)))")
                        }
                        at += wrote
                    }
                }
                bytes += Int64(count)
                counted(UInt64(count))
            }
            return (bytes, hash.finalize())
        }
        let output = open(to, O_WRONLY | O_CREAT | O_EXCL, mode_t(info.st_mode & 0o777))
        guard output >= 0 else { throw CLIError("Cannot write \(to): \(String(cString: strerror(errno)))") }
        let source: (bytes: Int64, digest: SHA256.Digest)
        do {
            source = try pass(from, write: output)
            // A full flush (the drive's own cache too): the source is deleted once every copy is in place.
            guard fcntl(output, F_FULLFSYNC) != -1 || fsync(output) == 0 else {
                throw CLIError("Cannot write \(to): \(String(cString: strerror(errno)))")
            }
        } catch {
            close(output)
            throw error
        }
        close(output)
        guard source.bytes == Int64(info.st_size) else { throw CLIError("\(from) changed while it was copied; try again") }
        let copied = try pass(to, write: nil)
        guard copied.bytes == source.bytes, copied.digest == source.digest else {
            throw CLIError("The copy of \(from) differs from the original; nothing was deleted")
        }
    }

    /// Deletes an item nothing uses (its parts, then a family folder left empty), and forgets an interrupted move's copy.
    static func remove(_ item: Item, settings: Settings) throws -> String {
        guard item.unused, item.deleteBlocked == nil else {
            throw ServingFailure(item.deleteBlocked ?? "Only an item nothing uses is removed", status: 403)
        }
        for part in item.pieces {
            do { try FileManager.default.removeItem(atPath: part.from) } catch {
                throw CLIError("Cannot delete \(part.from): \(error.localizedDescription)")
            }
            if part.to.contains("/") { rmdir((part.from as NSString).deletingLastPathComponent) }
        }
        if item.role == "staging" { try? setPending(settings, removing: [item.path]) }
        return "Removed \(item.name) (\(OrigamiPlan.gigabytes(item.bytes)))."
    }

    /// Explicit deletion is separate from cache disable and library removal.
    static func delete(_ item: Item, settings: Settings) throws -> String {
        guard item.readable, item.deleteBlocked == nil, ["model", "pack", "hot copy", "prompts"].contains(item.role) else {
            throw ServingFailure(item.deleteBlocked ?? "This item cannot be deleted", status: 409)
        }
        // Disable regeneration before deleting an active cache: the place that holds it is turned off (a copy in a folder
        // no longer used changes no setting). The owner is stopped by the scheduler.
        if item.role == "hot copy", case var target = try owner(item, settings: settings), let index = place(item, settings: target) {
            target.model = item.model
            var changes: [String: Any?] = ["hot_copy": "off"]
            if index >= 0 {
                var more = target.hotCopyMore
                more[index].size = 0
                changes = ["hot_copy_more": Settings.copyPlacesText(more)]
            }
            try Commands.modelSettings(&changes, target: target)
            if !changes.isEmpty { try Settings.write(changes, to: settings.configFile) }
        }
        var kept: [String] = []
        for part in item.pieces {
            // A model's folder may hold files that are not the model's (notes, another download): only its own go.
            guard item.role == "model", isFolder(part.from) else {
                try FileManager.default.removeItem(atPath: part.from)
                continue
            }
            for name in modelFiles(part.from) {
                let file = part.from + "/" + name
                if FileManager.default.fileExists(atPath: file) { try FileManager.default.removeItem(atPath: file) }
            }
            let left = children(part.from).filter { $0 != ".DS_Store" }.sorted()
            if left.isEmpty {
                try FileManager.default.removeItem(atPath: part.from)
            } else {
                kept.append("\(part.from) (\(left.joined(separator: ", ")))")
            }
        }
        if item.role == "model" { try forget(item, settings: settings) }
        return "Deleted " + item.name + (kept.isEmpty ? "" : "; kept its folder for the other files in it: " + kept.joined(separator: "; "))
    }
    /// The files of the model in `folder`: its GGUF parts, or its safetensors shards (from the index) and the files that
    /// describe it (config, tokenizer, template, licence, Origami's own records, those a download fetched).
    static func modelFiles(_ folder: String) -> [String] {
        let url = URL(fileURLWithPath: folder, isDirectory: true)
        if Models.isGGUF(folder), let parts = try? GGUFSource.parts(url) {
            return parts.files + ["origami-model.json", ".origami-download.json"]
        }
        var names = [
            "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "added_tokens.json", "chat_template.jinja", "chat_template.json", "preprocessor_config.json", "processor_config.json",
            "video_preprocessor_config.json", "merges.txt", "vocab.json", "LICENSE", "model.safetensors",
            "model.safetensors.index.json", "origami-model.json", "download-provenance.json", ".origami-download.json", GGUFExperts.file,
        ]
        let json = { (name: String) in
            (try? Data(contentsOf: url.appendingPathComponent(name))).flatMap {
                try? JSONSerialization.jsonObject(with: $0) as? [String: Any]
            }
        }
        names += ((json("model.safetensors.index.json")?["weight_map"] as? [String: String]) ?? [:]).values
        names += ((json(".origami-download.json")?["files"] as? [String: Any]) ?? [:]).keys
        // Only plain names in the folder itself: an index or record cannot point a delete elsewhere.
        return unique(names.filter { !$0.isEmpty && !$0.contains("/") && $0 != "." && $0 != ".." })
    }
    /// The settings of the registered model `item` belongs to, whose own places hold its copy: `settings` for the
    /// configured model or one not registered.
    private static func owner(_ item: Item, settings: Settings) throws -> Settings {
        guard let name = item.model, name != settings.model, ModelRegistry.entry(name, settings: settings) != nil else { return settings }
        return try Settings.load(flags: ["config": [settings.configFile.path], "data-dir": [settings.dataDir.path], "model": [name]])
    }
    /// The hot copy place that holds `item`: the index of its `hot_copy_more` entry, -1 for the first place (`hot_copy_dir`,
    /// a store, the data folder), nil for none (a folder no longer used).
    private static func place(_ item: Item, settings: Settings) -> Int? {
        guard let part = item.pieces.first, part.from.hasSuffix("/" + part.to) else { return nil }
        let folder = resolved(String(part.from.dropLast(part.to.count + 1)))
        if let index = settings.hotCopyMore.firstIndex(where: { resolved($0.folder.path) == folder }) { return index }
        let first = settings.storeFolders.map { resolved($0.path) } + [resolved(settings.dataDir.appendingPathComponent("family").path)]
        return first.contains { folder == $0 || folder.hasPrefix($0 + "/") } ? -1 : nil
    }
    static func forget(_ item: Item, settings: Settings) throws {
        guard item.role == "model" else { throw ServingFailure("Only a model can be removed from the library") }
        // A low-bit variant registered on this path is another model: it stays unless named.
        let removed = try ModelRegistry.update(settings) { entries in
            let removed = entries.filter {
                $0["path"] as? String == item.path && $0["experts"] == nil || $0["name"] as? String == item.model
            }
            entries.removeAll { entry in removed.contains { NSDictionary(dictionary: $0).isEqual(to: entry) } }
            return removed
        }
        // A Qwen low-bit variant's generated folder (links and two small files) goes with it, as `origami model remove`
        // does; the links' targets, the base's files and the experts file, stay.
        for entry in removed where entry["base"] != nil {
            if let path = entry["path"] as? String, ExpertVariant.owns(path, settings: settings) {
                try? FileManager.default.removeItem(atPath: path)
            }
        }
        if settings.model == item.model || settings.model.map(Settings.expand) == item.path {
            try Settings.write(["model": nil, "stopped": true], to: settings.configFile)
        }
    }
    static func relink(_ item: Item, to path: String, settings: Settings) throws -> String {
        guard item.role == "model", let real = Folders.real(path), Folders.root(of: real, Folders.roots(settings)) != nil else {
            throw ServingFailure("Choose an existing model on this host")
        }
        let url = URL(fileURLWithPath: real)
        let report = try ModelCompatibility.check(local: url)
        guard report.verdict == .runs else { throw ServingFailure("That location does not contain a supported model") }
        let entries = try ModelRegistry.entries(settings)
        guard let index = entries.firstIndex(where: { $0["name"] as? String == item.model }) else {
            throw ServingFailure("Add this model to the library first")
        }
        let manifest = entries[index]["manifest"] as? [String: Any]
        guard sameQwen(manifest, at: url) else { throw ServingFailure("This is a different model; add it separately") }
        if let family = entries[index]["family"] as? String, let gguf = try? GGUFSource(url),
            ModelFamily.detect(gguf: gguf.metadata)?.rawValue != family
        {
            throw ServingFailure("This location holds a different model family")
        }
        // What is keyed on the identity it had (its fingerprint or key as recorded) moves to the identity it has here.
        if let old = (manifest?["fingerprint"] ?? entries[index]["key"]) as? String {
            var probe = settings
            probe.model = item.model
            _ = try Refresh.rebind(url, before: old, relatedBefore: [:], settings: probe)
        }
        let followsPath = settings.model.map(Settings.expand) == item.path
        if followsPath { try Settings.write(["model": real], to: settings.configFile) }
        do {
            try ModelRegistry.update(settings) { entries in
                for i in entries.indices where entries[i]["name"] as? String == item.model { entries[i]["path"] = real }
            }
        } catch {
            if followsPath { try? Settings.write(["model": settings.model], to: settings.configFile) }
            throw error
        }
        return "Location updated"
    }
    /// Whether the Qwen model at `url` is the one `manifest` (its registry entry's) describes. The fingerprint binds each
    /// file's path: the same model elsewhere is known by what does not (its config, index and shard sizes, as
    /// `earlierFingerprint` hashes them), else by the size and shape `model add` recorded. True without a fingerprint.
    static func sameQwen(_ manifest: [String: Any]?, at url: URL) -> Bool {
        guard manifest?["fingerprint"] != nil else { return true }
        if let earlier = manifest?["earlier_fingerprint"] as? String { return ModelSource.earlierFingerprint(directory: url) == earlier }
        let shape = { (all: [String: Any]?) in (all ?? [:]).filter { ["bytes", "shards", "model_type"].contains($0.key) } }
        let report = shape((try? Diagnostics.inspectReport(url))?.report)
        return report.count == 3 && NSDictionary(dictionary: report).isEqual(to: shape(manifest))
    }

    // MARK: what Origami remembers

    /// The folders `stores`, `hot_copy_dir` and `context_dir` named before (newest last), which the map also reads for
    /// what Origami left there. Called before any of them changes.
    static func remember(_ settings: Settings) {
        let now =
            settings.stores.map(\.path) + [settings.hotCopyDir?.path, settings.promptCacheRoot.path].compactMap { $0 }
            + settings.hotCopyMore.map(\.folder.path)
        let all = unique(list(settings, "former-folders.json") + now)
        try? save(settings, "former-folders.json", all)
    }
    /// Every folder `remember` recorded (`former-folders.json`).
    static func formerFolders(_ settings: Settings) -> [String] { list(settings, "former-folders.json") }
    private static func former(_ settings: Settings) -> [String] {
        let current = Set(settings.storeFolders.map(\.path) + [settings.promptCacheRoot.path])
        return list(settings, "former-folders.json").filter { !current.contains($0) }
    }
    /// The staging copies of this process's move under way: listed as such, never offered for removal.
    nonisolated(unsafe) private static var live = Set<String>()
    private static let liveLock = NSLock()
    /// The staging copies of moves under way: one the server stopped during is listed (and removed) from here.
    private static func pending(_ settings: Settings) -> [String] { list(settings, "moving.json") }
    private static func setPending(_ settings: Settings, adding: [String] = [], removing: [String] = []) throws {
        try save(settings, "moving.json", unique(list(settings, "moving.json") + adding).filter { !removing.contains($0) })
    }
    private static func list(_ settings: Settings, _ file: String) -> [String] {
        (try? Data(contentsOf: settings.dataDir.appendingPathComponent(file))).flatMap {
            try? JSONSerialization.jsonObject(with: $0) as? [String]
        } ?? []
    }
    private static func save(_ settings: Settings, _ file: String, _ paths: [String]) throws {
        try FileManager.default.createDirectory(at: settings.dataDir, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: paths, options: [.prettyPrinted]).write(
            to: settings.dataDir.appendingPathComponent(file), options: .atomic)
    }

    // MARK: files

    /// Origami's own folders (the data and models folders, the settings file, the stores and hot copy's folder, the saved
    /// prompts'), resolved: what holds one is neither moved nor removed.
    private static func ownFolders(_ settings: Settings) -> [String] {
        ([settings.dataDir, settings.modelsDir, settings.configFile, settings.promptCacheRoot] + settings.storeFolders).map(\.path).filter(
            reachable
        ).map(resolved)
    }

    /// Whether `path` opens within 3 s (a drive under /Volumes that is unplugged, asleep or not allowed does not).
    private static func reachable(_ path: String) -> Bool {
        !path.hasPrefix("/Volumes/")
            || Folders.timed(3, on: path, { ![.notConnected, .noPermission].contains(Stores.probe(path)) }) == true
    }
    private static func isFolder(_ path: String) -> Bool {
        var info = stat()
        return stat(path, &info) == 0 && info.st_mode & S_IFMT == S_IFDIR
    }
    private static func children(_ path: String) -> [String] { (try? FileManager.default.contentsOfDirectory(atPath: path)) ?? [] }
    private static func unique(_ paths: [String]) -> [String] {
        var seen = Set<String>()
        return paths.filter { seen.insert($0).inserted }
    }
    /// Links and `..` resolved where something is there (only for a path whose drive answered).
    private static func resolved(_ path: String) -> String { Folders.real(path) ?? path }
    /// One path is the other or inside it.
    private static func overlaps(_ a: String, _ b: String) -> Bool { a == b || a.hasPrefix(b + "/") || b.hasPrefix(a + "/") }
    /// Where the symbolic links at or under `path` lead, resolved (at most 8 folders deep).
    private static func links(_ path: String, depth: Int = 0) -> [String] {
        var info = stat()
        guard lstat(path, &info) == 0 else { return [] }
        if info.st_mode & S_IFMT == S_IFLNK { return [resolved(path)] }
        guard info.st_mode & S_IFMT == S_IFDIR, depth < 8 else { return [] }
        return children(path).flatMap { links(path + "/" + $0, depth: depth + 1) }
    }
    /// Renames `from` to `to` only if nothing is at `to` (`renamex_np` with `RENAME_EXCL`). exFAT refuses that flag
    /// (ENOTSUP): there a plain rename follows a check that `to` is free, which only Origami's own move writes into.
    static func renameNew(_ from: String, _ to: String) -> Int32 {
        if renamex_np(from, to, UInt32(RENAME_EXCL)) == 0 { return 0 }
        guard errno == ENOTSUP else { return -1 }
        var info = stat()
        guard lstat(to, &info) != 0, errno == ENOENT else {
            errno = EEXIST
            return -1
        }
        return rename(from, to)
    }
    /// Whether `path` is or holds a symbolic link (or cannot be read).
    private static func linked(_ path: String, depth: Int = 0) -> Bool {
        var info = stat()
        guard lstat(path, &info) == 0 else { return true }
        if info.st_mode & S_IFMT == S_IFLNK { return true }
        return info.st_mode & S_IFMT == S_IFDIR && (depth >= 8 || children(path).contains { linked(path + "/" + $0, depth: depth + 1) })
    }

    /// The bytes of the files under `path` (links followed, as a move copies them) and what they take on the drive.
    static func size(_ path: String, depth: Int = 0) -> (bytes: UInt64, allocated: UInt64) {
        var info = stat()
        guard stat(path, &info) == 0 else { return (0, 0) }
        guard info.st_mode & S_IFMT == S_IFDIR else { return (UInt64(max(0, info.st_size)), UInt64(max(0, info.st_blocks)) * 512) }
        guard depth < 8 else { return (0, 0) }
        return children(path).reduce((UInt64(0), UInt64(0))) { sum, name in
            let one = size(path + "/" + name, depth: depth + 1)
            return (sum.0 + one.0, sum.1 + one.1)
        }
    }
}

// MARK: a renamed or remounted drive

extension Storage {
    /// A mounted volume: its folder and its volume UUID (nil: none known).
    struct Volume: Sendable { let path: String, uuid: String? }
    /// A path a setting names that is missing, and the same folder found on a mounted volume, with how it was matched.
    /// `setting`: `model <name>`, `model`, `stores`, `hot_copy_dir`, `hot_copy_more` or `context_dir` (a model's own
    /// copy places end in ` of <name>`).
    struct Relink: Sendable { let setting: String, from: String, to: String, how: String }

    /// The mounted volumes but the startup one. `ORIGAMI_TEST_VOLUMES` (`folder=uuid,…`) stands in for them in the checks.
    static func volumes() -> [Volume] {
        if let test = ProcessInfo.processInfo.environment["ORIGAMI_TEST_VOLUMES"] {
            return test.split(separator: ",").map { pair in
                let parts = pair.split(separator: "=", maxSplits: 1).map(String.init)
                return Volume(path: parts[0], uuid: parts.count > 1 ? parts[1] : nil)
            }
        }
        let keys: Set<URLResourceKey> = [.volumeUUIDStringKey]
        return (FileManager.default.mountedVolumeURLs(includingResourceValuesForKeys: Array(keys), options: [.skipHiddenVolumes]) ?? [])
            .filter { $0.path != "/" }.map { Volume(path: $0.path, uuid: (try? $0.resourceValues(forKeys: keys))?.volumeUUIDString) }
    }

    /// Every path the settings file and the model registry name: each model's, the `model` setting when it is a path, and
    /// each folder of `stores`, `hot_copy_dir`, `hot_copy_more` and `context_dir` (a model's own copy places too).
    private static func bound(_ settings: Settings) -> [(setting: String, path: String)] {
        var all: [(setting: String, path: String)] = []
        let config = (try? Settings.readConfig(settings.configFile)) ?? [:]
        func places(_ values: [String: Any], _ suffix: String) {
            if let hot = values["hot_copy_dir"] as? String, hot != "auto" { all.append(("hot_copy_dir" + suffix, Settings.expand(hot))) }
            if let more = values["hot_copy_more"] as? String {
                all += ((try? Settings.copyPlaces(more)) ?? []).map { ("hot_copy_more" + suffix, $0.folder.path) }
            }
        }
        for entry in (try? ModelRegistry.entries(settings)) ?? [] {
            guard let name = entry["name"] as? String, let path = entry["path"] as? String else { continue }
            all.append(("model " + name, path))
            places(entry, " of " + name)
        }
        if let model = (config["model"] as? String).map(Settings.expand), model.hasPrefix("/") { all.append(("model", model)) }
        all += ((config["stores"] as? [[String: Any]]) ?? []).compactMap { ($0["path"] as? String).map { ("stores", Settings.expand($0)) } }
        places(config, "")
        if let context = config["context_dir"] as? String { all.append(("context_dir", Settings.expand(context))) }
        return all
    }
    private static func noted(_ settings: Settings) -> [String: Any] {
        (try? Settings.readConfig(settings.dataDir.appendingPathComponent("volumes.json"))) ?? [:]
    }

    /// Notes in `<data_dir>/volumes.json` the volume (its folder and UUID) of each path a setting names that is there now,
    /// and for a model its identity there: what `relinks` knows a renamed or remounted drive by, and what a relink carries
    /// over from. Paths on the startup volume are not noted.
    static func noteVolumes(_ settings: Settings, volumes: [Volume]) {
        var known = noted(settings)
        let before = NSDictionary(dictionary: known)
        var seen = Set<String>()
        for (setting, path) in bound(settings) where seen.insert(path).inserted && reachable(path) {
            guard FileManager.default.fileExists(atPath: path),
                let volume = volumes.filter({ path.hasPrefix($0.path + "/") }).max(by: { $0.path.count < $1.path.count }),
                let uuid = volume.uuid
            else { continue }
            var record: [String: Any] = ["volume": volume.path, "uuid": uuid]
            if setting.hasPrefix("model") {
                var probe = settings
                probe.model = setting == "model" ? path : String(setting.dropFirst(6))
                let identities = Refresh.identities(URL(fileURLWithPath: path), settings: probe)
                record["model"] = probe.model
                record["own"] = identities.own
                record["related"] = identities.related
            }
            known[path] = record
        }
        guard !before.isEqual(to: known) else { return }
        try? FileManager.default.createDirectory(at: settings.dataDir, withIntermediateDirectories: true)
        try? JSONSerialization.data(withJSONObject: known, options: [.prettyPrinted, .sortedKeys]).write(
            to: settings.dataDir.appendingPathComponent("volumes.json"), options: .atomic)
    }

    /// The paths settings name that are missing but whose folder a mounted volume in `volumes` holds: on the volume of
    /// the UUID noted for the path (`noteVolumes`), at the same place under it; for a path never noted, at the same place
    /// under another volume than the `/Volumes/<name>` it named, when what is there is the same: the same model by what
    /// does not bind a path, a store or copy place holding Origami's copies, a context folder holding saved prompts. A path
    /// found in more than one place is left for Locate.
    static func relinks(_ settings: Settings, volumes: [Volume]) -> [Relink] {
        let known = noted(settings)
        let entries = (try? ModelRegistry.entries(settings)) ?? []
        func same(_ setting: String, _ path: String) -> Bool {
            if setting.hasPrefix("model") {
                let entry = entries.first { "model " + ($0["name"] as? String ?? "") == setting }
                if Models.isGGUF(path) {
                    let family = entry?["family"] as? String
                    return (try? GGUFSource(URL(fileURLWithPath: path))).flatMap { ModelFamily.detect(gguf: $0.metadata)?.rawValue }
                        .map { family == nil || $0 == family } ?? false
                }
                let manifest = entry?["manifest"] as? [String: Any]
                return FileManager.default.fileExists(atPath: path + "/config.json")
                    && (setting == "model" || manifest?["fingerprint"] != nil && sameQwen(manifest, at: URL(fileURLWithPath: path)))
            }
            if setting == "context_dir" { return !prompts(path, unused: false).isEmpty }
            return !Commands.origamiFolders(in: URL(fileURLWithPath: path)).isEmpty
                || children(path).contains { $0.range(of: "^family-[0-9a-f]{16}$", options: .regularExpression) != nil }
        }
        var found: [Relink] = []
        var seen = Set<String>()
        for (setting, path) in bound(settings)
        where seen.insert(setting + "\n" + path).inserted && !FileManager.default.fileExists(atPath: path) {
            var places: [(to: String, how: String)] = []
            if let record = known[path] as? [String: Any], let root = record["volume"] as? String, let uuid = record["uuid"] as? String,
                path.hasPrefix(root + "/")
            {
                places = volumes.filter { $0.uuid == uuid && $0.path != root }.map {
                    ($0.path + path.dropFirst(root.count), "the same drive by its volume UUID, now at \($0.path)")
                }
            } else {
                let parts = URL(fileURLWithPath: path).pathComponents
                guard parts.count > 3, parts[1] == "Volumes" else { continue }
                let rest = "/" + parts[3...].joined(separator: "/")
                places = volumes.filter { $0.path != "/Volumes/" + parts[2] }.map { ($0.path + rest, "the same contents at \($0.path)") }
                    .filter { FileManager.default.fileExists(atPath: $0.to) && same(setting, $0.to) }
            }
            places = places.filter { FileManager.default.fileExists(atPath: $0.to) }
            if places.count == 1 { found.append(Relink(setting: setting, from: path, to: places[0].to, how: places[0].how)) }
        }
        return found
    }

    /// Uses the places `relinks` found: rewrites each path in the settings file and the model registry, then moves what is
    /// keyed on a model's identity (its hot copy, counts, tuning, saved prompts) to its identity at the new place, through
    /// the re-bind a move uses, from the identity noted at the old one. One line per change.
    static func relink(_ found: [Relink], settings: Settings) throws -> [String] {
        remember(settings)
        let map = Dictionary(found.map { ($0.from, $0.to) }) { first, _ in first }
        func swapped(_ path: String) -> String { map[Settings.expand(path)] ?? path }
        func places(_ text: String) -> String? {
            guard let list = try? Settings.copyPlaces(text) else { return nil }
            return Settings.copyPlacesText(
                list.map { place in
                    var place = place
                    place.folder = URL(fileURLWithPath: swapped(place.folder.path), isDirectory: true)
                    return place
                })
        }
        let config = try Settings.readConfig(settings.configFile)
        var changes: [String: Any?] = [:]
        for key in ["model", "hot_copy_dir", "context_dir"] {
            if let value = config[key] as? String, swapped(value) != value { changes[key] = swapped(value) }
        }
        if let stores = config["stores"] as? [[String: Any]],
            stores.contains(where: { map[Settings.expand($0["path"] as? String ?? "")] != nil })
        {
            changes["stores"] = stores.map { store -> [String: Any] in
                var store = store
                if let path = store["path"] as? String { store["path"] = swapped(path) }
                return store
            }
        }
        if let more = config["hot_copy_more"] as? String, let text = places(more), text != more { changes["hot_copy_more"] = text }
        if !changes.isEmpty { try Settings.write(changes, to: settings.configFile) }
        try ModelRegistry.update(settings) { entries in
            for index in entries.indices {
                for key in ["path", "hot_copy_dir"] {
                    if let value = entries[index][key] as? String, swapped(value) != value { entries[index][key] = swapped(value) }
                }
                if let more = entries[index]["hot_copy_more"] as? String, let text = places(more), text != more {
                    entries[index]["hot_copy_more"] = text
                }
            }
        }
        var lines = found.map { "\($0.setting): \($0.from) → \($0.to)" }
        let known = noted(settings)
        var carried = Set<String>()
        for item in found where item.setting.hasPrefix("model") && carried.insert(item.from).inserted {
            let record = known[item.from] as? [String: Any] ?? [:]
            var probe = settings
            probe.model = record["model"] as? String ?? (item.setting == "model" ? item.to : String(item.setting.dropFirst(6)))
            if probe.model == item.from { probe.model = item.to }
            guard record["own"] is String || !(record["related"] as? [String: String] ?? [:]).isEmpty else {
                lines.append(
                    "\(probe.model ?? item.to): no identity was noted at its old place, so what is keyed on it is not carried over (its hot copy is built again while serving)"
                )
                continue
            }
            lines += try Refresh.rebind(
                URL(fileURLWithPath: item.to), before: record["own"] as? String,
                relatedBefore: record["related"] as? [String: String] ?? [:], settings: probe)
        }
        return lines
    }
}
