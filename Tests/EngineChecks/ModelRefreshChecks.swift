import CryptoKit
import Darwin
import Foundation
import OrigamiEngine

/// `origami refresh` (ROADMAP R15): model files rewritten in place keep their bytes, and what was keyed by their old
/// identity is found under the new one; interruptions, a write fault, a held lock and a replaced file.
func modelRefreshChecks() -> [(String, () throws -> Void)] {
    [
        ("model_refresh_family_rebinds", modelRefreshFamilyRebinds),
        ("model_refresh_qwen_rebinds", modelRefreshQwenRebinds),
        ("model_refresh_resumes", modelRefreshResumes),
        ("model_refresh_write_fault", modelRefreshWriteFault),
        ("model_refresh_write_fault_time_not_restored", modelRefreshWriteFaultTimeNotRestored),
        ("model_refresh_unwritable_leaves_no_journal", modelRefreshUnwritableLeavesNoJournal),
        ("model_refresh_refused_while_locked", modelRefreshRefusedWhileLocked),
        ("model_refresh_replaced_file_not_rebound", modelRefreshReplacedFileNotRebound),
        ("model_refresh_variant_rebinds", modelRefreshVariantRebinds),
    ]
}

/// A low-bit variant (its key binds the base's files and its experts file): refreshing the base carries the variant's
/// counts and hot copy to its new key too; refreshing the variant rewrites its experts file (bytes unchanged) and re-keys
/// both, also when stopped before the re-bind and run again. A rerun for the other model's files is refused meanwhile.
private func modelRefreshVariantRebinds() throws {
    let root = try temporary("variant")
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root.appendingPathComponent("model"), withIntermediateDirectories: true)
    let gguf = try syntheticFamilyGGUF(.glm, in: root.appendingPathComponent("model"))
    let parts = try GGUFSource.parts(gguf)
    let experts = root.appendingPathComponent("lowbit/experts.gguf")
    try FileManager.default.createDirectory(at: experts.deletingLastPathComponent(), withIntermediateDirectories: true)
    var bytes = Data(count: 3 << 20)
    bytes.withUnsafeMutableBytes { arc4random_buf($0.baseAddress!, 3 << 20) }
    try bytes.write(to: experts)
    let where_ = places(root)
    let baseKey = { FamilySupport.key(folder: parts.folder, files: parts.files) }
    let variantKey = { FamilySupport.key(folder: parts.folder, files: parts.files, experts: [experts.path]) }
    let base = ModelRefresh.Related(name: "base", identity: baseKey)
    let variant = ModelRefresh.Related(name: "variant", identity: variantKey)
    /// Counts and a hot copy (in `hot_copy_dir`) under the variant's current key, and counts under the base's.
    func seed() throws {
        let key = variantKey()
        try HotCopy.saveUses([3: 7], to: where_.data.appendingPathComponent("family/\(key)/uses.json"))
        try HotCopy.saveUses([3: 2], to: where_.data.appendingPathComponent("family/\(key)/misses.json"))
        try HotCopy.saveUses([1: 1], to: where_.data.appendingPathComponent("family/\(baseKey())/uses.json"))
        _ = try HotCopy.write(
            source: PatternSource(), keys: Array(0..<8), folder: where_.folders[1].appendingPathComponent("family-\(key)/hot"),
            identity: LocalModelIdentity.familyRepresentation(source: key))
    }
    func carried(_ what: String) throws {
        let key = variantKey()
        let family = where_.data.appendingPathComponent("family/\(key)")
        try refreshRequire(HotCopy.loadUses(family.appendingPathComponent("uses.json")) == [3: 7], "\(what): variant counts left behind")
        try refreshRequire(HotCopy.loadUses(family.appendingPathComponent("misses.json")) == [3: 2], "\(what): variant misses left behind")
        try refreshRequire(
            HotCopy.loadUses(where_.data.appendingPathComponent("family/\(baseKey())/uses.json")) == [1: 1], "\(what): base counts lost")
        let identity = LocalModelIdentity.familyRepresentation(source: key)
        try refreshRequire(
            try hotRecord(where_.folders[1].appendingPathComponent("family-\(key)/hot"), identity: identity) == 4,
            "\(what): variant hot copy left behind")
    }
    try seed()
    let (oldBase, oldVariant) = (baseKey(), variantKey())
    let first = try ModelRefresh.run(
        model: gguf, family: true, files: [gguf], identity: baseKey, places: where_, related: [variant])
    try refreshRequire(first.new != oldBase && variantKey() != oldVariant, "the base's refresh did not change both keys")
    try carried("base refresh")

    let (digest, stamp) = (try sha(experts), LocalModelIdentity.file(experts))
    let (beforeBase, beforeVariant) = (baseKey(), variantKey())
    struct Stop: Error {}
    var hooks = ModelRefresh.Hooks()
    hooks.beforeRebind = { throw Stop() }
    do {
        _ = try ModelRefresh.run(
            model: gguf, family: true, files: [gguf, experts], identity: variantKey, places: where_, related: [base], hooks: hooks)
        throw EngineError("model refresh: the interruption did not stop it")
    } catch is Stop {}
    do {
        _ = try ModelRefresh.run(model: gguf, family: true, files: [gguf], identity: baseKey, places: where_, related: [variant])
        throw EngineError("model refresh: the base's refresh resumed the variant's")
    } catch let error as EngineError where "\(error)".contains("other files") {}
    let second = try ModelRefresh.run(
        model: gguf, family: true, files: [gguf, experts], identity: variantKey, places: where_, related: [base])
    try refreshRequire(
        try sha(experts) == digest && LocalModelIdentity.file(experts) != stamp, "the experts file was not rewritten as it was")
    try refreshRequire(second.old == beforeVariant && second.new == variantKey() && baseKey() != beforeBase, "the keys did not change")
    try carried("variant refresh")
    try refreshRequire(ModelRefresh.unfinished(data: where_.data) == nil, "the journal stayed")
}

private func refreshRequire(_ value: @autoclosure () throws -> Bool, _ message: String) throws {
    if try !value() { throw EngineError("model refresh: " + message) }
}
private func sha(_ url: URL) throws -> String { SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined() }
private func temporary(_ name: String) throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("origami-refresh-\(name)-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    return root
}
private final class PatternSource: HotCopySource {
    func hotCopyRecordBytes(_ key: Int) -> Int? { (0..<8).contains(key) ? 16384 : nil }
    func readHotCopyRecord(_ key: Int, into: UnsafeMutableRawPointer) throws { memset(into, Int32(key + 1), 16384) }
}
/// Reads one record of the copy in `folder` for `identity`; nil when no copy of it is there.
private func hotRecord(_ folder: URL, identity: String) throws -> UInt8? {
    guard let copy = HotCopy.open(folder: folder, identity: identity) else { return nil }
    let buffer = UnsafeMutableRawBufferPointer.allocate(byteCount: 16384, alignment: 16384)
    defer { buffer.deallocate() }
    try copy.read(3, into: buffer)
    return buffer[100]
}
private func places(_ root: URL) -> ModelRefresh.Places {
    ModelRefresh.Places(
        data: root.appendingPathComponent("data"), prompts: root.appendingPathComponent("data/prompt-cache"),
        folders: [root.appendingPathComponent("store"), root.appendingPathComponent("hot-drive")])
}
/// A saved prompt (a family state) in `<prompts>/<epoch16>`, written through a cache rooted elsewhere and copied without
/// its `.lock`: a cache keeps its folder's lock until the process ends, as a server's does.
private func savePrompt(_ prompts: URL, epoch: String, tokens: [Int]) throws {
    let staging = prompts.deletingLastPathComponent().appendingPathComponent("staging-\(UUID().uuidString)")
    let cache = try PromptCache(
        root: staging, epoch: epoch, entries: 1, systemEntries: 0, entryBytes: PromptCache.entryDiskBytes(context: 8192) * 2,
        ledger: try MemoryLedger(ceiling: 1 << 30, transientMargin: 1, availableFloor: 1), capacity: 8192, minimumFreeBytes: 1)
    try cache.writeState(tokens: tokens, bytes: 4) { folder in
        try FamilyStateFile.write([], logits: [0.25], tokens: tokens, epoch: epoch, to: folder)
    }
    try FileManager.default.createDirectory(at: prompts, withIntermediateDirectories: true)
    let target = prompts.appendingPathComponent(String(epoch.prefix(16)))
    try FileManager.default.copyItem(at: staging, to: target)
    try FileManager.default.removeItem(at: target.appendingPathComponent(".lock"))
}
/// The saved prompt for `tokens` under `epoch`, restored as the family backend restores it; nil when none is found.
private func restorePrompt(_ prompts: URL, epoch: String, tokens: [Int]) throws -> [Float]? {
    let root = prompts.appendingPathComponent(String(epoch.prefix(16)))
    guard FileManager.default.fileExists(atPath: root.path) else { return nil }
    let cache = try PromptCache(
        root: root, epoch: epoch, entries: 1, systemEntries: 0, entryBytes: PromptCache.entryDiskBytes(context: 8192) * 2,
        ledger: try MemoryLedger(ceiling: 1 << 30, transientMargin: 1, availableFloor: 1), capacity: 8192, minimumFreeBytes: 1)
    guard let found = cache.candidates(for: tokens.map(Int64.init)).first else { return nil }
    let folder = cache.stateFolder(found.slot)
    let header = try FamilyStateFile.header(folder, epoch: epoch, prompt: tokens)
    return try FamilyStateFile.read(folder, header: header, into: [])
}
private func epochs(_ key: String) -> [String] {
    [SHA256.hash(data: Data(("fixture-epoch" + key).utf8)).map { String(format: "%02x", $0) }.joined()]
}

/// The synthetic GGUF with a hot copy in the data folder and in `hot_copy_dir`, a pack, usage counts, support files,
/// tuning and a saved prompt: after `refresh` the bytes are the same, the key is new and each is found under it.
private func modelRefreshFamilyRebinds() throws {
    let root = try temporary("family")
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.createDirectory(at: root.appendingPathComponent("model"), withIntermediateDirectories: true)
    let gguf = try syntheticFamilyGGUF(.glm, in: root.appendingPathComponent("model"))
    let parts = try GGUFSource.parts(gguf)
    let where_ = places(root)
    let key = { FamilySupport.key(folder: parts.folder, files: parts.files) }
    let old = key()
    let bytesBefore = try sha(gguf)
    let identity = LocalModelIdentity.familyRepresentation(source: old)
    let family = where_.data.appendingPathComponent("family/\(old)")
    _ = try HotCopy.write(source: PatternSource(), keys: Array(0..<8), folder: family.appendingPathComponent("hot"), identity: identity)
    _ = try HotCopy.write(
        source: PatternSource(), keys: Array(0..<8), folder: where_.folders[1].appendingPathComponent("family-\(old)/hot"),
        identity: identity)
    try HotCopy.saveUses([3: 7, 5: 2], to: family.appendingPathComponent("uses.json"))
    try HotCopy.saveUses([3: 4], to: family.appendingPathComponent("misses.json"))
    let pack = where_.folders[0].appendingPathComponent("family-\(old)/pack-Q4_K")
    try FileManager.default.createDirectory(at: pack, withIntermediateDirectories: true)
    try Data(#"{"identity":"\#(old)","target":12}"#.utf8).write(to: pack.appendingPathComponent("pack.json"))
    try FileManager.default.createDirectory(at: where_.data.appendingPathComponent("support/\(old)"), withIntermediateDirectories: true)
    try Data(#"{"entries":{"\#(old.prefix(12))@8192":{"identity":{"model":"\#(old.prefix(12))"}}}}"#.utf8).write(
        to: where_.data.appendingPathComponent("tuning.json"))
    let tokens = Array(1...40)
    try savePrompt(where_.prompts, epoch: epochs(old)[0], tokens: tokens)

    let outcome = try ModelRefresh.run(model: gguf, family: true, files: [gguf], identity: key, places: where_, epochs: epochs)
    let new = key()
    try refreshRequire(try sha(gguf) == bytesBefore, "the model's bytes changed")
    try refreshRequire(outcome.old == old && outcome.new == new && new != old, "the identity did not change")
    try refreshRequire(outcome.rewritten.count == 1, "the file was not rewritten once")
    let newIdentity = LocalModelIdentity.familyRepresentation(source: new)
    let newFamily = where_.data.appendingPathComponent("family/\(new)")
    try refreshRequire(try hotRecord(newFamily.appendingPathComponent("hot"), identity: newIdentity) == 4, "data-folder hot copy lost")
    try refreshRequire(
        try hotRecord(where_.folders[1].appendingPathComponent("family-\(new)/hot"), identity: newIdentity) == 4,
        "hot_copy_dir hot copy lost")
    try refreshRequire(HotCopy.loadUses(newFamily.appendingPathComponent("uses.json")) == [3: 7, 5: 2], "usage counts lost")
    try refreshRequire(HotCopy.loadUses(newFamily.appendingPathComponent("misses.json")) == [3: 4], "miss counts lost")
    let packJSON = try String(contentsOf: where_.folders[0].appendingPathComponent("family-\(new)/pack-Q4_K/pack.json"), encoding: .utf8)
    try refreshRequire(packJSON.contains(new) && !packJSON.contains(old), "the pack is not re-bound")
    try refreshRequire(FileManager.default.fileExists(atPath: where_.data.appendingPathComponent("support/\(new)").path), "support")
    let tuning = try String(contentsOf: where_.data.appendingPathComponent("tuning.json"), encoding: .utf8)
    try refreshRequire(tuning.contains("\(new.prefix(12))@8192") && !tuning.contains(old.prefix(12)), "tuning not re-keyed")
    try refreshRequire(try restorePrompt(where_.prompts, epoch: epochs(new)[0], tokens: tokens + [41]) == [0.25], "saved prompt lost")
    try refreshRequire(!FileManager.default.fileExists(atPath: where_.data.appendingPathComponent("family/\(old)").path), "old left")
    try refreshRequire(ModelRefresh.unfinished(data: where_.data) == nil, "the journal stayed")
}

/// A Qwen-shaped folder of three small shards with configuration (the fingerprint reads no tensor header).
private func qwenFixture(_ root: URL, shardBytes: Int = 3 << 20) throws -> (URL, [URL]) {
    let folder = root.appendingPathComponent("qwen")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    var config = ModelContract.geometry.mapValues { $0 as Any }
    config["model_type"] = "qwen4_exp"
    config["quantization"] = ["bits": 4, "group_size": 32]
    try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys]).write(to: folder.appendingPathComponent("config.json"))
    let names = (1...3).map { "model-0000\($0)-of-00003.safetensors" }
    let map = Dictionary(uniqueKeysWithValues: names.enumerated().map { ("tensor.\($0.offset)", $0.element) })
    try JSONSerialization.data(withJSONObject: ["weight_map": map]).write(to: folder.appendingPathComponent("model.safetensors.index.json"))
    for name in names {
        var bytes = Data(count: shardBytes)
        bytes.withUnsafeMutableBytes { arc4random_buf($0.baseAddress!, shardBytes) }
        try bytes.write(to: folder.appendingPathComponent(name))
    }
    return (folder, names.map { folder.appendingPathComponent($0) })
}

/// The Qwen fixture with a hot copy, a store header, routing counts, calibration and a saved prompt folder. The saves
/// move to the new fingerprint's folder, which records the old one; a server there finds them under its new key (as
/// `OrigamiRuntime.load` looks: its fingerprint, then the earlier ones) and reads them under the identity they were
/// written with. A re-bind that did not verify the bytes (a move or Locate) still drops them.
private func modelRefreshQwenRebinds() throws {
    let root = try temporary("qwen")
    defer { try? FileManager.default.removeItem(at: root) }
    let (folder, files) = try qwenFixture(root)
    let where_ = places(root)
    let old = try ModelSource.fingerprint(directory: folder)
    let hashes = try files.map(sha)
    let store = where_.folders[0].appendingPathComponent("qwen-\(old.prefix(12))")
    _ = try HotCopy.write(
        source: PatternSource(), keys: Array(0..<8), folder: store.appendingPathComponent("hot"), identity: "qwen \(old) affine4")
    try Data(#"{"format":"origami-store-v1","kind":"mirror","model_fingerprint":"\#(old)"}"#.utf8).write(
        to: store.appendingPathComponent("mirror.json"))
    try FileManager.default.createDirectory(at: where_.data, withIntermediateDirectories: true)
    try Data(#"{"uses":{}}"#.utf8).write(to: where_.data.appendingPathComponent("routing-\(old.prefix(12)).json"))
    try Data(#"{"\#(old.prefix(16))|disk|disk":{}}"#.utf8).write(to: where_.data.appendingPathComponent("calibration.json"))
    func epoch(_ fingerprint: String) -> String { PromptCanary.epoch(fingerprint + "settings", arithmetic: "digest") }
    let saved = where_.prompts.appendingPathComponent("\(old.prefix(12))-8192/\(epoch(old).prefix(16))")
    try FileManager.default.createDirectory(at: saved, withIntermediateDirectories: true)
    try refreshRequire(PromptCanary.write(.init(arithmetic: "digest", canary: "c", builds: ["b"]), to: saved), "fixture marker")
    try Data(#"{"epoch":"\#(epoch(old))","entries":{}}"#.utf8).write(to: saved.appendingPathComponent("index.json"))

    let outcome = try ModelRefresh.run(
        model: folder, family: false, files: files, identity: { try ModelSource.fingerprint(directory: folder) }, places: where_)
    let new = try ModelSource.fingerprint(directory: folder)
    try refreshRequire(try files.map(sha) == hashes, "the model's bytes changed")
    try refreshRequire(outcome.new == new && new != old && outcome.rewritten.count == 3, "the identity did not change")
    let moved = where_.folders[0].appendingPathComponent("qwen-\(new.prefix(12))")
    try refreshRequire(try hotRecord(moved.appendingPathComponent("hot"), identity: "qwen \(new) affine4") == 4, "hot copy lost")
    try refreshRequire(
        try String(contentsOf: moved.appendingPathComponent("mirror.json"), encoding: .utf8).contains(new), "store header not re-bound")
    try refreshRequire(
        FileManager.default.fileExists(atPath: where_.data.appendingPathComponent("routing-\(new.prefix(12)).json").path), "routing counts")
    try refreshRequire(
        try String(contentsOf: where_.data.appendingPathComponent("calibration.json"), encoding: .utf8).contains(new.prefix(16)),
        "calibration not re-keyed")
    let prompts = where_.prompts.appendingPathComponent("\(new.prefix(12))-8192")
    try refreshRequire(
        !FileManager.default.fileExists(atPath: saved.deletingLastPathComponent().path)
            && ModelRefresh.earlierFingerprints(prompts) == [old] && outcome.rebound.contains { $0.contains("-8192 to ") },
        "Qwen saved prompts not moved to the new fingerprint: \(outcome.rebound)")
    let earlier = ([new] + ModelRefresh.earlierFingerprints(prompts).reversed()).map { (epoch: epoch($0), model: $0) }
    let key = prompts.appendingPathComponent(String(epoch(new).prefix(16)))
    let written = PromptCanary.adopt(key, epoch: epoch(new), model: new, arithmetic: "digest", from: earlier)
    try refreshRequire(
        written == (epoch(old), old) && PromptCanary.marker(key)?.builds == ["b"]
            && (try? String(contentsOf: key.appendingPathComponent("index.json"), encoding: .utf8))?.contains(epoch(old)) == true,
        "the refreshed model's saves are not found under its new key with the identity they were written under")
    // Without verified bytes (a move, a Locate): dropped as before.
    let lines = try ModelRefresh.rebind(family: false, old: new, new: old, places: where_)
    try refreshRequire(
        !FileManager.default.fileExists(atPath: prompts.path) && lines.contains { $0.hasPrefix("dropped the saved prompts") },
        "saves moved by an unverified re-bind")
}

/// Interrupted after file 1 of 3, then between the rewrite and the re-bind: each rerun finishes, no file done is
/// rewritten again, and the hot copy is found under the final identity.
private func modelRefreshResumes() throws {
    let root = try temporary("resume")
    defer { try? FileManager.default.removeItem(at: root) }
    let (folder, files) = try qwenFixture(root)
    let where_ = places(root)
    let identity = { try ModelSource.fingerprint(directory: folder) }
    let old = try identity()
    let hashes = try files.map(sha)
    let store = where_.folders[0].appendingPathComponent("qwen-\(old.prefix(12))")
    _ = try HotCopy.write(
        source: PatternSource(), keys: Array(0..<8), folder: store.appendingPathComponent("hot"), identity: "qwen \(old) affine4")
    struct Stop: Error {}
    var hooks = ModelRefresh.Hooks()
    hooks.afterFile = { if $0 == 0 { throw Stop() } }
    do {
        _ = try ModelRefresh.run(model: folder, family: false, files: files, identity: identity, places: where_, hooks: hooks)
        throw EngineError("model refresh: the interruption did not stop it")
    } catch is Stop {}
    try refreshRequire(ModelRefresh.unfinished(data: where_.data) == folder.path, "no journal after an interruption")
    try refreshRequire(FileManager.default.fileExists(atPath: store.path), "re-bound before every file was done")
    hooks = ModelRefresh.Hooks()
    hooks.beforeRebind = { throw Stop() }
    var second: [String] = []
    hooks.write = { path, _, _ in if !second.contains(path) { second.append(path) } }
    do {
        _ = try ModelRefresh.run(model: folder, family: false, files: files, identity: identity, places: where_, hooks: hooks)
        throw EngineError("model refresh: the interruption did not stop it")
    } catch is Stop {}
    try refreshRequire(second == files.dropFirst().map { $0.resolvingSymlinksInPath().path }, "file 1 was rewritten again: \(second)")
    let third = try ModelRefresh.run(model: folder, family: false, files: files, identity: identity, places: where_)
    try refreshRequire(third.rewritten.isEmpty, "the last rerun rewrote files")
    let new = try identity()
    try refreshRequire(third.old == old && third.new == new && files.map(sha) == hashes, "bytes or identities wrong")
    let moved = where_.folders[0].appendingPathComponent("qwen-\(new.prefix(12))/hot")
    try refreshRequire(try hotRecord(moved, identity: "qwen \(new) affine4") == 4, "hot copy not re-bound after the reruns")
    try refreshRequire(ModelRefresh.unfinished(data: where_.data) == nil, "the journal stayed")
}

/// A write fault: the refresh stops naming the file and re-binds nothing, and a retry refuses instead of carrying the file over.
private func modelRefreshWriteFault() throws {
    let root = try temporary("fault")
    defer { try? FileManager.default.removeItem(at: root) }
    let (folder, files) = try qwenFixture(root)
    let where_ = places(root)
    let old = try ModelSource.fingerprint(directory: folder)
    let store = where_.folders[0].appendingPathComponent("qwen-\(old.prefix(12))")
    try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
    var hooks = ModelRefresh.Hooks()
    let target = files[1].resolvingSymlinksInPath().path
    hooks.write = { path, offset, buffer in if path == target && offset == 0 { buffer[7] ^= 0xFF } }
    do {
        _ = try ModelRefresh.run(
            model: folder, family: false, files: files, identity: { try ModelSource.fingerprint(directory: folder) }, places: where_,
            hooks: hooks)
        throw EngineError("model refresh: a write fault went unnoticed")
    } catch let error as EngineError {
        try refreshRequire("\(error)".contains(target) && "\(error)".contains("SHA-256"), "the fault does not name the file: \(error)")
    }
    try refreshRequire(FileManager.default.fileExists(atPath: store.path), "something was re-bound after a fault")
    try refreshRequire(ModelRefresh.unfinished(data: where_.data) == folder.path, "the journal is gone after a fault")
    // The retry must not take the changed bytes as the model's: refused, still nothing re-bound.
    do {
        _ = try ModelRefresh.run(
            model: folder, family: false, files: files, identity: { try ModelSource.fingerprint(directory: folder) }, places: where_)
        throw EngineError("model refresh: a retry carried a mismatched file over")
    } catch let error as EngineError {
        try refreshRequire("\(error)".contains(target) && "\(error)".contains("download that file again"), "the retry's refusal: \(error)")
    }
    try refreshRequire(FileManager.default.fileExists(atPath: store.path), "a retry re-bound after a fault")
    try refreshRequire(ModelRefresh.unfinished(data: where_.data) == folder.path, "the journal is gone after a refused retry")
}

/// A write fault on a file whose modification time cannot be put back (moved away while written): the mismatch is still
/// journaled, so a retry with the file back refuses instead of hashing the changed bytes as the model's.
private func modelRefreshWriteFaultTimeNotRestored() throws {
    let root = try temporary("fault-time")
    defer { try? FileManager.default.removeItem(at: root) }
    let (folder, files) = try qwenFixture(root)
    let where_ = places(root)
    let target = files[1].resolvingSymlinksInPath().path
    var hooks = ModelRefresh.Hooks()
    hooks.write = { path, offset, buffer in
        guard path == target && offset == 0 else { return }
        buffer[7] ^= 0xFF
        _ = rename(target, target + ".aside")
    }
    do {
        _ = try ModelRefresh.run(
            model: folder, family: false, files: files, identity: { try ModelSource.fingerprint(directory: folder) }, places: where_,
            hooks: hooks)
        throw EngineError("model refresh: a write fault went unnoticed")
    } catch let error as EngineError {
        try refreshRequire(error.code == "refresh_read_back_mismatch", "the mismatch was not reported first: \(error)")
    }
    try refreshRequire(rename(target + ".aside", target) == 0, "the moved file could not be put back")
    do {
        _ = try ModelRefresh.run(
            model: folder, family: false, files: files, identity: { try ModelSource.fingerprint(directory: folder) }, places: where_)
        throw EngineError("model refresh: a retry carried a mismatched file over")
    } catch let error as EngineError {
        try refreshRequire("\(error)".contains("read back differently when an earlier refresh"), "the retry's refusal: \(error)")
    }
}

/// A file it cannot open for writing (0444 here; a read-only drive or a privacy refusal alike): refused before any journal
/// exists, so the model still serves.
private func modelRefreshUnwritableLeavesNoJournal() throws {
    let root = try temporary("unwritable")
    defer { try? FileManager.default.removeItem(at: root) }
    let (folder, files) = try qwenFixture(root, shardBytes: 4096)
    let where_ = places(root)
    let target = files[1].resolvingSymlinksInPath().path
    try FileManager.default.setAttributes([.posixPermissions: 0o444], ofItemAtPath: target)
    do {
        _ = try ModelRefresh.run(
            model: folder, family: false, files: files, identity: { try ModelSource.fingerprint(directory: folder) }, places: where_)
        throw EngineError("model refresh: a file it cannot write was refreshed")
    } catch let error as EngineError where "\(error)".contains("for writing") {
        try refreshRequire("\(error)".contains(target), "the refusal does not name the file: \(error)")
    }
    try refreshRequire(ModelRefresh.unfinished(data: where_.data) == nil, "a journal was written for a file it cannot open")
}

/// A prompt cache lock held (by this process's cache, as a server holds its own): refused, naming the lock.
private func modelRefreshRefusedWhileLocked() throws {
    let root = try temporary("locked")
    defer { try? FileManager.default.removeItem(at: root) }
    let (folder, files) = try qwenFixture(root, shardBytes: 4096)
    let where_ = places(root)
    _ = try PromptCache(
        root: where_.prompts.appendingPathComponent("0123456789abcdef"), epoch: "fixture", entries: 1, systemEntries: 0,
        entryBytes: PromptCache.entryDiskBytes(context: 8192) * 2,
        ledger: try MemoryLedger(ceiling: 1 << 30, transientMargin: 1, availableFloor: 1), capacity: 8192,
        minimumFreeBytes: 1)
    let before = try ModelSource.fingerprint(directory: folder)
    do {
        _ = try ModelRefresh.run(
            model: folder, family: false, files: files, identity: { try ModelSource.fingerprint(directory: folder) }, places: where_)
        throw EngineError("model refresh: ran while the lock was held")
    } catch let error as EngineError {
        try refreshRequire("\(error)".contains("0123456789abcdef/.lock"), "the refusal does not name the lock: \(error)")
    }
    try refreshRequire(try ModelSource.fingerprint(directory: folder) == before, "files were touched while refused")
}

/// File 1 of 3 replaced by a different file of the same size (mtime restored) between two runs: nothing is re-bound, the
/// saved prompts included.
private func modelRefreshReplacedFileNotRebound() throws {
    let root = try temporary("replaced")
    defer { try? FileManager.default.removeItem(at: root) }
    let (folder, files) = try qwenFixture(root, shardBytes: 4096)
    let where_ = places(root)
    let identity = { try ModelSource.fingerprint(directory: folder) }
    let old = try identity()
    let store = where_.folders[0].appendingPathComponent("qwen-\(old.prefix(12))")
    try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
    let old12 = String(old.prefix(12))
    let saved = where_.prompts.appendingPathComponent("\(old12)-8192/0123456789abcdef")
    try FileManager.default.createDirectory(at: saved, withIntermediateDirectories: true)
    struct Stop: Error {}
    var hooks = ModelRefresh.Hooks()
    hooks.afterFile = { if $0 == 0 { throw Stop() } }
    do {
        _ = try ModelRefresh.run(model: folder, family: false, files: files, identity: identity, places: where_, hooks: hooks)
    } catch is Stop {}
    let date = try FileManager.default.attributesOfItem(atPath: files[0].path)[.modificationDate]!
    let other = folder.appendingPathComponent("other")
    try Data(repeating: 0x5A, count: 4096).write(to: other)
    guard rename(other.path, files[0].path) == 0 else { throw EngineError("model refresh: fixture replacement failed") }
    try FileManager.default.setAttributes([.modificationDate: date], ofItemAtPath: files[0].path)
    do {
        _ = try ModelRefresh.run(model: folder, family: false, files: files, identity: identity, places: where_)
        throw EngineError("model refresh: a replaced file was re-bound")
    } catch let error as EngineError {
        try refreshRequire("\(error)".contains(files[0].lastPathComponent), "the refusal does not name the file: \(error)")
    }
    try refreshRequire(FileManager.default.fileExists(atPath: store.path), "the old copies were moved")
    try refreshRequire(FileManager.default.fileExists(atPath: saved.path), "the old saved prompts were moved")
    try refreshRequire(
        (try? FileManager.default.contentsOfDirectory(atPath: where_.prompts.path))?.contains { !$0.hasPrefix(old12) } != true,
        "saved prompts were keyed on the changed files")
    try refreshRequire(ModelRefresh.unfinished(data: where_.data) == nil, "the journal stayed after the refusal")
}
