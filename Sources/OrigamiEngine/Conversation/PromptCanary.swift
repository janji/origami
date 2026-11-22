import CryptoKit
import Foundation

/// Saved prompts across upgrades. A prompt-cache folder is keyed on the arithmetic digest (`ArithmeticDigest`: the sources
/// that can change a saved state's or its logits' bits), not on the executable, so builds that compute the same bits
/// share it. Its marker (`origami-epoch.json`, beside the model, key and build fields the command-line tool writes there)
/// records the arithmetic digest, the canary (the SHA-256 of the bits of a fixed probe run on the real model from a cold
/// state: `qwenProbe`, `familyProbe`) and the builds (`build`: executable SHA-256 and macOS build) that reproduced it. A
/// build not listed runs the probe before any saved prompt is read when the folder holds one (`PromptCache.settle`) and
/// uses the folder only when its bits are equal; otherwise it saves in its own folder (`ownFolder`). A new or empty folder
/// lists the build that makes it at once, and that build writes the canary while idle once it saved something
/// (`PromptCache.canaryWhenIdle`).
public enum PromptCanary {
    package struct Marker: Codable, Equatable {
        package var arithmetic: String
        /// nil until the build that made the folder ran the probe while idle.
        package var canary: String?
        package var builds: [String]
        /// In a build's own folder: the shared folder (16 hex digits) whose canary this build did not reproduce.
        package var refused: String? = nil
        package init(arithmetic: String, canary: String?, builds: [String], refused: String? = nil) {
            (self.arithmetic, self.canary, self.builds, self.refused) = (arithmetic, canary, builds, refused)
        }
    }
    /// The folder's marker, shared with the command-line tool's fields (`FamilySetup.epochMarker`).
    package static let file = "origami-epoch.json"
    /// The arithmetic digest of this build (`ArithmeticDigest.value`).
    package static let arithmetic = ArithmeticDigest.value
    /// This executable's SHA-256 (read once).
    package static let executable = (try? OrigamiRuntime.executableDigest()) ?? "unknown-build"
    /// What must reproduce a canary: this executable on this macOS (whose update brings another Metal compiler and drivers).
    package static let build = executable + " " + ProcessInfo.processInfo.operatingSystemVersionString

    /// A prompt cache's epoch: SHA-256 of the arithmetic digest and `settings` (model, tokenizer, context, modes).
    package static func epoch(_ settings: String, arithmetic: String = arithmetic) -> String {
        OrigamiRuntime.hex(SHA256.hash(data: Data((arithmetic + settings).utf8)))
    }
    /// The folder a build uses after the shared folder `folder` refused its canary: a sibling named by 16 hex digits of
    /// SHA-256 of that folder's name and the executable's digest.
    package static func ownFolder(_ folder: URL, build: String) -> URL {
        folder.deletingLastPathComponent().appendingPathComponent(
            String(OrigamiRuntime.hex(SHA256.hash(data: Data((folder.lastPathComponent + "|" + build).utf8))).prefix(16)),
            isDirectory: true)
    }

    /// The canary fields of the folder's marker; nil without them (a folder from before upgrades kept saved prompts).
    package static func marker(_ folder: URL) -> Marker? {
        (try? Data(contentsOf: folder.appendingPathComponent(file))).flatMap { try? JSONDecoder().decode(Marker.self, from: $0) }
    }
    /// Writes `marker`'s fields into the folder's marker, keeping its other fields (staged and renamed over, 0600); false
    /// when it could not.
    package static func write(_ marker: Marker, to folder: URL) -> Bool {
        let url = folder.appendingPathComponent(file)
        var fields = (try? Data(contentsOf: url)).flatMap { (try? JSONSerialization.jsonObject(with: $0)) as? [String: Any] } ?? [:]
        guard let own = (try? JSONEncoder().encode(marker)).flatMap({ (try? JSONSerialization.jsonObject(with: $0)) as? [String: Any] })
        else { return false }
        // A field left out (nil) is removed, not kept from before.
        for key in ["canary", "refused"] { fields.removeValue(forKey: key) }
        fields.merge(own) { $1 }
        guard let data = try? JSONSerialization.data(withJSONObject: fields, options: [.sortedKeys]) else { return false }
        let staging = folder.appendingPathComponent(".\(file).\(UUID().uuidString)")
        guard FileManager.default.createFile(atPath: staging.path, contents: data, attributes: [.posixPermissions: 0o600]),
            rename(staging.path, url.path) == 0
        else {
            try? FileManager.default.removeItem(at: staging)
            return false
        }
        return true
    }

    /// What a build knows of `folder` without running its probe.
    package enum Known: Equatable {
        /// Its canary lists this build.
        case listed
        /// This build did not reproduce it earlier: its own folder.
        case refused(URL)
        /// A new folder, or one this build was never checked against.
        case unchecked
    }
    package static func known(_ folder: URL, build: String) -> Known {
        if marker(folder)?.builds.contains(build) == true { return .listed }
        let own = ownFolder(folder, build: build)
        // Only a canary run is a refusal (not a folder this build used while the shared one had no canary yet).
        if let mine = marker(own), mine.refused == folder.lastPathComponent, mine.builds.contains(build), mine.canary != nil {
            return .refused(own)
        }
        return .unchecked
    }
    /// The folder's index holds a saved entry.
    package static func holdsEntries(_ folder: URL) -> Bool {
        (try? Data(contentsOf: folder.appendingPathComponent("index.json"))).flatMap {
            try? JSONDecoder().decode(PromptCache.Index.self, from: $0)
        }.map { !$0.entries.isEmpty } ?? false
    }

    /// The reason code of a restore that found the prompt saved in a folder this build cannot read
    /// (`PromptCache.explainRestore`); its row's `reason_words` say which kind (`otherArithmetic`, `beforeUpgrades`).
    package static let savedByOtherBuild = "saved_by_other_build"
    package static let otherArithmetic = "it was saved by another Origami build whose arithmetic differs"
    package static let beforeUpgrades = "it was saved before upgrades kept saved prompts"

    /// Whether the running build can read the saved-prompt folder `folder` (a `<context_dir>/<epoch16>` or a Qwen
    /// `<context_dir>/<fingerprint12>-<context>/<epoch16>`), for the storage listing: nil when it can or may (the same
    /// arithmetic digest and a canary this build reproduced, or one not checked against it yet, `notCheckedYet`: a later
    /// probe may still prove it, so it is never removed as unusable), else why (`otherArithmetic`, `beforeUpgrades`). A
    /// folder for other settings of this build reads as readable: it is another model's, context's or mode's.
    package static func whyUnreadable(_ folder: URL, build: String = build, arithmetic: String = arithmetic) -> String? {
        guard let marker = marker(folder) else { return beforeUpgrades }
        if marker.builds.contains(build) && marker.arithmetic == arithmetic { return nil }
        // Another arithmetic; a build's own folder (its bits differed from its shared folder's); or a shared folder this
        // build's canary did not reproduce.
        let refused = known(folder, build: build) == .refused(ownFolder(folder, build: build))
        return marker.arithmetic != arithmetic || marker.refused != nil || refused ? otherArithmetic : nil
    }
    /// Words for a folder `whyUnreadable` keeps that this build has not been checked against yet (nil: none to say).
    package static func notCheckedYet(_ folder: URL, build: String = build, arithmetic: String = arithmetic) -> String? {
        guard whyUnreadable(folder, build: build, arithmetic: arithmetic) == nil, let marker = marker(folder),
            !marker.builds.contains(build)
        else { return nil }
        return marker.canary == nil
            ? "not checked yet: the build that saved these has not written their canary"
            : "not checked yet: this build's next load with these settings runs the canary first"
    }

    /// The text entries of the sibling folders of `root` this build cannot read (of this model: `sameModel` not false),
    /// each with why (`whyUnreadable`), so a restore can say a save exists but cannot be used.
    static func unreadableEntries(beside root: URL, build: String, sameModel: ((URL) -> Bool?)?)
        -> [(tokens: Int, prefix: String, why: String)]
    {
        let siblings =
            (try? FileManager.default.contentsOfDirectory(at: root.deletingLastPathComponent(), includingPropertiesForKeys: nil)) ?? []
        return siblings.filter {
            $0.lastPathComponent.range(of: "^[0-9a-f]{16}$", options: .regularExpression) != nil
                && $0.standardizedFileURL != root.standardizedFileURL && sameModel?($0) != false
        }.flatMap { folder -> [(tokens: Int, prefix: String, why: String)] in
            guard let why = whyUnreadable(folder, build: build),
                let data = try? Data(contentsOf: folder.appendingPathComponent("index.json")),
                let index = try? JSONDecoder().decode(PromptCache.Index.self, from: data)
            else { return [] }
            return index.entries.values.filter { $0.images == nil && $0.tokens > 0 }.map { ($0.tokens, $0.prefix, why) }
        }
    }

    /// The probe's token ids: fixed ordinary text tokens of every vocabulary Origami reads (100 to 31,999, wrapped in a
    /// smaller one), never an end of turn. Fed as ids, never through a chat template.
    package static func probeTokens(_ count: Int, vocabulary: Int) -> [Int] {
        let span = max(1, min(vocabulary, 32_000) - 100)
        return (0..<count).map { 100 + ($0 * 7_919 + 13) % span }
    }
    /// The probes' sizes, prompt tokens then decode steps. Qwen: three 256-token tiles (the last ragged) and the decode
    /// step. GLM and DeepSeek: a prompt above 8 rows (the prompt matrix kernels, not only the decode ones) and the decode
    /// step, kept short because every token reads its experts from disk.
    package static let qwenSize = (prompt: 600, decode: 3), familySize = (prompt: 16, decode: 2)
    /// Qwen's probe: the prompt, then the decode steps teacher-forced on fixed tokens (one more: the last is not read).
    package static func qwenProbe(vocabulary: Int) -> (prompt: [Int], decode: [Int]) {
        let ids = probeTokens(qwenSize.prompt + qwenSize.decode + 1, vocabulary: vocabulary)
        return (Array(ids.prefix(qwenSize.prompt)), Array(ids.dropFirst(qwenSize.prompt)))
    }
    /// GLM's and DeepSeek's probe: the prompt, then each decode token read alone.
    package static func familyProbe(vocabulary: Int) -> (prompt: [Int], decode: [Int]) {
        let ids = probeTokens(familySize.prompt + familySize.decode, vocabulary: vocabulary)
        return (Array(ids.prefix(familySize.prompt)), Array(ids.dropFirst(familySize.prompt)))
    }
    /// The canary: SHA-256 over each step's logits (raw little-endian bits) in order, then `state` (the final state's bytes,
    /// when cheap to read).
    package static func digest(logits: [[Float]], state: [UnsafeRawBufferPointer] = []) -> String {
        var hash = SHA256()
        for step in logits { step.withUnsafeBytes { hash.update(bufferPointer: $0) } }
        for part in state { hash.update(bufferPointer: part) }
        return OrigamiRuntime.hex(hash.finalize())
    }
}
