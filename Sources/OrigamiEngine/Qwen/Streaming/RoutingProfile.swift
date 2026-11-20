import Foundation

/// How often each expert was needed and how often it had to be read: every scope's routing adds one count per layer
/// and selected expert, and every demand that found its record outside memory adds a weighted miss. Kept per model in
/// the data folder across restarts, so the hot copy (`HotCopyKeeper`) and the further copies `origami setup` writes are
/// ranked by the model's own routing;
/// until it covers `minimumTokens`, the shipped ranking (from the derivative Origami was tuned on) is used.
/// Counts halve when they cover `2 * HotCopy.decayTokens` tokens (`record`), so recent use weighs more.
public final class RoutingProfile: @unchecked Sendable {
    public static let minimumTokens = 20_000
    private let lock = NSLock()
    private var counts: [UInt32]
    private var misses: [UInt32]
    private var windows: UInt64

    public init(counts: [UInt32]? = nil, missed: [UInt32]? = nil, windows: UInt64 = 0) {
        let size = ModelContract.layers * ModelContract.experts
        self.counts = counts?.count == size ? counts! : [UInt32](repeating: 0, count: size)
        self.misses = missed?.count == size ? missed! : [UInt32](repeating: 0, count: size)
        self.windows = windows
    }

    package func record(layer: Int, experts: [Int]) {
        lock.lock()
        defer { lock.unlock() }
        for expert in Set(experts) where (0..<ModelContract.experts).contains(expert) {
            counts[layer * ModelContract.experts + expert] &+= 1
        }
        windows &+= 1
        // Decay (`HotCopy.halved`): a file of the previous build halves here too until it is under the bound.
        while windows >= UInt64(2 * HotCopy.decayTokens * ModelContract.layers) {
            windows /= 2
            for i in counts.indices { counts[i] = counts[i] / 2 + counts[i] % 2 }
            for i in misses.indices { misses[i] = misses[i] / 2 + misses[i] % 2 }
        }
    }
    /// A demand that had to read its record: 4 for a decode miss (a token waits on it), 1 for a prompt miss (one read
    /// serves many rows).
    package func recordMiss(layer: Int, expert: Int, weight: UInt32) {
        guard (0..<ModelContract.layers).contains(layer), (0..<ModelContract.experts).contains(expert) else { return }
        lock.lock()
        defer { lock.unlock() }
        misses[layer * ModelContract.experts + expert] &+= weight
    }
    /// Weighted misses per expert (layer * 512 + expert).
    package var missed: [UInt32] {
        lock.lock()
        defer { lock.unlock() }
        return misses
    }

    /// Layer windows counted; a decode token is one window per layer.
    public var tokens: Int {
        lock.lock()
        defer { lock.unlock() }
        return Int(windows / UInt64(ModelContract.layers))
    }

    /// The `count` experts (layer * 512 + expert) of `layer` used most, ties by expert; nil until `minimumTokens`.
    package func mostUsed(layer: Int, count: Int) -> [Int]? {
        guard tokens >= Self.minimumTokens, (0..<ModelContract.layers).contains(layer) else { return nil }
        let base = layer * ModelContract.experts
        lock.lock()
        let uses = Array(counts[base..<base + ModelContract.experts])
        lock.unlock()
        return uses.indices.sorted { uses[$0] != uses[$1] ? uses[$0] > uses[$1] : $0 < $1 }.prefix(count).map { base + $0 }
    }

    /// Experts (layer * 512 + expert) most missed first, then most used, then in the shipped order; nil until
    /// `minimumTokens`. Misses lead because the expert cache already holds the most used experts: a mirror of those
    /// would serve few of the reads that reach the drive.
    public func ranking(minimumTokens: Int = RoutingProfile.minimumTokens) -> [Int]? {
        guard tokens >= minimumTokens else { return nil }
        lock.lock()
        let counts = self.counts
        let misses = self.misses
        lock.unlock()
        return Self.ranked(counts: counts, misses: misses)
    }
    private static func ranked(counts: [UInt32], misses: [UInt32]) -> [Int] {
        var shipped = [Int](repeating: Int.max, count: counts.count)
        for (position, key) in HotExpertRanking.keys.enumerated() { shipped[key] = position }
        return counts.indices.sorted {
            (misses[$0], counts[$0]) != (misses[$1], counts[$1])
                ? (misses[$0], counts[$0]) > (misses[$1], counts[$1])
                : (shipped[$0], $0) < (shipped[$1], $1)
        }
    }

    /// The counts a `HotCopyKeeper` keeps (uses and weighted misses per key, nonzero only): none before the first reply,
    /// so a new model's copy is learning until then, as the family engines' is. Keys these few counts leave tied keep the
    /// shipped order (`hotCopyChoice`), so the first copy is about the shipped ranking's, as 3.0's setup built it.
    package var hotCopyCounts: (uses: [Int: Int], misses: [Int: Int]) {
        lock.lock()
        let counts = self.counts
        let misses = self.misses
        lock.unlock()
        func nonzero(_ values: [UInt32]) -> [Int: Int] {
            Dictionary(uniqueKeysWithValues: values.indices.filter { values[$0] > 0 }.map { ($0, Int(values[$0])) })
        }
        return (nonzero(counts), nonzero(misses))
    }
    /// Qwen's hot copy for `counts` (`hotCopyCounts`): the `ranking` order of those counts, as many records of `record`
    /// bytes (padded to 16 KiB in the copy) as `budget` holds.
    package static func hotCopyChoice(_ counts: (uses: [Int: Int], misses: [Int: Int]), budget: UInt64, record: Int) -> [Int] {
        let size = ModelContract.layers * ModelContract.experts
        var uses = [UInt32](repeating: 0, count: size)
        var missed = uses
        for (key, count) in counts.uses where (0..<size).contains(key) { uses[key] = UInt32(clamping: count) }
        for (key, count) in counts.misses where (0..<size).contains(key) { missed[key] = UInt32(clamping: count) }
        let padded = UInt64((record + 16383) / 16384 * 16384)
        return Array(ranked(counts: uses, misses: missed).prefix(Int(budget / max(1, padded))))
    }

    /// Refuses a file whose arrays have the wrong length; one written before misses were counted loads with none.
    public static func load(_ url: URL) -> RoutingProfile? {
        func decoded(_ value: Any?) -> [UInt32]? {
            guard let encoded = value as? String, let raw = Data(base64Encoded: encoded),
                raw.count == ModelContract.layers * ModelContract.experts * 4
            else { return nil }
            return raw.withUnsafeBytes { bytes in
                (0..<(bytes.count / 4)).map { UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self)) }
            }
        }
        guard let data = try? Data(contentsOf: url), let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let windows = (object["windows"] as? NSNumber)?.uint64Value, let counts = decoded(object["counts"])
        else { return nil }
        let missed = decoded(object["missed"])
        guard missed != nil || object["missed"] == nil else { return nil }
        return RoutingProfile(counts: counts, missed: missed, windows: windows)
    }
    public func save(to url: URL) throws {
        lock.lock()
        let counts = self.counts
        let misses = self.misses
        let windows = self.windows
        lock.unlock()
        func encoded(_ values: [UInt32]) -> String { values.map(\.littleEndian).withUnsafeBytes { Data($0) }.base64EncodedString() }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: ["windows": windows, "counts": encoded(counts), "missed": encoded(misses)]).write(
            to: url, options: .atomic)
    }
}
