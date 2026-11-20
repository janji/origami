import Foundation
import OrigamiEngine

/// Usage counts decay (`HotCopy.halved`, `RoutingProfile.record`): after a workload shift the new experts overtake the
/// old within `2 * HotCopy.decayTokens` tokens, and do not without the halving; a halving never leaves fewer counts
/// than the ranking is trusted from; a profile written before the decay (the format is unchanged) loads and decays.
func usageCountsDecay() throws {
    func require(_ ok: Bool, _ what: @autoclosure () -> String) throws { if !ok { throw EngineError("usage decay: " + what()) } }
    let layers = ModelContract.layers
    let experts = ModelContract.experts
    let bound = 2 * HotCopy.decayTokens
    try require(HotCopy.decayTokens >= RoutingProfile.minimumTokens, "decay below the trust threshold")

    // Qwen: a profile of the previous build, 1,000,000 tokens of experts 0-9 on layer 0, in the old file format.
    var old = [UInt32](repeating: 0, count: layers * experts)
    for e in 0..<10 { old[e] = 1_000_000 }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-usage-decay-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    let file = folder.appendingPathComponent("routing.json")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    let encoded = old.map(\.littleEndian).withUnsafeBytes { Data($0) }.base64EncodedString()
    try JSONSerialization.data(withJSONObject: ["windows": 1_000_000 * layers, "counts": encoded]).write(to: file)
    guard let profile = RoutingProfile.load(file) else { throw EngineError("usage decay: the old file does not load") }
    try require(profile.tokens == 1_000_000, "old tokens \(profile.tokens)")
    // The workload shifts to experts 100-109 on layer 0 (other layers: one expert each).
    let shifted = Array(100..<110)
    var overtaken: Int?
    for token in 1...bound {
        profile.record(layer: 0, experts: shifted)
        for layer in 1..<layers { profile.record(layer: layer, experts: [0]) }
        try require(profile.tokens >= RoutingProfile.minimumTokens, "too few counts at token \(token)")
        if token % 500 == 0, let top = profile.mostUsed(layer: 0, count: 10), Set(top) == Set(shifted) {
            overtaken = token
            break
        }
    }
    try require(overtaken != nil, "the shifted experts did not overtake within \(bound) tokens")
    // The halved profile keeps its format: it saves and loads again with the same counts.
    try profile.save(to: file)
    let again = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any]
    try require(Set(again?.keys.map { $0 } ?? []) == ["windows", "counts", "missed"], "saved keys \(again?.keys.sorted() ?? [])")
    try require(RoutingProfile.load(file)?.mostUsed(layer: 0, count: 10) == profile.mostUsed(layer: 0, count: 10), "reload")

    // Family: per-reply merges of 1,000 tokens (8 uses per token here), as `FamilyServingBackend` does.
    let perToken = 8
    let floor = HotCopy.decayTokens * perToken
    for decays in [true, false] {
        var earlier = Dictionary(uniqueKeysWithValues: (0..<8).map { ($0, 1_000_000) })
        var model: [Int: Int] = [:]
        var halvedAt: [Int: Int] = [:]
        var passed: Int?
        for reply in 1...(bound / 1_000) {
            for key in 100..<108 { model[key, default: 0] += 1_000 }
            var uses = HotCopy.mergeUses(earlier, HotCopy.added(model, since: halvedAt))
            if decays, let halved = HotCopy.halved(uses, [:], floor: floor) {
                uses = halved.uses
                earlier = halved.uses
                halvedAt = model
                try require(uses.values.reduce(0, +) >= floor, "family total \(uses.values.reduce(0, +)) under \(floor)")
            }
            if passed == nil, (100..<108).allSatisfy({ (uses[$0] ?? 0) > (uses[0] ?? 0) }) { passed = reply * 1_000 }
        }
        try require((passed != nil) == decays, "family overtaking with decay \(decays): \(String(describing: passed))")
    }
    // Below the bound nothing changes; a key counted once stays counted.
    try require(HotCopy.halved([1: 5], [:], floor: 10) == nil, "halved below the bound")
    try require(
        HotCopy.halved([1: 1, 2: 39], [3: 1], floor: 10).map { $0.uses == [1: 1, 2: 10] && $0.misses == [3: 1] } == true, "rounding")
}
