import Foundation
import Metal
import OrigamiEngine

/// ROADMAP R14: after a hot copy grows in place, the counted bytes per source equal the bytes each file was asked for,
/// on every path (prompt read-ahead, demand, decode guesses), and status's split names the model's drive when it read.
/// Before the fix the split took `expert_hot_reads` (demand and read-ahead) against `expert_misses` (demand only), so a
/// copy serving the read-ahead hid every read from the model's drive.
func familyReadCountersAfterGrow() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family read counters: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-read-counters-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let weights = try FamilyWeights(gguf: GGUFSource(try expertFixtureGGUF(in: folder)), device: device)
    let experts = try FamilyExperts(weights: weights, budget: UInt64(8 * (212_992 + 114_688)))
    // Keys are layer * 4 + expert (layers 1 and 2). A copy of two records, then grown in place by two more.
    let hot = folder.appendingPathComponent("hot")
    let copy = try HotCopy.write(source: experts, keys: [6, 9], folder: hot, identity: "a")
    experts.attach(copy)
    _ = try experts.fetch(layer: 1, experts: [2], decode: true)
    let grown = try HotCopy.write(source: experts, keys: [4, 11], folder: hot, identity: "a", onto: copy)
    experts.attach(grown)
    let modelFiles = try weights.gguf.files.map { try weights.file($0) }
    func modelRead() -> UInt64 { modelFiles.reduce(0) { $0 + $1.readBytes } }
    func copyRead() -> UInt64 {
        var seen = Set<ObjectIdentifier>()
        return [copy.file, grown.file].reduce(0) { seen.insert(ObjectIdentifier($1)).inserted ? $0 + $1.readBytes : $0 }
    }
    let before = experts.counters
    let (modelBefore, copyBefore) = (modelRead(), copyRead())
    // A prompt: each layer's records read ahead (mostly from the copy) before they are asked for.
    experts.readAhead(layer: 2, experts: [1, 3])
    _ = try experts.fetch(layer: 2, experts: [1, 3], decode: false)
    experts.readAhead(layer: 1, experts: [0, 1])
    _ = try experts.fetch(layer: 1, experts: [0, 1], decode: false)
    // Decode: a guess, then demand from the model's file.
    experts.prefetch(layer: 2, experts: [0, 2])
    _ = try experts.fetch(layer: 2, experts: [0, 2], decode: true)
    _ = try experts.fetch(layer: 1, experts: [3], decode: true)
    let after = experts.counters
    let counted = (copy: UInt64(after.copyBytes - before.copyBytes), model: UInt64(after.modelBytes - before.modelBytes))
    let read = (copy: copyRead() - copyBefore, model: modelRead() - modelBefore)
    try require(
        counted.copy == read.copy && counted.model == read.model && read.model > 0,
        "counted \(counted) bytes against \(read) read from the copy and the model's file")
    let split = after.split
    try require(
        after.lookups - before.lookups == 7 && split.drive > 0 && abs(split.memory + split.hot + split.drive - 1) < 1e-9,
        "lookups \(after.lookups - before.lookups), split \(split)")
}

/// ROADMAP R14: status's read counters move while a request runs (not only at its start and end), and the running
/// request's phase counters split them into prompt reading and decode.
func familyStatusCountersDuringRequest() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family live counters: " + what) } }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-live-counters-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let path = try syntheticFamilyGGUF(.glm, in: folder)
    let tokenizer = try tinyFamilyTokenizer("{% for m in messages %}{{ m.content }}{% endfor %}", fillers: 2_048 - 261, fillerType: 4)
    let ledger = try MemoryLedger(ceiling: 64 << 30, transientMargin: 1, availableFloor: 1)
    let weights = try FamilyWeights(gguf: GGUFSource(path), device: device)
    let model = try makeFamilyModel(weights, expertBudget: nil, memoryLimit: 8 << 30, context: 8_192, ledger: ledger) { _ in }
    let backend = FamilyServingBackend(model: model, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: false)
    let interval = FamilyServingBackend.countersInterval
    FamilyServingBackend.countersInterval = 0
    defer { FamilyServingBackend.countersInterval = interval }
    let body: [String: Any] = [
        "model": "family", "messages": [["role": "user", "content": String(repeating: "status counters ", count: 20)]],
        "max_tokens": 12, "temperature": 0, "reasoning_effort": "none", "prompt_cache_policy": "no_store",
    ]
    let request = try ServingRequest.parse(JSONSerialization.data(withJSONObject: body), contextTier: 8_192, defaults: ServingDefaults())
    let control = GenerationControl()
    let start = backend.telemetrySnapshot.notes["expert_lookups"] as? Int ?? -1
    final class Seen: @unchecked Sendable {
        var lookups: [Int] = [], phases: [String: Double] = [:]
    }
    let observed = Seen()  // written on the request's thread only, read after it returns
    let statistics = try backend.prepare(request, cancellation: control).run { _ in
        let snapshot = backend.telemetrySnapshot
        if snapshot.boundary == "request" { observed.lookups.append(snapshot.notes["expert_lookups"] as? Int ?? -1) }
        observed.phases = control.progress.phaseCounters
        return true
    }
    let (seen, phases) = (observed.lookups, observed.phases)
    try require(start >= 0 && statistics.completionTokens > 1, "start \(start), \(statistics.completionTokens) tokens")
    try require(
        seen.count > 1 && seen.first! > start && seen.last! > seen.first!,
        "status during the request: \(seen) lookups (\(start) at its start)")
    try require(
        (phases["prefill_expert_lookups"] ?? 0) > 0 && (phases["decode_expert_lookups"] ?? 0) > 0,
        "phase counters during the request: \(phases)")
}
