import Foundation
import Metal
import OrigamiEngine

/// A family engine (GLM-5.3-Flash, DeepSeek-V4.1-Flash) on a real GGUF against reference logits (skipped unless set):
/// ORIGAMI_FAMILY_GGUF is the model, ORIGAMI_FAMILY_REFERENCE a folder of `<name>.tokens.json` (token ids) and
/// `<name>.logits.json` (DwarfStar's `--dump-logits` JSON, or any JSON with a `logits` array). Each prompt runs as one prefill, then again as a prefill of all but its last token and one
/// decode step. Passes when the top token matches, the logits' cosine is at least 0.99 (top-1 only against a
/// top-log-probability reference) and the decode step's logits match the prefill's (cosine 0.999). Prints the
/// per-prompt numbers.
func familyReferenceLogits() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let path = environment["ORIGAMI_FAMILY_GGUF"], let folder = environment["ORIGAMI_FAMILY_REFERENCE"] else {
        throw FixtureUnavailable(
            description: "set ORIGAMI_FAMILY_GGUF and ORIGAMI_FAMILY_REFERENCE to compare a family engine with reference logits")
    }
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let weights = try FamilyWeights(gguf: GGUFSource(URL(fileURLWithPath: path)), device: device)
    let budget = environment["ORIGAMI_FAMILY_EXPERT_GB"].flatMap(Double.init).map { UInt64($0 * 1e9) }  // nil: what is free
    let model = try makeFamilyModel(weights, expertBudget: budget, context: 4096) { FileHandle.standardError.write(Data(($0 + "\n").utf8)) }
    func cosine(_ a: [Float], _ b: [Float]) -> Double {
        var dot = 0.0
        var na = 0.0
        var nb = 0.0
        for (x, y) in zip(a, b) {
            dot += Double(x) * Double(y)
            na += Double(x) * Double(x)
            nb += Double(y) * Double(y)
        }
        return dot / (na.squareRoot() * nb.squareRoot())
    }
    func best(_ v: [Float], _ k: Int) -> [Int] { Array(v.indices.sorted { v[$0] > v[$1] }.prefix(k)) }
    var failures: [String] = []
    let only = environment["ORIGAMI_FAMILY_PROMPTS"].map { Set($0.split(separator: ",").map(String.init)) }
    let names = try FileManager.default.contentsOfDirectory(atPath: folder).filter { $0.hasSuffix(".logits.json") }.map {
        String($0.dropLast(".logits.json".count))
    }
    .filter { only?.contains($0) ?? true }.sorted()
    func swapUsed() -> Double {
        var usage = xsw_usage()
        var size = MemoryLayout<xsw_usage>.size
        return sysctlbyname("vm.swapusage", &usage, &size, nil, 0) == 0 ? Double(usage.xsu_used) / 1e9 : -1
    }
    let swapBefore = swapUsed()
    for name in names {
        let tokens =
            try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: folder + "/\(name).tokens.json"))) as! [Int]
        let reference =
            try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: folder + "/\(name).logits.json")))
            as! [String: Any]
        // A full logits array (DwarfStar), or the top log-probabilities (llama-server's n_probs): compared through ours.
        let top = (reference["top"] as? [[NSNumber]])?.map { (Int(truncating: $0[0]), $0[1].doubleValue) }
        let expected = (reference["logits"] as? [Any])?.map { ($0 as? NSNumber)?.floatValue ?? -.infinity } ?? []
        try model.reset()
        var started = Date()
        let logits = try model.forward(tokens)
        let prefillSeconds = Date().timeIntervalSince(started)
        try model.reset()
        _ = try model.forward(Array(tokens.dropLast()))
        started = Date()
        let stepped = try model.forward([tokens.last!])
        let stepSeconds = Date().timeIntervalSince(started)
        let d = cosine(stepped, logits)
        if let top = top {
            // Our log-softmax at the reference's top tokens.
            let mx = logits.max()!
            let lse = log(logits.reduce(0.0) { $0 + exp(Double($1 - mx)) }) + Double(mx)
            let ours = top.map { Double(logits[$0.0]) - lse }
            let t1 = best(logits, 1)[0]
            let overlap = Set(best(logits, 5)).intersection(top.prefix(5).map(\.0)).count
            let worst = zip(ours, top.map(\.1)).map { abs($0 - $1) }.max() ?? 0
            FileHandle.standardError.write(
                Data(
                    String(
                        format:
                            "family %@: %d tokens, prefill %.1f s, step %.2f s; top-1 %d (reference %d), top-5 overlap %d, max |log p diff| over the reference's top 20 %.3f, decode-vs-prefill cosine %.6f\n",
                        name, tokens.count, prefillSeconds, stepSeconds, t1, top[0].0, overlap, worst, d
                    ).utf8))
            if t1 != top[0].0 { failures.append("\(name): top-1 \(t1) vs \(top[0].0)") }
            if d < 0.999 { failures.append("\(name): decode step differs from prefill (cosine \(d))") }
            continue
        }
        let c = cosine(logits, expected)
        let t1 = best(logits, 1)[0]
        let r1 = best(expected, 1)[0]
        let overlap = Set(best(logits, 5)).intersection(best(expected, 5)).count
        let worst = zip(logits, expected).map { abs($0 - $1) }.max() ?? 0
        FileHandle.standardError.write(
            Data(
                String(
                    format:
                        "family %@: %d tokens, prefill %.1f s, step %.2f s; top-1 %d (reference %d), top-5 overlap %d, cosine %.5f, max |diff| %.3f, decode-vs-prefill cosine %.6f\n",
                    name, tokens.count, prefillSeconds, stepSeconds, t1, r1, overlap, c, worst, d
                ).utf8))
        if t1 != r1 || c < 0.99 { failures.append("\(name): top-1 \(t1) vs \(r1), cosine \(c)") }
        if d < 0.999 { failures.append("\(name): decode step differs from prefill (cosine \(d))") }
    }
    // Greedy decode after the first prompt (ORIGAMI_FAMILY_DECODE tokens, default 16): speed and where the time goes.
    let decode = Int(environment["ORIGAMI_FAMILY_DECODE"] ?? "16") ?? 16
    if decode > 0, let first = names.first {
        let tokens =
            try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: folder + "/\(first).tokens.json"))) as! [Int]
        try model.reset()
        var logits = try model.forward(tokens)
        var out: [Int] = []
        let before = model.timing
        let hits = (model as! FamilyRunner).experts.counters
        let started = Date()
        for _ in 0..<decode {
            let next = logits.indices.max { logits[$0] < logits[$1] }!
            out.append(next)
            logits = try model.forward([next])
        }
        let seconds = Date().timeIntervalSince(started)
        let after = model.timing
        let c = (model as! FamilyRunner).experts.counters
        FileHandle.standardError.write(
            Data(
                String(
                    format:
                        "family decode: %d tokens in %.1f s (%.2f tok/s); per token GPU wait %.3f s (GPU busy %.3f s), reads %.3f s, other %.3f s; hits %d misses %d; swap %.2f -> %.2f GB; tokens %@\n",
                    decode, seconds, Double(decode) / seconds, (after.gpuWait - before.gpuWait) / Double(decode),
                    (after.gpu - before.gpu) / Double(decode), (after.reads - before.reads) / Double(decode),
                    (after.total - before.total - (after.gpuWait - before.gpuWait) - (after.reads - before.reads)) / Double(decode),
                    c.hits - hits.hits, c.misses - hits.misses, swapBefore, swapUsed(), "\(out)"
                ).utf8))
    }
    let notes = model.statusNotes()
    FileHandle.standardError.write(Data("family experts: \(notes)\n".utf8))
    guard failures.isEmpty else { throw EngineError("Family reference: " + failures.joined(separator: "; ")) }
}
