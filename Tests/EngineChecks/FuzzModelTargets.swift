import Foundation
import OrigamiEngine

// The model-file fuzz targets (`Fuzz.swift`): a GGUF header, and a safetensors shard with its config.json. Inputs keep
// only the header: the weights after it are a virtual run of zero bytes whose length the input's first 8 bytes give,
// so a 20 KB input stands for a model file of any size.

/// Header bytes followed by `tail` zero bytes, read as a file of that size (short reads at the end, like `LocalBytes`).
struct FuzzVirtualBytes: ByteSource {
    let data: Data
    let tail: UInt64
    var size: UInt64 { UInt64(data.count) + tail }
    func read(at offset: UInt64, count: Int) throws -> Data {
        guard count >= 0 else { throw EngineError("negative read") }
        guard offset < size else { return Data() }
        let end = min(size, offset + UInt64(count))
        var out = Data(count: Int(end - offset))
        if offset < UInt64(data.count) {
            let stop = min(UInt64(data.count), end)
            out.replaceSubrange(0..<Int(stop - offset), with: data[Int(offset)..<Int(stop)])
        }
        return out
    }
}

private func fuzzU64(_ data: Data, _ at: Int) -> UInt64 {
    data.count >= at + 8 ? data.subdata(in: at..<(at + 8)).withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) } : 0
}
private func fuzzU32(_ data: Data, _ at: Int) -> UInt32 {
    data.count >= at + 4 ? data.subdata(in: at..<(at + 4)).withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self)) } : 0
}
private func le<T: FixedWidthInteger>(_ value: T) -> Data { withUnsafeBytes(of: value.littleEndian) { Data($0) } }

/// Spans of tensors with a known length must lie in their file and not overlap one another.
private func fuzzCheckSpans(_ spans: [(name: String, file: String, offset: Int64, length: Int64)], size: [String: UInt64]) throws {
    var byFile: [String: [(Int64, Int64, String)]] = [:]
    for span in spans {
        guard span.offset >= 0, UInt64(span.offset) <= size[span.file] ?? 0 else {
            throw FuzzFinding(description: "\(span.name) starts at \(span.offset), outside its file")
        }
        guard span.length >= 0 else { continue }
        guard span.length <= Int64(size[span.file] ?? 0) - span.offset else {
            throw FuzzFinding(description: "\(span.name) (\(span.offset), \(span.length) bytes) ends past its file")
        }
        if span.length > 0 { byFile[span.file, default: []].append((span.offset, span.length, span.name)) }
    }
    for (_, list) in byFile {
        let sorted = list.sorted { $0.0 < $1.0 }
        for (a, b) in zip(sorted, sorted.dropFirst()) where a.0 + a.1 > b.0 {
            throw FuzzFinding(description: "\(a.2) and \(b.2) overlap (\(a.0)+\(a.1) > \(b.0))")
        }
    }
}

// MARK: GGUF

/// A GGUF header taken apart where it can be: metadata payloads stay raw, tensor infos are fields.
private struct FuzzGGUF {
    var version: UInt32 = 3
    var tensorCount: UInt64?, keyCount: UInt64?
    var keys: [(key: String, type: UInt32, payload: Data)] = []
    var tensors: [(name: String, dims: [UInt64], type: UInt32, offset: UInt64)] = []
    var tail: UInt64 = 0

    /// The input's layout: 8 bytes of tail length, then the header.
    init?(_ input: Data) {
        guard input.count >= 32, input.subdata(in: 8..<12) == Data("GGUF".utf8) else { return nil }
        tail = fuzzU64(input, 0)
        let h = input.subdata(in: 8..<input.count)
        var at = 4
        func u32() -> UInt32? {
            defer { at += 4 }
            return at + 4 <= h.count ? fuzzU32(h, at) : nil
        }
        func u64() -> UInt64? {
            defer { at += 8 }
            return at + 8 <= h.count ? fuzzU64(h, at) : nil
        }
        func string() -> String? {
            guard let n = u64(), n < 1 << 20, at + Int(n) <= h.count else { return nil }
            defer { at += Int(n) }
            return String(decoding: h.subdata(in: at..<(at + Int(n))), as: UTF8.self)
        }
        /// The bytes of a value of `type`, or nil where it cannot be walked.
        func skip(_ type: UInt32, depth: Int = 0) -> Bool {
            if let width = [0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8][type] {
                at += width
                return at <= h.count
            }
            if type == 8 { return string() != nil }
            guard type == 9, depth == 0, let element = u32(), let count = u64(), count < 1 << 20 else { return false }
            for _ in 0..<count where !skip(element, depth: 1) { return false }
            return true
        }
        guard let version = u32(), let tensorCount = u64(), let keyCount = u64(), tensorCount < 1 << 16, keyCount < 1 << 16 else {
            return nil
        }
        self.version = version
        for _ in 0..<keyCount {
            guard let key = string(), let type = u32() else { return nil }
            let start = at
            guard skip(type) else { return nil }
            keys.append((key, type, h.subdata(in: start..<at)))
        }
        for _ in 0..<tensorCount {
            guard let name = string(), let n = u32(), n <= 8 else { return nil }
            var dims: [UInt64] = []
            for _ in 0..<n {
                guard let d = u64() else { return nil }
                dims.append(d)
            }
            guard let type = u32(), let offset = u64() else { return nil }
            tensors.append((name, dims, type, offset))
        }
        // Bytes after the tensor infos (padding) count towards the tail.
        tail &+= UInt64(h.count - at)
    }

    var data: Data {
        var h = Data("GGUF".utf8)
        h += le(version) + le(tensorCount ?? UInt64(tensors.count)) + le(keyCount ?? UInt64(keys.count))
        for (key, type, payload) in keys { h += le(UInt64(key.utf8.count)) + Data(key.utf8) + le(type) + payload }
        for (name, dims, type, offset) in tensors {
            h += le(UInt64(name.utf8.count)) + Data(name.utf8) + le(UInt32(dims.count))
            for d in dims { h += le(d) }
            h += le(type) + le(offset)
        }
        return le(tail) + h
    }

    var architecture: String {
        for (key, type, payload) in keys where key == "general.architecture" && type == 8 && payload.count >= 8 {
            return String(decoding: payload.dropFirst(8), as: UTF8.self)
        }
        return "glm5-next"
    }
}

private func ggufString(_ s: String) -> Data { le(UInt64(s.utf8.count)) + Data(s.utf8) }

/// Numeric keys the geometry, sampling and tokenizer readers take (after `<architecture>.` where they start with a dot).
private let ggufNumericKeys = [
    ".block_count", ".trunk_block_count", ".expert_count", ".expert_used_count", ".expert_shared_count", ".embedding_length",
    ".vocab_size", ".expert_feed_forward_length", ".feed_forward_length", ".leading_dense_block_count", ".context_length",
    ".attention.head_count", ".attention.head_count_kv", ".attention.key_length", ".attention.value_length", ".rope.dimension_count",
    ".attention.q_lora_rank", ".attention.kv_lora_rank", ".nextn_predict_layers", ".hyper_connection.count",
    ".attention.indexer.top_k", ".attention.indexer.head_count", ".attention.indexer.key_length", ".attention.sliding_window",
    ".attention.compress_ratios", ".attention.kv_source_layers", ".attention.indexer.source_layers", ".engram.layer_ids",
    ".ple.layers", ".ple.ngram_size", ".engram.head_count", ".engram.key_length", ".ssm.state_size", ".ssm.conv_kernel",
    ".linear_attention.head_count", ".layer_types", ".rope.freq_base", ".expert_weights_scale", ".attention.layer_norm_rms_epsilon",
    "general.alignment", "split.count", "split.no", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eot_token_id", "tokenizer.ggml.padding_token_id", "general.sampling.temp", "general.sampling.top_k",
    "general.sampling.top_p", ".engram.token_map", ".engram.multipliers", ".engram.primes", "general.file_type",
]

private func ggufMutate(_ rng: inout FuzzRNG, _ input: Data, _ corpus: [Data]) -> Data {
    guard var g = FuzzGGUF(input) else { return input }
    let arch = g.architecture
    func integerPayload(_ type: UInt32) -> Data {
        var value = rng.pick(fuzzInteresting)
        if rng.chance(30) { value = 0 &- value }
        switch type {
        case 0, 1, 7: return le(UInt8(truncatingIfNeeded: value))
        case 2, 3: return le(UInt16(truncatingIfNeeded: value))
        case 4, 5: return le(UInt32(truncatingIfNeeded: value))
        case 6: return le(rng.pick([Float.nan, .infinity, -1, 0, 1e30, 0.5, Float.leastNonzeroMagnitude]).bitPattern)
        case 12: return le(rng.pick([Double.nan, .infinity, -1, 0, 1e300, 0.5]).bitPattern)
        default: return le(value)
        }
    }
    func arrayPayload() -> Data {
        let element = rng.pick([UInt32(4), 5, 0, 10, 8, 9, 6, 7, 13])
        let count = rng.chance(80) ? UInt64(rng.below(70)) : rng.pick(fuzzInteresting)
        var payload = le(element) + le(count)
        for _ in 0..<min(count, 300) {
            payload += element == 8 ? ggufString(rng.pick(fuzzStrings)) : integerPayload(element)
        }
        return payload
    }
    for _ in 0..<(1 + rng.below(3)) {
        switch rng.below(13) {
        case 0 where !g.keys.isEmpty:
            // A string value: a config JSON mutated as a tree, any other string replaced.
            let i = rng.below(g.keys.count)
            guard g.keys[i].type == 8 else { continue }
            let text = String(decoding: g.keys[i].payload.dropFirst(8), as: UTF8.self)
            if let tree = fuzzParseJSON(Data(text.utf8)), rng.chance(80) {
                g.keys[i].payload = ggufString(fuzzJSONText(fuzzMutateJSON(&rng, tree)))
            } else {
                g.keys[i].payload = ggufString(rng.pick(fuzzStrings + ["glm5-next", "deepseek41", "qwen4exp", "llama", "gpt2", "glm4"]))
            }
        case 1:
            // A numeric key, of any integer or float type.
            let suffix = rng.pick(ggufNumericKeys)
            let key = suffix.hasPrefix(".") ? arch + suffix : suffix
            let type = rng.pick([UInt32(4), 5, 10, 11, 0, 1, 2, 3, 6, 12, 7])
            let entry = (key, rng.chance(15) ? UInt32(9) : type, rng.chance(15) ? arrayPayload() : integerPayload(type))
            let payload = entry.1 == 9 && entry.2.count < 12 ? arrayPayload() : entry.2
            if let i = g.keys.firstIndex(where: { $0.key == key }), rng.chance(70) {
                g.keys[i] = (key, entry.1, payload)
            } else {
                g.keys.insert((key, entry.1, payload), at: rng.below(g.keys.count + 1))
            }
        case 2 where !g.keys.isEmpty:
            g.keys[rng.below(g.keys.count)].type = rng.pick([UInt32(0), 4, 8, 9, 10, 12, 13, 255, 1 << 31])
        case 3 where !g.keys.isEmpty:
            let i = rng.below(g.keys.count)
            if rng.chance(50) { g.keys.append(g.keys[i]) } else { g.keys.remove(at: i) }
        case 4 where !g.keys.isEmpty:
            let i = rng.below(g.keys.count)
            g.keys[i].type = 9
            g.keys[i].payload = arrayPayload()
        case 5 where g.tensors.count > 1:
            g.tensors[rng.below(g.tensors.count)].name = g.tensors[rng.below(g.tensors.count)].name
        case 6 where !g.tensors.isEmpty:
            let i = rng.below(g.tensors.count)
            switch rng.below(4) {
            case 0: g.tensors[i].offset = g.tensors[rng.below(g.tensors.count)].offset
            case 1: g.tensors[i].offset &+= UInt64(rng.pick([1, 16, 31, 32]))
            case 2: g.tensors[i].offset = g.tail &- UInt64(rng.below(64))
            default: g.tensors[i].offset = rng.pick(fuzzInteresting)
            }
        case 7 where !g.tensors.isEmpty:
            let i = rng.below(g.tensors.count)
            if rng.chance(25) || g.tensors[i].dims.isEmpty {
                if rng.chance(50) { g.tensors[i].dims.append(rng.pick(fuzzInteresting)) } else { g.tensors[i].dims = [] }
            } else {
                g.tensors[i].dims[rng.below(g.tensors[i].dims.count)] = rng.chance(50) ? rng.pick(fuzzInteresting) : UInt64(rng.below(4096))
            }
        case 8 where !g.tensors.isEmpty:
            g.tensors[rng.below(g.tensors.count)].type = rng.pick([UInt32(0), 1, 2, 8, 10, 12, 14, 16, 23, 24, 30, 31, 39, 42, 43, 44, 255])
        case 9 where !g.tensors.isEmpty:
            if rng.chance(50) {
                g.tensors.remove(at: rng.below(g.tensors.count))
            } else {
                g.tensors.append(g.tensors[rng.below(g.tensors.count)])
            }
        case 10:
            if rng.chance(50) { g.tensorCount = rng.pick(fuzzInteresting) } else { g.keyCount = rng.pick(fuzzInteresting) }
        case 11:
            g.tail = rng.chance(50) ? UInt64(rng.below(4096)) : g.tail / UInt64(1 + rng.below(4)) + UInt64(rng.below(3))
        case 12: g.version = rng.pick([UInt32(1), 2, 3, 4, UInt32.max])
        default: continue
        }
    }
    return g.data
}

/// What a model-file reader does with a header before any weight is read: compatibility, geometry, sampling defaults,
/// conversion and resident-pack plans; with a tokenizer, the support files and the tokenizer itself.
private func ggufRun(_ input: Data) throws {
    guard input.count >= 8 else { return }
    let tail = fuzzU64(input, 0) % (1 << 32)
    let source = FuzzVirtualBytes(data: input.subdata(in: 8..<input.count), tail: tail)
    let folder = CheckFixtures.scratch.appendingPathComponent("fuzz-gguf-\(getpid())", isDirectory: true)
    let gguf: GGUFSource
    do { gguf = try GGUFSource(folder: folder, parts: [("model.gguf", source)]) } catch is EngineError { return }
    try fuzzCheckSpans(
        gguf.tensors.values.map { ($0.name, $0.file, $0.offset, $0.length) }, size: ["model.gguf": source.size])
    _ = ModelCompatibility.check(gguf: gguf, source: "fuzz")
    _ = FamilySupport.stopTokens(gguf)
    _ = SamplingDefaults.recommended(gguf.metadata, splitModes: true)
    _ = CheckpointConverter.ggufProblem(gguf)
    _ = CheckpointConverter.keepsExpertsNative(gguf)
    if let geometry = try? ModelGeometry(gguf: gguf) {
        _ = try? FamilyResidentPack.plan(gguf, geometry: geometry)
        for table in gguf.tensors.values where table.name.contains("engram") {
            _ = DeepSeekModel.engramRowBytes(table, dim: geometry.int("engram_head_dim") ?? 0)
        }
    }
    // The tokenizer path reads the file by URL: a sparse file of the same size.
    guard input.range(of: Data("tokenizer.ggml.tokens".utf8)) != nil else { return }
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    let url = folder.appendingPathComponent("model.gguf")
    try source.data.write(to: url)
    let handle = try FileHandle(forWritingTo: url)
    try handle.truncate(atOffset: source.size)
    try handle.close()
    guard let support = try? FamilySupport.prepare(url, into: folder.appendingPathComponent("support")) else { return }
    final class Box: @unchecked Sendable { var stack: TokenizerStack? }
    let box = Box()
    let done = DispatchSemaphore(value: 0)
    Task {
        box.stack = try? await TokenizerStack(modelDirectory: support)
        done.signal()
    }
    done.wait()
    if let stack = box.stack {
        _ = try? stack.encodeChat(messages: [["role": .string("user"), "content": .string("Hi <|user|> é 😀")]])
        _ = try? stack.decode(Array(0..<300))
    }
}

/// The seeds: the synthetic GLM and DeepSeek headers the end-to-end checks run, and a tokenizer-only GGUF.
nonisolated(unsafe) private var ggufSeedCache: [Data]?
private func ggufSeeds() throws -> [Data] {
    if let ggufSeedCache { return ggufSeedCache }
    let folder = CheckFixtures.scratch.appendingPathComponent("fuzz-gguf-seeds-\(getpid())", isDirectory: true)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    var paths = [try syntheticFamilyGGUF(.glm, in: folder, weights: false), try syntheticFamilyGGUF(.deepseek, in: folder, weights: false)]
    var shifted = 0
    let alphabet = (0..<256).map { b -> String in
        if (33...126).contains(b) || (161...172).contains(b) || (174...255).contains(b) { return String(UnicodeScalar(UInt8(b))) }
        shifted += 1
        return String(UnicodeScalar(UInt32(255 + shifted))!)
    }
    let tokens = alphabet + ["Hi", "<think>", "</think>", "<|user|>", "<|assistant|>"]
    let tokenizer = folder.appendingPathComponent("tokenizer.gguf")
    try familyGGUF(
        [
            ("general.architecture", "glm5-next"), ("tokenizer.ggml.model", "gpt2"), ("tokenizer.ggml.pre", "glm4"),
            ("tokenizer.ggml.tokens", tokens), ("tokenizer.ggml.merges", ["H i"]),
            ("tokenizer.ggml.token_type", [Int](repeating: 1, count: 257) + [3, 3, 3, 3]), ("tokenizer.ggml.eos_token_id", 259),
            (
                "tokenizer.chat_template",
                try String(contentsOf: CheckFixtures.file("models/glm-5.3-flash/chat_template.jinja"), encoding: .utf8)
            ),
        ], tensors: [("token_embd.weight", [261, 32], 0)], to: tokenizer)
    paths.append(tokenizer)
    var seeds: [Data] = []
    for path in paths {
        let file = try Data(contentsOf: path)
        // The header ends where the walker stops: everything after it becomes the virtual tail.
        guard let g = FuzzGGUF(le(UInt64(0)) + file) else { throw EngineError("fuzz: cannot walk the seed \(path.lastPathComponent)") }
        seeds.append(g.data)
    }
    ggufSeedCache = seeds
    return seeds
}

func ggufFuzzTarget() -> FuzzTarget { FuzzTarget(name: "gguf", seeds: ggufSeeds, mutate: ggufMutate, run: ggufRun) }

// MARK: safetensors and config.json

/// The input's layout: u32 config length, the config JSON, u64 tail length, the shard's header bytes.
private func safetensorsSplit(_ input: Data) -> (config: Data, tail: UInt64, shard: Data)? {
    guard input.count >= 12 else { return nil }
    let length = Int(fuzzU32(input, 0))
    guard length <= input.count - 12 else { return nil }
    return (input.subdata(in: 4..<(4 + length)), fuzzU64(input, 4 + length), input.subdata(in: (12 + length)..<input.count))
}
private func safetensorsJoin(_ config: Data, _ tail: UInt64, _ shard: Data) -> Data { le(UInt32(config.count)) + config + le(tail) + shard }

private func safetensorsMutate(_ rng: inout FuzzRNG, _ input: Data, _ corpus: [Data]) -> Data {
    guard var (config, tail, shard) = safetensorsSplit(input) else { return input }
    switch rng.below(5) {
    case 0, 1:
        if let tree = fuzzParseJSON(config) { config = Data(fuzzJSONText(fuzzMutateJSON(&rng, tree)).utf8) }
    case 2, 3:
        // The shard's JSON header as a tree, its length prefix rewritten (or not).
        let length = Int(min(fuzzU64(shard, 0), UInt64(Int32.max)))
        guard shard.count >= 8, length <= shard.count - 8, let tree = fuzzParseJSON(shard.subdata(in: 8..<(8 + length))) else { break }
        let json = Data(fuzzJSONText(fuzzMutateJSON(&rng, tree)).utf8)
        let prefix = rng.chance(85) ? UInt64(json.count) : rng.pick(fuzzInteresting)
        shard = le(prefix) + json + shard.subdata(in: (8 + length)..<shard.count)
    default:
        tail = rng.chance(50) ? UInt64(rng.below(1 << 20)) : tail / 2
    }
    return safetensorsJoin(config, tail, shard)
}

private func safetensorsRun(_ input: Data) throws {
    guard let (config, tail, shard) = safetensorsSplit(input) else { return }
    let source = FuzzVirtualBytes(data: shard, tail: tail % (1 << 40))
    var headers: [String: TensorSpan] = [:]
    do { headers = try ModelSource.header(of: source, file: "model.safetensors") } catch is EngineError {}
    let length = Int64(fuzzU64(shard, 0) % (1 << 62))
    for span in headers.values where span.offset < 8 + length {
        throw FuzzFinding(description: "a tensor starts at \(span.offset), inside the \(length)-byte header")
    }
    try fuzzCheckSpans(headers.map { ($0.key, $0.value.file, $0.value.offset, $0.value.length) }, size: ["model.safetensors": source.size])
    _ = try? ModelContract.validate(config: config)
    _ = ModelContract.formats(config: config)
    guard let object = fuzzParseJSON(config) as? [String: Any] else { return }
    _ = ModelCompatibility.check(config: object, headers: headers, source: "fuzz")
    _ = try? ModelGeometry(config: object)
}

/// The seeds: each fixture config with a shard header holding layer 0's MLP tensors of the MLX checkpoint.
private func safetensorsSeeds() throws -> [Data] {
    let layout = try JSONSerialization.jsonObject(with: Data(contentsOf: CheckFixtures.file("models/mlx-community-4bit/layer0-mlp.json")))
    let tensors = ((layout as? [String: Any])?["tensors"] as? [String: [String: Any]]) ?? [:]
    var header: [String: Any] = ["__metadata__": ["format": "mlx"]]
    var offset = 0
    for (name, entry) in tensors.sorted(by: { $0.key < $1.key }) {
        let length = entry["length"] as? Int ?? 0
        header[name] = ["dtype": entry["dtype"] ?? "BF16", "shape": entry["shape"] ?? [], "data_offsets": [offset, offset + length]]
        offset += length
    }
    let json = try JSONSerialization.data(withJSONObject: header, options: .sortedKeys)
    let shard = le(UInt64(json.count)) + json
    return try ["official-bf16", "mlx-community-4bit", "glm-5.3-flash", "deepseek-v4.1-flash"].map { model in
        safetensorsJoin(try Data(contentsOf: CheckFixtures.file("models/\(model)/config.json")), UInt64(offset), shard)
    }
}

func safetensorsFuzzTarget() -> FuzzTarget {
    FuzzTarget(name: "safetensors", seeds: safetensorsSeeds, mutate: safetensorsMutate, run: safetensorsRun)
}
