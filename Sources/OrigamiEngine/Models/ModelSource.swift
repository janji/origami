import CryptoKit
import Foundation

/// How a checkpoint stores its quantized matrices (MLX affine 4-bit): one group size for the dense
/// weights and the experts. The PLE table is 4-bit group 32 in every supported build; the small
/// matrices the engine keeps as floats (router, gates, block inject) may use any affine layout and are
/// dequantized at load. This is the one place that decides whether a layout can run at all; adding
/// one means a case here and kernels that read it (see INTERNALS.md, "Adding a quantization layout").
public enum QuantizationScheme: String, Codable, Sendable {
    /// MLX affine 4-bit, group 64, as carloslfu/Qwen3.8-Flash-Next-MLX-4bit. Fast path.
    case affine4Group64 = "affine-4bit-g64"
    /// MLX affine 4-bit, group 32, as mlx-community/Qwen3.8-Flash-Next-4bit.
    case affine4Group32 = "affine-4bit-g32"
    /// Dense matrices affine 4-bit group 64, experts NVFP4 as NVIDIA's ModelOpt release stores them (converted
    /// once by `origami model add`, experts kept bit for bit).
    case nvfp4Experts = "nvfp4-experts"
    /// Dense matrices affine 4-bit group 64, routed experts read from a GGUF in its own encodings (`GGUFExperts`:
    /// `origami model add` of a GGUF whose experts are under 4.5 bits per weight). `experts` and the record sizes
    /// below stay the affine ones the other expert kernels compile with; the GGUF records are `GGUFExperts`'.
    case ggufExperts = "gguf-experts"

    public var group: Int { self == .affine4Group32 ? 32 : 64 }
    /// The checkpoint's own expert record format.
    public var experts: AffineFormat { self == .nvfp4Experts ? .nvfp4 : AffineFormat(bits: 4, group: group) }
    /// The three tensors of each expert projection, in record order.
    public var expertParts: [String] { self == .nvfp4Experts ? ["weight", "scales", "global_scale"] : ["weight", "scales", "biases"] }
    /// Bytes of one packed expert record (gate, up and down projections with their scales and biases).
    public var expertRecordBytes: Int { experts.expertRecordBytes }
}

/// MLX affine quantization of a matrix: `bits` per weight, a bf16 scale and bias per `group` weights.
public struct AffineFormat: Codable, Hashable, Sendable {
    public let bits, group: Int
    /// NVFP4 (experts only): E2M1 codes, an E4M3 scale byte per 16 weights and one float32 global scale per matrix
    /// in place of the biases; the value is code × scale × global.
    public let fp4: Bool
    public init(bits: Int, group: Int, fp4: Bool = false) {
        self.bits = bits
        self.group = group
        self.fp4 = fp4
    }
    public static let nvfp4 = AffineFormat(bits: 4, group: 16, fp4: true)
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        bits = try c.decode(Int.self, forKey: .bits)
        group = try c.decode(Int.self, forKey: .group)
        fp4 = try c.decodeIfPresent(Bool.self, forKey: .fp4) ?? false
    }
    /// Affine formats encode as before (no `fp4` key), so existing store headers stay byte-identical.
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(bits, forKey: .bits)
        try c.encode(group, forKey: .group)
        if fp4 { try c.encode(fp4, forKey: .fp4) }
    }
    private enum CodingKeys: String, CodingKey { case bits, group, fp4 }
    /// Bytes of each part of an `outputs` x `width` matrix: codes, scales, then biases (NVFP4: the global scale).
    public func codeBytes(width: Int, outputs: Int) -> Int { outputs * width * bits / 8 }
    public func scaleBytes(width: Int, outputs: Int) -> Int { outputs * width / group * (fp4 ? 1 : 2) }
    public func biasBytes(width: Int, outputs: Int) -> Int { fp4 ? 4 : outputs * width / group * 2 }
    public func bytes(width: Int, outputs: Int) -> Int {
        codeBytes(width: width, outputs: outputs) + scaleBytes(width: width, outputs: outputs) + biasBytes(width: width, outputs: outputs)
    }
    /// One expert record: the gate, up and down projections back to back.
    public var expertRecordBytes: Int { ExpertProjection.allCases.reduce(0) { $0 + bytes(width: $1.width, outputs: $1.outputs) } }
}

/// A local Qwen3.8-Flash-Next checkpoint folder in MLX safetensors format. The
/// folder is only read; tensor spans come straight from the safetensors headers.
public struct ModelSource: Sendable {
    public let directory: URL
    /// Canonical tensor names (`language_model.…`) to their byte spans in the shard files.
    public let tensors: [String: TensorSpan]
    /// Shard file name to size; a shard of another size is a different (or damaged) checkpoint.
    public let shardSizes: [String: UInt64]
    /// Modification time of each shard when this source was opened: a shard rewritten while
    /// serving is refused instead of read.
    public let shardModified: [String: Int64]
    /// Local checkpoint incarnation: config, index, external-bank manifest and each weight file's resolved path,
    /// file identity and change times. Copies/moves deliberately invalidate reuse; this is not a content checksum.
    public let fingerprint: String
    public let quantization: QuantizationScheme
    /// The GGUF the routed experts are read from (`gguf-experts`), or nil.
    public let ggufExperts: GGUFExperts?
    /// Bytes of one expert record as this checkpoint stores it (the GGUF layout's slot for `gguf-experts`).
    public var expertRecordBytes: Int { ggufExperts?.recordBytes ?? quantization.expertRecordBytes }
    /// What the routed experts are read from: the GGUF holding them, else the checkpoint's folder.
    public var expertsLocation: URL { ggufExperts?.files.first.map { URL(fileURLWithPath: $0.path) } ?? directory }
    /// Expert record `key` (layer * 512 + expert) as a hot copy holds it: the record, for GGUF experts without the cache
    /// slot's padding (its layer's own bytes; a slot is the largest layer's). Nil for a key the model does not have.
    package func hotCopyRecordBytes(_ key: Int) -> Int? {
        guard (0..<(ModelContract.layers * ModelContract.experts)).contains(key) else { return nil }
        guard let gguf = ggufExperts else { return quantization.expertRecordBytes }
        return key % ModelContract.experts < gguf.experts ? gguf.layers[key / ModelContract.experts].bytes : nil
    }
    /// A hot copy of every routed expert: each record padded to 16 KiB, as a copy stores it.
    package var hotCopyWholeBytes: UInt64 {
        (0..<(ModelContract.layers * ModelContract.experts)).reduce(0) {
            $0 + UInt64(((hotCopyRecordBytes($1) ?? 0) + 16383) / 16384 * 16384)
        }
    }
    /// The checkpoint's global affine format and the modules the config gives another one (canonical names).
    public let globalFormat: AffineFormat
    public let moduleFormats: [String: AffineFormat]
    /// The zero-centred norms (`CheckpointConverter.plusOneNorms`) are stored as w, as in transformers and
    /// mlx-community's builds, instead of the 1 + w Origami computes with (carloslfu's build, Origami's conversions):
    /// `PackStore.readTensor` adds the 1 as the converter does.
    public let zeroCentredNorms: Bool
    /// The affine format of a quantized module (a tensor name without `.weight`/`.scales`/`.biases`).
    public func format(ofModule name: String) -> AffineFormat { moduleFormats[Self.canonical(name)] ?? globalFormat }
    /// Total bytes of all shards.
    public var totalBytes: UInt64 { shardSizes.values.reduce(0, +) }
    /// Human name for logs and `origami status`: the Hugging Face repo when known, else the folder name.
    public var displayName: String {
        // Downloaded by Origami: the repo id is recorded next to the files.
        if let data = try? Data(contentsOf: directory.appendingPathComponent(".origami-download.json")),
            let repo = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["repo"] as? String
        {
            return repo
        }
        let parts = directory.standardizedFileURL.pathComponents
        if let index = parts.lastIndex(where: { $0.hasPrefix("models--") }) {
            return parts[index].dropFirst("models--".count).replacingOccurrences(of: "--", with: "/")
        }
        return directory.lastPathComponent
    }

    public init(directory: URL) throws {
        let directory = directory.standardizedFileURL.resolvingSymlinksInPath()
        self.directory = directory
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            let gguf =
                directory.pathExtension.lowercased() == "gguf"
                || ((try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []).contains {
                    $0.lowercased().hasSuffix(".gguf")
                }
            throw EngineError(
                gguf
                    ? "\(directory.path) is a GGUF; `origami model add \(directory.path)` converts it once into the layout Origami runs"
                    : "No config.json in \(directory.path); is this a model folder?", code: "model_not_found")
        }
        let config = try Data(contentsOf: configURL)
        quantization = try ModelContract.validate(config: config)
        (globalFormat, moduleFormats) = ModelContract.formats(config: config)

        // Sharded checkpoints have an index; single-file ones have one model.safetensors.
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        let indexData: Data
        let shards: [String]
        if let data = try? Data(contentsOf: indexURL) {
            guard let map = ((try? JSONSerialization.jsonObject(with: data)) as? [String: Any])?["weight_map"] as? [String: String] else {
                throw EngineError("model.safetensors.index.json has no weight_map", code: "model_incompatible")
            }
            indexData = data
            shards = Set(map.values).sorted()
        } else {
            indexData = Data()
            shards = ["model.safetensors"]
        }

        var tensors: [String: TensorSpan] = [:]
        var sizes: [String: UInt64] = [:]
        var modified: [String: Int64] = [:]
        for shard in shards {
            guard !shard.contains("/"), !shard.hasPrefix(".") else {
                throw EngineError("Unsafe shard name \(shard)", code: "model_incompatible")
            }
            let url = directory.appendingPathComponent(shard)
            // stat follows symlinks: Hugging Face cache snapshots link each file to a blob.
            var info = stat()
            guard stat(url.path, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG else {
                throw EngineError("Missing model shard \(shard); the download may be incomplete", code: "model_incomplete")
            }
            sizes[shard] = UInt64(info.st_size)
            modified[shard] = Self.modificationNanoseconds(url)
            for (name, span) in try Self.header(of: url, file: shard) {
                tensors[Self.canonical(name)] = span
            }
        }
        self.tensors = tensors
        shardSizes = sizes
        shardModified = modified
        var manifest = Data()
        if quantization == .ggufExperts {
            let url = directory.appendingPathComponent(GGUFExperts.file)
            guard let data = try? Data(contentsOf: url) else {
                throw EngineError(
                    "\(GGUFExperts.file) is missing from \(directory.path); add the GGUF again with `origami model add`",
                    code: "model_incomplete")
            }
            let experts = try GGUFExperts(manifest: data)
            let root = (try? JSONSerialization.jsonObject(with: config)) as? [String: Any] ?? [:]
            guard ((root["text_config"] as? [String: Any] ?? root)["num_experts"] as? Int) == experts.experts else {
                throw EngineError(
                    "config.json's num_experts differs from the \(experts.experts) experts of \(GGUFExperts.file)",
                    code: "model_incompatible")
            }
            ggufExperts = experts
            manifest = data
        } else {
            ggufExperts = nil
        }
        try Self.requireTensors(tensors, scheme: quantization)
        zeroCentredNorms = try Self.storesZeroCentredNorms(tensors, directory: directory)

        fingerprint = Self.fingerprint(
            directory: directory, config: config, index: indexData, manifest: manifest,
            shards: shards, sizes: sizes, externalFiles: ggufExperts?.files.map { URL(fileURLWithPath: $0.path) } ?? [])
    }

    private static func fingerprint(
        directory: URL, config: Data, index: Data, manifest: Data, shards: [String],
        sizes: [String: UInt64], externalFiles: [URL]
    ) -> String {
        var hash = SHA256()
        // The manifest names the GGUF banks the experts are read from: another one is another model.
        hash.update(data: config)
        hash.update(data: index)
        hash.update(data: manifest)
        for shard in shards {
            hash.update(data: Data("\(shard):\(sizes[shard]!)\n".utf8))
            hash.update(data: LocalModelIdentity.file(directory.appendingPathComponent(shard)))
        }
        for file in externalFiles.sorted(by: { $0.path < $1.path }) { hash.update(data: LocalModelIdentity.file(file)) }
        LocalModelIdentity.updateSupport(&hash, directory: directory)
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// The fingerprint Origami 2.x gave this folder (its config, index, GGUF-experts manifest and shard sizes, before each
    /// file's local identity was bound in), only to recognise the copies it made; nil when the folder cannot be read.
    package static func earlierFingerprint(directory: URL) -> String? {
        guard let config = try? Data(contentsOf: directory.appendingPathComponent("config.json")) else { return nil }
        let index = (try? Data(contentsOf: directory.appendingPathComponent("model.safetensors.index.json"))) ?? Data()
        let map =
            index.isEmpty ? nil : ((try? JSONSerialization.jsonObject(with: index)) as? [String: Any])?["weight_map"] as? [String: String]
        guard index.isEmpty || map != nil else { return nil }
        var hash = SHA256()
        hash.update(data: config)
        hash.update(data: index)
        // 2.x hashed the manifest of a GGUF-experts model, which alone has one.
        hash.update(data: (try? Data(contentsOf: directory.appendingPathComponent(GGUFExperts.file))) ?? Data())
        for shard in map.map({ Set($0.values).sorted() }) ?? ["model.safetensors"] {
            var info = stat()
            guard !shard.contains("/"), stat(directory.appendingPathComponent(shard).path, &info) == 0 else { return nil }
            hash.update(data: Data("\(shard):\(info.st_size)\n".utf8))
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// The fingerprint `init` gives a folder, without reading tensor headers or weight payloads:
    /// for matching a pack to a model found on this Mac.
    package static func fingerprint(directory: URL) throws -> String {
        let config = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let index = (try? Data(contentsOf: directory.appendingPathComponent("model.safetensors.index.json"))) ?? Data()
        let map =
            index.isEmpty ? nil : ((try? JSONSerialization.jsonObject(with: index)) as? [String: Any])?["weight_map"] as? [String: String]
        guard index.isEmpty || map != nil else {
            throw EngineError("model.safetensors.index.json has no weight_map", code: "model_incompatible")
        }
        let shards = map.map { Set($0.values).sorted() } ?? ["model.safetensors"]
        let manifest =
            try ModelContract.validate(config: config) == .ggufExperts
            ? try Data(contentsOf: directory.appendingPathComponent(GGUFExperts.file)) : Data()
        let externalFiles = manifest.isEmpty ? [] : try GGUFExperts(manifest: manifest).files.map { URL(fileURLWithPath: $0.path) }
        var sizes: [String: UInt64] = [:]
        for shard in shards {
            guard !shard.contains("/"), !shard.hasPrefix(".") else {
                throw EngineError("Unsafe shard name \(shard)", code: "model_incompatible")
            }
            var info = stat()
            guard stat(directory.appendingPathComponent(shard).path, &info) == 0 else {
                throw EngineError("Missing model shard \(shard)", code: "model_incomplete")
            }
            sizes[shard] = UInt64(info.st_size)
        }
        return fingerprint(
            directory: directory, config: config, index: index, manifest: manifest, shards: shards,
            sizes: sizes, externalFiles: externalFiles)
    }

    /// Check fixture: explicit spans over small files, without config or geometry checks.
    package init(
        fixture directory: URL, spans: [String: Any], fingerprint: String = String(repeating: "0", count: 64),
        formats: [String: AffineFormat] = [:], zeroCentredNorms: Bool = false, gguf: GGUFExperts? = nil
    ) throws {
        let tensors = try JSONDecoder().decode([String: TensorSpan].self, from: JSONSerialization.data(withJSONObject: spans))
        var sizes: [String: UInt64] = [:]
        var modified: [String: Int64] = [:]
        for span in tensors.values {
            var info = stat()
            sizes[span.file] = stat(directory.appendingPathComponent(span.file).path, &info) == 0 ? UInt64(info.st_size) : 0
            modified[span.file] = Self.modificationNanoseconds(directory.appendingPathComponent(span.file))
        }
        self.directory = directory
        self.tensors = tensors
        shardSizes = sizes
        shardModified = modified
        self.fingerprint = fingerprint
        quantization = gguf == nil ? .affine4Group64 : .ggufExperts
        ggufExperts = gguf
        globalFormat = AffineFormat(bits: 4, group: 64)
        moduleFormats = formats
        self.zeroCentredNorms = zeroCentredNorms
    }

    /// Whether the attention q/k norms are stored as w: trained zero-centred weights sit near 0 (mean about -0.07
    /// in mlx-community's build), stored as 1 + w near 1 (about 0.93 in carloslfu's), so the mean over all of them
    /// against 0.5 tells the two apart. A checkpoint without them keeps 1 + w.
    package static func storesZeroCentredNorms(_ tensors: [String: TensorSpan], directory: URL) throws -> Bool {
        var sum: Double = 0
        var count = 0
        for (name, span) in tensors
        where name.hasPrefix("language_model.model.layers.")
            && (name.hasSuffix(".self_attn.q_norm.weight") || name.hasSuffix(".self_attn.k_norm.weight"))
        {
            guard span.dtype == "BF16", span.shape.count == 1, span.length == Int64(span.shape[0] * 2), span.length <= 65_536 else {
                throw EngineError("Norm \(name) is not a bf16 vector", code: "model_incompatible")
            }
            let handle = try FileHandle(forReadingFrom: directory.appendingPathComponent(span.file))
            defer { try? handle.close() }
            try handle.seek(toOffset: UInt64(span.offset))
            guard let bytes = try handle.read(upToCount: Int(span.length)), bytes.count == Int(span.length) else {
                throw EngineError("Could not read norm \(name)", code: "model_incomplete")
            }
            bytes.withUnsafeBytes { raw in for value in raw.bindMemory(to: UInt16.self) { sum += Double(BF16.decode(value)) } }
            count += bytes.count / 2
        }
        return count > 0 && sum / Double(count) < 0.5
    }

    static func modificationNanoseconds(_ url: URL) -> Int64? {
        var info = stat()
        guard stat(url.path, &info) == 0 else { return nil }
        // A file-derived value: an out-of-range date must not trap (as in PackStore).
        let (scaled, overflow) = Int64(info.st_mtimespec.tv_sec).multipliedReportingOverflow(by: 1_000_000_000)
        let (total, carry) = scaled.addingReportingOverflow(Int64(info.st_mtimespec.tv_nsec))
        return overflow || carry ? nil : total
    }

    /// MLX conversions name text weights `language_model.model.…`; text-only exports drop the prefix.
    /// A tensor or module name as Origami looks it up: under `language_model.` (or `vision_tower.`), and the PLE table's
    /// shards as `shard_<n>`, which mlx-community's builds name `shards.<n>`.
    package static func canonical(_ name: String) -> String {
        let prefixed = name.hasPrefix("language_model.") || name.hasPrefix("vision_tower.") ? name : "language_model." + name
        return prefixed.contains(".ngram_embedding.shards.")
            ? prefixed.replacingOccurrences(of: ".ngram_embedding.shards.", with: ".ngram_embedding.shard_") : prefixed
    }

    /// Bytes per element of the safetensors dtypes with a fixed width; a tensor of another dtype is not size-checked.
    static let dtypeBytes = [
        "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
        "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8,
    ]
    /// A model's config written as JSON again. Foundation reads a number beyond a double's range (`1e400`) as infinity
    /// and raises an uncatchable exception writing it, so such a config is refused here.
    package static func configJSON(_ config: [String: Any], options: JSONSerialization.WritingOptions = []) throws -> Data {
        guard JSONSerialization.isValidJSONObject(config) else {
            throw EngineError("config.json holds a number out of range", code: "model_incompatible")
        }
        return try JSONSerialization.data(withJSONObject: config, options: options)
    }

    /// The element count of `shape`; nil for a negative dimension or a count that overflows.
    static func elementCount(_ shape: [Int]) -> Int? {
        var count = 1
        for dimension in shape {
            let (product, overflow) = count.multipliedReportingOverflow(by: dimension)
            guard dimension >= 0, !overflow else { return nil }
            count = product
        }
        return count
    }

    /// Parses one safetensors header: 8-byte little-endian length, then a JSON table of byte ranges. The file is
    /// measured here (through a symlink, as in a Hugging Face cache); every tensor must lie inside it, and one of a
    /// fixed-width dtype must fill exactly its shape, so readers can size buffers from the shape.
    package static func header(of url: URL, file: String) throws -> [String: TensorSpan] {
        try header(of: try LocalBytes(url), file: file)
    }

    /// The same from any byte source (a Hugging Face file by HTTP range): two reads, no weights.
    package static func header(of source: any ByteSource, file: String) throws -> [String: TensorSpan] {
        let size = source.size
        guard size >= 8, case let prefix = try source.read(at: 0, count: 8), prefix.count == 8 else {
            throw EngineError("\(file) is not a safetensors file", code: "model_incompatible")
        }
        let length = prefix.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) }
        guard length > 0, length <= 64 * 1024 * 1024, length <= size - 8,
            case let json = try source.read(at: 8, count: Int(length)), json.count == Int(length),
            let table = (try? JSONSerialization.jsonObject(with: json)) as? [String: Any]
        else {
            throw EngineError("\(file) has an invalid safetensors header", code: "model_incompatible")
        }
        let base = Int64(8 + length)
        let end = Int64(size)
        var spans: [String: TensorSpan] = [:]
        for (name, value) in table where name != "__metadata__" {
            guard let entry = value as? [String: Any], let dtype = entry["dtype"] as? String,
                let shape = entry["shape"] as? [Int], let elements = elementCount(shape),
                let offsets = entry["data_offsets"] as? [Int64],
                offsets.count == 2, offsets[0] >= 0, offsets[1] >= offsets[0], offsets[1] <= end - base,
                dtypeBytes[dtype].map({ elements.multipliedReportingOverflow(by: $0) == (Int(offsets[1] - offsets[0]), false) }) ?? true
            else {
                throw EngineError("\(file): invalid tensor entry \(name)", code: "model_incompatible")
            }
            spans[name] = TensorSpan(
                file: file, dtype: dtype, offset: base + offsets[0],
                length: offsets[1] - offsets[0], shape: shape)
        }
        // No two tensors may share bytes, as the safetensors format requires.
        let ordered = spans.filter { $0.value.length > 0 }.sorted { $0.value.offset < $1.value.offset }
        for (a, b) in zip(ordered, ordered.dropFirst()) where a.value.offset + a.value.length > b.value.offset {
            throw EngineError("\(file): tensors \(a.key) and \(b.key) overlap", code: "model_incompatible")
        }
        return spans
    }

    /// The tensors the expert and PLE readers address by computed name must exist with the scheme's geometry.
    /// Everything else is checked when it is first loaded.
    static func requireTensors(_ tensors: [String: TensorSpan], scheme: QuantizationScheme) throws {
        // GGUF experts are not checkpoint tensors (`GGUFExperts` checks their banks).
        var record = scheme == .ggufExperts ? scheme.expertRecordBytes : 0
        for layer in 0..<ModelContract.layers where scheme != .ggufExperts {
            var layerBytes = 0
            for projection in ["gate", "up", "down"] {
                for part in scheme.expertParts {
                    let name = "language_model.model.layers.\(layer).mlp.switch_mlp.\(projection)_proj.\(part)"
                    guard let span = tensors[name], span.shape.first == ModelContract.experts,
                        span.length % Int64(ModelContract.experts) == 0
                    else {
                        throw EngineError("Checkpoint lacks expert bank \(name)", code: "model_incompatible")
                    }
                    layerBytes += Int(span.length) / ModelContract.experts
                }
            }
            if layer == 0 { record = layerBytes }
            guard layerBytes == record else { throw EngineError("Expert layout differs between layers", code: "model_incompatible") }
        }
        guard record == scheme.expertRecordBytes else {
            throw EngineError(
                "Expert records are \(record) bytes; \(scheme.rawValue) expects \(scheme.expertRecordBytes)", code: "model_incompatible")
        }
        for shard in 0..<ModelContract.pleShards {
            for (part, bytes) in [("weight", 80), ("scales", 10), ("biases", 10)] {
                let name = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_\(shard).\(part)"
                guard let span = tensors[name], span.shape.first == ModelContract.pleRowsPerShard,
                    span.length == Int64(ModelContract.pleRowsPerShard * bytes)
                else {
                    throw EngineError("Checkpoint lacks PLE table \(name) in the expected layout", code: "model_incompatible")
                }
            }
        }
    }
}
