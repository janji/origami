import CryptoKit
import Foundation

/// Converts an unquantized Qwen3.8-Flash-Next checkpoint (the official bf16 release, and fine-tunes saved the same
/// way in the transformers layout) or the official FP8 release (experts in E4M3 with block scales, one tensor per expert)
/// once into the MLX 4-bit layout Origami runs. The result is byte-identical to the MLX
/// conversion of the same weights (checked tensor by tensor against carloslfu/Qwen3.8-Flash-Next-MLX-4bit): MLX's affine
/// quantizer and rounding at group 64, the PLE n-gram tables at group 32; zero-centred norms stored as 1 + w; conv
/// kernels as [C, K, 1]; stacked experts split into switch_mlp gate, up and down; the vision tower copied; MTP dropped.
/// The source is only read. Output goes to `<destination>.partial` and is renamed when complete; a restart keeps the
/// shards already finished.
public struct CheckpointConverter: Sendable {
    /// One output tensor of a shard: its MLX name, safetensors dtype and shape; `offset` is set by `finish`.
    struct Tensor: Sendable {
        let name, dtype: String
        let shape: [Int]
        var offset: Int64 = 0
        var bytes: Int64 { Int64(shape.reduce(1, *)) * Int64(ModelSource.dtypeBytes[dtype] ?? 0) }
    }
    /// How a source tensor becomes output tensors (indices into its shard's `tensors`), from output row `row` on;
    /// an FP8 source has its block scales in `scale`.
    struct Work: Sendable {
        enum Kind: Sendable {
            case copy, plusOne
            /// Rows of the source quantized at `group`: codes, scales and biases.
            case quantize(group: Int)
            /// Stacked [experts, 2 × inner, hidden]: the first half of each expert's rows is gate, the second up.
            case gateUp(group: Int)
            /// Decoded and written as bf16 (a GGUF source's float tensors).
            case floats
            /// These bytes as they are (metadata a GGUF keeps outside its tensors).
            case literal(Data)
        }
        let source: TensorSpan, kind: Kind, outputs: [Int]
        var row = 0, scale: TensorSpan? = nil, block: [Int] = []
        /// An MLX affine source's biases (its scales are `scale`).
        var bias: TensorSpan? = nil
        /// Output row r (column c) takes source row (column) `rowOrder[r]` (`columnOrder[c]`): llama.cpp's head reorders, undone.
        var rowOrder: [Int]? = nil, columnOrder: [Int]? = nil
        /// x becomes log(-x): llama.cpp stores the GDN decay A_log as -exp(A_log).
        var negativeLog = false
        /// x becomes 1 + x (a zero-centred norm of an F16 or F32 source, written through `.floats`).
        var plusOne = false
        /// `.copy` and `.literal` write at this byte offset of their output (an NVFP4 expert's slice of the stacked bank, a
        /// pruned router's zero rows).
        var byteOffset: Int64 = 0
    }
    /// One output safetensors file: its tensors in write order, the works that fill them, and its header.
    struct Shard: Sendable {
        let file: String
        var tensors: [Tensor] = [], works: [Work] = []
        var header = Data()
        var bytes: Int64 { Int64(header.count) + tensors.reduce(0) { $0 + $1.bytes } }
    }

    public let source: URL
    public let sourceBytes, outputBytes: UInt64
    let shards: [Shard]
    let config: Data
    /// Where the tokenizer, template and other small files come from (the source itself, or for a GGUF the support
    /// folder), and a chat template that overrides theirs (a GGUF's own).
    let extras: URL, template: String?
    /// Hash of the source's index, config and each file's size and modification time: a finished conversion records it
    /// in `origami-model.json`.
    public let identity: String
    /// "unquantized", "FP8" or "NVFP4", and the format chain the manifest records.
    public let sourceKind, chain: String
    /// A GGUF whose routed experts stay in it: the `GGUFExperts` manifest written beside the converted weights.
    public internal(set) var expertManifest: Data?

    /// Reads the source's config and safetensors headers. Throws when it is not a convertible Qwen3.8-Flash-Next
    /// checkpoint; the converted layout passes `ModelContract.validate` before a byte is written.
    public init(source: URL) throws { try self.init(source: source, checkContract: true) }

    /// `checkContract: false` lets the checks convert small synthetic checkpoints that cannot pass the contract.
    package init(source: URL, checkContract: Bool) throws {
        func refuse(_ why: String) -> EngineError {
            EngineError("Origami cannot convert this checkpoint: \(why)", code: "model_incompatible")
        }
        self.source = source
        extras = source
        template = nil
        let configData = try Data(contentsOf: source.appendingPathComponent("config.json"))
        guard var config = (try? JSONSerialization.jsonObject(with: configData)) as? [String: Any] else {
            throw refuse("config.json is not JSON")
        }
        var block: [Int]?
        var nvfp4 = false
        // An MLX checkpoint at another width than Origami's (8- or 2-bit, or 4-bit at another group size): its codes are
        // decoded (q × scale + bias per group) and quantized again to affine 4-bit; per-module widths in its config apply.
        var affine: (bits: Int, group: Int, modules: [String: Any])?
        if let quantization = (config["quantization_config"] ?? config["quantization"]) as? [String: Any] {
            if quantization["quant_method"] == nil, (quantization["mode"] as? String ?? "affine") == "affine",
                let bits = quantization["bits"] as? Int, let group = quantization["group_size"] as? Int
            {
                guard [2, 4, 8].contains(bits), group > 0 else {
                    throw refuse(
                        "its MLX \(bits)-bit codes are packed in a layout Origami does not read yet; 8-, 4- and 2-bit MLX checkpoints are converted"
                    )
                }
                affine = (bits, group, quantization)
            } else if quantization["quant_method"] as? String == "modelopt" {
                // NVIDIA's ModelOpt release: NVFP4 experts (kept), FP8 PLE table with one scale (decoded), MTP dropped.
                try Self.checkModelOpt(quantization)
                nvfp4 = true
            } else {
                // FP8 with block scales (the official FP8 release) is decoded; other quantized checkpoints run as they are or not at all.
                guard quantization["quant_method"] as? String == "fp8", let size = quantization["weight_block_size"] as? [Int],
                    size.count == 2,
                    size.allSatisfy({ $0 > 0 })
                else {
                    throw refuse(
                        "it is already quantized (\(quantization["quant_method"] ?? quantization["mode"] ?? "\(quantization["bits"] ?? "?")-bit")); unquantized, FP8 and NVIDIA NVFP4 checkpoints are converted"
                    )
                }
                block = size
            }
        }
        sourceKind = affine.map { "MLX \($0.bits)-bit (group \($0.group))" } ?? (nvfp4 ? "NVFP4" : block == nil ? "unquantized" : "FP8")
        let floatChain =
            nvfp4
            ? "nvfp4 experts kept (E2M1, E4M3 scale per 16, float32 global scale); bf16 -> affine 4-bit group 64; "
                + "PLE n-gram tables fp8 e4m3 (one scale) -> affine 4-bit group 32, MLX rounding"
            : (block.map { "fp8 e4m3 (blocks \($0[0])x\($0[1])) -> " } ?? "unquantized -> ")
                + "affine 4-bit group 64 (PLE n-gram tables group 32), MLX rounding"
        chain =
            affine.map { "affine \($0.bits)-bit group \($0.group) -> affine 4-bit group 64 (PLE n-gram tables group 32), MLX rounding" }
            ?? floatChain
        let indexData = try? Data(contentsOf: source.appendingPathComponent("model.safetensors.index.json"))
        let weightMap =
            indexData.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }?["weight_map"] as? [String: String]
        let files = weightMap.map { Array(Set($0.values)).sorted() } ?? ["model.safetensors"]
        // The index is untrusted: a shard name is read here and written under the same name in the destination.
        guard files.allSatisfy({ !$0.isEmpty && !$0.contains("/") && !$0.hasPrefix(".") }) else {
            throw refuse("its index names a shard outside the folder")
        }
        // Sizes and modification times too: a checkpoint saved again in place converts again, and a restart never
        // mixes the shards of two versions.
        var hash = SHA256()
        hash.update(data: indexData ?? Data())
        hash.update(data: configData)
        var shards: [Shard] = []
        var total: UInt64 = 0
        var overrides: [String: Any] = [:]
        var headers: [String: [String: TensorSpan]] = [:]
        for file in files {
            let url = source.appendingPathComponent(file)
            // stat follows symlinks: Hugging Face cache snapshots link each file to a blob.
            var info = stat()
            guard stat(url.path, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG else { throw refuse("\(file) is missing") }
            total += UInt64(info.st_size)
            hash.update(data: Data("\(file):\(info.st_size):\(ModelSource.modificationNanoseconds(url) ?? 0)\n".utf8))
            headers[file] = try ModelSource.header(of: url, file: file)
        }
        identity = hash.finalize().map { String(format: "%02x", $0) }.joined()
        // NVFP4 expert banks may span two files: planned over all headers, written into the shard of expert 0's file.
        let banks = nvfp4 ? try Self.nvfp4Banks(headers) : [:]
        if nvfp4, checkContract, banks.count != ModelContract.layers * 3 {
            throw refuse("it has \(banks.count) NVFP4 expert banks; Qwen3.8-Flash-Next has \(ModelContract.layers * 3)")
        }
        let consumed = Set(banks.values.flatMap { $0.names })
        for file in files {
            var shard = Shard(file: file)
            var header = headers[file]!.filter { !consumed.contains($0.key) }
            for (base, bank) in banks.sorted(by: { $0.key < $1.key }) where bank.file == file {
                try Self.planNVFP4(base, bank, into: &shard)
                overrides[String(base.dropFirst("language_model.".count))] = ["group_size": 16, "bits": 4, "mode": "nvfp4"]
            }
            try Self.planExperts(&header, into: &shard, block: block)
            if let affine { try Self.planAffine(&header, affine, into: &shard, overrides: &overrides) }
            for (name, span) in header.sorted(by: { $0.key < $1.key }) {
                if nvfp4, name.hasSuffix(".ngram_embedding.weight_scale") { continue }  // read with each FP8 shard
                let scale =
                    nvfp4 && span.dtype == "F8_E4M3" && name.contains(".ngram_embedding.shard_")
                    ? header[String(name[..<name.range(of: ".shard_")!.lowerBound]) + ".weight_scale"] : header[name + "_scale_inv"]
                try Self.plan(
                    name, span, into: &shard, overrides: &overrides, block: nvfp4 && scale != nil ? span.shape : block, scale: scale)
            }
            guard !shard.tensors.isEmpty else { continue }
            try Self.finish(&shard)
            shards.append(shard)
        }
        var quantization: [String: Any] = ["group_size": 64, "bits": 4]
        quantization.merge(overrides) { $1 }
        config["quantization"] = quantization
        config["quantization_config"] = quantization
        self.config = try ModelSource.configJSON(config, options: [.prettyPrinted, .sortedKeys])
        if checkContract { _ = try ModelContract.validate(config: self.config) }
        self.shards = shards
        sourceBytes = total
        outputBytes = UInt64(shards.reduce(Int64(0)) { $0 + $1.bytes })
    }

    /// Places a shard's tensors and writes its safetensors header.
    static func finish(_ shard: inout Shard) throws {
        var offset: Int64 = 0
        for i in shard.tensors.indices {
            shard.tensors[i].offset = offset
            offset += shard.tensors[i].bytes
        }
        var table: [String: Any] = ["__metadata__": ["format": "mlx"]]
        for t in shard.tensors { table[t.name] = ["dtype": t.dtype, "shape": t.shape, "data_offsets": [t.offset, t.offset + t.bytes]] }
        var json = try JSONSerialization.data(withJSONObject: table, options: [.sortedKeys])
        json.append(contentsOf: [UInt8](repeating: 0x20, count: (8 - json.count % 8) % 8))
        var length = UInt64(json.count).littleEndian
        shard.header = Data(bytes: &length, count: 8) + json
    }

    /// The MLX name of a transformers tensor, nil for one Origami drops (MTP).
    static func mlxName(_ name: String) -> String? {
        if name.hasPrefix("mtp.") { return nil }
        if name == "lm_head.weight" { return "language_model.lm_head.weight" }
        if name.hasPrefix("model.language_model.") { return "language_model.model." + name.dropFirst("model.language_model.".count) }
        if name.hasPrefix("model.visual.") { return "vision_tower." + name.dropFirst("model.visual.".count) }
        return name
    }
    static let plusOneNorms = [
        ".q_norm.weight", ".k_norm.weight", ".q_layernorm.weight", ".k_layernorm.weight", ".hc_norm.weight",
        ".norm_conv.weight", ".norm_key.weight", ".norm_query.weight",
    ]

    static func add(_ name: String, _ dtype: String, _ shape: [Int], to shard: inout Shard) -> Int {
        shard.tensors.append(Tensor(name: name, dtype: dtype, shape: shape))
        return shard.tensors.count - 1
    }
    static func quantized(_ base: String, _ shape: [Int], _ group: Int, of name: String, to shard: inout Shard) throws -> [Int] {
        // No zero dimension: `run` sizes its blocks by the row width.
        guard let columns = shape.last, shape.allSatisfy({ $0 > 0 }), columns % group == 0, columns % 8 == 0 else {
            throw EngineError(
                "Cannot quantize \(name): its shape \(shape) does not split into rows of groups of \(group)", code: "model_incompatible")
        }
        let rows = Array(shape.dropLast())
        return [
            add(base + ".weight", "U32", rows + [columns / 8], to: &shard),
            add(base + ".scales", "BF16", rows + [columns / group], to: &shard),
            add(base + ".biases", "BF16", rows + [columns / group], to: &shard),
        ]
    }
    /// The block scales an FP8 tensor needs, checked against its shape.
    private static func fp8Scale(_ name: String, _ span: TensorSpan, _ scale: TensorSpan?, _ block: [Int]?) throws -> TensorSpan {
        // One block over the whole tensor: a single scale, stored as a scalar by ModelOpt.
        guard let block, let scale, span.shape.count == 2, ["BF16", "F32"].contains(scale.dtype), scale.file == span.file,
            scale.shape == [(span.shape[0] + block[0] - 1) / block[0], (span.shape[1] + block[1] - 1) / block[1]]
                || (block == span.shape && scale.shape.reduce(1, *) == 1)
        else {
            throw EngineError(
                "Origami cannot convert \(name): its FP8 values lack matching block scales in the same file", code: "model_incompatible")
        }
        return scale
    }

    /// ModelOpt's per-module algorithms: NVFP4 group 16 on the routed experts, FP8 on the PLE table, anything on MTP.
    static func checkModelOpt(_ quantization: [String: Any]) throws {
        guard let layers = quantization["quantized_layers"] as? [String: Any], !layers.isEmpty else {
            throw EngineError(
                "Origami cannot convert this checkpoint: its ModelOpt config names no quantized layers", code: "model_incompatible")
        }
        for (name, value) in layers {
            let entry = value as? [String: Any] ?? [:]
            let algo = entry["quant_algo"] as? String ?? "?"
            let fine =
                name.hasPrefix("mtp.")
                || (name.hasSuffix(".mlp.experts") && algo == "NVFP4" && entry["group_size"] as? Int == 16)
                || (name.hasSuffix(".ple.ple_embedding.ngram_embedding") && algo == "FP8")
            guard fine else {
                throw EngineError(
                    "Origami cannot convert this checkpoint: \(name) is \(algo); NVFP4 experts and an FP8 PLE table are converted",
                    code: "model_incompatible")
            }
        }
    }

    /// An NVFP4 expert bank: per expert its codes (U8 [outputs, width / 2]), E4M3 scales ([outputs, width / 16]) and
    /// float32 global scale; `file` holds expert 0's codes.
    struct NVFP4Bank { var file: String, codes: [TensorSpan], scales: [TensorSpan], globals: [TensorSpan], names: [String] }

    /// The banks of ModelOpt's per-expert tensors (`…mlp.experts.<e>.<projection>.weight`, `.weight_scale`,
    /// `.weight_scale_2`; `.input_scale` is for activation quantization, which Origami does not do), keyed by output base.
    static func nvfp4Banks(_ headers: [String: [String: TensorSpan]]) throws -> [String: NVFP4Bank] {
        let all = headers.values.reduce(into: [String: TensorSpan]()) { $0.merge($1) { a, _ in a } }
        var experts: [String: [Int: String]] = [:]
        for name in all.keys where !name.hasPrefix("mtp.") && name.hasSuffix(".weight") {
            guard let range = name.range(of: ".mlp.experts.") else { continue }
            let parts = name[range.upperBound...].split(separator: ".")
            guard parts.count == 3, let expert = Int(parts[0]), ["gate_proj", "up_proj", "down_proj"].contains(String(parts[1])) else {
                continue
            }
            experts[String(name[..<range.upperBound]) + String(parts[1]), default: [:]][expert] = String(name.dropLast(".weight".count))
        }
        var banks: [String: NVFP4Bank] = [:]
        for (key, stems) in experts {
            let projection = String(key.split(separator: ".").last!)
            let prefix = String(key.dropLast(projection.count))
            guard let out = mlxName(prefix + projection) else { continue }
            let base = String(out.dropLast(("experts." + projection).count)) + "switch_mlp." + projection
            var bank = NVFP4Bank(file: "", codes: [], scales: [], globals: [], names: [])
            for expert in 0..<stems.count {
                guard let stem = stems[expert], let codes = all[stem + ".weight"], let scales = all[stem + ".weight_scale"],
                    let global = all[stem + ".weight_scale_2"], codes.dtype == "U8", codes.shape.count == 2,
                    scales.dtype == "F8_E4M3", scales.shape == [codes.shape[0], codes.shape[1] * 2 / 16],
                    global.dtype == "F32", global.shape.reduce(1, *) == 1,
                    codes.shape == bank.codes.first?.shape ?? codes.shape
                else {
                    throw EngineError(
                        "Origami cannot convert \(key): expert \(expert) lacks NVFP4 codes, E4M3 scales and a global scale of one shape",
                        code: "model_incompatible")
                }
                if expert == 0 { bank.file = codes.file }
                bank.codes.append(codes)
                bank.scales.append(scales)
                bank.globals.append(global)
                bank.names += [".weight", ".weight_scale", ".weight_scale_2", ".input_scale"].map { stem + $0 }
            }
            banks[base] = bank
        }
        return banks
    }

    /// Stacks a bank into MLX's NVFP4 tensors (`weight` U32 [experts, outputs, width / 8], `scales` U8, plus
    /// `global_scale` F32 [experts]): the bytes are copied, little-endian nibbles already in MLX's order.
    static func planNVFP4(_ base: String, _ bank: NVFP4Bank, into shard: inout Shard) throws {
        let experts = bank.codes.count
        let outputs = bank.codes[0].shape[0]
        let width = bank.codes[0].shape[1] * 2
        let codes = add(base + ".weight", "U32", [experts, outputs, width / 8], to: &shard)
        let scales = add(base + ".scales", "U8", [experts, outputs, width / 16], to: &shard)
        let globals = add(base + ".global_scale", "F32", [experts], to: &shard)
        for e in 0..<experts {
            shard.works.append(Work(source: bank.codes[e], kind: .copy, outputs: [codes], byteOffset: Int64(e * outputs * width / 2)))
            shard.works.append(Work(source: bank.scales[e], kind: .copy, outputs: [scales], byteOffset: Int64(e * outputs * width / 16)))
            shard.works.append(Work(source: bank.globals[e], kind: .copy, outputs: [globals], byteOffset: Int64(e * 4)))
        }
    }

    /// One tensor per expert (`…mlp.experts.<e>.gate_proj.weight`, the FP8 release's layout) stacks into `switch_mlp`:
    /// a work per expert, written at its rows. Every expert of a projection must sit in one file.
    private static func planExperts(_ header: inout [String: TensorSpan], into shard: inout Shard, block: [Int]?) throws {
        var groups: [String: [Int: String]] = [:]
        for name in header.keys {
            guard name.hasSuffix(".weight"), let range = name.range(of: ".mlp.experts.") else { continue }
            let parts = name[range.upperBound...].split(separator: ".")
            guard parts.count == 3, let expert = Int(parts[0]), ["gate_proj", "up_proj", "down_proj"].contains(String(parts[1])) else {
                continue
            }
            groups[String(name[..<range.upperBound]) + String(parts[1]), default: [:]][expert] = name
        }
        for (key, experts) in groups.sorted(by: { $0.key < $1.key }) {
            let projection = String(key.split(separator: ".").last!)
            let prefix = String(key.dropLast(projection.count))
            guard let out = mlxName(prefix + projection) else { continue }
            let names = (0..<experts.count).compactMap { experts[$0] }
            guard names.count == experts.count, let shape = header[names[0]]?.shape, shape.count == 2,
                names.allSatisfy({ header[$0]?.shape == shape })
            else {
                throw EngineError(
                    "Origami cannot convert \(key): its experts are not numbered 0 to \(experts.count - 1) with one shape, or some sit in another file",
                    code: "model_incompatible")
            }
            let base = String(out.dropLast(("experts." + projection).count)) + "switch_mlp." + projection
            let outputs = try quantized(base, [names.count] + shape, 64, of: key, to: &shard)
            for (expert, name) in names.enumerated() {
                let span = header.removeValue(forKey: name)!
                let fp8 = span.dtype == "F8_E4M3"
                guard fp8 || ["BF16", "F16", "F32"].contains(span.dtype) else {
                    throw EngineError(
                        "Origami cannot convert \(name): its \(span.dtype) values are not converted yet", code: "model_incompatible")
                }
                let scale = fp8 ? try fp8Scale(name, span, header.removeValue(forKey: name + "_scale_inv"), block) : nil
                shard.works.append(
                    Work(
                        source: span, kind: .quantize(group: 64), outputs: outputs, row: expert * shape[0], scale: scale, block: block ?? []
                    ))
            }
        }
    }

    /// An MLX affine source's packed weights (`U32` codes with `.scales` and `.biases`), taken out of `header`: each as
    /// a source of `bits` and `group` (its module's own in the config, else the checkpoint's), quantized again to
    /// affine 4-bit group 64 (32 for the PLE n-gram tables), or written as bf16 where Origami keeps the module in floats.
    private static func planAffine(
        _ header: inout [String: TensorSpan], _ affine: (bits: Int, group: Int, modules: [String: Any]),
        into shard: inout Shard, overrides: inout [String: Any]
    ) throws {
        for name in header.keys.sorted() where name.hasSuffix(".weight") && header[name]?.dtype == "U32" {
            let base = String(name.dropLast(".weight".count))
            guard let span = header[name], let scales = header[base + ".scales"], let biases = header[base + ".biases"] else { continue }
            let module = (affine.modules[base] ?? affine.modules[String(base.drop { $0 != "." }.dropFirst())]) as? [String: Any]
            let bits = module?["bits"] as? Int ?? affine.bits
            let group = module?["group_size"] as? Int ?? affine.group
            guard [2, 4, 8].contains(bits), let packed = span.shape.last, packed * 32 % bits == 0 else {
                throw EngineError(
                    "Origami cannot convert \(name): its \(bits)-bit codes are packed in a layout it does not read yet",
                    code: "model_incompatible")
            }
            let shape = span.shape.dropLast() + [packed * 32 / bits]
            let source = TensorSpan(
                file: span.file, dtype: "AFFINE:\(bits):\(group)", offset: span.offset, length: span.length, shape: Array(shape))
            for key in [name, base + ".scales", base + ".biases"] { header.removeValue(forKey: key) }
            guard let out = mlxName(name) else { continue }
            if out.hasPrefix("vision_tower.") || out.contains(".conv1d.")
                || ModelContract.floatModules.contains(where: { out.contains($0 + ".") })
            {
                shard.works.append(
                    Work(source: source, kind: .floats, outputs: [add(out, "BF16", Array(shape), to: &shard)], scale: scales, bias: biases))
                continue
            }
            let target = String(out.dropLast(".weight".count))
            let outGroup = out.contains(".ple.ple_embedding.ngram_embedding.") ? 32 : 64
            if outGroup != 64 { overrides[String(target.dropFirst("language_model.".count))] = ["group_size": outGroup, "bits": 4] }
            shard.works.append(
                Work(
                    source: source, kind: .quantize(group: outGroup),
                    outputs: try quantized(target, Array(shape), outGroup, of: name, to: &shard),
                    scale: scales, bias: biases))
        }
    }

    private static func plan(
        _ name: String, _ span: TensorSpan, into shard: inout Shard, overrides: inout [String: Any], block: [Int]?,
        scale: TensorSpan?
    ) throws {
        guard let out = mlxName(name) else { return }
        if name.hasSuffix("_scale_inv"), block != nil { return }  // read with its FP8 tensor
        func add(_ name: String, _ dtype: String, _ shape: [Int]) -> Int { Self.add(name, dtype, shape, to: &shard) }
        func quantized(_ base: String, _ shape: [Int], _ group: Int) throws -> [Int] {
            try Self.quantized(base, shape, group, of: name, to: &shard)
        }
        let fp8 = span.dtype == "F8_E4M3"
        guard ["BF16", "F16", "F32", "I64", "I32"].contains(span.dtype) || fp8 else {
            throw EngineError("Origami cannot convert \(name): its \(span.dtype) values are not converted yet", code: "model_incompatible")
        }
        let floatSource = ["BF16", "F16", "F32"].contains(span.dtype)
        if out.hasSuffix(".mlp.experts.gate_up_proj"), span.shape.count == 3, span.shape[1] % 2 == 0, floatSource {
            let base = String(out.dropLast("experts.gate_up_proj".count)) + "switch_mlp."
            let half = [span.shape[0], span.shape[1] / 2, span.shape[2]]
            shard.works.append(
                Work(
                    source: span, kind: .gateUp(group: 64),
                    outputs: try quantized(base + "gate_proj", half, 64) + quantized(base + "up_proj", half, 64)))
        } else if out.hasSuffix(".mlp.experts.down_proj"), span.shape.count == 3, floatSource {
            let base = String(out.dropLast("experts.down_proj".count)) + "switch_mlp.down_proj"
            shard.works.append(Work(source: span, kind: .quantize(group: 64), outputs: try quantized(base, span.shape, 64)))
        } else if out.hasSuffix(".weight"), span.shape.count >= 2, floatSource || fp8, !out.hasPrefix("vision_tower."),
            !out.contains(".conv1d."),
            !ModelContract.floatModules.contains(where: { out.contains($0 + ".") })
        {
            let base = String(out.dropLast(".weight".count))
            let group = out.contains(".ple.ple_embedding.ngram_embedding.") ? 32 : 64
            if group != 64 { overrides[String(base.dropFirst("language_model.".count))] = ["group_size": group, "bits": 4] }
            shard.works.append(
                Work(
                    source: span, kind: .quantize(group: group), outputs: try quantized(base, span.shape, group),
                    scale: fp8 ? try fp8Scale(name, span, scale, block) : nil, block: block ?? []))
        } else if fp8 {
            throw EngineError(
                "Origami cannot convert \(name): it keeps this tensor in floats and does not decode FP8 into them yet",
                code: "model_incompatible")
        } else {
            // [C, 1, K] and [C, K, 1] share one memory order: only the shape changes.
            let shape =
                out.contains(".conv1d.weight") && span.shape.count == 3 && span.shape[1] == 1
                ? [span.shape[0], span.shape[2], 1] : span.shape
            let plusOne = !out.hasPrefix("vision_tower.") && plusOneNorms.contains(where: { out.hasSuffix($0) })
            if floatSource && span.dtype != "BF16" {
                // F16 and F32 floats are written as bf16, the format the engine reads, 1 + w taken before rounding.
                shard.works.append(Work(source: span, kind: .floats, outputs: [add(out, "BF16", shape)], plusOne: plusOne))
            } else {
                shard.works.append(
                    Work(source: span, kind: plusOne && span.dtype == "BF16" ? .plusOne : .copy, outputs: [add(out, span.dtype, shape)]))
            }
        }
    }

    /// Writes the converted checkpoint to `destination`, reporting bytes of output written.
    public func convert(to destination: URL, progress: ((UInt64, UInt64) -> Void)? = nil) throws {
        let fm = FileManager.default
        // Only an earlier conversion is replaced, never a folder of another origin that has the same name.
        if fm.fileExists(atPath: destination.path), !fm.fileExists(atPath: destination.appendingPathComponent("origami-model.json").path) {
            throw EngineError(
                "\(destination.path) already exists and is not an Origami conversion; move it or choose another --name",
                code: "destination_exists")
        }
        let partial = destination.deletingLastPathComponent().appendingPathComponent(
            destination.lastPathComponent + ".partial", isDirectory: true)
        try fm.createDirectory(at: partial, withIntermediateDirectories: true)
        let progressFile = partial.appendingPathComponent("conversion-progress.json")
        var done = Set<String>()
        if let data = try? Data(contentsOf: progressFile), let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            object["source"] as? String == identity
        {
            done = Set(object["done"] as? [String] ?? [])
        }
        var written = UInt64(shards.filter { done.contains($0.file) }.reduce(Int64(0)) { $0 + $1.bytes })
        for shard in shards where !done.contains(shard.file) {
            // Open per output shard (an NVFP4 bank spans at most two source files), so descriptors never pile up.
            var sources: [String: ExactFile] = [:]
            let url = partial.appendingPathComponent(shard.file)
            fm.createFile(atPath: url.path, contents: nil)
            let output = try FileHandle(forWritingTo: url)
            defer { try? output.close() }
            try output.truncate(atOffset: 0)
            try output.write(contentsOf: shard.header)
            let base = Int64(shard.header.count)
            written += UInt64(base)
            func write(_ index: Int, row: Int64, _ data: UnsafeRawBufferPointer) throws {
                let t = shard.tensors[index]
                // A work fills only its own tensor: a source that disagrees with its plan fails, never overwrites a neighbour.
                guard row >= 0, row + Int64(data.count) <= t.bytes else {
                    throw EngineError(
                        "Origami cannot convert this checkpoint: \(t.name) overflows its planned size", code: "model_incompatible")
                }
                try output.seek(toOffset: UInt64(base + t.offset + row))
                try output.write(
                    contentsOf: Data(
                        bytesNoCopy: UnsafeMutableRawPointer(mutating: data.baseAddress!), count: data.count, deallocator: .none))
            }
            for work in shard.works {
                if case .literal(let data) = work.kind {
                    try data.withUnsafeBytes { try write(work.outputs[0], row: work.byteOffset, $0) }
                    continue
                }
                let file =
                    try sources[work.source.file] ?? ExactFile(url: source.appendingPathComponent(work.source.file), bypassCache: true)
                sources[work.source.file] = file
                try Self.run(work, from: file, write: write)
                written += UInt64(work.outputs.reduce(Int64(0)) { $0 + shard.tensors[$1].bytes })
                progress?(written, outputBytes)
            }
            try output.synchronize()
            done.insert(shard.file)
            try JSONSerialization.data(withJSONObject: ["source": identity, "done": done.sorted()]).write(
                to: progressFile, options: .atomic)
        }
        // Everything but the weights: the config with its quantization block, the index, tokenizer, template, other configs
        // and the licence; not other weight formats or partial downloads, which the free-space check does not count.
        try config.write(to: partial.appendingPathComponent("config.json"))
        var weightMap: [String: String] = [:]
        for shard in shards { for t in shard.tensors { weightMap[t.name] = shard.file } }
        try JSONSerialization.data(
            withJSONObject: ["metadata": ["total_size": outputBytes], "weight_map": weightMap], options: [.prettyPrinted, .sortedKeys]
        )
        .write(to: partial.appendingPathComponent("model.safetensors.index.json"))
        let kept = [".json", ".jinja", ".txt", ".model", ".tiktoken"]
        for name in try fm.contentsOfDirectory(atPath: extras.path)
        where !name.hasPrefix(".") && !["config.json", "model.safetensors.index.json"].contains(name)
            && (kept.contains { name.hasSuffix($0) } || ["LICENSE", "README", "NOTICE"].contains { name.uppercased().hasPrefix($0) })
            && !name.hasSuffix(".part")
        {
            var isDirectory: ObjCBool = false
            guard fm.fileExists(atPath: extras.appendingPathComponent(name).path, isDirectory: &isDirectory), !isDirectory.boolValue else {
                continue
            }
            let target = partial.appendingPathComponent(name)
            try? fm.removeItem(at: target)
            // The file itself, not a (cache-relative) link that would break in the new folder.
            try fm.copyItem(at: extras.appendingPathComponent(name).resolvingSymlinksInPath(), to: target)
        }
        if let template { try Data(template.utf8).write(to: partial.appendingPathComponent("chat_template.jinja")) }
        try expertManifest?.write(to: partial.appendingPathComponent(GGUFExperts.file))
        // The folder name only: the manifest travels with the model, the full path stays in the private models.json.
        let manifest: [String: Any] = [
            "converted_from": source.lastPathComponent, "source_identity": identity,
            "converted": ISO8601DateFormatter().string(from: Date()),
            "chain": chain,
        ]
        try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys]).write(
            to: partial.appendingPathComponent("origami-model.json"))
        try? fm.removeItem(at: progressFile)
        if fm.fileExists(atPath: destination.path) { try fm.removeItem(at: destination) }
        try fm.moveItem(at: partial, to: destination)
    }

    /// E4M3 (FP8 without infinities, NaN at S.1111.111) as floats.
    package static let e4m3: [Float] = (0..<256).map { byte in
        let sign: Float = byte & 0x80 == 0 ? 1 : -1
        let exponent = (byte >> 3) & 0xF
        let mantissa = Float(byte & 7)
        if exponent == 0xF && byte & 7 == 7 { return .nan }
        return sign
            * (exponent == 0 ? mantissa / 8 * 0x1p-6 : (1 + mantissa / 8) * Float(sign: .plus, exponent: Int(exponent) - 7, significand: 1))
    }
    /// FP8 rows `rows` of a [_, columns] tensor times their block scales (`scales`, row-major over blocks of `block`).
    package static func decodeFP8(
        _ source: UnsafeRawPointer, rows: Range<Int>, columns: Int, scales: [Float], block: [Int], into values: inout [Float]
    ) {
        let blocksPerRow = (columns + block[1] - 1) / block[1]
        for (local, row) in rows.enumerated() {
            let scaleRow = row / block[0] * blocksPerRow
            for column in 0..<columns {
                values[local * columns + column] =
                    e4m3[Int(source.load(fromByteOffset: local * columns + column, as: UInt8.self))]
                    * scales[scaleRow + column / block[1]]
            }
        }
    }

    /// MLX affine rows `rows` of a [_, columns] tensor: each code of `bits` (packed least-significant bit first in 32-bit
    /// words, as MLX packs 2-, 4- and 8-bit codes) times its group's scale plus its bias.
    package static func decodeAffine(
        _ source: UnsafeRawPointer, rows: Range<Int>, columns: Int, bits: Int, group: Int, scales: [Float], biases: [Float],
        into values: inout [Float]
    ) {
        let rowBytes = columns * bits / 8
        let perByte = 8 / bits
        let mask = (1 << bits) - 1
        let groups = columns / group
        for (local, row) in rows.enumerated() {
            let base = source + local * rowBytes
            for c in 0..<columns {
                let q = (Int(base.load(fromByteOffset: c / perByte, as: UInt8.self)) >> ((c % perByte) * bits)) & mask
                let g = row * groups + c / group
                values[local * columns + c] = scales[g] * Float(q) + biases[g]
            }
        }
    }

    private static func run(_ work: Work, from file: ExactFile, write: (Int, Int64, UnsafeRawBufferPointer) throws -> Void) throws {
        let span = work.source
        let element = span.dtype == "F32" ? 4 : ["F8_E4M3", "U8"].contains(span.dtype) ? 1 : 2
        // A GGUF tensor's rows are whole blocks of its encoding, decoded to floats before anything else; an MLX affine
        // source's rows are its packed codes.
        let gguf = span.dtype.hasPrefix("GGUF:") ? Int(span.dtype.dropFirst(5)) : nil
        let affine = span.dtype.hasPrefix("AFFINE:") ? span.dtype.split(separator: ":").dropFirst().compactMap { Int($0) } : nil
        func rowBytes(_ columns: Int) -> Int {
            if let affine { return columns * affine[0] / 8 }
            return gguf.flatMap { GGUFQuant.sizes[$0] }.map { columns / $0.block * $0.bytes } ?? columns * element
        }
        func floats(_ tensor: TensorSpan) throws -> [Float] {
            var raw = [UInt8](repeating: 0, count: Int(tensor.length))
            try raw.withUnsafeMutableBytes { try file.read(offset: tensor.offset, into: $0) }
            return raw.withUnsafeBytes { bytes in
                tensor.dtype == "F32"
                    ? Array(bytes.bindMemory(to: Float.self))
                    : tensor.dtype == "F16"
                        ? bytes.bindMemory(to: Float16.self).map(Float.init)
                        : bytes.bindMemory(to: UInt16.self).map(BF16.decode)
            }
        }
        // An MLX affine source's scales and biases, read once.
        let affineScales = try affine == nil ? nil : work.scale.map(floats)
        let affineBiases = try affine == nil ? nil : work.bias.map(floats)
        // FP8: the block scales, read once (they are small), as floats.
        let blockScales =
            try affine != nil
            ? nil
            : work.scale.map { scale -> [Float] in
                var raw = [UInt8](repeating: 0, count: Int(scale.length))
                try raw.withUnsafeMutableBytes { try file.read(offset: scale.offset, into: $0) }
                return raw.withUnsafeBytes { bytes in
                    scale.dtype == "F32" ? Array(bytes.bindMemory(to: Float.self)) : bytes.bindMemory(to: UInt16.self).map(BF16.decode)
                }
            }
        /// Rows `first..<first + count` as floats, or nil when the quantizer reads the source's own floats.
        func decoded(_ raw: UnsafeRawPointer, first: Int, count: Int, columns: Int, always: Bool = false) -> [Float]? {
            if let affine, let affineScales, let affineBiases {
                var values = [Float](repeating: 0, count: count * columns)
                decodeAffine(
                    raw, rows: first..<(first + count), columns: columns, bits: affine[0], group: affine[1], scales: affineScales,
                    biases: affineBiases,
                    into: &values)
                return values
            }
            if let blockScales {
                var values = [Float](repeating: 0, count: count * columns)
                decodeFP8(raw, rows: first..<(first + count), columns: columns, scales: blockScales, block: work.block, into: &values)
                return values
            }
            guard gguf != nil || always else { return nil }
            var values = [Float](repeating: 0, count: count * columns)
            values.withUnsafeMutableBufferPointer { out in
                switch span.dtype {
                case "F32": for i in 0..<out.count { out[i] = raw.loadUnaligned(fromByteOffset: i * 4, as: Float.self) }
                case "F16": for i in 0..<out.count { out[i] = Float(raw.loadUnaligned(fromByteOffset: i * 2, as: Float16.self)) }
                case "BF16": for i in 0..<out.count { out[i] = BF16.decode(raw.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)) }
                default: GGUFQuant.dequantize(raw, type: gguf!, count: count * columns, into: out.baseAddress!)
                }
            }
            return values
        }
        /// The whole tensor as floats, with llama.cpp's transforms undone.
        func whole() throws -> (values: [Float], rows: Int, columns: Int) {
            let columns = span.shape.last ?? 1
            let rows = span.shape.dropLast().reduce(1, *)
            var raw = [UInt8](repeating: 0, count: Int(span.length))
            try raw.withUnsafeMutableBytes { try file.read(offset: span.offset, into: $0) }
            var values = raw.withUnsafeBytes { decoded($0.baseAddress!, first: 0, count: rows, columns: columns, always: true)! }
            if work.negativeLog { values = values.map { log(-$0) } }
            if work.plusOne { values = values.map { $0 + 1 } }
            if let order = work.rowOrder {
                var permuted = values
                for (row, from) in order.enumerated() {
                    permuted.replaceSubrange(
                        (row * columns)..<((row + 1) * columns), with: values[(from * columns)..<((from + 1) * columns)])
                }
                values = permuted
            }
            if let order = work.columnOrder {
                var permuted = values
                for row in 0..<rows {
                    for (column, from) in order.enumerated() { permuted[row * columns + column] = values[row * columns + from] }
                }
                values = permuted
            }
            return (values, rows, columns)
        }
        switch work.kind {
        case .literal: break  // written by `convert` without a source file
        case .floats:
            let (values, _, columns) = try whole()
            try values.map(BF16.encode).withUnsafeBytes { try write(work.outputs[0], Int64(work.row * columns * 2), $0) }
        case .copy, .plusOne:
            // In pieces of at most 64 MB, so even the largest tensor needs little memory.
            let piece = 64 * 1024 * 1024 / element * element
            var buffer = [UInt8](repeating: 0, count: Int(min(Int64(piece), span.length)))
            var done: Int64 = 0
            while done < span.length {
                let count = Int(min(Int64(piece), span.length - done))
                try buffer.withUnsafeMutableBytes { raw in
                    try file.read(offset: span.offset + done, into: UnsafeMutableRawBufferPointer(rebasing: raw[0..<count]))
                    if case .plusOne = work.kind {
                        let values = raw.bindMemory(to: UInt16.self)
                        for i in 0..<(count / 2) { values[i] = BF16.encode(BF16.decode(values[i]) + 1) }
                    }
                    try write(work.outputs[0], work.byteOffset + done, UnsafeRawBufferPointer(rebasing: raw[0..<count]))
                }
                done += Int64(count)
            }
        case .quantize(let group), .gateUp(let group):
            let columns = span.shape.last!
            let rows = span.shape.dropLast().reduce(1, *)
            func quantized(_ source: UnsafeRawPointer, dtype: String, count: Int) -> ([UInt32], [UInt16], [UInt16]) {
                var codes = [UInt32](repeating: 0, count: count * columns / 8)
                var scales = [UInt16](repeating: 0, count: count * columns / group)
                var biases = scales
                codes.withUnsafeMutableBufferPointer { c in
                    scales.withUnsafeMutableBufferPointer { s in
                        biases.withUnsafeMutableBufferPointer { b in
                            quantize(
                                source, dtype: dtype, rows: count, columns: columns, group: group, codes: c.baseAddress!,
                                scales: s.baseAddress!, biases: b.baseAddress!)
                        }
                    }
                }
                return (codes, scales, biases)
            }
            func emit(_ parts: ([UInt32], [UInt16], [UInt16]), _ outputs: ArraySlice<Int>, outputRow: Int, sourceRows: Range<Int>) throws {
                let cw = columns / 8
                let gw = columns / group
                try parts.0.withUnsafeBytes {
                    try write(
                        outputs[outputs.startIndex], Int64(outputRow * cw * 4),
                        UnsafeRawBufferPointer(rebasing: $0[(sourceRows.lowerBound * cw * 4)..<(sourceRows.upperBound * cw * 4)]))
                }
                try parts.1.withUnsafeBytes {
                    try write(
                        outputs[outputs.startIndex + 1], Int64(outputRow * gw * 2),
                        UnsafeRawBufferPointer(rebasing: $0[(sourceRows.lowerBound * gw * 2)..<(sourceRows.upperBound * gw * 2)]))
                }
                try parts.2.withUnsafeBytes {
                    try write(
                        outputs[outputs.startIndex + 2], Int64(outputRow * gw * 2),
                        UnsafeRawBufferPointer(rebasing: $0[(sourceRows.lowerBound * gw * 2)..<(sourceRows.upperBound * gw * 2)]))
                }
            }
            if work.rowOrder != nil || work.columnOrder != nil {
                // Reordered heads: the whole matrix at once (the reordered ones are under 110 MB as floats).
                let values = try whole().values
                try emit(
                    values.withUnsafeBytes { quantized($0.baseAddress!, dtype: "F32", count: rows) }, work.outputs[0..<3],
                    outputRow: work.row, sourceRows: 0..<rows)
                return
            }
            // A block of rows: whole experts for gate/up (2 × inner rows each), else up to about 32 MB.
            let expertRows = span.shape.count == 3 ? span.shape[1] : 1
            let perRow = rowBytes(columns)
            let block = { () -> Int in
                if case .gateUp = work.kind { return expertRows * max(1, (32 << 20) / (expertRows * perRow)) }
                return max(1, (32 << 20) / perRow)
            }()
            var source = [UInt8](repeating: 0, count: min(block, rows) * perRow)
            for first in stride(from: 0, to: rows, by: block) {
                let count = min(block, rows - first)
                let parts = try source.withUnsafeMutableBytes { raw -> ([UInt32], [UInt16], [UInt16]) in
                    try file.read(
                        offset: span.offset + Int64(first * perRow),
                        into: UnsafeMutableRawBufferPointer(rebasing: raw[0..<(count * perRow)]))
                    guard let floats = decoded(raw.baseAddress!, first: first, count: count, columns: columns) else {
                        return quantized(raw.baseAddress!, dtype: span.dtype, count: count)
                    }
                    return floats.withUnsafeBytes { quantized($0.baseAddress!, dtype: "F32", count: count) }
                }
                if case .gateUp = work.kind {
                    let inner = expertRows / 2
                    for e in 0..<(count / expertRows) {
                        let expert = first / expertRows + e
                        let local = e * expertRows
                        try emit(parts, work.outputs[0..<3], outputRow: expert * inner, sourceRows: local..<(local + inner))
                        try emit(parts, work.outputs[3..<6], outputRow: expert * inner, sourceRows: (local + inner)..<(local + expertRows))
                    }
                } else {
                    try emit(parts, work.outputs[0..<3], outputRow: work.row + first, sourceRows: 0..<count)
                }
            }
        }
    }

    /// The quantizer's buffers, shared by its parallel blocks. Safe to send: the caller keeps them alive for the call,
    /// every block only reads the source and writes the codes, scales and biases of its own rows.
    private struct QuantizerBuffers: @unchecked Sendable {
        let source: UnsafeRawPointer, codes: UnsafeMutablePointer<UInt32>, scales, biases: UnsafeMutablePointer<UInt16>
    }

    /// MLX's affine quantizer, value for value: each group's scale is set so the endpoint farther from zero is exact,
    /// with Metal's rounding (half away from zero); 4-bit codes packed eight to a word, scales and biases in bf16.
    package static func quantize(
        _ source: UnsafeRawPointer, dtype: String, rows: Int, columns: Int, group: Int,
        codes: UnsafeMutablePointer<UInt32>, scales: UnsafeMutablePointer<UInt16>, biases: UnsafeMutablePointer<UInt16>
    ) {
        let groups = columns / group
        let element = dtype == "F32" ? 2 : dtype == "F16" ? 1 : 0
        let buffers = QuantizerBuffers(source: source, codes: codes, scales: scales, biases: biases)
        DispatchQueue.concurrentPerform(iterations: (rows + 63) / 64) { blockIndex in
            let (source, codes, scales, biases) = (buffers.source, buffers.codes, buffers.scales, buffers.biases)
            var values = [Float](repeating: 0, count: group)
            for row in (blockIndex * 64)..<min(rows, blockIndex * 64 + 64) {
                for g in 0..<groups {
                    let start = row * columns + g * group
                    switch element {
                    case 2: for i in 0..<group { values[i] = source.load(fromByteOffset: (start + i) * 4, as: Float.self) }
                    case 1: for i in 0..<group { values[i] = Float(source.load(fromByteOffset: (start + i) * 2, as: Float16.self)) }
                    default: for i in 0..<group { values[i] = BF16.decode(source.load(fromByteOffset: (start + i) * 2, as: UInt16.self)) }
                    }
                    var low = values[0]
                    var high = values[0]
                    for v in values {
                        low = min(low, v)
                        high = max(high, v)
                    }
                    var scale = max((high - low) / 15, Float(1e-7))
                    let side = abs(low) > abs(high)
                    scale = side ? scale : -scale
                    let edge = side ? low : high
                    let q0 = (edge / scale).rounded(.toNearestOrAwayFromZero)
                    if q0 != 0 { scale = edge / q0 }
                    let bias: Float = q0 == 0 ? 0 : edge
                    scales[row * groups + g] = BF16.encode(scale)
                    biases[row * groups + g] = BF16.encode(bias)
                    for word in 0..<(group / 8) {
                        var packed: UInt32 = 0
                        for k in 0..<8 {
                            let code = min(max(((values[word * 8 + k] - bias) / scale).rounded(.toNearestOrAwayFromZero), 0), 15)
                            packed |= UInt32(code) << (4 * k)
                        }
                        codes[(row * columns + g * group) / 8 + word] = packed
                    }
                }
            }
        }
    }
}
