import CoreFoundation
import Foundation
import Metal

/// Routed experts kept in a Qwen GGUF's own encodings (the `gguf-experts` scheme). `origami model add` converts the rest
/// of a GGUF whose expert banks average under 4.5 bits per weight (or that keeps fewer than 512 experts) as it converts
/// any GGUF and, instead of requantizing the banks, writes `gguf-experts.json` beside the converted weights: per layer
/// and projection the GGUF file, the bank's data offset, the bytes from one expert to the next, its ggml type and shape.
/// `PackStore` reads a record's three spans straight from the GGUF.
///
/// A record is a 16-byte header (magic, layer, expert, 0), then gate, up and down in their GGUF bytes, each from a 16-byte
/// boundary (ggml's kernels take 16-byte aligned weights). A cache slot holds the largest layer's record, or a size class's
/// largest when the cache keeps several (`ExpertCache.SizeClasses`). The header
/// lets every kernel that gets only a record's buffer and offset (decode groups, prefill jobs, the layer command) find its
/// layer's types without the layer being passed through each of them.
public struct GGUFExperts: Sendable, Equatable {
    public static let file = "gguf-experts.json", format = "origami-gguf-experts-v1"
    static let magic: UInt32 = 0x5847_474F, headerBytes = 16
    /// One projection of one layer: its bank in a GGUF file and its place in a record.
    public struct Part: Sendable, Equatable {
        public let path: String
        /// The bank's data offset in the file, and the bytes from one expert's matrix to the next.
        public let offset, stride: Int
        /// ggml type, bytes of one expert's matrix, its rows (outputs) and columns (inputs), and its offset in a record.
        public let type, bytes, outputs, width, record: Int
    }
    /// A layer's gate, up and down, and the bytes of its records.
    public struct Layer: Sendable, Equatable {
        public let parts: [Part]
        public let bytes: Int
    }
    /// A GGUF file the banks are in, with the size and modification time it had when this layout was read.
    public struct File: Sendable, Equatable {
        public let path: String
        public let size: UInt64
        public let modified: Int64?
    }
    /// Routed experts per layer in the GGUF (512, or fewer for a pruned one); the engine's other expert ids are never routed.
    public let experts: Int
    public let layers: [Layer]
    public let files: [File]
    /// The largest layer's record: the largest cache slot.
    public let recordBytes: Int

    /// The architecture's shape of each projection: (outputs, inputs).
    static let shapes = [(640, 2560), (640, 2560), (2560, 640)]

    /// `parts`: per layer, gate, up and down as (path, offset, stride, type); shapes are the architecture's.
    private init(experts: Int, files: [File], parts: [[(path: String, offset: Int, stride: Int, type: Int)]]) {
        self.experts = experts
        self.files = files
        layers = parts.map { list in
            var cursor = Self.headerBytes
            let made = zip(list, Self.shapes).map { part, shape -> Part in
                let (block, size) = GGUFQuant.sizes[part.type]!
                let bytes = shape.0 * (shape.1 / block) * size
                defer { cursor += (bytes + 15) / 16 * 16 }
                return Part(
                    path: part.path, offset: part.offset, stride: part.stride, type: part.type, bytes: bytes,
                    outputs: shape.0, width: shape.1, record: cursor)
            }
            return Layer(parts: made, bytes: cursor)
        }
        recordBytes = layers.map(\.bytes).max() ?? Self.headerBytes
    }

    /// A layout without files, for checks: `types[layer]` are gate, up and down's ggml types.
    package init(experts: Int, types: [[Int]]) {
        self.init(experts: experts, files: [], parts: types.map { $0.map { (path: "", offset: 0, stride: 0, type: $0) } })
    }

    /// Reads and checks a manifest (untrusted): every shape is the architecture's, every type has a kernel, every bank lies
    /// inside its file, and the files are there with the sizes recorded.
    public init(manifest data: Data) throws {
        func bad(_ why: String) -> EngineError {
            EngineError("\(Self.file) is not usable: \(why); add the GGUF again with `origami model add`", code: "model_incompatible")
        }
        func int(_ value: Any?) -> Int? {
            guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), let exact = Int(exactly: number.doubleValue)
            else { return nil }
            return exact
        }
        guard data.count <= 1 << 20, let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
            root["format"] as? String == Self.format
        else { throw bad("it is not an \(Self.format) manifest") }
        guard let experts = int(root["experts"]), (10...ModelContract.experts).contains(experts) else {
            throw bad("its expert count is not 10 to 512")
        }
        guard let listed = root["files"] as? [[String: Any]], (1...64).contains(listed.count),
            let layers = root["layers"] as? [[String: Any]], layers.count == ModelContract.layers
        else {
            throw bad("it does not list the GGUF files and \(ModelContract.layers) layers")
        }
        var files: [File] = []
        for entry in listed {
            guard let path = entry["path"] as? String, path.hasPrefix("/"), !path.contains("/../"), let size = int(entry["size"]), size > 0
            else {
                throw bad("a file entry is not an absolute path with a size")
            }
            var info = stat()
            guard stat(path, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG, UInt64(info.st_size) == UInt64(size) else {
                throw EngineError(
                    "The GGUF \(path) this model's experts are read from is missing or changed (it had \(size) bytes); "
                        + "put it back or add the GGUF again with `origami model add`", code: "model_incomplete")
            }
            files.append(File(path: path, size: UInt64(size), modified: ModelSource.modificationNanoseconds(URL(fileURLWithPath: path))))
        }
        var parts: [[(path: String, offset: Int, stride: Int, type: Int)]] = []
        for (index, layer) in layers.enumerated() {
            var list: [(path: String, offset: Int, stride: Int, type: Int)] = []
            for (name, shape) in zip(["gate", "up", "down"], Self.shapes) {
                guard let part = layer[name] as? [String: Any], let path = part["path"] as? String,
                    let file = files.first(where: { $0.path == path }),
                    let offset = int(part["offset"]), let stride = int(part["stride"]), let type = int(part["type"]),
                    let rows = int(part["rows"]), let columns = int(part["columns"]), let bytes = int(part["expert_bytes"])
                else {
                    throw bad("layer \(index) lacks its \(name) bank")
                }
                guard GGMLMatrixKernels.types.contains(type), let (block, size) = GGUFQuant.sizes[type], block > 1 else {
                    throw bad("layer \(index) \(name) is \(GGUFQuant.name(type)), which Origami's expert kernels do not run")
                }
                // Bounded by the file's size first, so the bank's end cannot overflow.
                guard (rows, columns) == shape, columns % block == 0, bytes == rows * (columns / block) * size, stride >= bytes,
                    offset >= 0, UInt64(offset) <= file.size, UInt64(stride) <= file.size,
                    UInt64(offset + (experts - 1) * stride + bytes) <= file.size
                else {
                    throw bad("layer \(index) \(name) bank does not fit its shape or file")
                }
                list.append((path, offset, stride, type))
            }
            parts.append(list)
        }
        self.init(experts: experts, files: files, parts: parts)
    }

    /// The manifest the converter writes: `layers[l]` holds gate, up and down as (path, offset, stride, type, rows, columns).
    package static func manifest(
        experts: Int, files: [(path: String, size: UInt64)],
        layers: [[(path: String, offset: Int, stride: Int, type: Int, rows: Int, columns: Int)]]
    ) throws -> Data {
        let entries = layers.map { list in
            Dictionary(
                uniqueKeysWithValues: zip(["gate", "up", "down"], list).map { name, part -> (String, Any) in
                    let (block, size) = GGUFQuant.sizes[part.type] ?? (1, 0)
                    return (
                        name,
                        [
                            "path": part.path, "offset": part.offset, "stride": part.stride, "type": part.type,
                            "type_name": GGUFQuant.name(part.type),
                            "rows": part.rows, "columns": part.columns, "expert_bytes": part.rows * (part.columns / block) * size,
                        ]
                    )
                })
        }
        let root: [String: Any] = [
            "format": format, "experts": experts, "files": files.map { ["path": $0.path, "size": $0.size] }, "layers": entries,
        ]
        return try JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
    }

    /// Writes record `expert` of `layer`'s header into `slot` and zeroes its padding. `slot` is one cache slot, or the
    /// layer's own bytes (a record as a hot copy holds it: the same bytes without the slot's padding after them).
    package func prepare(_ slot: UnsafeMutableRawBufferPointer, layer: Int, expert: Int) {
        precondition((0..<layers.count).contains(layer) && (layers[layer].bytes...recordBytes).contains(slot.count))
        memset(slot.baseAddress!, 0, Self.headerBytes)
        for (i, value) in [Self.magic, UInt32(layer), UInt32(expert)].enumerated() {
            slot.storeBytes(of: value.littleEndian, toByteOffset: i * 4, as: UInt32.self)
        }
        var cursor = Self.headerBytes
        for part in layers[layer].parts + [
            Part(path: "", offset: 0, stride: 0, type: 0, bytes: 0, outputs: 0, width: 0, record: slot.count)
        ] {
            if part.record > cursor { memset(slot.baseAddress! + cursor, 0, part.record - cursor) }
            cursor = part.record + part.bytes
        }
    }

    /// The layer of the record at `offset` in `buffer` (shared storage), from its header.
    func layer(of buffer: MTLBuffer, offset: Int) throws -> Layer {
        guard offset >= 0, offset % 16 == 0, offset <= buffer.length - Self.headerBytes else {
            throw EngineError("A GGUF expert record is outside its buffer")
        }
        let header = buffer.contents().advanced(by: offset)
        let magic = UInt32(littleEndian: header.load(as: UInt32.self))
        let layer = Int(UInt32(littleEndian: header.load(fromByteOffset: 4, as: UInt32.self)))
        guard magic == Self.magic, (0..<layers.count).contains(layer) else {
            throw EngineError("A cache slot does not hold a GGUF expert record")
        }
        // A smaller size class's slot holds the layer's record, not the largest one.
        guard layers[layer].bytes <= buffer.length - offset else { throw EngineError("A GGUF expert record is outside its buffer") }
        return layers[layer]
    }

    /// Bytes of every expert of every layer in the GGUF.
    public var bankBytes: UInt64 { UInt64(experts) * layers.reduce(0) { $0 + UInt64($1.parts.reduce(0) { $0 + $1.bytes }) } }
    /// The banks' ggml types and how many banks use each, most used first ("IQ3_S ×26, …").
    public var typeSummary: String {
        var counts: [Int: Int] = [:]
        for layer in layers { for part in layer.parts { counts[part.type, default: 0] += 1 } }
        return counts.sorted { ($0.value, $1.key) > ($1.value, $0.key) }.map { "\(GGUFQuant.name($0.key)) ×\($0.value)" }.joined(
            separator: ", ")
    }
    /// Average bits per weight of the banks.
    public var bitsPerWeight: Double {
        Double(bankBytes) * 8 / Double(experts * layers.reduce(0) { $0 + $1.parts.reduce(0) { $0 + $1.outputs * $1.width } })
    }
}

extension GGUFExperts {
    /// Layers whose router and shared-expert gate `variant` compares with the base's.
    static let comparedLayers = [0, 11, 23, 35, 47]

    /// The manifest of a low-bit variant of `base`: its routed experts read from `gguf` (some or all parts of a split set;
    /// only the banks are read while serving), everything else from `base`, which must be an exact 4-bit group-64 model.
    /// Checks that the GGUF is the same checkpoint quantized differently, from headers and about 13 MB of reads:
    /// Qwen3.8-Flash-Next without folded rotations, the base's geometry, every token of the base's tokenizer at the same
    /// id, the router and shared-expert gate of `comparedLayers` equal value for value, and banks of shapes and types the
    /// expert kernels run. Throws naming what differs.
    package static func variant(gguf: GGUFSource, base: ModelSource) throws -> Data {
        func refuse(_ why: String) -> EngineError {
            EngineError("These experts do not belong with \(base.displayName): \(why)", code: "model_incompatible")
        }
        guard base.quantization == .affine4Group64 else {
            throw refuse("the base must be an exact 4-bit group-64 MLX model, not \(base.quantization.rawValue)")
        }
        guard case .string("qwen4exp")? = gguf.metadata["general.architecture"] else {
            throw refuse("the GGUF is not Qwen3.8-Flash-Next (qwen4exp)")
        }
        if gguf.metadata.keys.contains(where: { $0.hasPrefix("prism.hadamard.") }) {
            throw refuse("Hadamard rotations are folded into the GGUF's weights")
        }
        func json(_ file: String) throws -> [String: Any] {
            guard let data = try? Data(contentsOf: base.directory.appendingPathComponent(file)),
                let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            else { throw refuse("the base's \(file) cannot be read") }
            return object
        }
        let root = try json("config.json")
        let config = root["text_config"] as? [String: Any] ?? root
        let nextn = gguf.metadata["qwen4exp.nextn_predict_layers"]?.integer ?? 0
        for (key, ggufKey) in [
            ("num_hidden_layers", "block_count"), ("hidden_size", "embedding_length"), ("num_experts", "expert_count"),
            ("num_experts_per_tok", "expert_used_count"), ("moe_intermediate_size", "expert_feed_forward_length"),
            ("shared_expert_intermediate_size", "expert_shared_feed_forward_length"), ("num_attention_heads", "attention.head_count"),
            ("num_key_value_heads", "attention.head_count_kv"),
        ] {
            let ours = (config[key] as? NSNumber)?.int64Value
            let theirs = gguf.metadata["qwen4exp." + ggufKey]?.integer.map { ggufKey == "block_count" ? $0 - nextn : $0 }
            guard let ours, ours == theirs else {
                throw refuse(
                    "its \(ggufKey) is \(theirs.map(String.init) ?? "missing"), the base's \(key) \(ours.map(String.init) ?? "missing")")
            }
        }
        let tokenizer = try json("tokenizer.json")
        let tokens = gguf.tokenizer["tokenizer.ggml.tokens"] ?? []
        var expected = ((tokenizer["model"] as? [String: Any])?["vocab"] as? [String: Int] ?? [:]).map { ($0.value, $0.key) }
        for added in tokenizer["added_tokens"] as? [[String: Any]] ?? [] {
            if let id = added["id"] as? Int, let content = added["content"] as? String { expected.append((id, content)) }
        }
        guard !expected.isEmpty, !tokens.isEmpty else { throw refuse("there is no tokenizer to compare (the base's or the GGUF's)") }
        for (id, text) in expected.sorted(by: { $0.0 < $1.0 }) {
            guard id >= 0, id < tokens.count, tokens[id] == .string(text) else {
                throw refuse("token \(id) is \(text.debugDescription) in the base and not in the GGUF: another tokenizer")
            }
        }
        // Values of a float tensor at `offset` in `url`: `count` of ggml `type`.
        func values(_ url: URL, offset: Int64, count: Int, type: Int) throws -> [Float] {
            let bytes = count * (GGUFQuant.sizes[type]?.bytes ?? 0)
            guard [0, 1, 30].contains(type), count > 0, bytes <= 64 << 20 else {
                throw refuse("\(url.lastPathComponent) holds no float tensor there")
            }
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            try handle.seek(toOffset: UInt64(offset))
            guard let data = try handle.read(upToCount: bytes), data.count == bytes else {
                throw refuse("\(url.lastPathComponent) is cut short")
            }
            var out = [Float](repeating: 0, count: count)
            data.withUnsafeBytes { raw in
                out.withUnsafeMutableBufferPointer {
                    GGUFQuant.dequantize(raw.baseAddress!, type: type, count: count, into: $0.baseAddress!)
                }
            }
            return out
        }
        for layer in comparedLayers {
            for (ours, theirs, what) in [
                ("mlp.gate.weight", "ffn_gate_inp.weight", "router"),
                ("mlp.shared_expert_gate.weight", "ffn_gate_inp_shexp.weight", "shared-expert gate"),
            ] {
                guard let span = base.tensors["language_model.model.layers.\(layer).\(ours)"],
                    let type = ["F32": 0, "F16": 1, "BF16": 30][span.dtype]
                else {
                    // shortcut: a base whose router is quantized cannot be compared value for value; compare dequantized values if one appears.
                    throw refuse("the base's layer \(layer) \(what) is not a float tensor Origami can compare")
                }
                guard let t = gguf.tensors["blk.\(layer).\(theirs)"] else {
                    throw refuse("blk.\(layer).\(theirs) is not in the GGUF parts given")
                }
                let count = span.shape.reduce(1, *)
                guard t.shape.reduce(1, *) == count,
                    try values(base.directory.appendingPathComponent(span.file), offset: span.offset, count: count, type: type)
                        == values(gguf.folder.appendingPathComponent(t.file), offset: t.offset, count: count, type: t.type)
                else {
                    throw refuse("layer \(layer)'s \(what) differs from the base's: another checkpoint or a fine-tune")
                }
            }
        }
        let experts = Int(gguf.metadata["qwen4exp.expert_count"]?.integer ?? 0)
        var layers: [[(path: String, offset: Int, stride: Int, type: Int, rows: Int, columns: Int)]] = []
        for layer in 0..<ModelContract.layers {
            // As the converter reads them (`CheckpointConverter(gguf:)`): a fused gate/up bank is two parts.
            func bank(_ name: String, part: Int = 0, of parts: Int = 1) throws -> (
                path: String, offset: Int, stride: Int, type: Int, rows: Int, columns: Int
            ) {
                guard let t = gguf.tensors[name] else { throw refuse("\(name) is not in the GGUF parts given") }
                guard GGMLMatrixKernels.types.contains(t.type), let size = GGUFQuant.sizes[t.type], size.block > 1 else {
                    throw refuse("\(name) is \(GGUFQuant.name(t.type)), which Origami's expert kernels do not run")
                }
                guard t.shape.count == 3, t.shape[0] == experts, t.shape[1] % parts == 0, t.shape[2] % size.block == 0,
                    Int64(experts * t.shape[1] * (t.shape[2] / size.block * size.bytes)) == t.length
                else { throw refuse("\(name) is not a bank of \(experts) experts") }
                let rowBytes = t.shape[2] / size.block * size.bytes
                return (
                    gguf.folder.appendingPathComponent(t.file).standardizedFileURL.path,
                    Int(t.offset) + part * t.shape[1] / parts * rowBytes,
                    t.shape[1] * rowBytes, t.type, t.shape[1] / parts, t.shape[2]
                )
            }
            let b = "blk.\(layer)."
            layers.append(
                (gguf.tensors[b + "ffn_gate_up_exps.weight"] != nil
                    ? [try bank(b + "ffn_gate_up_exps.weight", part: 0, of: 2), try bank(b + "ffn_gate_up_exps.weight", part: 1, of: 2)]
                    : [try bank(b + "ffn_gate_exps.weight"), try bank(b + "ffn_up_exps.weight")]) + [try bank(b + "ffn_down_exps.weight")])
            for (part, shape) in zip(layers[layer], shapes) where (part.rows, part.columns) != shape {
                throw refuse("layer \(layer)'s banks are \(part.rows) × \(part.columns), not the architecture's \(shape.0) × \(shape.1)")
            }
        }
        let files = Set(layers.joined().map(\.path)).sorted().map { path -> (path: String, size: UInt64) in
            var info = stat()
            return (path, stat(path, &info) == 0 ? UInt64(info.st_size) : 0)
        }
        let data = try manifest(experts: experts, files: files, layers: layers)
        _ = try GGUFExperts(manifest: data)  // as the engine reads it
        return data
    }
}

/// The Qwen engine's GGUF expert arithmetic on ggml's matrix kernels (`GGMLMatrixKernels`), as llama.cpp computes a GGUF's
/// experts: activations widened from bf16 to f32, gate and up, SiLU(gate) × up in f32, down. Decode groups take a layer's
/// records at their canonical ranks (`GatheredExpertDecode`); prefill tiles take rows `PrefillMoEWorkspace` gathers
/// (`FusedExpertExecution`). Other weights and other arithmetic than the MLX schedules, so NUMERIC by construction.
final class GGUFExpertKernels {
    let matrices: GGMLMatrixKernels
    private let widen, swiglu, swigluRanks: MTLComputePipelineState

    init(device: MTLDevice) throws {
        matrices = try GGMLMatrixKernels(device: device)
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.mathMode = .safe
        let library = try device.makeLibrary(source: Self.source, options: options)
        guard let w = library.makeFunction(name: "gguf_expert_widen"), let s = library.makeFunction(name: "gguf_expert_swiglu"),
            let r = library.makeFunction(name: "gguf_expert_swiglu_ranks")
        else {
            throw EngineError("Missing GGUF expert kernels")
        }
        widen = try device.makeComputePipelineState(function: w)
        swiglu = try device.makeComputePipelineState(function: s)
        swigluRanks = try device.makeComputePipelineState(function: r)
    }

    private func elementwise(
        _ e: MTLComputeCommandEncoder, _ pipeline: MTLComputePipelineState, _ a: (MTLBuffer, Int), _ b: (MTLBuffer, Int),
        count: Int, bytes: (Int, Int)
    ) throws {
        guard count > 0, a.1 >= 0, b.1 >= 0, a.1 + count * bytes.0 <= a.0.length, b.1 + count * bytes.1 <= b.0.length else {
            throw EngineError("GGUF expert elementwise buffers are too short")
        }
        e.setComputePipelineState(pipeline)
        e.setBuffer(a.0, offset: a.1, index: 0)
        e.setBuffer(b.0, offset: b.1, index: 1)
        var n = UInt32(count)
        e.setBytes(&n, length: 4, index: 2)
        e.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(256, pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
    }
    /// The f32 values of `count` bf16 values.
    func encodeWiden(_ e: MTLComputeCommandEncoder, _ bf16: MTLBuffer, into floats: MTLBuffer, count: Int) throws {
        try elementwise(e, widen, (bf16, 0), (floats, 0), count: count, bytes: (2, 4))
    }
    /// gate = SiLU(gate) × up over `count` values from the offsets.
    func encodeSwiGLU(_ e: MTLComputeCommandEncoder, gate: MTLBuffer, gateOffset: Int, up: MTLBuffer, upOffset: Int, count: Int) throws {
        try elementwise(e, swiglu, (gate, gateOffset), (up, upOffset), count: count, bytes: (4, 4))
    }

    /// Output z = part of record z times its input rows: input z at `inputOffset + z * inputStride` (0 shares one input),
    /// output z at `outputOffset + z * outputStride`. One dispatch for the group where ggml's batch kernel applies (up to 8
    /// rows), else one per record; the same bits either way.
    func encode(
        _ e: MTLComputeCommandEncoder, _ part: GGUFExperts.Part, records: [(buffer: MTLBuffer, offset: Int)],
        input: MTLBuffer, inputOffset: Int, inputStride: Int, rows: Int, output: MTLBuffer, outputOffset: Int, outputStride: Int
    ) throws {
        let weights = records.map { (buffer: $0.buffer, offset: $0.offset + part.record) }
        if weights.count > 1, rows <= 8, matrices.batches(type: part.type) {
            return try matrices.encodeBatch(
                e, type: part.type, matrices: weights, width: part.width, outputs: part.outputs,
                input: input, inputOffset: inputOffset, inputStride: inputStride, rows: rows,
                output: output, outputOffset: outputOffset, outputStride: outputStride)
        }
        for (z, weight) in weights.enumerated() {
            try matrices.encode(
                e, type: part.type, weights: weight.buffer, weightOffset: weight.offset, width: part.width, outputs: part.outputs,
                input: input, inputOffset: inputOffset + z * inputStride, rows: rows,
                output: output, outputOffset: outputOffset + z * outputStride)
        }
    }

    /// One decode group: `records` (one layer's) over the bf16 row `input`, record i at canonical rank `ranks[i]`. `floats`
    /// receives the f32 row (2560 values), `gate` and `up` hold [10][640] (gate ends as the product) and `down` [10][2560],
    /// each record writing only its rank's spans, so any split of the ten into groups gives the same bytes.
    func encodeGroup(
        _ command: MTLCommandBuffer, input: MTLBuffer, records: [(buffer: MTLBuffer, offset: Int)], ranks: [Int],
        floats: MTLBuffer, gate: MTLBuffer, up: MTLBuffer, down: MTLBuffer, layout: GGUFExperts
    ) throws {
        guard let first = records.first, records.count == ranks.count else { throw EngineError("Invalid GGUF expert decode group") }
        let layer = try layout.layer(of: first.buffer, offset: first.offset)
        guard try records.allSatisfy({ try layout.layer(of: $0.buffer, offset: $0.offset) == layer }) else {
            throw EngineError("A GGUF expert decode group mixes layers")
        }
        guard let e = command.makeComputeCommandEncoder() else { throw EngineError("GGUF expert decode encoder unavailable") }
        defer { e.endEncoding() }
        try encodeWiden(e, input, into: floats, count: 2560)
        // Consecutive ranks share a dispatch; a resident-first group of scattered ranks takes one per run.
        let order = ranks.indices.sorted { ranks[$0] < ranks[$1] }
        var runs: [[Int]] = []
        for i in order {
            if let last = runs.last?.last, ranks[i] == ranks[last] + 1 { runs[runs.count - 1].append(i) } else { runs.append([i]) }
        }
        let (g, u, d) = (layer.parts[0], layer.parts[1], layer.parts[2])
        for run in runs {
            let at = ranks[run[0]]
            let group = run.map { records[$0] }
            try encode(
                e, g, records: group, input: floats, inputOffset: 0, inputStride: 0, rows: 1, output: gate, outputOffset: at * 640 * 4,
                outputStride: 640 * 4)
            try encode(
                e, u, records: group, input: floats, inputOffset: 0, inputStride: 0, rows: 1, output: up, outputOffset: at * 640 * 4,
                outputStride: 640 * 4)
            try encodeSwiGLU(e, gate: gate, gateOffset: at * 640 * 4, up: up, upOffset: at * 640 * 4, count: run.count * 640)
            try encode(
                e, d, records: group, input: gate, inputOffset: at * 640 * 4, inputStride: 640 * 4, rows: 1,
                output: down, outputOffset: at * 2560 * 4, outputStride: 2560 * 4)
        }
    }

    /// Queued decode: `encodeGroup` for records not known when it is encoded. `records` holds ten GPU addresses and `ranks` ten
    /// UInt32 ranks, as many used as the grids say (`grids` at `offset`: two (x, records, 1) UInt32 triples, read when it
    /// runs, as the tile8 kernels read them); all are `layer`'s records (the residency table and the host fill only take
    /// that layer's). Each record runs the kernels `encodeGroup` runs for it on the same spans, so the bytes are the same.
    func encodeDeferredGroup(
        _ command: MTLCommandBuffer, layer: GGUFExperts.Layer, input: MTLBuffer, records: MTLBuffer, ranks: MTLBuffer,
        grids: MTLBuffer, offset: Int, floats: MTLBuffer, gate: MTLBuffer, up: MTLBuffer, down: MTLBuffer
    ) throws {
        guard gate.length >= 10 * 640 * 4, up.length >= 10 * 640 * 4, ranks.length >= 40 else {
            throw EngineError("GGUF expert decode buffers are too short")
        }
        guard let e = command.makeComputeCommandEncoder() else { throw EngineError("GGUF expert decode encoder unavailable") }
        defer { e.endEncoding() }
        try encodeWiden(e, input, into: floats, count: 2560)
        let (g, u, d) = (layer.parts[0], layer.parts[1], layer.parts[2])
        for (part, output) in [(g, gate), (u, up)] {
            try matrices.encodeGathered(
                e, type: part.type, part: part.record, width: part.width, outputs: part.outputs, records: records, ranks: ranks,
                grid: (grids, offset), input: floats, inputOffset: 0, inputStride: 0, output: output, outputOffset: 0,
                outputStride: 640 * 4)
        }
        e.setComputePipelineState(swigluRanks)
        e.setBuffer(gate, offset: 0, index: 0)
        e.setBuffer(up, offset: 0, index: 1)
        e.setBuffer(ranks, offset: 0, index: 2)
        e.dispatchThreadgroups(
            indirectBuffer: grids, indirectBufferOffset: offset, threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        try matrices.encodeGathered(
            e, type: d.type, part: d.record, width: d.width, outputs: d.outputs, records: records, ranks: ranks,
            grid: (grids, offset + 12), input: gate, inputOffset: 0, inputStride: 640 * 4, output: down, outputOffset: 0,
            outputStride: 2560 * 4)
    }

    /// Prefill tiles of one layer, each one record over `rows` gathered bf16 rows in `input`: widened into `output`, gate and
    /// up into `gate` and `up`, the product into `gate`, down into `output` (f32 [rows][2560], what the scatter reads).
    /// Stages are separated by barriers, so the tiles of a wave run side by side on a concurrent encoder.
    struct Tile {
        let record: (buffer: MTLBuffer, offset: Int)
        let rows: Int
        let input, gate, up, output: MTLBuffer
    }
    func encodeTiles(_ e: MTLComputeCommandEncoder, _ tiles: [Tile], layout: GGUFExperts) throws {
        let layers = try tiles.map { try layout.layer(of: $0.record.buffer, offset: $0.record.offset) }
        for t in tiles { try encodeWiden(e, t.input, into: t.output, count: t.rows * 2560) }
        e.memoryBarrier(scope: .buffers)
        for (t, layer) in zip(tiles, layers) {
            for (part, out) in [(layer.parts[0], t.gate), (layer.parts[1], t.up)] {
                try encode(
                    e, part, records: [t.record], input: t.output, inputOffset: 0, inputStride: 0, rows: t.rows, output: out,
                    outputOffset: 0, outputStride: 0)
            }
        }
        e.memoryBarrier(scope: .buffers)
        for t in tiles { try encodeSwiGLU(e, gate: t.gate, gateOffset: 0, up: t.up, upOffset: 0, count: t.rows * 640) }
        e.memoryBarrier(scope: .buffers)
        for (t, layer) in zip(tiles, layers) {
            try encode(
                e, layer.parts[2], records: [t.record], input: t.gate, inputOffset: 0, inputStride: 0, rows: t.rows, output: t.output,
                outputOffset: 0, outputStride: 0)
        }
    }

    private static let source = #"""
        #include <metal_stdlib>
        using namespace metal;
        // y = the f32 value of bf16 x (the engine narrows activations to bf16; ggml's kernels take f32).
        kernel void gguf_expert_widen(device const ushort *x [[buffer(0)]], device float *y [[buffer(1)]], constant uint &n [[buffer(2)]],
                                      uint i [[thread_position_in_grid]]) {
            if (i < n) y[i] = as_type<float>(uint(x[i]) << 16);
        }
        // gate = SiLU(gate) * up, in f32 (ggml's SwiGLU).
        kernel void gguf_expert_swiglu(device float *gate [[buffer(0)]], device const float *up [[buffer(1)]], constant uint &n [[buffer(2)]],
                                       uint i [[thread_position_in_grid]]) {
            if (i < n) { float g = gate[i]; gate[i] = g / (1.0f + exp(-g)) * up[i]; }
        }
        // gguf_expert_swiglu over the 640 values of each grid row's rank (queued decode: the indirect grid of 20 x records
        // threadgroups of 32 that the tile8 product kernel reads).
        kernel void gguf_expert_swiglu_ranks(device float *gate [[buffer(0)]], device const float *up [[buffer(1)]],
                                             constant uint *ranks [[buffer(2)]], uint2 t [[thread_position_in_grid]]) {
            if (t.x >= 640) return;  // the grid is 640 wide today; a wider one must not touch the next rank's values
            uint i = ranks[t.y] * 640 + t.x;
            float g = gate[i]; gate[i] = g / (1.0f + exp(-g)) * up[i];
        }
        """#
}
