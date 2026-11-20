import Foundation

/// Byte ranges of one model file, wherever it lies: headers are read this way so a checkpoint can be judged without
/// its weights (a Hugging Face file is read by HTTP range).
package protocol ByteSource: Sendable {
    var size: UInt64 { get }
    func read(at offset: UInt64, count: Int) throws -> Data
}

package struct LocalBytes: ByteSource {
    let url: URL
    package let size: UInt64
    /// Measured through a symlink (a Hugging Face cache); a file that cannot be measured is an error, not size 0.
    package init(_ url: URL) throws {
        self.url = url
        guard let size = try FileManager.default.attributesOfItem(atPath: url.resolvingSymlinksInPath().path)[.size] as? NSNumber else {
            throw EngineError("\(url.path) has no size", code: "model_incompatible")
        }
        self.size = size.uint64Value
    }
    package func read(at offset: UInt64, count: Int) throws -> Data {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        try handle.seek(toOffset: offset)
        return try handle.read(upToCount: count) ?? Data()
    }
}

/// The Hugging Face Hub: a repo's file list and byte ranges of its files. Uses `HF_TOKEN` or the token
/// `huggingface-cli login` saved, for gated repos; the token is only ever sent to huggingface.co.
package struct HubRepo: Sendable {
    package let repo: String, revision: String
    package struct File: Sendable { package let path: String, size: UInt64 }

    package init(_ repo: String, revision: String = "main") {
        self.repo = repo
        self.revision = revision
    }

    package static var token: String? {
        if let token = ProcessInfo.processInfo.environment["HF_TOKEN"], !token.isEmpty { return token }
        let saved = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/token")
        return (try? String(contentsOf: saved, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// One synchronous request (the CLI and checks call this off the serving path).
    static func fetch(_ url: URL, range: ClosedRange<UInt64>? = nil) throws -> (Data, HTTPURLResponse) {
        var request = URLRequest(url: url, timeoutInterval: 60)
        if let range { request.setValue("bytes=\(range.lowerBound)-\(range.upperBound)", forHTTPHeaderField: "Range") }
        if let host = url.host, host == "huggingface.co" || host.hasSuffix(".huggingface.co"), let token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        final class Box: @unchecked Sendable { var result: Result<(Data, HTTPURLResponse), Error> = .failure(EngineError("no response")) }
        let box = Box()
        let done = DispatchSemaphore(value: 0)
        for attempt in 0..<4 {
            URLSession.shared.dataTask(with: request) { data, response, error in
                if let error {
                    box.result = .failure(error)
                } else if let response = response as? HTTPURLResponse, let data {
                    box.result = .success((data, response))
                }
                done.signal()
            }.resume()
            done.wait()
            switch box.result {
            case .success(let (data, response)) where (200..<300).contains(response.statusCode): return (data, response)
            case .success(let (_, response)) where response.statusCode == 429 || response.statusCode >= 500:
                Thread.sleep(forTimeInterval: Double(1 << attempt))
            case .success(let (_, response)):
                throw EngineError(
                    "\(url.absoluteString) answered HTTP \(response.statusCode)"
                        + ([401, 403].contains(response.statusCode) ? " (a gated or private repo: set HF_TOKEN)" : ""),
                    code: "hub_unavailable")
            case .failure(let error):
                if attempt == 3 { throw EngineError("\(url.absoluteString): \(error.localizedDescription)", code: "hub_unavailable") }
                Thread.sleep(forTimeInterval: Double(1 << attempt))
            }
        }
        throw EngineError("\(url.absoluteString) kept failing", code: "hub_unavailable")
    }

    /// Every file in the repo with its size (LFS size for weights), following the tree API's pages.
    package func files() throws -> [File] {
        var next: URL? = URL(string: "https://huggingface.co/api/models/\(repo)/tree/\(revision)?recursive=true&expand=false")
        var files: [File] = []
        while let url = next {
            let (data, response) = try Self.fetch(url)
            guard let entries = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
                throw EngineError("\(repo): the Hub's file list is not JSON", code: "hub_unavailable")
            }
            for entry in entries where entry["type"] as? String == "file" {
                guard let path = entry["path"] as? String else { continue }
                let size = ((entry["lfs"] as? [String: Any])?["size"] as? NSNumber ?? entry["size"] as? NSNumber)?.uint64Value ?? 0
                files.append(File(path: path, size: size))
            }
            next =
                (response.value(forHTTPHeaderField: "Link") ?? "").components(separatedBy: ",").lazy
                .filter { $0.contains("rel=\"next\"") }
                .compactMap { part -> URL? in
                    guard let open = part.firstIndex(of: "<"), let close = part.firstIndex(of: ">") else { return nil }
                    return URL(string: String(part[part.index(after: open)..<close]))
                }.first
        }
        return files
    }

    package func url(_ path: String) -> URL {
        URL(
            string: "https://huggingface.co/\(repo)/resolve/\(revision)/"
                + path.split(separator: "/").map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }.joined(
                    separator: "/"))!
    }
    package func data(_ path: String) throws -> Data { try Self.fetch(url(path)).0 }
    package func bytes(_ file: File) -> HubBytes { HubBytes(url: url(file.path), size: file.size) }
}

package struct HubBytes: ByteSource {
    let url: URL
    package let size: UInt64
    package func read(at offset: UInt64, count: Int) throws -> Data {
        guard count > 0, offset < size else { return Data() }
        let last = min(size - 1, offset + UInt64(count) - 1)
        let (data, response) = try HubRepo.fetch(url, range: offset...last)
        // A server that ignores Range sends the whole file; take the part asked for.
        if response.statusCode == 206 { return data }
        guard data.count > Int(offset) else {
            throw EngineError("\(url.absoluteString) ended before byte \(offset)", code: "hub_unavailable")
        }
        return data.subdata(in: Int(offset)..<min(data.count, Int(last) + 1))
    }
}

/// Whether Origami runs a checkpoint, judged from its headers alone (config.json, safetensors index and headers, or
/// GGUF metadata and tensor table): its family and geometry, its tensor types by role, and a verdict. A refusal says
/// what differs; nothing is refused for a repo name, a publisher's naming or shard layout, or a type some kernel reads.
public struct CompatibilityReport: Codable, Sendable {
    public enum Verdict: String, Codable, Sendable { case runs, converts, refused }
    public let source: String
    public let format: String
    public let family: ModelFamily?
    public let architecture: String
    public let geometry: ModelGeometry?
    /// Tensor types by role ("Q4_K": 129 …), and bytes by role.
    public let types: [String: [String: Int]]
    public let bytes: [String: Int64]
    public let totalBytes: Int64
    public let verdict: Verdict
    public let reason: String
}

public enum ModelCompatibility {
    /// ggml types the family engines read for matrices (dense and experts) on the GPU.
    package static let matrixTypes = GGMLMatrixKernels.types
    /// Families with a serving engine in this build (GLM and DeepSeek through the family engines, from GGUF files).
    package static let servedFamilies: Set<ModelFamily> = [.qwen, .glm, .deepseek]

    /// A GGUF model (all its parts), local or remote.
    package static func check(gguf: GGUFSource, source: String) -> CompatibilityReport {
        var types: [String: [String: Int]] = [:]
        var bytes: [String: Int64] = [:]
        var total: Int64 = 0
        for tensor in gguf.tensors.values {
            let role = TensorRole.of(tensor.name).rawValue
            types[role, default: [:]][GGUFQuant.name(tensor.type), default: 0] += 1
            bytes[role, default: 0] += max(0, tensor.length)
            total += max(0, tensor.length)
        }
        var architecture = "missing"
        if case .string(let value)? = gguf.metadata["general.architecture"] { architecture = value }
        func report(_ family: ModelFamily?, _ geometry: ModelGeometry?, _ verdict: CompatibilityReport.Verdict, _ reason: String)
            -> CompatibilityReport
        {
            CompatibilityReport(
                source: source, format: "gguf", family: family, architecture: architecture, geometry: geometry,
                types: types, bytes: bytes, totalBytes: total, verdict: verdict, reason: reason)
        }
        if case .string("adapter")? = gguf.metadata["general.type"] {
            return report(
                ModelFamily.detect(gguf: gguf.metadata), nil, .refused, "it is a LoRA adapter, not a model; Origami does not apply adapters"
            )
        }
        let geometry: ModelGeometry
        do { geometry = try ModelGeometry(gguf: gguf) } catch {
            return report(ModelFamily.detect(gguf: gguf.metadata), nil, .refused, "\(error)")
        }
        if geometry.family == .qwen {
            if let problem = CheckpointConverter.ggufProblem(gguf) { return report(.qwen, geometry, .refused, problem) }
            return report(.qwen, geometry, .converts, "converts once into the 4-bit layout (origami model add)")
        }
        if let problem = familyProblem(gguf, geometry) { return report(geometry.family, geometry, .refused, problem) }
        guard servedFamilies.contains(geometry.family) else {
            return report(
                geometry.family, geometry, .refused,
                "every tensor has a kernel, but this build has no \(geometry.family.displayName) engine yet")
        }
        return report(geometry.family, geometry, .runs, "runs as stored")
    }

    /// What stops a GLM or DeepSeek GGUF: a missing part of the model, or a matrix type no kernel reads.
    package static func familyProblem(_ gguf: GGUFSource, _ geometry: ModelGeometry) -> String? {
        let names = Set(gguf.tensors.keys)
        // Types first: a file in a format no kernel reads is refused for that, not for a tensor its format names otherwise.
        // Integer tensors are data (hash tables, Engram metadata, FP8 bytes of a lookup table); as a matrix they are not.
        let integers: Set<Int> = [24, 25, 26, 27]
        let matrices: Set<TensorRole> = [.experts, .sharedExpert, .denseFFN, .recurrent, .attention, .output, .embeddings]
        // A matrix needs a GPU kernel; other tensors are decoded on the CPU.
        let unread = gguf.tensors.values.filter { tensor in
            let role = TensorRole.of(tensor.name)
            guard ![.mtp, .vision].contains(role), !integers.contains(tensor.type) else { return false }
            return matrices.contains(role) && tensor.shape.count >= 2
                ? !matrixTypes.contains(tensor.type) : !GGUFQuant.readable.contains(tensor.type) && !matrixTypes.contains(tensor.type)
        }
        if let first = unread.min(by: { $0.name < $1.name }) {
            let kinds = Set(unread.map { GGUFQuant.name($0.type) }).sorted().joined(separator: ", ")
            return "\(unread.count) tensors are \(kinds), which no Origami kernel reads (first: \(first.name))"
        }
        if let raw = gguf.tensors.values.first(where: {
            integers.contains($0.type) && matrices.contains(TensorRole.of($0.name)) && $0.shape.count >= 2
        }) {
            return
                "its matrices are stored as raw \(GGUFQuant.name(raw.type)) (FP8 bytes without a ggml type, such as \(raw.name)), which no kernel reads"
        }
        let present = Set(
            names.compactMap { name -> Int? in
                name.hasPrefix("blk.") ? name.split(separator: ".").dropFirst().first.flatMap { Int($0) } : nil
            })
        if let gap = (0..<geometry.layers).first(where: { !present.contains($0) }) {
            return
                "it holds tensors for \(present.filter { $0 < geometry.layers }.count) of the model's \(geometry.layers) layers (none for layer \(gap)): "
                + "part of a model split by layer, or an incomplete file"
        }
        for layer in geometry.moeLayers {
            for part in ["gate", "up", "down"] where !names.contains("blk.\(layer).ffn_\(part)_exps.weight") {
                return "layer \(layer) has no routed experts (blk.\(layer).ffn_\(part)_exps.weight)"
            }
        }
        if !names.contains("token_embd.weight") { return "it has no token embeddings" }
        if !names.contains("output.weight") { return "it has no output head" }
        // One Engram table for every engram layer (Unsloth's) or one per layer (vcruz305's ggml rows, DwarfStar's I8 rows).
        if geometry.family == .deepseek, !names.contains("per_layer_token_embd.weight"),
            let layer = geometry.lists["engram_layer_ids"]?.first(where: { !names.contains("blk.\($0).engram_embd.weight") })
        {
            return "it has no Engram table (per_layer_token_embd.weight or blk.\(layer).engram_embd.weight)"
        }
        return nil
    }

    /// A safetensors checkpoint from its config.json and headers (`headers` maps tensor names to spans).
    package static func check(config: [String: Any], headers: [String: TensorSpan], source: String) -> CompatibilityReport {
        var types: [String: [String: Int]] = [:]
        var bytes: [String: Int64] = [:]
        var total: Int64 = 0
        for (name, span) in headers {
            let role = TensorRole.of(name).rawValue
            types[role, default: [:]][span.dtype, default: 0] += 1
            bytes[role, default: 0] += span.length
            total += span.length
        }
        let text = config["text_config"] as? [String: Any] ?? config
        let architecture = (config["model_type"] ?? text["model_type"]).map { "\($0)" } ?? "missing"
        func report(_ family: ModelFamily?, _ geometry: ModelGeometry?, _ verdict: CompatibilityReport.Verdict, _ reason: String)
            -> CompatibilityReport
        {
            CompatibilityReport(
                source: source, format: "safetensors", family: family, architecture: architecture, geometry: geometry,
                types: types, bytes: bytes, totalBytes: total, verdict: verdict, reason: reason)
        }
        let geometry: ModelGeometry
        do { geometry = try ModelGeometry(config: config) } catch {
            return report(ModelFamily.detect(config: config), nil, .refused, "\(error)")
        }
        let quantization = (config["quantization_config"] ?? config["quantization"]) as? [String: Any]
        if geometry.family == .qwen {
            do {
                _ = try ModelContract.validate(config: ModelSource.configJSON(config))
                return report(.qwen, geometry, .runs, "runs as stored (MLX 4-bit)")
            } catch {
                let text = "\(error)"
                return report(.qwen, geometry, text.contains("origami model add") ? .converts : .refused, text)
            }
        }
        // A new family's safetensors: what its quantization is, and whether a reader for it exists.
        let format = describe(quantization: quantization, headers: headers)
        if let refusal = format.refusal { return report(geometry.family, geometry, .refused, refusal) }
        // FamilyConverter reads GLM-5.3-Flash in FP8 with block scales or unquantized.
        let method = (quantization?["quant_method"] as? String)?.lowercased()
        let unread = Set(headers.values.map(\.dtype)).subtracting(["F8_E4M3", "BF16", "F16", "F32"]).sorted()
        let readable = method == "fp8" || (method == nil && quantization?["bits"] == nil)
        if geometry.family == .glm, readable, unread.isEmpty {
            return report(
                .glm, geometry, .converts,
                "\(format.name): converts once into a GGUF with 2-bit experts (origami model add; --expert-bits 4 or 8 for Q4_K or Q8_0)")
        }
        guard geometry.family == .glm else {
            return report(
                geometry.family, geometry, .refused,
                "\(format.name): no \(geometry.family.displayName) safetensors converter yet; its GGUF files run as stored")
        }
        return report(
            .glm, geometry, .refused,
            (readable ? "\(format.name) with \(unread.joined(separator: ", ")) tensors" : format.name)
                + ": the GLM-5.3-Flash converter reads FP8 (E4M3 with block scales), BF16, F16 and F32 weights")
    }

    /// A safetensors checkpoint's weight encoding, and a refusal when no Origami reader exists for it.
    static func describe(quantization: [String: Any]?, headers: [String: TensorSpan]) -> (name: String, refusal: String?) {
        let method = (quantization?["quant_method"] as? String)?.lowercased()
        let dtypes = Set(headers.values.map(\.dtype))
        switch method {
        case nil:
            if quantization?["bits"] != nil {
                return ("MLX \(quantization?["bits"] ?? "?")-bit group \(quantization?["group_size"] ?? "?")", nil)
            }
            return (dtypes.contains("F8_E4M3") ? "FP8 (no config block)" : "unquantized (\(dtypes.sorted().joined(separator: ", ")))", nil)
        case "fp8": return ("FP8 E4M3 with block scales", nil)
        case "modelopt": return ("NVIDIA ModelOpt (\(quantization?["quant_algo"] ?? "FP4"))", nil)
        case "compressed-tensors": return ("compressed-tensors (\(quantization?["format"] ?? "pack-quantized"))", nil)
        case "mxfp4": return ("MXFP4", nil)
        case "exl3", "exl2":
            return (
                method!.uppercased(),
                "\(method!.uppercased()) stores ExLlama's trellis codes, which only ExLlama's CUDA kernels decode; no Metal reader exists"
            )
        case "gptq", "awq", "hqq", "bitsandbytes", "aqlm", "quip", "vptq":
            return (method!.uppercased(), "\(method!.uppercased()) weights need a converter Origami does not have yet")
        default:
            return (method!, "the quantization method \(method!) is unknown to Origami")
        }
    }

    // MARK: where the headers come from

    /// Why `repo`, judged by `reports` (`check(hub:)`), is not downloaded by `model add` or `setup`, in the dashboard's
    /// words; nil for one model Origami runs or converts. A repo of GGUF files or of several checkpoints holds several
    /// models: one is downloaded by hand.
    package static func downloadRefusal(_ repo: String, reports: [CompatibilityReport]) -> String? {
        guard let first = reports.first else { return "\(repo) holds no safetensors or GGUF model" }
        if reports.contains(where: { $0.format.lowercased().contains("gguf") }) {
            return "\(repo) holds GGUF files: download the one you want, then add it from its folder"
        }
        guard reports.count == 1 else {
            let folders = reports.map { $0.source == repo ? "the top folder" : String($0.source.dropFirst(repo.count + 1)) }
            return "\(repo) holds \(reports.count) safetensors checkpoints (\(folders.joined(separator: ", "))): download the "
                + "folder of the one you want, then add it from that folder (`origami model add <folder>`)"
        }
        return first.family == nil || first.verdict == .refused ? "Origami cannot use \(repo): \(first.reason)" : nil
    }

    /// Every model in a Hugging Face repo, each judged from its headers: one report per GGUF model (a file or a split
    /// set, grouped by folder and name) and per safetensors checkpoint (a folder with config.json). Vision
    /// projectors, MTP heads and importance matrices beside them are not models.
    package static func check(hub repo: HubRepo) throws -> [CompatibilityReport] {
        let files = try repo.files()
        var reports: [CompatibilityReport] = []
        func folder(_ path: String) -> String { path.contains("/") ? String(path[..<path.lastIndex(of: "/")!]) : "" }
        func name(_ path: String) -> String { String(path.split(separator: "/").last ?? "") }
        let ggufs = files.filter { file in
            let n = name(file.path).lowercased()
            return n.hasSuffix(".gguf") && !n.hasPrefix("mmproj") && !n.hasPrefix("mtp") && !n.contains("imatrix")
        }
        let groups = Dictionary(grouping: ggufs) { file in folder(file.path) + "/" + GGUFSource.modelPrefix(name(file.path)) }
        for (key, parts) in groups.sorted(by: { $0.key < $1.key }) {
            let ordered = parts.sorted { $0.path < $1.path }
            let source = "\(repo.repo)/\(key.hasPrefix("/") ? String(key.dropFirst()) : key)"
            do {
                let gguf = try GGUFSource(
                    folder: repo.url(folder(ordered[0].path)), parts: ordered.map { (name($0.path), repo.bytes($0) as any ByteSource) })
                reports.append(check(gguf: gguf, source: source))
            } catch {
                reports.append(
                    CompatibilityReport(
                        source: source, format: "gguf", family: nil, architecture: "?", geometry: nil, types: [:], bytes: [:],
                        totalBytes: Int64(ordered.reduce(0) { $0 + $1.size }), verdict: .refused, reason: "\(error)"))
            }
        }
        for config in files where name(config.path) == "config.json" {
            let base = folder(config.path)
            let prefix = base.isEmpty ? "" : base + "/"
            let shards = files.filter { folder($0.path) == base && $0.path.hasSuffix(".safetensors") }
            guard !shards.isEmpty else { continue }
            let source = repo.repo + (base.isEmpty ? "" : "/" + base)
            do {
                guard let json = try JSONSerialization.jsonObject(with: repo.data(prefix + "config.json")) as? [String: Any] else {
                    throw EngineError("config.json is not JSON")
                }
                var headers: [String: TensorSpan] = [:]
                for shard in shards {
                    for (name, span) in try ModelSource.header(of: repo.bytes(shard), file: name(shard.path)) { headers[name] = span }
                }
                reports.append(check(config: json, headers: headers, source: source))
            } catch {
                reports.append(
                    CompatibilityReport(
                        source: source, format: "safetensors", family: nil, architecture: "?", geometry: nil, types: [:], bytes: [:],
                        totalBytes: Int64(shards.reduce(0) { $0 + $1.size }), verdict: .refused, reason: "\(error)"))
            }
        }
        return reports
    }

    /// A local model: a .gguf file (with its split siblings) or a folder holding one GGUF model or a safetensors checkpoint.
    package static func check(local url: URL) throws -> CompatibilityReport {
        var isFolder: ObjCBool = false
        FileManager.default.fileExists(atPath: url.path, isDirectory: &isFolder)
        let config = url.appendingPathComponent("config.json")
        let shards =
            isFolder.boolValue
            ? (try FileManager.default.contentsOfDirectory(atPath: url.path)).filter { $0.hasSuffix(".safetensors") } : []
        // A config.json without shards (beside GGUF files, or weights not downloaded) is not a checkpoint to judge.
        if !shards.isEmpty, FileManager.default.fileExists(atPath: config.path) {
            guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: config)) as? [String: Any] else {
                throw EngineError("\(config.path) is not JSON", code: "model_incompatible")
            }
            var headers: [String: TensorSpan] = [:]
            for shard in shards {
                for (name, span) in try ModelSource.header(of: try LocalBytes(url.appendingPathComponent(shard)), file: shard) {
                    headers[name] = span
                }
            }
            return check(config: json, headers: headers, source: url.path)
        }
        return check(gguf: try GGUFSource(url), source: url.path)
    }
}
