import Foundation

/// llama.cpp's GGUF format, read for conversion: one file or a split set (`…-00001-of-00004.gguf`). Keeps scalar
/// and short-array metadata and where every tensor lies; long arrays are skipped unless asked for (`tokenizer:`,
/// `arrays:`).
package struct GGUFSource: Sendable {
    /// A metadata value as stored (GGUF types 0-12; arrays of up to 1,024 elements, or longer ones asked for).
    package enum Value: Sendable, Equatable {
        case int(Int64)
        case uint(UInt64)
        case float(Double)
        case bool(Bool)
        case string(String)
        case array([Value])
        package var integer: Int64? {
            switch self {
            case .int(let v): return v
            case .uint(let v): return Int64(exactly: v)
            default: return nil
            }
        }
        package var integers: [Int64]? {
            if case .array(let values) = self { return values.compactMap(\.integer) }
            return nil
        }
    }
    package struct Tensor: Sendable {
        /// Shape in PyTorch order (GGUF lists the fastest dimension first).
        package let name: String, shape: [Int], type: Int, file: String
        package let offset, length: Int64
    }
    package let folder: URL
    package let files: [String]
    package private(set) var metadata: [String: Value] = [:]
    /// The tokenizer's long arrays (tokens, merges, token types), kept only when asked for (`tokenizer: true`).
    package private(set) var tokenizer: [String: [Value]] = [:]
    private var keepTokenizer = false
    /// Long arrays kept by key suffix (`arrays:`), such as DeepSeek's Engram token map.
    package private(set) var arrays: [String: [Value]] = [:]
    private var keepArrays: Set<String> = []
    package private(set) var tensors: [String: Tensor] = [:]
    /// Bytes of all parts.
    package private(set) var bytes: UInt64 = 0
    /// A low-bit variant's experts file parts (absolute paths; `withExperts`); empty for a GGUF as stored.
    package private(set) var expertFiles: [String] = []

    /// Where a tensor's `file` is: one of `files` in `folder`, or an experts file's absolute path (`withExperts`).
    package func url(_ file: String) -> URL { file.hasPrefix("/") ? URL(fileURLWithPath: file) : folder.appendingPathComponent(file) }

    /// This GGUF with the routed-expert banks (`blk.L.ffn_{gate,up,down}_exps.weight`) of `layers` taken from `experts`,
    /// everything else as stored here (`FamilyExpertVariant` checks the pairing).
    package func withExperts(from experts: GGUFSource, layers: [Int]) -> GGUFSource {
        var merged = self
        for layer in layers {
            for part in ["gate", "up", "down"] {
                let name = "blk.\(layer).ffn_\(part)_exps.weight"
                guard let t = experts.tensors[name] else { continue }
                merged.tensors[name] = Tensor(
                    name: name, shape: t.shape, type: t.type, file: experts.url(t.file).standardizedFileURL.path, offset: t.offset,
                    length: t.length)
            }
        }
        merged.expertFiles = experts.files.map { experts.url($0).standardizedFileURL.path }
        return merged
    }

    /// `url`: a .gguf file (its split siblings are found) or a folder holding one model's parts. Vision projectors
    /// (`mmproj…`) and MTP heads (`mtp…`) beside it are not part of the language model. `partial`: the parts present are
    /// enough (low-bit experts need only the part their banks are in).
    package init(_ url: URL, tokenizer: Bool = false, arrays: Set<String> = [], partial: Bool = false) throws {
        keepTokenizer = tokenizer
        keepArrays = arrays
        let parts = try Self.parts(url)
        folder = parts.folder
        files = parts.files
        for file in files { try read(file, from: try LocalBytes(folder.appendingPathComponent(file))) }
        // llama.cpp reads a file as split only when `split.count` is above 1 (some converters write 0 in a single file).
        if !partial, let count = metadata["split.count"]?.integer, max(count, 1) != Int64(files.count) {
            throw Self.refuse("\(Self.modelPrefix(files[0])) has \(files.count) of its \(count) parts")
        }
    }
    private static func refuse(_ why: String) -> EngineError {
        EngineError("Origami cannot read this GGUF: \(why)", code: "model_incompatible")
    }

    /// The files `init` reads for `url`: the folder and the model's parts in split order, from their names, and a part the
    /// names lack from the headers of the files named on their own beside it or in the folder above (`completed`).
    package static func parts(_ url: URL) throws -> (folder: URL, files: [String]) {
        var isFolder: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isFolder) else { throw refuse("\(url.path) does not exist") }
        let folder = isFolder.boolValue ? url : url.deletingLastPathComponent()
        func models(_ names: [String]) -> [String] {
            names.filter { $0.lowercased().hasSuffix(".gguf") && !$0.lowercased().hasPrefix("mmproj") && !$0.lowercased().hasPrefix("mtp") }
        }
        let all = models(try FileManager.default.contentsOfDirectory(atPath: folder.path))
        func set(_ model: String) -> [String] {
            let named = all.filter { modelPrefix($0) == model }.sorted()
            guard let count = named.first.flatMap({ splitName($0)?.count }), named.count < count,
                let parts = try? named.map({ ($0, try LocalBytes(folder.appendingPathComponent($0)) as any ByteSource) })
            else { return named }
            let above = models((try? FileManager.default.contentsOfDirectory(atPath: folder.deletingLastPathComponent().path)) ?? [])
            let others = (all + above.map { "../" + $0 }).filter { splitName($0) == nil }.compactMap { name in
                (try? LocalBytes(folder.appendingPathComponent(name))).map { (name, $0 as any ByteSource) }
            }
            return completed(parts, others: others).map(\.file)
        }
        var chosen = isFolder.boolValue ? Set(all.map(modelPrefix)) : [modelPrefix(url.lastPathComponent)]
        // A folder holding one split set and its part named on its own holds one model.
        let sets = Set(all.filter { splitName($0) != nil }.map(modelPrefix))
        if chosen.count > 1, sets.count == 1, let model = sets.first, Set(set(model)) == Set(all) { chosen = [model] }
        guard chosen.count == 1, let model = chosen.first else {
            throw refuse(
                chosen.isEmpty ? "\(folder.path) holds no model .gguf file" : "\(folder.path) holds several models; pass one .gguf file")
        }
        return (folder, set(model))
    }

    /// A split part's number and the set's count, from its name (`…-00002-of-00006.gguf` → 2, 6).
    package static func splitName(_ name: String) -> (no: Int, count: Int)? {
        guard let range = name.range(of: "-\\d+-of-\\d+\\.gguf$", options: [.regularExpression, .caseInsensitive]) else { return nil }
        let numbers = name[range].dropFirst().dropLast(5).components(separatedBy: "-of-").compactMap { Int($0) }
        return numbers.count == 2 ? (numbers[0], numbers[1]) : nil
    }

    /// A split set with the parts its names lack found among `others` by header: a file whose `split.no`, `split.count`
    /// and `split.tensors.count` (llama.cpp's split keys) make it a missing part of this set. ISTA-DASLab's DeepSeek builds
    /// share one Engram file, named on its own, as their last part. The set in split order; as given when none is missing.
    package static func completed(
        _ named: [(file: String, bytes: any ByteSource)], others: [(file: String, bytes: any ByteSource)]
    ) -> [(file: String, bytes: any ByteSource)] {
        guard let count = named.first.flatMap({ splitName($0.file)?.count }), named.count < count, !others.isEmpty,
            let tensors = try? splitPlace(named[named.count - 1].bytes)?.tensors
        else { return named }
        var byNumber = Dictionary(named.compactMap { part in splitName(part.file).map { ($0.no - 1, part) } }) { first, _ in first }
        for other in others {
            guard let place = try? splitPlace(other.bytes), place.count == count, place.tensors == tensors,
                (0..<Int64(count)).contains(place.no), byNumber[Int(place.no)] == nil
            else { continue }
            byNumber[Int(place.no)] = other
        }
        return byNumber.sorted { $0.key < $1.key }.map(\.value)
    }

    /// A file's own place in its split set, from its header's metadata alone; nil when it names none.
    static func splitPlace(_ source: any ByteSource) throws -> (no: Int64, count: Int64, tensors: Int64)? {
        let reader = GGUFReader(source)
        guard try reader.bytes(4) == Data("GGUF".utf8), [2, 3].contains(try reader.integer(UInt32.self)) else { return nil }
        _ = try reader.integer(UInt64.self)
        let keys = try reader.integer(UInt64.self)
        guard keys <= 1 << 16 else { return nil }
        var split: [String: Int64] = [:]
        for _ in 0..<keys where split.count < 3 {
            let key = try reader.string()
            if let value = try reader.value(try reader.integer(UInt32.self))?.integer, key.hasPrefix("split.") { split[key] = value }
        }
        guard let no = split["split.no"], let count = split["split.count"], let tensors = split["split.tensors.count"] else { return nil }
        return (no, count, tensors)
    }

    /// Header only, from wherever the parts are (a Hugging Face repo by HTTP range): metadata and tensor spans,
    /// no weights. `parts` are one model's files in split order.
    package init(folder: URL, parts: [(file: String, bytes: any ByteSource)]) throws {
        self.folder = folder
        files = parts.map(\.file)
        for part in parts { try read(part.file, from: part.bytes) }
        if let count = metadata["split.count"]?.integer, max(count, 1) != Int64(files.count) {
            throw EngineError("Origami cannot read this GGUF: it has \(files.count) of its \(count) parts", code: "model_incompatible")
        }
    }

    /// A split set's shared name (`…-00001-of-00004.gguf` or `…-01-of-04.gguf` → `…`); a single file is its own name.
    package static func modelPrefix(_ name: String) -> String {
        name.range(of: "-\\d+-of-\\d+\\.gguf$", options: [.regularExpression, .caseInsensitive]).map { String(name[..<$0.lowerBound]) }
            ?? name
    }

    private mutating func read(_ file: String, from source: any ByteSource) throws {
        let reader = GGUFReader(source)
        guard try reader.bytes(4) == Data("GGUF".utf8) else { throw EngineError("\(file) is not a GGUF file", code: "model_incompatible") }
        let version = try reader.integer(UInt32.self)
        guard version == 2 || version == 3 else {
            throw EngineError("\(file) is GGUF version \(version); Origami reads versions 2 and 3", code: "model_incompatible")
        }
        let count = try reader.integer(UInt64.self)
        let keys = try reader.integer(UInt64.self)
        // Far above any model's (DeepSeek's ~2,000 tensors, ~60 keys): a corrupt count would otherwise walk a large file's
        // weights as empty entries, for minutes.
        guard count <= 1 << 20, keys <= 1 << 16 else {
            throw EngineError("\(file) claims \(count) tensors and \(keys) metadata keys", code: "model_incompatible")
        }
        // The first part's metadata is the model's, as llama.cpp reads it: a later part (some are edited on their own,
        // e.g. a different general.architecture spelling) only adds keys the first lacks. Alignment is per file.
        var alignment: UInt64 = 32
        for _ in 0..<keys {
            let key = try reader.string()
            let type = try reader.integer(UInt32.self)
            if keepTokenizer && type == 9 && key.hasPrefix("tokenizer.ggml.") {
                let value = try reader.array()
                if tokenizer[key] == nil { tokenizer[key] = value }
                continue
            }
            if type == 9 && keepArrays.contains(where: { key.hasSuffix("." + $0) }) {
                let value = try reader.array()
                if arrays[key] == nil { arrays[key] = value }
                continue
            }
            guard let value = try reader.value(type) else { continue }
            if key == "general.alignment", let own = value.integer { alignment = UInt64(exactly: own) ?? 0 }
            if metadata[key] == nil { metadata[key] = value }
        }
        var infos: [(String, [Int], Int, UInt64)] = []
        for _ in 0..<count {
            let name = try reader.string(maximum: 256)  // ggml allows 63 bytes; some forks more
            let dimensions = try reader.integer(UInt32.self)
            // ggml's own limits: at most four dimensions, each below 2^63.
            guard dimensions <= 4 else { throw EngineError("\(file): \(name) has \(dimensions) dimensions", code: "model_incompatible") }
            let ne = try (0..<dimensions).map { _ -> Int in
                guard let size = Int(exactly: try reader.integer(UInt64.self)) else {
                    throw EngineError("\(file): \(name) has a dimension too large", code: "model_incompatible")
                }
                return size
            }
            infos.append((name, ne.reversed(), Int(try reader.integer(UInt32.self)), try reader.integer(UInt64.self)))
        }
        guard alignment > 0, alignment & (alignment - 1) == 0 else {
            throw EngineError("\(file): alignment \(alignment) is not a power of two", code: "model_incompatible")
        }
        let start = (reader.position + alignment - 1) / alignment * alignment
        let size = reader.size
        bytes += size
        // As ggml reads it: names unique across the parts, every offset aligned, no two tensors sharing bytes.
        var spans: [(start: UInt64, end: UInt64, name: String)] = []
        for (name, shape, type, offset) in infos {
            guard tensors[name] == nil else { throw EngineError("\(file): tensor \(name) is listed twice", code: "model_incompatible") }
            guard let values = ModelSource.elementCount(shape) else {
                throw EngineError("\(file): \(name) has too many elements", code: "model_incompatible")
            }
            var length: Int64 = -1
            if let quant = GGUFQuant.sizes[type] {
                // Rows are whole blocks (as ggml requires), so row and tensor sizes below are exact.
                if let row = shape.last, row % quant.block != 0 {
                    throw EngineError(
                        "\(file): \(name) has rows of \(row) values, not whole \(GGUFQuant.name(type)) blocks of \(quant.block)",
                        code: "model_incompatible")
                }
                let (total, overflow) = (values / quant.block).multipliedReportingOverflow(by: quant.bytes)
                guard !overflow else { throw EngineError("\(file): \(name) has too many elements", code: "model_incompatible") }
                length = Int64(total)
            }
            guard start <= size, offset <= size - start, length < 0 || UInt64(length) <= size - start - offset else {
                throw EngineError(
                    "\(file) is shorter than its header says (\(name) lies past the end of the file): the download is incomplete",
                    code: "model_incompatible")
            }
            guard offset % alignment == 0 else {
                throw EngineError("\(file): \(name) starts at \(offset), not a multiple of \(alignment)", code: "model_incompatible")
            }
            tensors[name] = Tensor(name: name, shape: shape, type: type, file: file, offset: Int64(start + offset), length: length)
            if length > 0 { spans.append((offset, offset + UInt64(length), name)) }
        }
        spans.sort { $0.start < $1.start }
        for (a, b) in zip(spans, spans.dropFirst()) where a.end > b.start {
            throw EngineError("\(file): tensors \(a.name) and \(b.name) overlap", code: "model_incompatible")
        }
    }
}

/// Sequential reads through a GGUF header: 256 KiB first (a later split part's whole header), doubling to 8 MiB.
private final class GGUFReader {
    private let source: any ByteSource
    private var buffer = Data(), base: UInt64 = 0, cursor = 0, chunk = 256 << 10
    /// The file's size (measured through a symlink): no string or array may claim more than remains.
    let size: UInt64
    var position: UInt64 { base + UInt64(cursor) }
    private var remaining: UInt64 { size > position ? size - position : 0 }
    init(_ source: any ByteSource) {
        self.source = source
        size = source.size
    }
    func bytes(_ count: Int) throws -> Data {
        while buffer.count - cursor < count {
            buffer = buffer.subdata(in: cursor..<buffer.count)
            base += UInt64(cursor)
            cursor = 0
            let at = base + UInt64(buffer.count)
            let more = at < source.size ? try source.read(at: at, count: Int(min(UInt64(max(count, chunk)), source.size - at))) : Data()
            guard !more.isEmpty else { throw EngineError("GGUF header ends early", code: "model_incompatible") }
            buffer.append(more)
            chunk = min(chunk * 2, 8 << 20)
        }
        defer { cursor += count }
        return buffer.subdata(in: cursor..<(cursor + count))
    }
    /// Moves past `count` bytes without reading the ones not yet buffered (a long array over HTTP).
    func skip(_ count: Int) {
        let buffered = buffer.count - cursor
        if count <= buffered {
            cursor += count
            return
        }
        base += UInt64(buffer.count) + UInt64(count - buffered)
        buffer = Data()
        cursor = 0
    }
    func integer<T: FixedWidthInteger>(_ type: T.Type) throws -> T {
        try bytes(MemoryLayout<T>.size).withUnsafeBytes { T(littleEndian: $0.loadUnaligned(as: T.self)) }
    }
    func string(maximum: UInt64 = 1 << 24) throws -> String {
        let length = try integer(UInt64.self)
        // 16 MiB: far above any chat template or config, so a corrupt length cannot make one string take gigabytes.
        guard length <= maximum, length <= remaining else {
            throw EngineError("GGUF string of \(length) bytes", code: "model_incompatible")
        }
        return String(decoding: try bytes(Int(length)), as: UTF8.self)
    }
    /// A whole array, however long (the tokenizer's).
    func array() throws -> [GGUFSource.Value] {
        let element = try integer(UInt32.self)
        let count = try integer(UInt64.self)
        guard element != 9 else { throw EngineError("GGUF nested arrays are not supported", code: "model_incompatible") }
        // Every element takes at least one byte (a string its 8-byte length), so the file bounds the count.
        guard count <= Self.arrayLimit, count <= remaining / UInt64(element == 8 ? 8 : Self.widths[element] ?? 1) else {
            throw EngineError("GGUF array of \(count) values", code: "model_incompatible")
        }
        return try (0..<count).compactMap { _ in try value(element) }
    }
    /// A metadata value; nil for an array too long to be worth keeping (it is read past).
    func value(_ type: UInt32) throws -> GGUFSource.Value? {
        switch type {
        case 0: return .uint(UInt64(try integer(UInt8.self)))
        case 1: return .int(Int64(try integer(Int8.self)))
        case 2: return .uint(UInt64(try integer(UInt16.self)))
        case 3: return .int(Int64(try integer(Int16.self)))
        case 4: return .uint(UInt64(try integer(UInt32.self)))
        case 5: return .int(Int64(try integer(Int32.self)))
        case 6: return .float(Double(Float(bitPattern: try integer(UInt32.self))))
        case 7: return .bool(try integer(UInt8.self) != 0)
        case 8: return .string(try string())
        case 9:
            let element = try integer(UInt32.self)
            let count = try integer(UInt64.self)
            // ggml refuses nested arrays too; recursing into them would let a small file exhaust the stack.
            guard element != 9 else { throw EngineError("GGUF nested arrays are not supported", code: "model_incompatible") }
            guard count <= 1024 else {
                // The tokenizer's vocabulary and merges: read past them.
                if let width = Self.widths[element] {
                    guard count <= remaining / UInt64(width) else {
                        throw EngineError("GGUF array larger than the file", code: "model_incompatible")
                    }
                    skip(Int(count) * width)
                } else {
                    guard count <= Self.arrayLimit else { throw EngineError("GGUF array of \(count) values", code: "model_incompatible") }
                    for _ in 0..<count { _ = try value(element) }
                }
                return nil
            }
            return .array(try (0..<count).compactMap { _ in try value(element) })
        case 10: return .uint(try integer(UInt64.self))
        case 11: return .int(try integer(Int64.self))
        case 12: return .float(Double(bitPattern: try integer(UInt64.self)))
        default: throw EngineError("GGUF metadata type \(type) is unknown", code: "model_incompatible")
        }
    }
    /// The most elements an array of strings may have: sixteen times the largest vocabulary or merge list, so a corrupt
    /// count cannot walk gigabytes of weights as empty strings.
    static let arrayLimit: UInt64 = 1 << 22
    /// Bytes of each fixed-width metadata type.
    static let widths: [UInt32: Int] = [0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8]
}

/// ggml's tensor encodings: block sizes, names, and dequantization for the ones Origami reads, written to compute
/// exactly what llama.cpp's gguf-py does (`engine-checks` `gguf_dequantize` compares them bit for bit).
package enum GGUFQuant {
    /// Values and bytes per block, for every type ggml defines.
    package static let sizes: [Int: (block: Int, bytes: Int)] = [
        0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
        10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
        16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
        24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2), 34: (256, 54), 35: (256, 66),
        39: (32, 17), 40: (64, 36), 41: (128, 18), 42: (64, 18), 43: (32, 33),
    ]
    package static let names: [Int: String] = [
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
        13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
        22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
        35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0", 43: "MXFP8",
    ]
    /// The types Origami dequantizes; the others (ternary, Q1_0, the intermediate Q8_1/Q8_K, integers) are refused by name.
    package static let readable: Set<Int> = [0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 30, 39, 42, 43]
    package static func name(_ type: Int) -> String { names[type] ?? "type \(type)" }

    /// OCP E4M3 (fn): no infinities, 0x7F/0xFF are NaN, exponent bias 7, subnormals below.
    static func e4m3(_ b: UInt8) -> Float {
        let sign: Float = b & 0x80 != 0 ? -1 : 1
        let exponent = Int((b >> 3) & 15)
        let mantissa = Float(b & 7)
        if b & 0x7F == 0x7F { return .nan }
        return sign
            * (exponent == 0 ? mantissa / 8 * 0x1p-6 : (1 + mantissa / 8) * Float(sign: .plus, exponent: exponent - 7, significand: 1))
    }
    static let iq4nl: [Float] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
    static let e2m1: [Float] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]

    /// Decodes `count` values (whole blocks) of `type` from `source` into `out`.
    package static func dequantize(_ source: UnsafeRawPointer, type: Int, count: Int, into out: UnsafeMutablePointer<Float>) {
        func byte(_ offset: Int) -> UInt8 { source.load(fromByteOffset: offset, as: UInt8.self) }
        func half(_ offset: Int) -> Float { Float(Float16(bitPattern: source.loadUnaligned(fromByteOffset: offset, as: UInt16.self))) }
        func word(_ offset: Int) -> UInt32 { source.loadUnaligned(fromByteOffset: offset, as: UInt32.self) }
        func short(_ offset: Int) -> UInt16 { source.loadUnaligned(fromByteOffset: offset, as: UInt16.self) }
        func sign(_ bits: UInt8, _ j: Int) -> Float { (bits >> UInt8(j)) & 1 == 0 ? 1 : -1 }
        switch type {
        case 0: for i in 0..<count { out[i] = source.loadUnaligned(fromByteOffset: i * 4, as: Float.self) }
        case 1: for i in 0..<count { out[i] = half(i * 2) }
        case 30:
            for i in 0..<count { out[i] = Float(bitPattern: UInt32(source.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)) << 16) }
        default: break
        }
        guard let (block, bytes) = sizes[type], block > 1 else { return }
        for b in 0..<(count / block) {
            let at = b * bytes
            let o = out + b * block
            switch type {
            case 2, 3:  // Q4_0, Q4_1: d (and m), 16 bytes of nibbles: low nibbles are values 0-15, high 16-31
                let d = half(at)
                let m: Float = type == 3 ? half(at + 2) : 0
                let qs = at + (type == 3 ? 4 : 2)
                for j in 0..<16 {
                    let q = byte(qs + j)
                    if type == 2 {
                        o[j] = d * Float(Int(q & 15) - 8)
                        o[j + 16] = d * Float(Int(q >> 4) - 8)
                    } else {
                        o[j] = d * Float(q & 15) + m
                        o[j + 16] = d * Float(q >> 4) + m
                    }
                }
            case 6, 7:  // Q5_0, Q5_1: d (and m), 32 high bits, 16 bytes of low nibbles
                let d = half(at)
                let m: Float = type == 7 ? half(at + 2) : 0
                let hi = word(at + (type == 7 ? 4 : 2))
                let qs = at + (type == 7 ? 8 : 6)
                for j in 0..<32 {
                    let low = j < 16 ? byte(qs + j) & 15 : byte(qs + j - 16) >> 4
                    let q = Int(low) | Int((hi >> UInt32(j)) & 1) << 4
                    o[j] = type == 6 ? d * Float(q - 16) : d * Float(q) + m
                }
            case 8:  // Q8_0
                let d = half(at)
                for j in 0..<32 { o[j] = Float(Int8(bitPattern: byte(at + 2 + j))) * d }
            case 42:  // Q2_0: d, then 2-bit codes four to a byte from the low bits; value (code - 1) * d
                let d = half(at)
                for j in 0..<64 { o[j] = Float(Int((byte(at + 2 + j / 4) >> UInt8(2 * (j % 4))) & 3) - 1) * d }
            case 10:  // Q2_K: 16 scale bytes, 64 bytes of 2-bit codes, d, dmin
                let d = half(at + 80)
                let dmin = half(at + 82)
                for p in 0..<256 {
                    let group = p / 16
                    let scale = byte(at + group)
                    let q = Float((byte(at + 16 + (p / 128) * 32 + p % 32) >> UInt8(2 * ((p % 128) / 32))) & 3)
                    o[p] = (d * Float(scale & 15)) * q - dmin * Float(scale >> 4)
                }
            case 11:  // Q3_K: 32 bytes of high-bit mask, 64 bytes of 2-bit codes, 12 bytes of 6-bit scales, d
                let d = half(at + 108)
                var scales = [Float](repeating: 0, count: 16)
                for i in 0..<16 {
                    let low = (byte(at + 96 + i % 8) >> UInt8(4 * (i / 8))) & 15
                    let high = (byte(at + 104 + i % 4) >> UInt8(2 * (i / 4))) & 3
                    scales[i] = d * Float(Int8(bitPattern: low | high << 4) - 32)
                }
                for p in 0..<256 {
                    let low = Int((byte(at + 32 + (p / 128) * 32 + p % 32) >> UInt8(2 * ((p % 128) / 32))) & 3)
                    let mask = Int((byte(at + p % 32) >> UInt8(p / 32)) & 1) ^ 1
                    o[p] = scales[p / 16] * Float(low - (mask << 2))
                }
            case 12, 13:  // Q4_K, Q5_K: d, dmin, 12 bytes of 6-bit scales and mins, (Q5_K: 32 bytes of fifth bits,) nibbles
                let d = half(at)
                let dmin = half(at + 2)
                let packed = at + 4
                func sc(_ i: Int) -> (UInt8, UInt8) {
                    i < 4
                        ? (byte(packed + i) & 63, byte(packed + 4 + i) & 63)
                        : (
                            (byte(packed + 4 + i) & 15) | ((byte(packed + i - 4) >> 2) & 0x30),
                            (byte(packed + 4 + i) >> 4) | ((byte(packed + i) >> 2) & 0x30)
                        )
                }
                let qh = at + 16
                let qs = at + (type == 13 ? 48 : 16)
                for g in 0..<8 {
                    let (s, m) = sc(g)
                    let scale = d * Float(s)
                    let minimum = dmin * Float(m)
                    for j in 0..<32 {
                        var q = Int((byte(qs + (g / 2) * 32 + j) >> UInt8(4 * (g % 2))) & 15)
                        if type == 13 { q |= Int((byte(qh + j) >> UInt8(g)) & 1) << 4 }
                        o[g * 32 + j] = scale * Float(q) - minimum
                    }
                }
            case 14:  // Q6_K: 128 bytes of low nibbles, 64 bytes of high 2-bit pairs, 16 int8 scales, d
                let d = half(at + 208)
                for p in 0..<256 {
                    let c = p / 128
                    let within = p % 128
                    let low = (byte(at + c * 64 + within % 64) >> UInt8(4 * (within / 64))) & 15
                    let high = (byte(at + 128 + c * 32 + within % 32) >> UInt8(2 * (within / 32))) & 3
                    o[p] = (d * Float(Int8(bitPattern: byte(at + 192 + p / 16)))) * Float(Int(low | high << 4) - 32)
                }
            case 20:  // IQ4_NL: d, 16 bytes of nibbles into a fixed codebook
                let d = half(at)
                for j in 0..<16 {
                    o[j] = d * iq4nl[Int(byte(at + 2 + j) & 15)]
                    o[j + 16] = d * iq4nl[Int(byte(at + 2 + j) >> 4)]
                }
            case 23:  // IQ4_XS: d, 16 high scale bits, 4 bytes of low scale nibbles, 128 bytes of nibbles
                let d = half(at)
                let high = UInt16(byte(at + 2)) | UInt16(byte(at + 3)) << 8
                for g in 0..<8 {
                    let low = (byte(at + 4 + g / 2) >> UInt8(4 * (g % 2))) & 15
                    let top = UInt8((high >> UInt16(2 * g)) & 3)
                    let scale = d * Float(Int8(bitPattern: low | top << 4) - 32)
                    for t in 0..<32 { o[g * 32 + t] = scale * iq4nl[Int((byte(at + 8 + g * 16 + t % 16) >> UInt8(4 * (t / 16))) & 15)] }
                }
            case 16:  // IQ2_XXS: d; per 32 values a word of four grid indices, then four 7-bit sign indices and a 4-bit scale
                let d = half(at)
                for ib in 0..<8 {
                    let aux = word(at + 6 + 8 * ib)
                    let db = d * (0.5 + Float(aux >> 28)) * 0.25
                    for l in 0..<4 {
                        let entry = Int(byte(at + 2 + 8 * ib + l)) * 8
                        let signs = ksigns[Int((aux >> UInt32(7 * l)) & 127)]
                        for j in 0..<8 { o[32 * ib + 8 * l + j] = db * Float(iq2xxsGrid[entry + j]) * sign(signs, j) }
                    }
                }
            case 17, 22:  // IQ2_XS: d, 32 words of a 9-bit grid index and a 7-bit sign index, 16 4-bit scales.
                // IQ2_S: d, 32 low index bytes, 32 sign bytes, 2 high index bits per entry, 16 4-bit scales.
                let d = half(at)
                let grid = type == 17 ? iq2xsGrid : iq2sGrid
                for i in 0..<32 {
                    let db = d * (0.5 + Float((byte(at + (type == 17 ? 66 : 74) + i / 4) >> UInt8(4 * (i / 2 % 2))) & 15)) * 0.25
                    let entry: Int
                    let signs: UInt8
                    if type == 17 {
                        let q = short(at + 2 + 2 * i)
                        entry = Int(q & 511) * 8
                        signs = ksigns[Int(q >> 9)]
                    } else {
                        entry = (Int(byte(at + 2 + i)) | Int((byte(at + 66 + i / 4) >> UInt8(2 * (i % 4))) & 3) << 8) * 8
                        signs = byte(at + 34 + i)
                    }
                    for j in 0..<8 { o[8 * i + j] = db * Float(grid[entry + j]) * sign(signs, j) }
                }
            case 18:  // IQ3_XXS: d, 64 indices of four-value grid entries; per 32 values a word of four 7-bit sign indices and a 4-bit scale
                let d = half(at)
                for ib in 0..<8 {
                    let aux = word(at + 66 + 4 * ib)
                    let db = d * (0.5 + Float(aux >> 28)) * 0.5
                    for t in 0..<32 {
                        let signs = ksigns[Int((aux >> UInt32(7 * (t / 8))) & 127)]
                        o[32 * ib + t] = db * Float(iq3xxsGrid[Int(byte(at + 2 + 8 * ib + t / 4)) * 4 + t % 4]) * sign(signs, t % 8)
                    }
                }
            case 21:  // IQ3_S: d, 64 low index bytes, 64 high index bits, 256 sign bits, 8 4-bit scales (odd multiples of d)
                let d = half(at)
                for p in 0..<256 {
                    let e = p / 4
                    let s = (byte(at + 106 + p / 64) >> UInt8(4 * (p / 32 % 2))) & 15
                    let entry = Int(byte(at + 2 + e)) | Int((byte(at + 66 + e / 8) >> UInt8(e % 8)) & 1) << 8
                    o[p] = d * Float(1 + 2 * Int(s)) * Float(iq3sGrid[entry * 4 + p % 4]) * sign(byte(at + 74 + p / 8), p % 8)
                }
            case 19:  // IQ1_S: d, 32 low index bytes; per 32 values a word of four 3-bit high indices, a 3-bit scale, the delta's sign
                let d = half(at)
                for ib in 0..<8 {
                    let qh = short(at + 34 + 2 * ib)
                    let dl = d * Float(2 * Int((qh >> 12) & 7) + 1)
                    let delta: Float = qh & 0x8000 == 0 ? 0.125 : -0.125
                    for l in 0..<4 {
                        let entry = (Int(byte(at + 2 + 4 * ib + l)) | Int((qh >> UInt16(3 * l)) & 7) << 8) * 8
                        for j in 0..<8 { o[32 * ib + 8 * l + j] = dl * (Float(Int(iq1sGrid[entry + j]) - 1) + delta) }
                    }
                }
            case 29:  // IQ1_M: 32 low index bytes, per entry a nibble of 3 high index bits and the delta's sign, 16 3-bit scales,
                // and d spread over the top nibbles of the four scale words
                let d = Float(
                    Float16(
                        bitPattern: short(at + 48) >> 12 | (short(at + 50) >> 8) & 0xF0 | (short(at + 52) >> 4) & 0xF00 | short(at + 54)
                            & 0xF000))
                for i in 0..<32 {
                    let h = (byte(at + 32 + i / 2) >> UInt8(4 * (i % 2))) & 15
                    let k = i / 2
                    let dl = d * Float(2 * Int((short(at + 48 + 2 * (k / 4)) >> UInt16(3 * (k % 4))) & 7) + 1)
                    let delta: Float = h & 8 == 0 ? 0.125 : -0.125
                    let entry = (Int(byte(at + i)) | Int(h & 7) << 8) * 8
                    for j in 0..<8 { o[8 * i + j] = dl * (Float(Int(iq1sGrid[entry + j]) - 1) + delta) }
                }
            case 43:  // MXFP8: an E8M0 scale and 32 E4M3 codes (ggml's dequantize_row_mxfp8; a scale of 0xFF is NaN)
                let e = UInt32(byte(at))
                let d = e == 0xFF ? Float.nan : Float(bitPattern: e == 0 ? 0x0040_0000 : e << 23)
                for j in 0..<32 { o[j] = e4m3(byte(at + 1 + j)) * d }
            case 39:  // MXFP4: an E8M0 scale (halved) and 16 bytes of E2M1 nibbles
                let e = UInt32(byte(at))
                let d = Float(bitPattern: e < 2 ? UInt32(0x0020_0000) << e : (e - 1) << 23)
                for j in 0..<16 {
                    o[j] = d * e2m1[Int(byte(at + 1 + j) & 15)]
                    o[j + 16] = d * e2m1[Int(byte(at + 1 + j) >> 4)]
                }
            default: return
            }
        }
    }
}
