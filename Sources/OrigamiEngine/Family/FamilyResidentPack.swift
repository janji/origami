import CryptoKit
import Foundation

/// Explicit, non-exact dense-weight experiment: target Q8_0 -> Q4_K.
/// Experts, embeddings, vocabulary heads, routers and vectors retain their source representation.
/// Packs are built separately; model files are read-only. Target verification remains the caller's job.
package final class FamilyResidentPack: @unchecked Sendable {
    package enum Conversion: String, Codable, Sendable {
        case targetQ4
        var sourceType: Int { 8 }
        package var targetType: Int { 12 }
        var identityPrefix: String { "|resident-q4-v1=" }
    }
    package struct Layout: Codable, Equatable, Sendable {
        package let name: String
        package let width, outputs: Int
        package let offset, bytes, sourceBytes: Int64
    }
    private struct Record: Codable {
        let layout: Layout
        let sha256: String
    }
    // A missing conversion is the original target Q4 format; existing packs retain their identity.
    private struct Index: Codable {
        let version: Int
        let source: String
        let conversion: Conversion?
        let records: [Record]
    }
    package static let dataName = "resident.pack", indexName = "resident.json"
    package let sourceIdentity: String, representationIdentity: String
    package let conversion: Conversion
    package let payloadBytes, savedBytes: Int64
    private let file: ExactFile
    private let records: [String: Record]
    private static func hex(_ hash: SHA256.Digest) -> String { hash.map { String(format: "%02x", $0) }.joined() }

    /// Target plans exclude auxiliary MTP blocks.
    package static func plan(_ gguf: GGUFSource, geometry: ModelGeometry, conversion: Conversion = .targetQ4) throws -> [Layout] {
        // GLM's trunk matrices are Q8_0 like DeepSeek's.
        guard geometry.family == .deepseek || geometry.family == .glm else {
            throw EngineError("Resident conversion supports DeepSeek, and GLM target weights")
        }
        var result: [Layout] = []
        var offset: Int64 = 0
        for t in gguf.tensors.values.sorted(by: { $0.name < $1.name }) {
            let fields = t.name.split(separator: ".")
            let selected =
                fields.count >= 3 && fields[0] == "blk" && Int(fields[1]).map { (0..<geometry.layers).contains($0) } == true
                && !t.name.contains("_exps.") && !t.name.contains("engram_embd")
                && !t.name.contains("ffn_gate_inp") && !t.name.contains("router")
            guard selected, t.shape.count == 2, t.type == conversion.sourceType,
                let width = t.shape.last, width > 0, width % 256 == 0, t.shape[0] > 0
            else { continue }
            guard width <= 1 << 20 else { throw EngineError("Resident conversion row exceeds bounded workspace: \(t.name)") }
            let outputs = t.shape[0]
            let inputRow = GGUFQuantize.bytes(conversion.sourceType, count: width)
            let outputRow = GGUFQuantize.bytes(conversion.targetType, count: width)
            let (sourceBytes, sourceOverflow) = Int64(inputRow).multipliedReportingOverflow(by: Int64(outputs))
            let (bytes, overflow) = Int64(outputRow).multipliedReportingOverflow(by: Int64(outputs))
            let (end, offsetOverflow) = offset.addingReportingOverflow(bytes)
            guard !sourceOverflow, !overflow, !offsetOverflow, sourceBytes == t.length, bytes < sourceBytes,
                bytes <= Int64(Int.max)
            else { throw EngineError("Invalid resident matrix span: \(t.name)") }
            result.append(Layout(name: t.name, width: width, outputs: outputs, offset: offset, bytes: bytes, sourceBytes: sourceBytes))
            offset = end
        }
        guard !result.isEmpty else { throw EngineError("No eligible resident matrices in this model") }
        return result
    }

    package static func open(
        _ folder: URL, gguf: GGUFSource, geometry: ModelGeometry,
        conversion: Conversion = .targetQ4
    ) throws -> FamilyResidentPack {
        let expected = try plan(gguf, geometry: geometry, conversion: conversion)
        let identity = FamilySupport.key(gguf)
        let header = try ExactFile(url: folder.appendingPathComponent(indexName))
        guard header.identity.size > 0, header.identity.size <= 4 << 20 else { throw EngineError("Invalid resident pack index size") }
        let bytes = try header.read(offset: 0, length: Int(header.identity.size))
        try header.validateIdentity()
        let index = try JSONDecoder().decode(Index.self, from: bytes)
        guard index.version == 1, index.source == identity, (index.conversion ?? .targetQ4) == conversion,
            index.records.map(\.layout) == expected,
            index.records.allSatisfy({
                $0.sha256.count == 64 && $0.sha256.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
            })
        else {
            throw EngineError("Resident pack does not match the source model and conversion plan")
        }
        let file = try ExactFile(url: folder.appendingPathComponent(dataName))
        let total = expected.last!.offset + expected.last!.bytes
        guard file.identity.size == UInt64(total) else { throw EngineError("Resident pack payload has the wrong extent") }
        return FamilyResidentPack(file: file, index: index, indexBytes: bytes, total: total)
    }

    private init(file: ExactFile, index: Index, indexBytes: Data, total: Int64) {
        self.file = file
        sourceIdentity = index.source
        conversion = index.conversion ?? .targetQ4
        representationIdentity = conversion.identityPrefix + Self.hex(SHA256.hash(data: indexBytes))
        records = Dictionary(uniqueKeysWithValues: index.records.map { ($0.layout.name, $0) })
        payloadBytes = total
        savedBytes = index.records.reduce(0) { $0 + $1.layout.sourceBytes - $1.layout.bytes }
    }
    package func layout(_ name: String) -> Layout? { records[name]?.layout }
    package var status: [String: Any] {
        [
            "enabled": true, "experimental": true, "correctness": "non_exact", "matrices": records.count,
            "payload_bytes": payloadBytes, "payload_saved_bytes": savedBytes, "representation": representationIdentity,
            "conversion": conversion.rawValue, "target_type": conversion.targetType,
        ]
    }
    /// Hash the exact buffer that will be executed. There is no second payload allocation or read.
    package func read(_ name: String, into output: UnsafeMutableRawBufferPointer) throws {
        guard let record = records[name], output.count == Int(record.layout.bytes), let pointer = output.baseAddress else {
            throw EngineError("Resident pack read does not match its validated layout")
        }
        try file.validateIdentity()
        try file.read(offset: record.layout.offset, into: output)
        let digest = Self.hex(SHA256.hash(data: Data(bytesNoCopy: pointer, count: output.count, deallocator: .none)))
        guard digest == record.sha256 else { throw EngineError("Resident pack checksum mismatch: \(name)") }
        try file.validateIdentity()
    }

    /// Publish a new folder only on success. Existing destinations are never replaced. Each chunk is 64 source
    /// rows plus one decoded row and its quantized output, keeping conversion memory independent of model size.
    package static func build(
        _ gguf: GGUFSource, geometry: ModelGeometry, into destination: URL,
        conversion: Conversion = .targetQ4,
        progress: (Int, Int) -> Void = { _, _ in }
    ) throws {
        let layout = try plan(gguf, geometry: geometry, conversion: conversion)
        let manager = FileManager.default
        guard !manager.fileExists(atPath: destination.path) else { throw EngineError("Resident pack destination already exists") }
        let sourceIdentity = FamilySupport.key(gguf)
        let parent = destination.deletingLastPathComponent()
        try manager.createDirectory(at: parent, withIntermediateDirectories: true)
        let stage = parent.appendingPathComponent(".resident-stage-" + UUID().uuidString, isDirectory: true)
        try manager.createDirectory(at: stage, withIntermediateDirectories: false)
        defer { try? manager.removeItem(at: stage) }
        let outputURL = stage.appendingPathComponent(dataName)
        guard manager.createFile(atPath: outputURL.path, contents: nil) else { throw EngineError("Cannot create resident pack payload") }
        let output = try FileHandle(forWritingTo: outputURL)
        defer { try? output.close() }
        var sources: [String: ExactFile] = [:]
        var records: [Record] = []
        for (number, item) in layout.enumerated() {
            guard let t = gguf.tensors[item.name] else { throw EngineError("Resident plan lost source tensor") }
            if sources[t.file] == nil { sources[t.file] = try ExactFile(url: gguf.folder.appendingPathComponent(t.file)) }
            let source = sources[t.file]!
            try source.validateIdentity()
            let inputRow = GGUFQuantize.bytes(conversion.sourceType, count: item.width)
            let outputRow = GGUFQuantize.bytes(conversion.targetType, count: item.width)
            var raw = Data(count: inputRow * min(64, item.outputs))
            var packed = Data(count: outputRow * min(64, item.outputs))
            var values = [Float](repeating: 0, count: item.width)
            var verified = values
            var hash = SHA256()
            for first in stride(from: 0, to: item.outputs, by: 64) {
                let count = min(64, item.outputs - first)
                try raw.withUnsafeMutableBytes {
                    try source.read(offset: t.offset + Int64(first * inputRow), into: .init(start: $0.baseAddress, count: count * inputRow))
                }
                try raw.withUnsafeBytes { bytes in
                    try packed.withUnsafeMutableBytes { target in
                        for row in 0..<count {
                            try values.withUnsafeMutableBufferPointer { floats in
                                GGUFQuant.dequantize(
                                    bytes.baseAddress! + row * inputRow, type: conversion.sourceType, count: item.width,
                                    into: floats.baseAddress!)
                                guard floats.allSatisfy(\.isFinite) else { throw EngineError("Non-finite resident source: \(item.name)") }
                                GGUFQuantize.encode(
                                    floats.baseAddress!, count: item.width, type: conversion.targetType,
                                    into: target.baseAddress! + row * outputRow)
                            }
                            try verified.withUnsafeMutableBufferPointer { floats in
                                GGUFQuant.dequantize(
                                    target.baseAddress! + row * outputRow, type: conversion.targetType, count: item.width,
                                    into: floats.baseAddress!)
                                guard floats.allSatisfy(\.isFinite) else {
                                    throw EngineError("Resident weights exceed finite quantized range: \(item.name)")
                                }
                            }
                        }
                    }
                }
                let chunk = packed.prefix(count * outputRow)
                hash.update(data: chunk)
                try output.write(contentsOf: chunk)
            }
            try source.validateIdentity()
            records.append(Record(layout: item, sha256: hex(hash.finalize())))
            progress(number + 1, layout.count)
        }
        try output.synchronize()
        try output.close()
        for source in sources.values { try source.validateIdentity() }
        guard FamilySupport.key(gguf) == sourceIdentity else { throw EngineError("Model source changed during resident conversion") }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(
            Index(
                version: 1, source: sourceIdentity, conversion: conversion == .targetQ4 ? nil : conversion,
                records: records)
        ).write(to: stage.appendingPathComponent(indexName), options: .atomic)
        _ = try open(stage, gguf: gguf, geometry: geometry, conversion: conversion)
        try manager.moveItem(at: stage, to: destination)
    }
}
