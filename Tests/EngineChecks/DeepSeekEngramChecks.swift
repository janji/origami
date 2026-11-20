import Foundation
import OrigamiEngine

private func engramRequire(_ value: Bool, _ message: String) throws {
    guard value else { throw EngineError("Engram rows: " + message) }
}

/// Independent full-history oracle rebuilds each n-gram, rather than carrying the implementation's rolling hash.
/// Include missing token-map entries, signed IDs, wrapping multiplication, and prompt/chunk boundaries.
func deepSeekEngramHashesMatchFullHistory() throws {
    let tokenMap = (0..<13).map { Int32($0 * 3 - 4) }
    let pad: Int64 = -7
    for ngram in [2, 4, 8, 16] {
        for heads in [1, 8] {
            let cols = (ngram - 1) * heads
            let multipliers = (0..<(2 * ngram)).map { UInt64.max - UInt64($0 * 97) }
            let primes = (0..<(2 * cols)).map { UInt64(101 + $0 * 2) }
            let offsets = (0..<(2 * cols)).map { Int64($0 * 1009) }
            func rows(_ tokens: [Int], _ history: [Int], _ layer: Int) -> [Int64] {
                DeepSeekEngramHash.rows(
                    tokens: tokens, history: history, layer: layer, ngram: ngram, heads: heads,
                    multipliers: multipliers, primes: primes, offsets: offsets, tokenMap: tokenMap, pad: pad)
            }
            func original(_ tokens: [Int], _ history: [Int], _ layer: Int) -> [Int64] {
                let all = history + tokens
                var result: [Int64] = []
                for index in tokens.indices {
                    for length in 2...ngram {
                        var hash: UInt64 = 0
                        var blocked = false
                        for offset in 0..<length {
                            let at = history.count + index - offset
                            let token = at >= 0 ? all[at] : -1
                            blocked = blocked || !(0..<tokenMap.count).contains(token)
                            let value = blocked ? pad : Int64(tokenMap[token])
                            hash ^= UInt64(bitPattern: value) &* multipliers[layer * ngram + offset]
                        }
                        for head in 0..<heads {
                            let column = layer * cols + (length - 2) * heads + head
                            result.append(Int64(hash % primes[column]) + offsets[column])
                        }
                    }
                }
                return result
            }
            for history in [[], [1], [1, -1, 2], (0..<2048).map { $0 % 17 }] {
                for tokens in [[2], [1, 3, 2, 0, 5, 8, 11], [99, 1, -1, 2]] {
                    for layer in 0..<2 {
                        try engramRequire(rows(tokens, history, layer) == original(tokens, history, layer), "suffix changed a row hash")
                        let split = min(2, tokens.count)
                        let first = Array(tokens.prefix(split))
                        let rest = Array(tokens.dropFirst(split))
                        try engramRequire(
                            rows(first, history, layer) + rows(rest, history + first, layer) == rows(tokens, history, layer),
                            "chunking changed a row hash")
                    }
                }
            }
        }
    }
}

/// Actual tiny IQ4_NL file reads and dequantization: two reads give the same bits in row order and stay within their
/// destination, and a short file fails the read.
/// Buffers are CPU allocations; this check creates no Metal device or command queue.
func deepSeekEngramReadBytesAndFailure() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-engram-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let path = folder.appendingPathComponent("rows.gguf")
    let name = "per_layer_token_embd.weight"
    var raw = Data()
    for row in 0..<64 {
        for block in 0..<8 {
            var scale = Float16(Float(row + 1) / 32).bitPattern.littleEndian
            withUnsafeBytes(of: &scale) { raw.append(contentsOf: $0) }
            raw.append(contentsOf: (0..<16).map { UInt8((row * 11 + block * 7 + $0) & 255) })
        }
    }
    try familyGGUF([("general.architecture", "deepseek41")], tensors: [(name, [64, 256], 20)], data: [name: raw], to: path)
    let table = try GGUFSource(path).tensors[name]!
    let file = try ExactFile(url: path, bypassCache: false)
    let ids = (0..<24).map { Int64(($0 * 17 + 5) % 64) }
    let count = ids.count * 256
    final class Destination: @unchecked Sendable {
        let pointer: UnsafeMutablePointer<Float>, count: Int
        init(_ count: Int) {
            self.count = count
            pointer = .allocate(capacity: count + 1)
            pointer.initialize(repeating: -123, count: count + 1)
        }
        deinit {
            pointer.deinitialize(count: count + 1)
            pointer.deallocate()
        }
        var bits: [UInt32] { (0..<count).map { pointer[$0].bitPattern } }
    }
    let expected = Destination(count)
    let actual = Destination(count)
    try DeepSeekModel.loadEngramRows(ids, table: table, file: file, dim: 256, into: expected.pointer)
    try DeepSeekModel.loadEngramRows(ids, table: table, file: file, dim: 256, into: actual.pointer)
    try engramRequire(
        actual.bits == expected.bits && actual.pointer[count] == -123, "row order/bytes changed or exceeded its destination")
    let writer = try FileHandle(forWritingTo: path)
    defer { try? writer.close() }
    try writer.truncate(atOffset: UInt64(table.offset))
    var refused = false
    do {
        try DeepSeekModel.loadEngramRows(ids, table: table, file: file, dim: 256, into: actual.pointer)
    } catch let error as EngineError where error.description.contains("pread failed") { refused = true }
    try engramRequire(refused, "short file read became a usable Engram result")
}
