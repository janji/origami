import CryptoKit
import Darwin
import Foundation
import Metal

/// One saved family-engine state (GLM-5.3-Flash, DeepSeek-V4.1-Flash) in a prompt-cache slot: `state.bin` holds the
/// model's state parts (`FamilyModel.stateParts`) in order, then the last token's logits when it holds a whole prompt;
/// `state.json` the epoch, the tokens, each part's size and the SHA-256 of the tokens and every byte. Both are written
/// under staging names and renamed, the header last; a read goes straight into the model's buffers and is checked
/// before the state is used.
package enum FamilyStateFile {
    package struct Header: Codable, Equatable {
        let epoch: String
        let tokens: [Int]
        let parts: [Int]
        let logits: Int
        let sha256: String
        package init(epoch: String, tokens: [Int], parts: [Int], logits: Int, sha256: String) {
            self.epoch = epoch
            self.tokens = tokens
            self.parts = parts
            self.logits = logits
            self.sha256 = sha256
        }
    }
    package static func dataURL(_ folder: URL) -> URL { folder.appendingPathComponent("state.bin") }
    package static func headerURL(_ folder: URL) -> URL { folder.appendingPathComponent("state.json") }
    /// Bytes a state takes: its parts and logits.
    package static func bytes(_ parts: [(buffer: MTLBuffer, bytes: Int)], logits: [Float]?) -> UInt64 {
        UInt64(parts.reduce(0) { $0 + $1.bytes } + (logits?.count ?? 0) * 4)
    }

    private static func digest(_ tokens: [Int]) -> SHA256 {
        var hash = SHA256()
        tokens.map { Int64($0).littleEndian }.withUnsafeBufferPointer { hash.update(bufferPointer: UnsafeRawBufferPointer($0)) }
        return hash
    }
    private static let chunk = 64 << 20

    /// Writes `parts` (read from the model's shared buffers, which the GPU has finished) for `tokens` into `folder`.
    package static func write(
        _ parts: [(buffer: MTLBuffer, bytes: Int)], logits: [Float]?, tokens: [Int], epoch: String, to folder: URL,
        checkCancellation: () throws -> Void = {}
    ) throws {
        try checkCancellation()
        guard parts.allSatisfy({ $0.bytes >= 0 && $0.bytes <= $0.buffer.length }) else {
            throw EngineError("A saved state's parts exceed their buffers", code: "prompt_cache_mismatch")
        }
        let data = dataURL(folder)
        let header = headerURL(folder)
        let stagingData = folder.appendingPathComponent(".state.bin.staging")
        let stagingHeader = folder.appendingPathComponent(".state.json.staging")
        // The header is the publication marker. Cancellation or any later failure must leave no staging files and
        // must never make a partial payload visible as a reusable state. An in-flight syscall itself is not aborted.
        defer {
            try? FileManager.default.removeItem(at: stagingData)
            try? FileManager.default.removeItem(at: stagingHeader)
        }
        guard FileManager.default.createFile(atPath: stagingData.path, contents: nil, attributes: [.posixPermissions: 0o600]) else {
            throw EngineError("Cannot write a saved state in \(folder.path)", code: "storage_unavailable")
        }
        var hash = digest(tokens)
        do {
            let handle = try FileHandle(forWritingTo: stagingData)
            defer { try? handle.close() }
            // State buffers remain immutable until this synchronous save finishes.
            // Write them directly; copying each 64 MiB chunk into Data increases
            // memory traffic and competes with the model's expert cache.
            guard fcntl(handle.fileDescriptor, F_NOCACHE, 1) == 0 else {
                throw EngineError("Cannot configure saved-state writes: errno=\(errno)", code: "storage_unavailable")
            }
            func put(_ base: UnsafeRawPointer, _ count: Int) throws {
                var done = 0
                while done < count {
                    try checkCancellation()
                    let piece = UnsafeRawBufferPointer(start: base + done, count: min(chunk, count - done))
                    hash.update(bufferPointer: piece)
                    var written = 0
                    while written < piece.count {
                        try checkCancellation()
                        let amount = Darwin.write(handle.fileDescriptor, piece.baseAddress!.advanced(by: written), piece.count - written)
                        if amount < 0 && errno == EINTR { continue }
                        guard amount > 0 else { throw EngineError("Saved-state write failed: errno=\(errno)", code: "storage_unavailable") }
                        written += amount
                    }
                    done += piece.count
                }
            }
            for part in parts { try put(part.buffer.contents(), part.bytes) }
            if let logits { try logits.withUnsafeBytes { try put($0.baseAddress!, $0.count) } }
            try checkCancellation()
            try handle.synchronize()
        } catch {
            try? FileManager.default.removeItem(at: stagingData)
            throw error
        }
        let sum = hash.finalize().map { String(format: "%02x", $0) }.joined()
        let encoded = try JSONEncoder().encode(
            Header(epoch: epoch, tokens: tokens, parts: parts.map(\.bytes), logits: logits?.count ?? 0, sha256: sum))
        try encoded.write(to: stagingHeader, options: .atomic)
        try checkCancellation()
        // Keep a valid previous state usable throughout staging. Remove its publication marker only once every
        // new byte is ready; if removal itself fails, do not replace its payload under the still-valid old header.
        if unlink(header.path) != 0 && errno != ENOENT {
            throw EngineError("Cannot replace a saved-state header: errno=\(errno)", code: "storage_unavailable")
        }
        guard rename(stagingData.path, data.path) == 0, rename(stagingHeader.path, header.path) == 0 else {
            throw EngineError(
                "Cannot finish a saved state in \(folder.path): \(String(cString: strerror(errno)))", code: "storage_unavailable")
        }
    }

    /// The header in `folder`, when it is this epoch's and its tokens begin `prompt`.
    package static func header(_ folder: URL, epoch: String, prompt: [Int]) throws -> Header {
        let header = try JSONDecoder().decode(Header.self, from: Data(contentsOf: headerURL(folder)))
        guard header.epoch == epoch, header.tokens.count <= prompt.count, Array(prompt.prefix(header.tokens.count)) == header.tokens else {
            throw EngineError("A saved state belongs to another model, build or prompt", code: "prompt_cache_mismatch")
        }
        return header
    }

    /// Reads a saved state into `parts` (the model's buffers at the header's position, reset first by the caller) and
    /// returns its logits; throws when a size or the digest does not match, the buffers then holding nothing usable.
    package static func read(
        _ folder: URL, header: Header, into parts: [(buffer: MTLBuffer, bytes: Int)],
        expectedLogits: Int? = nil,
        checkCancellation: () throws -> Void = {}
    ) throws -> [Float]? {
        try checkCancellation()
        guard parts.map(\.bytes) == header.parts, parts.allSatisfy({ $0.bytes >= 0 && $0.bytes <= $0.buffer.length }),
            header.logits >= 0, header.logits <= Int.max / MemoryLayout<Float>.stride,
            expectedLogits.map({ $0 >= 0 && (header.logits == 0 || header.logits == $0) }) ?? true
        else {
            throw EngineError("A saved state's parts do not fit this model", code: "prompt_cache_mismatch")
        }
        let handle = try FileHandle(forReadingFrom: dataURL(folder))
        defer { try? handle.close() }
        // Header sizes are untrusted. Reject overflow, truncation and extra bytes before touching live state or
        // allocating logits; otherwise a tiny corrupt JSON header can request an enormous array before its digest
        // is checked. Serving additionally binds the only nonzero permitted logits count to the model vocabulary.
        var expectedBytes = UInt64(header.logits) * UInt64(MemoryLayout<Float>.stride)
        for part in header.parts {
            let (sum, overflow) = expectedBytes.addingReportingOverflow(UInt64(part))
            guard !overflow else { throw EngineError("Saved-state size overflow", code: "prompt_cache_mismatch") }
            expectedBytes = sum
        }
        var file = stat()
        guard fstat(handle.fileDescriptor, &file) == 0, file.st_size >= 0, UInt64(file.st_size) == expectedBytes else {
            throw EngineError("A saved state's payload size does not match its header", code: "prompt_cache_mismatch")
        }
        guard fcntl(handle.fileDescriptor, F_NOCACHE, 1) == 0 else {
            throw EngineError("Cannot configure saved-state reads: errno=\(errno)", code: "storage_unavailable")
        }
        var hash = digest(header.tokens)
        func take(_ base: UnsafeMutableRawPointer, _ count: Int) throws {
            var done = 0
            while done < count {
                try checkCancellation()
                let amount = Darwin.read(handle.fileDescriptor, base.advanced(by: done), min(chunk, count - done))
                if amount < 0 && errno == EINTR { continue }
                guard amount > 0 else {
                    throw EngineError("Saved-state read failed or truncated: errno=\(errno)", code: "prompt_cache_mismatch")
                }
                hash.update(bufferPointer: UnsafeRawBufferPointer(start: base.advanced(by: done), count: amount))
                done += amount
            }
        }
        for part in parts { try take(part.buffer.contents(), part.bytes) }
        var logits = header.logits > 0 ? [Float](repeating: 0, count: header.logits) : nil
        if logits != nil { try logits!.withUnsafeMutableBytes { try take($0.baseAddress!, $0.count) } }
        try checkCancellation()
        guard try handle.read(upToCount: 1)?.isEmpty ?? true,
            hash.finalize().map({ String(format: "%02x", $0) }).joined() == header.sha256
        else {
            throw EngineError("A saved state does not match its digest", code: "prompt_cache_mismatch")
        }
        return logits
    }
}
