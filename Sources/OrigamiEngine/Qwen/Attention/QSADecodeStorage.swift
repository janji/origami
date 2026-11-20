import Foundation
import Metal

/// Request-owned mirrors, never part of value-semantic QSAState/checkpoints.
/// Immutable committed CPU state remains authoritative. A storage object is
/// bound by LayerMath to one active request and layer, seeded once, and advanced
/// only after the unchanged pinned kernels join. Errors permanently retire it.
public final class QSADecodeStorage {
    public struct Snapshot: Codable, Sendable {
        public let capacityTokens, committedTokens: Int
        public let seeded, invalidated: Bool
        public let uploadedBytes: UInt64
    }
    public let capacityTokens: Int
    package let keys, values: MTLBuffer
    private let lock = NSLock()
    private var committedTokens = 0
    private var seeded = false
    private var invalidated = false
    private var pendingTokens = 0
    private var encodedCommand: MTLCommandBuffer?
    private var uploadedBytes: UInt64 = 0
    // Bounded query/selection/vector-partial scratch is also held only for this
    // request/layer, avoiding allocator churn without shared writable scratch.
    private var scratch: [String: MTLBuffer] = [:]

    init(capacityTokens: Int, keys: MTLBuffer, values: MTLBuffer) throws {
        guard (1...ModelContract.maximumContext).contains(capacityTokens),
            keys.length >= capacityTokens * 512 * 2, values.length >= capacityTokens * 512 * 2
        else {
            throw EngineError("Invalid QSA decode storage capacity")
        }
        self.capacityTokens = capacityTokens
        self.keys = keys
        self.values = values
    }

    /// Locks the storage until complete(). Call only after attention geometry
    /// and causal selection have validated. Every submitted command joins
    /// before complete(), including the failure path after command submission.
    package func stage(oldKeys: [UInt16], newKeys: [UInt16], oldValues: [UInt16], newValues: [UInt16]) throws {
        lock.lock()
        guard !invalidated, newKeys.count == 512, newValues.count == 512,
            oldKeys.count % 512 == 0, oldKeys.count == oldValues.count,
            oldKeys.count / 512 < capacityTokens,
            !seeded || oldKeys.count / 512 == committedTokens
        else {
            lock.unlock()
            throw EngineError("QSA decode mirror frontier is invalid or does not match committed state")
        }
        pendingTokens = oldKeys.count / 512 + 1
        if !seeded {
            oldKeys.withUnsafeBytes { if !$0.isEmpty { keys.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) } }
            oldValues.withUnsafeBytes { if !$0.isEmpty { values.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) } }
            uploadedBytes += UInt64((oldKeys.count + oldValues.count) * 2)
        }
        newKeys.withUnsafeBytes { keys.contents().advanced(by: oldKeys.count * 2).copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        newValues.withUnsafeBytes {
            values.contents().advanced(by: oldValues.count * 2).copyMemory(from: $0.baseAddress!, byteCount: $0.count)
        }
        uploadedBytes += 2048
    }

    package func complete(success: Bool) {
        precondition(encodedCommand == nil, "Encoded QSA append requires its ownership ticket")
        retire(success: success)
    }

    private func retire(success: Bool) {
        if success {
            committedTokens = pendingTokens
            seeded = true
        } else {
            invalidated = true
            scratch.removeAll()
        }
        pendingTokens = 0
        lock.unlock()
    }

    package struct EncodedAppend {
        fileprivate let owner: QSADecodeStorage
        fileprivate let command: MTLCommandBuffer
    }

    /// Keep CPU state authoritative while staging an append from provisional
    /// device projections. The lock is retained on this serial execution thread
    /// until finishEncoded(), after the submitted command joins on every path.
    /// Only suffix bytes are written; committed prefix bytes remain immutable.
    package func stageEncoded(
        _ command: MTLCommandBuffer, oldKeys: [UInt16], oldValues: [UInt16],
        newKeys: MTLBuffer, newValues: MTLBuffer, rows: Int
    ) throws -> EncodedAppend {
        guard oldKeys.count % 512 == 0, oldValues.count == oldKeys.count else { throw EngineError("Encoded QSA prefix geometry differs") }
        return try stageEncoded(command, oldTokens: oldKeys.count / 512, newKeys: newKeys, newValues: newValues, rows: rows) { key, value in
            oldKeys.withUnsafeBytes { if !$0.isEmpty { key.copyMemory(from: $0.baseAddress!, byteCount: $0.count) } }
            oldValues.withUnsafeBytes { if !$0.isEmpty { value.copyMemory(from: $0.baseAddress!, byteCount: $0.count) } }
        }
    }
    func stageEncoded(
        _ command: MTLCommandBuffer, state: QSAState,
        newKeys: MTLBuffer, newValues: MTLBuffer, rows: Int
    ) throws -> EncodedAppend {
        try state.validate()
        return try stageEncoded(command, oldTokens: state.offset, newKeys: newKeys, newValues: newValues, rows: rows) { key, value in
            try state.copyKeys(to: key)
            try state.copyValues(to: value)
        }
    }

    private func stageEncoded(
        _ command: MTLCommandBuffer, oldTokens: Int,
        newKeys: MTLBuffer, newValues: MTLBuffer, rows: Int,
        seed: (UnsafeMutableRawPointer, UnsafeMutableRawPointer) throws -> Void
    ) throws -> EncodedAppend {
        guard (1...256).contains(rows), rows <= capacityTokens,
            oldTokens >= 0,
            oldTokens <= capacityTokens - rows,
            command.commandQueue.device.registryID == keys.device.registryID, command.status == .notEnqueued,
            newKeys.device.registryID == keys.device.registryID, newValues.device.registryID == keys.device.registryID,
            newKeys.length >= rows * 512 * 2, newValues.length >= rows * 512 * 2,
            newKeys !== keys, newKeys !== values, newValues !== keys, newValues !== values,
            lock.try()
        else { throw EngineError("Invalid/busy encoded QSA append") }
        guard !invalidated, pendingTokens == 0, encodedCommand == nil,
            !seeded || oldTokens == committedTokens
        else {
            lock.unlock()
            throw EngineError("Encoded QSA mirror differs from committed frontier")
        }
        if !seeded {
            do { try seed(keys.contents(), values.contents()) } catch {
                invalidated = true
                lock.unlock()
                throw error
            }
            uploadedBytes += UInt64((oldTokens * 512) * 4)
        }
        guard let blit = command.makeBlitCommandEncoder() else {
            lock.unlock()
            throw EngineError("Encoded QSA append encoder unavailable")
        }
        blit.copy(from: newKeys, sourceOffset: 0, to: keys, destinationOffset: (oldTokens * 512) * 2, size: rows * 512 * 2)
        blit.copy(from: newValues, sourceOffset: 0, to: values, destinationOffset: (oldTokens * 512) * 2, size: rows * 512 * 2)
        blit.endEncoding()
        pendingTokens = oldTokens + rows
        encodedCommand = command
        return EncodedAppend(owner: self, command: command)
    }

    /// Success is permitted only after the CPU state append also validates.
    /// A discarded/unsubmitted command can invalidate without submission, but
    /// a live writer can never unlock or retire the mirror early.
    package func finishEncoded(_ ticket: EncodedAppend, success: Bool) throws {
        guard ticket.owner === self, encodedCommand === ticket.command, pendingTokens > 0 else {
            throw EngineError("Encoded QSA append ownership differs")
        }
        let status = ticket.command.status
        guard success ? status == .completed : (status == .notEnqueued || status == .completed || status == .error) else {
            throw EngineError("Cannot retire an unjoined QSA append")
        }
        encodedCommand = nil
        retire(success: success)
    }

    /// Invoked only while stage() holds the storage lock. Metal resources own
    /// their ledger reservation independently, even if a caller retains one.
    func buffer(name: String, bytes: Int, allocate: (Int) throws -> MTLBuffer) throws -> MTLBuffer {
        guard ["query", "ids", "counts", "output", "intermediate", "sums", "maxima"].contains(name) else {
            throw EngineError("Unexpected QSA decode scratch role")
        }
        if let value = scratch[name], value.length >= bytes { return value }
        let value = try allocate(bytes)
        scratch[name] = value
        return value
    }

    /// The MTP verify step's rejected token: the frontier moves back; the next append overwrites the suffix.
    func rollback(to tokens: Int) throws {
        lock.lock()
        defer { lock.unlock() }
        guard pendingTokens == 0, encodedCommand == nil, tokens >= 0, tokens <= committedTokens else {
            throw EngineError("QSA mirror rollback outside its committed frontier")
        }
        committedTokens = tokens
    }
    public func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(
            capacityTokens: capacityTokens, committedTokens: committedTokens,
            seeded: seeded, invalidated: invalidated, uploadedBytes: uploadedBytes)
    }
}
