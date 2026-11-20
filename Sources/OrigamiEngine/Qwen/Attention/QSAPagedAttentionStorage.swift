import Foundation
import Metal
import ObjectiveC

private final class QSAPageTableAllocation {
    private let ledger: MemoryLedger, reservation: MemoryLedger.Reservation
    init(ledger: MemoryLedger, bytes: UInt64) throws {
        self.ledger = ledger
        reservation = try ledger.reserve(bytes: bytes, category: "qsa_shared_page_tables")
    }
    deinit { try? ledger.release(reservation) }
}
private enum QSAPageTableAssociation { nonisolated(unsafe) static var key: UInt8 = 0 }

/// Request/layer-owned argument table over immutable CPU/Metal prefix pages.
/// The current uncommitted suffix is read from the producer's GPU projections.
/// No complete KV copy is allocated or uploaded. Indirect buffers and the arena
/// charge remain alive until the enclosing command joins, including failures.
public final class QSAPagedAttentionStorage {
    public struct Snapshot: Codable, Sendable {
        public let capacityTokens, committedTokens: Int
        public let invalidated: Bool
        public let tableBytes, sharedPrefixBytes, descriptorUpdates, transactions: UInt64
    }
    static let maximumPages = 1024
    package let capacityTokens: Int, table: MTLBuffer
    private let header: MTLBuffer, arguments: MTLArgumentEncoder
    private let allocation: QSAPageTableAllocation
    private let lock = NSLock()
    private struct Binding { let buffer: MTLBuffer, offset: Int }
    private var oldKeys: [Binding] = [], oldValues: [Binding] = []
    private var active: MTLCommandBuffer?
    private var seeded = false, invalidated = false
    private var committedTokens = 0, updates: UInt64 = 0, transactions: UInt64 = 0

    init(device: MTLDevice, ledger: MemoryLedger, arguments: MTLArgumentEncoder, capacityTokens: Int) throws {
        guard device.argumentBuffersSupport == .tier2, (1...ModelContract.maximumContext).contains(capacityTokens),
            capacityTokens <= Self.maximumPages * 256, arguments.encodedLength > 0, arguments.encodedLength <= 131072
        else {
            throw EngineError("Invalid shared QSA argument-table geometry")
        }
        let rounded = (arguments.encodedLength + 16383) / 16384 * 16384
        // Includes Array capacity/COW/bridging for old/new page resource lists
        // at the complete 1024-page geometry, not only the pages currently used.
        let allocation = try QSAPageTableAllocation(ledger: ledger, bytes: UInt64(rounded + 16384 + 1_048_576))
        guard let table = device.makeBuffer(length: rounded, options: .storageModeShared), table.allocatedSize <= rounded,
            let header = device.makeBuffer(length: 16384, options: .storageModeShared), header.allocatedSize <= 16384
        else {
            throw EngineError("Shared QSA table allocation exceeds admission", code: "insufficient_memory")
        }
        objc_setAssociatedObject(table, &QSAPageTableAssociation.key, allocation, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        objc_setAssociatedObject(header, &QSAPageTableAssociation.key, allocation, .OBJC_ASSOCIATION_RETAIN_NONATOMIC)
        table.contents().initializeMemory(as: UInt8.self, repeating: 0, count: table.length)
        header.contents().initializeMemory(as: UInt8.self, repeating: 0, count: header.length)
        self.capacityTokens = capacityTokens
        self.table = table
        self.header = header
        self.arguments = arguments
        self.allocation = allocation
    }

    /// One staged transaction: the table now describes `tokens` keys for `command`; `resources` must be
    /// made resident by every encoder that reads through the table. `finish` must follow the join.
    package struct Read {
        let owner: QSAPagedAttentionStorage, command: MTLCommandBuffer
        let tokens: Int
        let resources: [MTLResource]
        let fileLease: QSAFileStaging.Lease?
    }
    package func stage(
        _ command: MTLCommandBuffer, state: QSAState, newKeys: MTLBuffer, newValues: MTLBuffer, rows: Int,
        selected: [[UInt32]]? = nil
    ) throws -> Read {
        try state.validate()
        guard let pages = state.pages, (1...256).contains(rows), state.offset <= capacityTokens - rows,
            command.status == .notEnqueued, command.commandQueue.device.registryID == table.device.registryID,
            newKeys.device.registryID == table.device.registryID, newValues.device.registryID == table.device.registryID,
            newKeys !== newValues, newKeys.length >= rows * 512 * 2, newValues.length >= rows * 512 * 2,
            lock.try()
        else { throw EngineError("Invalid or busy shared QSA transaction") }
        defer { lock.unlock() }
        guard active == nil, !invalidated, !seeded || committedTokens == state.offset else {
            throw EngineError("Shared QSA table has a different committed frontier")
        }
        let keys: [Binding]
        let values: [Binding]
        let fileLease: QSAFileStaging.Lease?
        if case .file(let k) = pages.keys, case .file(let v) = pages.values {
            guard rows > 2 || selected != nil else {
                throw EngineError("File-KV vector attention requires joined/host-validated selection")
            }
            let staging = try pages.arena.staging(device: table.device)
            let lease = try staging.stage(command: command, keys: k, values: v, rows: rows, selected: selected)
            fileLease = lease
            keys = k.pages.indices.map { Binding(buffer: lease.buffer, offset: $0 * QSAFilePageStore.pageBytes) }
            values = v.pages.indices.map {
                Binding(buffer: lease.buffer, offset: QSAFileStaging.frameBytes / 2 + $0 * QSAFilePageStore.pageBytes)
            }
        } else {
            fileLease = nil
            keys = try pages.keys.metalBuffers(device: table.device).map { Binding(buffer: $0, offset: 0) }
            values = try pages.values.metalBuffers(device: table.device).map { Binding(buffer: $0, offset: 0) }
        }
        guard keys.count == (state.offset + 255) / 256, values.count == keys.count, keys.count <= Self.maximumPages else {
            throw EngineError("Shared QSA page coverage differs")
        }
        arguments.setArgumentBuffer(table, offset: 0)
        defer { arguments.setArgumentBuffer(nil, offset: 0) }
        for (buffers, prior, base) in [(keys, oldKeys, 0), (values, oldValues, Self.maximumPages)] {
            for index in 0..<max(buffers.count, prior.count) {
                let next = index < buffers.count ? buffers[index] : nil
                if index >= prior.count || next == nil || next!.buffer !== prior[index].buffer || next!.offset != prior[index].offset {
                    arguments.setBuffer(next?.buffer, offset: next?.offset ?? 0, index: base + index)
                    updates += 1
                }
            }
        }
        arguments.setBuffer(newKeys, offset: 0, index: 2048)
        arguments.setBuffer(newValues, offset: 0, index: 2049)
        arguments.setBuffer(header, offset: 0, index: 2050)
        header.contents().storeBytes(of: UInt32(state.offset), as: UInt32.self)
        oldKeys = keys
        oldValues = values
        active = command
        transactions += 1
        var seen = Set<ObjectIdentifier>()
        let resources = (keys.map(\.buffer) + values.map(\.buffer) + [newKeys, newValues, header]).filter {
            seen.insert(ObjectIdentifier($0)).inserted
        }
        return Read(
            owner: self, command: command, tokens: state.offset + rows,
            resources: resources.map { $0 as MTLResource }, fileLease: fileLease)
    }
    package func finish(_ read: Read, success: Bool) throws {
        lock.lock()
        defer { lock.unlock() }
        guard read.owner === self, active === read.command,
            success
                ? read.command.status == .completed
                : [.notEnqueued, .completed, .error].contains(read.command.status)
        else {
            throw EngineError("Shared QSA reader has not joined or belongs to another transaction")
        }
        try read.fileLease?.finish(command: read.command, success: success)
        if success {
            committedTokens = read.tokens
            seeded = true
        } else {
            invalidated = true
            oldKeys.removeAll()
            oldValues.removeAll()
        }
        active = nil
    }
    /// The MTP verify step's rejected token: the frontier moves back (the header takes each stage's offset).
    func rollback(to tokens: Int) throws {
        lock.lock()
        defer { lock.unlock() }
        guard active == nil, tokens >= 0, tokens <= committedTokens else {
            throw EngineError("Shared QSA rollback outside its committed frontier")
        }
        committedTokens = tokens
    }
    public func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        var seen = Set<ObjectIdentifier>()
        let buffers = (oldKeys + oldValues).map(\.buffer).filter { seen.insert(ObjectIdentifier($0)).inserted }
        return Snapshot(
            capacityTokens: capacityTokens, committedTokens: committedTokens, invalidated: invalidated,
            tableBytes: UInt64(table.allocatedSize + header.allocatedSize),
            sharedPrefixBytes: buffers.reduce(0) { $0 + UInt64($1.length) },
            descriptorUpdates: updates, transactions: transactions)
    }

    static let preamble = #"""
        #include <metal_stdlib>
        using namespace metal;
        #if ORIGAMI_PAGED_KV
        // Argument table: 1024 key and 1024 value page pointers (256 tokens x 512 BF16 each), the
        // uncommitted suffix rows written by this command's projections, and the committed token count.
        // Token t < offset reads page t / 256 at (t % 256) * 512; later tokens read the suffix.
        struct OrigamiPagedKV {
            array<device const ushort *, 1024> keys [[id(0)]];
            array<device const ushort *, 1024> values [[id(1024)]];
            device const ushort *newKeys [[id(2048)]];
            device const ushort *newValues [[id(2049)]];
            constant uint *offset [[id(2050)]];
        };
        #define ORIGAMI_KV_ARGS constant OrigamiPagedKV &
        inline ushort origami_key_bits(ORIGAMI_KV_ARGS table, uint i) {
            uint token = i / 512, column = i % 512, old = table.offset[0];
            return token < old ? table.keys[token / 256][(token % 256) * 512 + column] : table.newKeys[(token - old) * 512 + column];
        }
        inline ushort origami_value_bits(ORIGAMI_KV_ARGS table, uint i) {
            uint token = i / 512, column = i % 512, old = table.offset[0];
            return token < old ? table.values[token / 256][(token % 256) * 512 + column] : table.newValues[(token - old) * 512 + column];
        }
        // Never dispatched: it exists so the host can make the argument encoder for OrigamiPagedKV.
        kernel void origami_paged_argument_layout(ORIGAMI_KV_ARGS table [[buffer(0)]], device ushort *out [[buffer(1)]],
            uint i [[thread_position_in_grid]]) { out[i] = origami_key_bits(table, i) ^ origami_value_bits(table, i); }
        #else
        #define ORIGAMI_KV_ARGS device const ushort *
        inline ushort origami_key_bits(ORIGAMI_KV_ARGS keys, uint i) { return keys[i]; }
        inline ushort origami_value_bits(ORIGAMI_KV_ARGS values, uint i) { return values[i]; }
        #endif
        """#
}
