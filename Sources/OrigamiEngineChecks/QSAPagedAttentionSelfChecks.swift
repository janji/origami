import Foundation
import Metal
import OrigamiEngine

/// Model-free checks of attention over shared pages: Metal page and argument-table lifetime, and bitwise
/// parity of the paged (memory and file) libraries with the host-array path up to 262,144 keys.
public enum QSAPagedAttentionSelfChecks {
    private static func require(_ value: Bool, _ message: String) throws {
        if !value { throw EngineError(message) }
    }
    private static func rejected(_ body: () throws -> Void) throws {
        do { try body() } catch is EngineError { return }
        throw EngineError("Expected shared QSA ownership refusal")
    }
    public static func lifetime() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw EngineError("No Metal device") }
        let ledger = try MemoryLedger(ceiling: 4_000_000, transientMargin: 1_000_000)
        var escaped: MTLBuffer?
        try autoreleasepool {
            let arena = try QSAPageArena(ledger: ledger, maximumBytes: 1_000_000)
            let vector = try QSAPagedVector(arena: arena, pageElements: 256 * 512).appending([0x3f80, 0x4000, 0x4040])
            escaped = try vector.metalBuffers(device: device)[0]
            try require(escaped!.contents().load(as: UInt16.self) == 0x3f80, "Shared Metal page did not alias published bytes")
            let appended = try vector.appending([0x4080])
            let newer = try appended.metalBuffers(device: device)[0]
            try require(
                newer !== escaped && escaped!.contents().assumingMemoryBound(to: UInt16.self)[2] == 0x4040,
                "COW append changed an existing GPU reader")
        }
        try autoreleasepool {
            try require(
                ledger.snapshot().reserved == 1_000_000 && escaped!.contents().load(as: UInt16.self) == 0x3f80,
                "GPU reader lost allocation or arena charge with CPU vector retirement")
            escaped = nil
        }
        try require(ledger.snapshot().reserved == 0, "Shared allocation outlived final GPU reader")

        // An argument buffer can also escape its Swift table owner. Its charge
        // must survive that final reader; cancelled tables must refuse reuse.
        let tableLedger = try MemoryLedger(ceiling: 8_000_000, transientMargin: 1_000_000)
        var escapedTable: MTLBuffer?
        var tableCharge: UInt64 = 0
        try autoreleasepool {
            let attention = try PinnedAttention(ledger: tableLedger, architecture: .medium, batched: true)
            let storage = try attention.makePagedStorage(capacityTokens: 1)
            tableCharge = tableLedger.snapshot().reserved
            let arena = try QSAPageArena(ledger: tableLedger, maximumBytes: 1_000_000)
            let state = QSAState(arena: arena)
            let scratch = try tableLedger.reserve(bytes: 32768, category: "paged_lifetime_fixture")
            defer { try? tableLedger.release(scratch) }
            guard let queue = device.makeCommandQueue(), let command = queue.makeCommandBuffer(),
                let key = device.makeBuffer(length: 16384, options: .storageModeShared),
                let value = device.makeBuffer(length: 16384, options: .storageModeShared)
            else {
                throw EngineError("Shared table fixture resources unavailable")
            }
            let read = try storage.stage(command, state: state, newKeys: key, newValues: value, rows: 1)
            try rejected { _ = try storage.stage(command, state: state, newKeys: key, newValues: value, rows: 1) }
            try rejected { try storage.finish(read, success: true) }
            try storage.finish(read, success: false)
            try require(storage.snapshot().invalidated, "Cancelled shared table remained reusable")
            try rejected { _ = try storage.stage(command, state: state, newKeys: key, newValues: value, rows: 1) }
            escapedTable = storage.table
        }
        try require(
            escapedTable != nil && tableCharge > 0 && tableLedger.snapshot().reserved == tableCharge,
            "Escaped argument table lost or duplicated its allocation charge")
        autoreleasepool { escapedTable = nil }
        try require(tableLedger.snapshot().reserved == 0, "Shared argument table failed final retirement")
    }
    public static func attentionParity() throws {
        try parity(
            cases: [
                (4, 262144), (1, 32769), (2, 8193), (9, 4097), (256, 768), (3, 3),
                (1, 1), (2, 257), (1, 1023),
            ], fileBacked: false)
    }
    public static func fileAttentionParity() throws {
        try parity(cases: [(256, 262144), (1, 262144), (2, 8193), (9, 4097), (1, 1023), (1, 1)], fileBacked: true)
    }
    public static func registerDecodeParity() throws {
        try parity(
            cases: [
                (1, 1025), (1, 2049), (1, 8191), (1, 8192), (1, 8193),
                (1, 32767), (1, 32768), (2, 8192), (1, 32769),
            ], fileBacked: false, registerDecode: true)
        try parity(cases: [(1, 8192), (1, 32768)], fileBacked: true, registerDecode: true)
    }
    public static func prefillReuseParity() throws { try prefillReuseParity(exact: false) }
    /// `exact`: both libraries run the direct stripe PV (`qsa_exact_kernels`); the paged form is compared
    /// with the nonpaged one of the same flag, whose bytes the stripe check ties to the original control.
    package static func prefillReuseParity(exact: Bool) throws {
        try parity(cases: [(256, 7680), (256, 4096)], fileBacked: false, exactKernels: exact)
        try parity(cases: [(256, 7680), (256, 4096)], fileBacked: true, exactKernels: exact)
    }
    private static func parity(cases: [(Int, Int)], fileBacked: Bool, registerDecode: Bool = false, exactKernels: Bool = false) throws {
        // Maximum domain/tile first. File cases additionally include asymmetric
        // NaN/Inf values, signed zero/subnormal and an empty vector selection.
        for (rows, keys) in cases {
            try autoreleasepool {
                let ledger = try MemoryLedger(ceiling: 4_000_000_000, transientMargin: 16_000_000)
                let scratch = try ledger.reserve(bytes: 2 * UInt64(keys) * 512 * 4 + 64_000_000, category: "paged_attention_fixture_arrays")
                defer { try? ledger.release(scratch) }
                let file =
                    fileBacked ? try QSAFilePageStore(parent: CheckFixtures.scratch, diskBytes: 768 * 1024 * 1024, ledger: ledger) : nil
                let arena = try QSAPageArena(
                    ledger: ledger, maximumBytes: UInt64(keys) * (fileBacked ? 64 : 2112) + 16_777_216, fileStore: file)
                var state = QSAState(arena: arena)
                func bits(start: Int, rows: Int, value: Bool) -> [UInt16] {
                    (start * 512..<(start + rows) * 512).map { index in
                        let token = index / 512
                        let column = index % 512
                        if fileBacked && value {
                            if token == 7719 && column == 7 { return 0x7fc1 }
                            if token == 131071 && column == 8 { return 0x7f80 }
                            if token == 0 && column == 3 { return 0x8000 }
                            if token == 1 && column == 3 { return 0x0001 }
                        }
                        let base = ((token + column) % 2 == 0) ? 0x3e00 : 0xbe00
                        return UInt16(base + (token * (value ? 43 : 31) + column * (value ? 19 : 13)) % 509)
                    }
                }
                let old = keys - rows
                for start in stride(from: 0, to: old, by: 256) {
                    let count = min(256, old - start)
                    let end = start + count
                    try state.append(
                        keys: bits(start: start, rows: count, value: false), values: bits(start: start, rows: count, value: true),
                        pooledKeys: [UInt16](repeating: 0x3f80, count: (end / 4 - start / 4) * 128),
                        rawTail: [UInt16](repeating: 0x3f80, count: end % 4 * 128), tokens: count)
                }
                let q = (0..<rows * 6144).map { BF16.decode(BF16.encode(Float(($0 * 31 + 13) % 509 - 254) / 513)) }
                let newK = bits(start: old, rows: rows, value: false)
                let newV = bits(start: old, rows: rows, value: true)
                let positions: [[UInt32]] = (0..<rows).map { row in
                    if rows > 2 && row == 1 || fileBacked && rows == 1 && keys == 1023 { return [] }
                    if keys <= 768 { return (0...old + row).map(UInt32.init) }
                    var candidates: [Int] = [0, 1, 255, 256, 257, 4095, 8191, 32767, 65535, 131071, old - 1, old + row]
                    if fileBacked && row == 0 { candidates.append(7719) }
                    let valid = candidates.filter { $0 >= 0 && $0 <= old + row }
                    return Set(valid).sorted().map { UInt32($0) }
                }
                let attention = try PinnedAttention(ledger: ledger, architecture: .medium, batched: true, registerDecode: registerDecode)
                attention.qsaExactKernels = exactKernels
                if exactKernels { try attention.prepareExactKernels() }
                let expected = try attention.attention(
                    queries: q, oldKeys: state.keys, newKeys: newK,
                    oldValues: state.values, newValues: newV, selectedPositions: positions)
                let storage = try attention.makePagedStorage(capacityTokens: keys)
                let actual = try attention.attentionPaged(
                    queries: q, state: state, newKeys: newK,
                    newValues: newV, selectedPositions: positions, storage: storage)
                try require(
                    attention.registerDecodeDispatches == (registerDecode && rows == 1 && (1025...32768).contains(keys) ? 1 : 0),
                    "Register attention dispatch/fallback was not exercised")
                try require(
                    attention.prefillReuseDispatches == (rows == 256 && (256...7680).contains(keys) ? 2 : 0),
                    "Shared prefill reuse dispatch/fallback was not exercised")
                try require(
                    attention.qsaExactPVTiles == (exactKernels && rows == 256 && (256...7680).contains(keys) ? 2 * 768 : 0)
                        && attention.qsaExactQKTiles == 0
                        && attention.qsaExactFallbacks
                            == (exactKernels && rows > 2 && !(rows == 256 && (256...7680).contains(keys)) ? 2 : 0),
                    "Exact stripe PV counters differ across the nonpaged/paged libraries rows=\(rows) keys=\(keys)")
                try require(
                    expected.count == actual.count
                        && zip(expected, actual).allSatisfy {
                            (fileBacked || $0.isFinite && $1.isFinite) && $0.bitPattern == $1.bitPattern
                        },
                    "Shared QSA changed attention bits at rows=\(rows) keys=\(keys)")
                let report = storage.snapshot()
                try require(
                    report.committedTokens == keys && report.tableBytes <= 147456
                        && report.sharedPrefixBytes
                            <= (fileBacked ? UInt64(QSAFileStaging.frameBytes) : UInt64((old + 255) / 256) * 2 * 262144),
                    "Shared QSA table geometry or byte accounting differs")
                try rejected {
                    _ = try attention.attentionPaged(
                        queries: q, state: state, newKeys: newK,
                        newValues: newV, selectedPositions: positions, storage: storage)
                }
            }
        }
    }
}
