import Foundation
import Metal
import OrigamiEngine

/// Model-free checks of the request-owned decode mirror: bitwise parity with the unmirrored path,
/// frontier/failure retirement, ledger ownership of escaped buffers, encoded-append ownership.
package enum QSADecodeStorageSelfChecks {
    private static func require(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
        guard try condition() else { throw EngineError(message) }
    }
    private static func rejects(_ body: () throws -> Void) throws {
        do { try body() } catch is EngineError { return }
        throw EngineError("Invalid QSA mirror operation was accepted")
    }

    package static func parity() throws {
        let ledger = try MemoryLedger(ceiling: 600_000_000, transientMargin: 1_000_000)
        let attention = try PinnedAttention(ledger: ledger, architecture: .medium, batched: true)
        // Cross vector/partial and 128/256-partition boundaries with the exact
        // same kernels and complete key domain in retained and legacy calls.
        for start in [0, 31, 1023, 8191, 32767] {
            try autoreleasepool {
                var keys = (0..<(start * 512)).map { BF16.encode(Float(($0 * 7 % 41) - 20) / 16) }
                var values = (0..<(start * 512)).map { BF16.encode(Float(($0 * 11 % 37) - 18) / 32) }
                let storage = try attention.makeDecodeStorage(capacityTokens: start + 2)
                for step in 0..<2 {
                    let newKeys = (0..<512).map { BF16.encode(Float(($0 + step) % 17 - 8) / 8) }
                    let newValues = (0..<512).map { BF16.encode(Float(($0 + step * 3) % 23 - 11) / 16) }
                    let queries = (0..<6144).map { Float(($0 * 5 + step) % 31 - 15) / 16 }
                    let count = start + step + 1
                    var positions = Array(stride(from: 0, to: count - 1, by: 19)).map(UInt32.init)
                    positions.append(UInt32(count - 1))
                    let expected = try attention.attention(
                        queries: queries, oldKeys: keys, newKeys: newKeys,
                        oldValues: values, newValues: newValues, selectedPositions: [positions])
                    let actual = try attention.attention(
                        queries: queries, oldKeys: keys, newKeys: newKeys,
                        oldValues: values, newValues: newValues, selectedPositions: [positions], decodeStorage: storage)
                    try require(expected.map(\.bitPattern) == actual.map(\.bitPattern), "Resident QSA changed pinned attention arithmetic")
                    keys += newKeys
                    values += newValues
                    let snapshot = storage.snapshot()
                    try require(
                        snapshot.committedTokens == count && snapshot.seeded && !snapshot.invalidated,
                        "Resident QSA frontier was not committed after joined completion")
                    try require(snapshot.uploadedBytes == UInt64(count * 2048), "Resident QSA reuploaded already committed KV")
                }
            }
            try require(ledger.snapshot().reserved == 0, "Request-owned QSA buffers survived retirement")
        }
    }

    package static func frontierAndFailure() throws {
        let ledger = try MemoryLedger(ceiling: 16_000_000, transientMargin: 1_000_000)
        let attention = try PinnedAttention(ledger: ledger, architecture: .medium)
        try autoreleasepool {
            let storage = try attention.makeDecodeStorage(capacityTokens: 3)
            let row = [UInt16](repeating: 0x3f80, count: 512)
            try storage.stage(oldKeys: row, newKeys: row, oldValues: row, newValues: row)
            storage.complete(success: true)
            try rejects { try storage.stage(oldKeys: row, newKeys: row, oldValues: row, newValues: row) }
            let old = row + row
            try storage.stage(oldKeys: old, newKeys: row, oldValues: old, newValues: row)
            storage.complete(success: false)
            try require(storage.snapshot().invalidated, "Canceled/failed staged QSA mirror remains reusable")
            try rejects { try storage.stage(oldKeys: old, newKeys: row, oldValues: old, newValues: row) }
        }
        try require(ledger.snapshot().reserved == 0, "Failed QSA mirror leaked its ledger reservation")
    }

    package static func escapedBufferAndDenial() throws {
        let ledger = try MemoryLedger(ceiling: 16_000_000, transientMargin: 1_000_000)
        let attention = try PinnedAttention(ledger: ledger, architecture: .medium)
        var escaped: MTLBuffer?
        try autoreleasepool {
            let storage = try attention.makeDecodeStorage(capacityTokens: 128)
            escaped = storage.keys
        }
        try require(escaped != nil && ledger.snapshot().reserved > 0, "Escaped resident QSA Metal buffer lost its ledger owner")
        escaped = nil
        try require(ledger.snapshot().reserved == 0, "Escaped QSA buffer did not retire")
        let denied = try MemoryLedger(ceiling: 16_384, transientMargin: 0)
        let constrained = try PinnedAttention(ledger: denied, architecture: .medium)
        try autoreleasepool { try rejects { _ = try constrained.makeDecodeStorage(capacityTokens: 1) } }
        try require(denied.snapshot().reserved == 0, "Second QSA allocation denial leaked the first buffer")
    }

    package static func encodedAppendOwnership() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("Metal unavailable")
        }
        let ledger = try MemoryLedger(ceiling: 16_000_000, transientMargin: 1_000_000)
        let fixture = try ledger.reserve(bytes: 1_000_000, category: "qsa_append_fixture")
        let attention = try PinnedAttention(ledger: ledger, architecture: .medium)
        try autoreleasepool {
            let storage = try attention.makeDecodeStorage(capacityTokens: 6)
            var keys = (0..<512).map { UInt16($0 &+ 100) }
            var values = (0..<512).map { UInt16($0 &+ 900) }
            for rows in [3, 1] {
                let k = (0..<(rows * 512)).map { UInt16(($0 * 17 + keys.count) % 65536) }
                let v = (0..<(rows * 512)).map { UInt16(($0 * 31 + values.count) % 65536) }
                let kb = try DecodeFrontierSelfChecks.buffer(k, device)
                let vb = try DecodeFrontierSelfChecks.buffer(v, device)
                let command = queue.makeCommandBuffer()!
                let ticket = try storage.stageEncoded(command, oldKeys: keys, oldValues: values, newKeys: kb, newValues: vb, rows: rows)
                try rejects { try storage.finishEncoded(ticket, success: true) }
                try rejects {
                    _ = try storage.stageEncoded(command, oldKeys: keys, oldValues: values, newKeys: kb, newValues: vb, rows: rows)
                }
                let other = try attention.makeDecodeStorage(capacityTokens: 6)
                try rejects { try other.finishEncoded(ticket, success: false) }
                try DecodeFrontierSelfChecks.joined(command)
                // CPU state publication precedes mirror commit. Every committed
                // prefix byte is included, not merely the newly written suffix.
                keys += k
                values += v
                for (buffer, expected) in [(storage.keys, keys), (storage.values, values)] {
                    let actual = Array(
                        UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt16.self), count: expected.count))
                    try require(actual == expected, "Device QSA append changed prefix/suffix bytes")
                }
                try storage.finishEncoded(ticket, success: true)
                try rejects { try storage.finishEncoded(ticket, success: true) }
                let snapshot = storage.snapshot()
                try require(
                    snapshot.committedTokens == keys.count / 512 && snapshot.seeded && !snapshot.invalidated,
                    "Encoded QSA append was not committed")
                try require(snapshot.uploadedBytes == 2048, "Encoded QSA append reuploaded committed KV/device suffix")
            }
            let row = try DecodeFrontierSelfChecks.buffer([UInt16](repeating: 0xffff, count: 512), device)
            let command = queue.makeCommandBuffer()!
            try rejects { _ = try storage.stageEncoded(command, oldKeys: [], oldValues: [], newKeys: row, newValues: row, rows: 1) }
            try rejects {
                _ = try storage.stageEncoded(command, oldKeys: keys, oldValues: values, newKeys: storage.keys, newValues: row, rows: 1)
            }
            let ticket = try storage.stageEncoded(command, oldKeys: keys, oldValues: values, newKeys: row, newValues: row, rows: 1)
            command.enqueue()
            try rejects { try storage.finishEncoded(ticket, success: false) }
            try DecodeFrontierSelfChecks.joined(command)
            try storage.finishEncoded(ticket, success: false)
            try require(
                storage.snapshot().invalidated && storage.snapshot().committedTokens == 5, "Joined canceled QSA append became committed")
            try rejects {
                _ = try storage.stageEncoded(
                    queue.makeCommandBuffer()!, oldKeys: keys, oldValues: values, newKeys: row, newValues: row, rows: 1)
            }

            let discarded = try attention.makeDecodeStorage(capacityTokens: 1)
            let unused = queue.makeCommandBuffer()!
            let pending = try discarded.stageEncoded(unused, oldKeys: [], oldValues: [], newKeys: row, newValues: row, rows: 1)
            try discarded.finishEncoded(pending, success: false)
            try require(
                discarded.snapshot().invalidated && discarded.snapshot().committedTokens == 0, "Unsubmitted QSA append remained reusable")
            // The discarded command is never submitted; its encoded readers
            // retain all backing until this autorelease pool retires.
        }
        try require(ledger.snapshot().reserved == 1_000_000, "Encoded QSA mirror leaked its reader allocation")
        try ledger.release(fixture)
    }
}
