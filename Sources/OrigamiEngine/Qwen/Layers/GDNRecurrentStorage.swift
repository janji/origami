import Darwin
import Foundation
import Metal

/// One shared CPU/Metal allocation, charged once until its final state or
/// checkpoint reader retires. Layout: [recurrence output (outputCount FP32) | state 48 x 128 x 128 FP32
/// (value head, value row, key column) | optional 3 x 10240 BF16 convolution tail]. Only the producing
/// joined command may write it; every subsequent recurrence creates a different allocation, so a
/// checkpoint or MTP snapshot can share it without copying.
package final class GDNRecurrentStorage: @unchecked Sendable {
    package let buffer: MTLBuffer
    package let byteOffset: Int
    let convolutionByteOffset: Int?
    let count = 48 * 128 * 128
    package let allocationBytes: UInt64
    private let ledger: MemoryLedger
    private let reservation: MemoryLedger.Reservation

    package init(device: MTLDevice, outputCount: Int, ledger: MemoryLedger, retainConvolutionTail: Bool = false) throws {
        guard outputCount > 0, outputCount <= 4096 * 6144 else {
            throw EngineError("Invalid resident GDN output extent")
        }
        byteOffset = outputCount * 4
        // Use the host page alignment before reserving, including trailing
        // padding and the small recurrent output retained in the same buffer.
        convolutionByteOffset = retainConvolutionTail ? byteOffset + count * 4 : nil
        let page = Int(getpagesize())
        let requested = byteOffset + count * 4 + (retainConvolutionTail ? 3 * 10240 * 2 : 0)
        let bytes = ((requested + page - 1) / page) * page
        guard bytes <= device.maxBufferLength else { throw EngineError("Resident GDN exceeds Metal buffer limit") }
        let reservation = try ledger.reserve(bytes: UInt64(bytes), category: "gdn_resident_state")
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            try ledger.release(reservation)
            throw EngineError("Resident GDN allocation failed", code: "insufficient_memory")
        }
        self.buffer = buffer
        self.ledger = ledger
        self.reservation = reservation
        allocationBytes = UInt64(bytes)
    }
    deinit { try? ledger.release(reservation) }
    func withBytes<Result>(_ body: (UnsafeRawBufferPointer) throws -> Result) rethrows -> Result {
        try withExtendedLifetime(self) {
            try body(UnsafeRawBufferPointer(start: buffer.contents().advanced(by: byteOffset), count: count * 4))
        }
    }
    package func withConvolutionBytes<Result>(_ body: (UnsafeRawBufferPointer) throws -> Result) rethrows -> Result {
        precondition(convolutionByteOffset != nil)
        return try withExtendedLifetime(self) {
            try body(UnsafeRawBufferPointer(start: buffer.contents().advanced(by: convolutionByteOffset!), count: 3 * 10240 * 2))
        }
    }
    func array() -> [Float] {
        withBytes { Array($0.bindMemory(to: Float.self)) }
    }
}
