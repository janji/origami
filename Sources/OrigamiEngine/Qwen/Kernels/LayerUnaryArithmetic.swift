import Foundation

/// Immutable unary policy for BF16 inputs (sigmoid, SiLU, softplus with a BF16 result; sigmoid with an
/// FP32 result). `pinned` evaluates the GPU kernel `bf16_unary_pinned` once over all 65,536 BF16 inputs,
/// so a host lookup equals the GPU's value bit for bit (BITWISE; `pinned_unary_tables_domain_lifetime`)
/// without a dispatch per activation. `hostReference` (no tables) uses LayerArithmetic's libm code.
/// Algebra stays safe; only elementary functions use Metal's fast namespace.
public struct LayerUnaryArithmetic: Sendable {
    public static let pinnedStorageBytes: UInt64 = 655_360
    public static let hostReference = LayerUnaryArithmetic(backing: nil)
    private let backing: UnaryTableBacking?

    public var policy: String { backing == nil ? "host_libm_reference" : "pinned_safe_fast_unary_tables_v1" }

    /// MLX core (commit 1f8e74e): metal_kernel.h compiles with Safe math, and device.cpp leaves
    /// floating-point functions at Metal's default, Fast (MTLLibrary.h). Keep those two compiler
    /// choices independent. The generated library contains only unary primitives, so this policy
    /// cannot alter matrix or reduction arithmetic.
    public static func pinned(ledger: MemoryLedger) throws -> LayerUnaryArithmetic {
        let reservation = try ledger.reserve(bytes: pinnedStorageBytes, category: "unary_lookup_tables")
        var transferred = false
        defer { if !transferred { try? ledger.release(reservation) } }
        let scratch = try ledger.reserve(bytes: 2_097_152, category: "unary_table_generation_workspace")
        defer { try? ledger.release(scratch) }
        return try autoreleasepool {
            let kernels = try ReferenceKernels.pinnedUnaryKernels()
            let inputs = (0..<65_536).map { BF16.decode(UInt16($0)) }
            let sigmoid = try kernels.pinnedUnary(x: inputs, operation: .sigmoid, native: true).map(BF16.encode)
            let silu = try kernels.pinnedUnary(x: inputs, operation: .silu, native: true).map(BF16.encode)
            let softplus = try kernels.pinnedUnary(x: inputs, operation: .softplus, native: true).map(BF16.encode)
            let wideSigmoid = try kernels.pinnedUnary(x: inputs, operation: .sigmoidFP32, native: true)
            let owner = UnaryTableBacking(
                sigmoid: sigmoid, silu: silu, softplus: softplus,
                wideSigmoid: wideSigmoid, ledger: ledger, reservation: reservation)
            transferred = true
            // Only immutable Swift arrays escape. Metal autoreleases drain
            // before the outer scratch reservation is retired on either path.
            return LayerUnaryArithmetic(backing: owner)
        }
    }

    @inline(__always) public func sigmoid(_ x: Float) -> Float {
        guard let backing else { return LayerArithmetic.sigmoid(x) }
        return BF16.decode(backing.sigmoid[Int(BF16.encode(x))])
    }

    @inline(__always) public func silu(_ x: Float) -> Float {
        guard let backing else { return LayerArithmetic.silu(x) }
        return BF16.decode(backing.silu[Int(BF16.encode(x))])
    }

    @inline(__always) public func softplus(_ x: Float) -> Float {
        guard let backing else { return LayerArithmetic.softplus(x) }
        return BF16.decode(backing.softplus[Int(BF16.encode(x))])
    }

    /// FP32 output over the BF16 input domain. GDN widens its already-BF16 z
    /// projection before evaluating sigmoid; no BF16 rounding follows sigmoid.
    @inline(__always) public func sigmoidFP32(_ x: Float) -> Float {
        guard let backing else { return LayerArithmetic.sigmoidFP32(x) }
        return backing.wideSigmoid[Int(BF16.encode(x))]
    }
}

/// A copied arithmetic value can outlive its ReferenceKernels instance. Bind
/// its reservation to the immutable arrays, not the originating kernel owner.
private final class UnaryTableBacking: @unchecked Sendable {
    let sigmoid, silu, softplus: [UInt16]
    let wideSigmoid: [Float]
    private let ledger: MemoryLedger
    private let reservation: MemoryLedger.Reservation

    init(
        sigmoid: [UInt16], silu: [UInt16], softplus: [UInt16], wideSigmoid: [Float],
        ledger: MemoryLedger, reservation: MemoryLedger.Reservation
    ) {
        self.sigmoid = sigmoid
        self.silu = silu
        self.softplus = softplus
        self.wideSigmoid = wideSigmoid
        self.ledger = ledger
        self.reservation = reservation
    }

    deinit { try? ledger.release(reservation) }
}
