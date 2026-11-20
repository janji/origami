import Foundation
import Metal
import OrigamiEngine

private func denseRequire(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    guard try condition() else { throw EngineError(message) }
}
private func denseReject(_ body: () throws -> Void) throws {
    do { try body() } catch { return }
    throw EngineError("Invalid pinned dense input was accepted")
}

func pinnedDenseDispatchUsesSourceSchedules() throws {
    // Independently recorded host-dispatch outcomes at the projection shapes;
    // includes scalar dot and the N=1 GEMV switch.
    for (n, k, p24, p256) in [(4, 10240, 32, 32), (48, 2560, 32, 16), (512, 2560, 16, 2), (640, 2560, 8, 2)] {
        let a = try PinnedDenseLinear.plan(rows: 24, width: k, outputs: n)
        let b = try PinnedDenseLinear.plan(rows: 256, width: k, outputs: n)
        try denseRequire(
            a.kind == .splitK && a.splitKPartitions == p24 && a.splitKPartitionSize * p24 == k, "Wrong 24-row dense split-K dispatch")
        try denseRequire(
            b.kind == .splitK && b.splitKPartitions == p256 && b.splitKPartitionSize * p256 == k, "Wrong 256-row dense split-K dispatch")
    }
    let scalar = try PinnedDenseLinear.plan(rows: 1, width: 2560, outputs: 1)
    try denseRequire(
        scalar.kind == .dot && scalar.matrixIsInput && scalar.reductionSIMDGroups == 16,
        "Contiguous scalar projection missed the source dot-product specialization")
    for rows in [2, 8, 24, 160, 161, 256] {
        let plan = try PinnedDenseLinear.plan(rows: rows, width: 2560, outputs: 1)
        try denseRequire(
            plan.kind == .gemv && plan.matrixIsInput && plan.reductionSIMDGroups == (rows <= 160 ? 8 : 1),
            "Shared gate used the wrong stock GEMV reduction")
    }
    for (rows, k, n) in [(0, 2560, 48), (257, 2560, 48), (1, 128, 48), (1, 2560, 4), (1, 10240, 48), (1, 2560, 0)] {
        try denseReject { _ = try PinnedDenseLinear.plan(rows: rows, width: k, outputs: n) }
    }
}

func pinnedDenseInputsAndLedgerStayBounded() throws {
    let ledger = try MemoryLedger(ceiling: 500_000_000, transientMargin: 100_000_000)
    let kernels = try PinnedDenseLinear(ledger: ledger)
    try denseReject { _ = try kernels.linear([1], weights: [1], width: 2560, outputs: 48, outputBF16: true) }
    try denseReject { _ = try kernels.linear([Float](repeating: 0, count: 2560), weights: [1], width: 2560, outputs: 48, outputBF16: true) }
    let tiny = try MemoryLedger(ceiling: 16_384, transientMargin: 1)
    let denied = try PinnedDenseLinear(ledger: tiny)
    try denseReject {
        _ = try denied.linear(
            [Float](repeating: 0, count: 2560), weights: [Float](repeating: 0, count: 48 * 2560),
            width: 2560, outputs: 48, outputBF16: true)
    }
    try denseRequire(tiny.snapshot().reserved == 0 && ledger.snapshot().reserved == 0, "Rejected dense call leaked reservation")
}
