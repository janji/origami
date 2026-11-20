import Darwin
import Foundation
import OrigamiEngine

private func reductionRequire(_ value: Bool, _ message: String) throws {
    if !value { throw EngineError(message) }
}

func pinnedReductionsPreserveShapesAndSourceOrder() throws {
    let ledger = try MemoryLedger()
    let kernels = try PinnedReductions(ledger: ledger)
    try autoreleasepool {
        let group = 2560
        let width = group * 4
        let input = (0..<(3 * width)).map { Float(($0 / group) % 4 + 1) }
        let weights = (0..<width).map { Float($0 / group + 1) / 2 }
        let actual = try kernels.groupedRMS(x: input, weights: weights, groupSize: group)
        let expected = (0..<3).flatMap { _ in weights }
        try reductionRequire(actual == expected, "Grouped RMS mixed distinct groups or repeated weight rows")
        let fp32 = try kernels.softmax(rows: .init(repeating: 0, count: 2 * 10), width: 10)
        try reductionRequire(fp32.allSatisfy { $0.bitPattern == Float(0.1).bitPattern }, "Uniform FP32 softmax changed")
        let bf16 = try kernels.softmax(rows: .init(repeating: 0, count: 2 * 10), width: 10, outputBF16: true)
        try reductionRequire(bf16.allSatisfy { $0 == BF16.decode(BF16.encode(0.1)) }, "BF16 softmax failed its output boundary")
        // Sequential ranks sum to 0; the pinned 8 partials produce 2:
        // p0+p8 cancels first, then p1=1, p9=0, p2=1 are retained.
        let rankValues: [Float] = [16_777_216, 1, 1, 0, 0, 0, 0, 0, -16_777_216, 0]
        let experts = rankValues.flatMap { [Float](repeating: $0, count: 129) }
        let routed = try kernels.weightedExpertReduce10(experts: experts, weights: .init(repeating: 1, count: 10), hiddenSize: 129)
        try reductionRequire(routed == [Float](repeating: 2, count: 129), "Expert rank10 reduction lost its source partial order")
    }
    try reductionRequire(ledger.snapshot().reserved == 0, "Pinned reduction resources escaped synchronous retirement")
}

func pinnedReductionsRejectInvalidOrUnadmittedWork() throws {
    let ledger = try MemoryLedger()
    let kernels = try PinnedReductions(ledger: ledger)
    let cases: [() throws -> Void] = [
        { _ = try kernels.softmax(rows: [1], width: 0) },
        { _ = try kernels.softmax(rows: [1], width: 4097) },
        { _ = try kernels.softmax(rows: [1, 2, 3], width: 2) },
        { _ = try kernels.groupedRMS(x: [1], weights: [1], groupSize: 1) },
        {
            _ = try kernels.groupedRMS(
                x: .init(repeating: 1, count: 128), weights: .init(repeating: 1, count: 128), groupSize: 128, epsilon: .nan)
        },
        { _ = try kernels.weightedExpertReduce10(experts: .init(repeating: 1, count: 10), weights: [1], hiddenSize: 1) },
    ]
    for operation in cases {
        var rejected = false
        do { try operation() } catch is EngineError { rejected = true }
        try reductionRequire(rejected, "Invalid reduction geometry was accepted")
    }
    try reductionRequire(ledger.snapshot().reserved == 0, "Rejected reduction retained memory")
    // One page fits, the second allocation must fail. All earlier resources
    // must release their attached reservations before the throwing call returns.
    let limited = try MemoryLedger(ceiling: UInt64(getpagesize()) + 1, transientMargin: 0)
    let denied = try PinnedReductions(ledger: limited)
    var rejected = false
    do { _ = try denied.softmax(rows: .init(repeating: 1, count: 10), width: 10) } catch let error as EngineError {
        rejected = error.code == "insufficient_memory"
    }
    try reductionRequire(rejected && limited.snapshot().reserved == 0, "Reduction admission failure leaked its first buffer")
}
