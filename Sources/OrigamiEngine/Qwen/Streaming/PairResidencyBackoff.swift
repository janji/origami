/// A resident-only pair must replay the whole pair when any expert is missing. Repeatedly trying that path on an
/// SSD-streamed checkpoint adds discarded GPU work to every verification. Probe again after 1, 3, 7, 15, then at
/// most 31 streamed pairs, so a warmer cache can recover without a permanent decision or a per-layer cache scan.
/// This selects an execution schedule only; every streamed pair still evaluates both complete target rows.
package struct PairResidencyBackoff {
    private var misses = 0
    private var remaining = 0

    package init() {}

    package mutating func shouldProbe() -> Bool {
        guard remaining > 0 else { return true }
        remaining -= 1
        return false
    }

    package mutating func recordMiss() {
        misses = min(misses + 1, 5)
        remaining = (1 << misses) - 1
    }

    package mutating func recordHit() { self = .init() }
}
