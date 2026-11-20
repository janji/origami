import Foundation

/// A model's usage curve for its hot copy: for a byte budget, the share of counted expert uses that the records a copy
/// of that size holds account for. `order` is the keeper's own ranking (its `choose` with no size limit), and a copy of a
/// budget takes that order's records that fit, skipping the rest, as `HotCopy.choose` does: the figure describes the
/// copy that would really be built. `full` is the budget at which the curve stops growing (every counted record held).
package struct HotCopyCurve: Sendable {
    /// What stopped the automatic size: no counts yet, the drive's free space, a flat curve, or the whole expert set.
    package enum Limit: String, Sendable {
        case noCounts = "no_counts"
        case freeSpace = "free_space"
        case flat
        case wholeSet = "whole_set"
    }
    private let records: [(bytes: UInt64, uses: Double)]
    /// Expert uses counted (the curve's 100%).
    package let counted: Double
    package let full: UInt64

    package init(uses: [Int: Int], order: [Int], size: (Int) -> UInt64) {
        records = order.map { (size($0), Double(uses[$0] ?? 0)) }
        counted = uses.values.reduce(0) { $0 + Double($1) }
        var total: UInt64 = 0
        var full: UInt64 = 0
        for record in records {
            total += record.bytes
            if record.uses > 0 { full = total }
        }
        self.full = full
    }
    /// A family model's curve, in `HotCopy.choose`'s order: most missed, then most used.
    package init(uses: [Int: Int], misses: [Int: Int], source: HotCopySource) {
        self.init(
            uses: uses, order: HotCopy.choose(uses, misses: misses, source: source, budget: .max),
            size: { HotCopy.size($0, source: source) ?? 0 })
    }

    /// The share (0...1) of counted uses a copy of `budget` bytes holds; 0 with none counted.
    package func coverage(_ budget: UInt64) -> Double {
        guard counted > 0 else { return 0 }
        var total: UInt64 = 0
        var held = 0.0
        for record in records where total + record.bytes <= budget {
            total += record.bytes
            held += record.uses
        }
        return held / counted
    }

    /// The dashboard's popularity graph: `order`'s records in at most `bars` runs of equal numbers of records, each as
    /// [records, bytes, summed uses, its first record's uses, its last's], from the same records `coverage` sums.
    package func popularity(bars: Int = 240) -> [String: Any] {
        let n = records.count
        let count = min(bars, n)
        let runs = (0..<count).map { records[$0 * n / count..<($0 + 1) * n / count] }
        return [
            "bars": runs.map {
                [
                    $0.count, Int($0.reduce(0) { $0 + $1.bytes }), Int($0.reduce(0) { $0 + $1.uses }), Int($0.first!.uses),
                    Int($0.last!.uses),
                ]
            },
            "counted_uses": Int(counted), "experts": n, "bytes": records.reduce(0) { $0 + $1.bytes },
        ]
    }

    /// The automatic size, from `floor` (the size without counts) upward in steps of a tenth of `whole` (the model's
    /// expert bytes) while each step pays: it adds at least a fifth of the coverage the same bytes would add if every
    /// expert were used alike (2% per 10%; a last, shorter step up to the room is taken). Never beyond `room` (the drive's free space above the keeper's reserve),
    /// `whole`, or `full`; never below `floor` (clamped to `room`).
    package func automatic(floor: UInt64, room: UInt64, whole: UInt64) -> (bytes: UInt64, limit: Limit) {
        let limit = min(room, whole)
        let start = min(floor, limit)
        guard counted > 0, whole > 0 else { return (start, .noCounts) }
        let step = max(whole / 10, 1)
        var bytes = start
        var low = coverage(bytes)
        var stop: Limit = whole <= room ? .wholeSet : .freeSpace
        while bytes < limit {
            let next = limit - bytes > step ? bytes + step : limit
            let high = coverage(next)
            // A last step shorter than a tenth only reaches the room: too few records fit it to judge the curve.
            if next - bytes == step, (high - low) * Double(whole) < 0.2 * Double(step) {
                stop = .flat
                break
            }
            (bytes, low) = (next, high)
        }
        if full < bytes, bytes > start { return (max(start, full), .flat) }
        return (bytes, stop)
    }
}
