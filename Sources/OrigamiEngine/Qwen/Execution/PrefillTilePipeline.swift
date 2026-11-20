import Foundation

/// Two-slot tile pipeline for whole prefill (`prefill_double_buffer`): tile t+1 is issued (encoded
/// and submitted into the other workspace) before tile t is finished (joined, flag-checked and
/// published), so the host work between tiles overlaps the GPU. Tiles finish strictly in order.
/// A tile whose check requires the original replay first has its already issued successor, which
/// was encoded from that tile's provisional state, joined and discarded; the successor is then
/// re-issued from the published host state. The pipeline owns no GPU objects: every closure
/// error, including one found at a deferred join, propagates from the call that made it, and the
/// caller's own joined cleanup covers whatever is still in flight.
package enum PrefillTilePipeline {
    package struct Counts: Equatable {
        /// Tiles finished only after their successor was issued.
        var deferred = 0
        /// Tiles that required the original replay; each discarded and re-issued its successor when it had one.
        var replays = 0, reissued = 0
        package init(deferred: Int = 0, replays: Int = 0, reissued: Int = 0) {
            self.deferred = deferred
            self.replays = replays
            self.reissued = reissued
        }
    }
    /// - issue: encodes and submits `tile` into workspace slot `slot`, chained from `predecessor`
    ///   when that tile is still in flight (nil for the first tile and after a replay).
    /// - finish: joins `tile`; true when it was checked and published, false when it was
    ///   discarded and the original replay is required.
    /// - discard: joins and discards an issued successor of a replayed tile before the replay.
    /// - replay: the host original path for `tile`.
    package static func run<Ticket>(
        tiles: Int,
        issue: (_ tile: Int, _ slot: Int, _ predecessor: Ticket?) throws -> Ticket,
        finish: (_ tile: Int, _ ticket: Ticket) throws -> Bool,
        discard: (_ tile: Int, _ ticket: Ticket) throws -> Void,
        replay: (_ tile: Int) throws -> Void
    ) throws -> Counts {
        guard tiles >= 0 else { throw EngineError("Prefill tile pipeline requires a non-negative tile count") }
        var counts = Counts()
        var tile = 0
        var inFlight: (tile: Int, ticket: Ticket)?
        while tile < tiles {
            guard let previous = inFlight else {
                inFlight = (tile, try issue(tile, tile % 2, nil))
                tile += 1
                continue
            }
            let next = try issue(tile, tile % 2, previous.ticket)
            inFlight = nil
            if try finish(previous.tile, previous.ticket) {
                counts.deferred += 1
                inFlight = (tile, next)
                tile += 1
            } else {
                // The successor read the invalid provisional state: retire it before the host
                // replay publishes, then re-issue it unchained on the next pass.
                try discard(tile, next)
                try replay(previous.tile)
                counts.replays += 1
                counts.reissued += 1
            }
        }
        if let last = inFlight {
            inFlight = nil
            if try !finish(last.tile, last.ticket) {
                try replay(last.tile)
                counts.replays += 1
            }
        }
        return counts
    }
}
