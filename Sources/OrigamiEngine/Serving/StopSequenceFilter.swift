import Foundation

/// Request `stop` strings over the streamed reply text, with the pinned SlotStream `Engine.feed` semantics:
/// stops apply only after `</think>` when the reply opens in reasoning, the earliest match wins, and a possible
/// stop prefix (up to the longest stop minus one Unicode scalar, or seven scalars while waiting for `</think>`)
/// is held back so it never leaks into emitted text. At most four stops of 128 scalars. No model or tokenizer
/// dependency. Derived from the MIT-licensed SlotStream implementation; its notice is in THIRD_PARTY_NOTICES.md.
public final class StopSequenceFilter {
    private let stops: [String]
    private let holdBack: Int
    private var pending = ""
    private var inReasoning: Bool
    public private(set) var didStop = false
    /// The stop string that ended the reply (the earliest match).
    public private(set) var matched: String?
    public var bufferedScalarCount: Int { pending.unicodeScalars.count }
    public init(stops: [String], reasoningOpen: Bool) throws {
        guard stops.count <= 4, stops.allSatisfy({ $0.unicodeScalars.count <= 128 && $0.utf8.count <= 512 }) else {
            throw ServingFailure("stop supports at most four strings of 128 Unicode scalars each")
        }
        self.stops = stops.filter { !$0.isEmpty }
        holdBack = max(0, (self.stops.map { $0.unicodeScalars.count }.max() ?? 1) - 1)
        inReasoning = reasoningOpen && !self.stops.isEmpty
    }
    public func push(_ piece: String) -> String { feed(piece, final: false) }
    public func finish() -> String { feed("", final: true) }
    private func feed(_ piece: String, final: Bool) -> String {
        guard !didStop else { return "" }
        pending += piece
        var output = ""
        if inReasoning {
            if let end = pending.range(of: "</think>", options: .literal) {
                output += pending[..<end.upperBound]
                pending = String(pending[end.upperBound...])
                inReasoning = false
            } else {
                return release(final ? pending.unicodeScalars.count : max(0, pending.unicodeScalars.count - 7))
            }
        }
        var earliest: String.Index?
        // Literal (scalar-exact) matching: canonical-equivalence matches could span more scalars than the holdback.
        for stop in stops {
            if let match = pending.range(of: stop, options: .literal), earliest == nil || match.lowerBound < earliest! {
                earliest = match.lowerBound
                matched = stop
            }
        }
        if let earliest {
            output += pending[..<earliest]
            pending = ""
            didStop = true
            return output
        }
        output += release(final ? pending.unicodeScalars.count : max(0, pending.unicodeScalars.count - holdBack))
        return output
    }
    private func release(_ count: Int) -> String {
        let scalars = pending.unicodeScalars
        let result = String(String.UnicodeScalarView(scalars.prefix(count)))
        pending = String(String.UnicodeScalarView(scalars.dropFirst(count)))
        return result
    }
}
