import Foundation
import OrigamiEngine

func stopSequenceHoldbackMatchesAcrossEveryScalarBoundary() throws {
    let raw = "STOP inside reasoning</think>\nAnswer 🌧️end discarded"
    let scalars = Array(raw.unicodeScalars)
    let expected = "STOP inside reasoning</think>\nAnswer "
    for boundary in 0...scalars.count {
        let filter = try StopSequenceFilter(stops: ["STOP", "🌧️end"], reasoningOpen: true)
        var result = filter.push(String(String.UnicodeScalarView(scalars.prefix(boundary))))
        result += filter.push(String(String.UnicodeScalarView(scalars.dropFirst(boundary))))
        result += filter.finish()
        guard result == expected, filter.didStop, filter.bufferedScalarCount == 0,
            filter.push("more").isEmpty
        else { throw EngineError("Stop filter changed at scalar boundary \(boundary)") }
    }
    let fragmented = try StopSequenceFilter(stops: ["STOP", "🌧️end"], reasoningOpen: true)
    var result = ""
    for scalar in scalars {
        result += fragmented.push(String(scalar))
        guard fragmented.bufferedScalarCount <= 7 else { throw EngineError("Stop holdback exceeded scalar bound") }
    }
    guard result == expected, fragmented.didStop else { throw EngineError("Scalar-fragmented stop leaked") }
    let earliest = try StopSequenceFilter(stops: ["bar", "foo"], reasoningOpen: false)
    guard earliest.push("afoobar") == "a", earliest.didStop else { throw EngineError("Stop did not choose earliest match") }
}

func stopSequenceFlushPreservesUnmatchedSuffixAndReasoning() throws {
    let filter = try StopSequenceFilter(stops: ["done"], reasoningOpen: false)
    let prefix = filter.push("abcdon")
    guard prefix + filter.finish() == "abcdon", !filter.didStop else { throw EngineError("Incomplete stop prefix disappeared") }
    let reasoning = try StopSequenceFilter(stops: ["done"], reasoningOpen: true)
    let first = reasoning.push("done before a missing closing tag")
    guard first + reasoning.finish() == "done before a missing closing tag", !reasoning.didStop else {
        throw EngineError("Stop truncated unfinished reasoning")
    }
    let disabled = try StopSequenceFilter(stops: [""], reasoningOpen: true)
    guard disabled.push("unfiltered") == "unfiltered", disabled.bufferedScalarCount == 0 else {
        throw EngineError("Empty stop should have no holdback")
    }
    for stops in [[String](repeating: "x", count: 5), [String(repeating: "😀", count: 129)]] {
        do {
            _ = try StopSequenceFilter(stops: stops, reasoningOpen: false)
            throw EngineError("Oversized stop accepted")
        } catch is ServingFailure {}
    }
}

/// Stops match literally, scalar for scalar, at every split of the text: a canonically equivalent spelling (é as e plus
/// a combining accent) is not a stop, and `</think>` followed by a combining mark still ends the reasoning.
func stopSequenceMatchesLiterally() throws {
    for (raw, stops, reasoning, expected, stopped) in [
        ("cafe\u{301} ok", ["\u{E9}"], false, "cafe\u{301} ok", false),
        ("a</think>\u{301}x tail", ["x"], true, "a</think>\u{301}", true),
    ] {
        let scalars = Array(raw.unicodeScalars)
        for boundary in 0...scalars.count {
            let filter = try StopSequenceFilter(stops: stops, reasoningOpen: reasoning)
            var result = filter.push(String(String.UnicodeScalarView(scalars.prefix(boundary))))
            result += filter.push(String(String.UnicodeScalarView(scalars.dropFirst(boundary))))
            result += filter.finish()
            guard Array(result.unicodeScalars) == Array(expected.unicodeScalars), filter.didStop == stopped else {
                throw EngineError(
                    "Stop \(stops) over \(Array(raw.unicodeScalars)) split at \(boundary) gave \(Array(result.unicodeScalars))")
            }
        }
    }
}
