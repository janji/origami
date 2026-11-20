import Foundation

/// Ends a think block early: at the request's thinking budget, or once only the answer room is left
/// (max(1024, 15% of the output budget)), so a capped reply ends with an answer, not `length` and no content.
/// The answer room is at most half the output budget (rounded up), so a small cap still thinks, as asked or not.
public final class ThinkCloser {
    public static func limit(maximumTokens: Int, budget: Int?) -> Int {
        min(budget ?? .max, maximumTokens - min(max(1024, maximumTokens * 15 / 100), (maximumTokens + 1) / 2))
    }
    private let limit: Int, budget: Int?, end: Int, closing: [Int]
    private var open: Bool
    public private(set) var closedBy: String?
    public init(open: Bool, maximumTokens: Int, budget: Int?, end: Int, closing: [Int]) {
        self.open = open
        self.budget = budget
        self.end = end
        self.closing = closing
        limit = Self.limit(maximumTokens: maximumTokens, budget: budget)
    }
    /// Tokens to write instead of sampling, given the output so far; nil to sample.
    public func next(_ output: [Int]) -> [Int]? {
        guard open else { return nil }
        if output.last == end {
            open = false
            return nil
        }
        guard output.count >= limit else { return nil }
        open = false
        closedBy = limit == budget ? "budget" : "answer_room"
        return closing
    }
}
