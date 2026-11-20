import Foundation

/// The prompt size from which both engines read a layer's experts before routing asks for them (Qwen's broad scan,
/// GLM's and DeepSeek's read-ahead). It selects reads only; what is computed never changes.
package enum PromptReadRows {
    /// The expected share of such a layer read that no row uses (0.1%: half an expert of 512). A model-free bound,
    /// not a measurement.
    package static let unusedShare = 0.001
    /// The fewest rows whose layer union, under evenly spread top-k routing, leaves at most `unusedShare` of a layer's
    /// experts unused: (1 − k/E)^rows ≤ unusedShare (Qwen 10 of 512: 351 rows; GLM 8 of 288: 246; DeepSeek 6 of 384:
    /// 439). Skewed routing leaves more unused: Qwen's coverage gate catches it; Family's unused reads cost only reads.
    package static func minimum(expertsPerToken k: Int, experts: Int) -> Int {
        guard k < experts else { return 1 }
        return max(1, Int((log(unusedShare) / log(1 - Double(k) / Double(experts))).rounded(.up)))
    }
}
