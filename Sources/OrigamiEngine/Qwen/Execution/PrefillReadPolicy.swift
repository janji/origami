import Foundation

/// Broad reads are useful only after routing has demonstrated that a large scope
/// uses nearly the whole model. Small suffixes always read their actual unions.
/// This policy selects I/O only; row geometry and arithmetic never change.
package struct PrefillReadPolicy {
    /// `PromptReadRows.minimum` for this model's top-k and experts per layer (351 rows for 10 of 512; a pruned GGUF's
    /// fewer experts lower it). Skewed routing is caught by the coverage gate.
    package static var minimumBroadRows: Int {
        PromptReadRows.minimum(expertsPerToken: ModelContract.geometry["num_experts_per_tok"]!, experts: ModelLayout.routedExperts)
    }
    package static let minimumCoverage = 0.75
    private var learnedCoverage: Double?
    private var rows = 0, layers = 0, experts = 0
    package private(set) var usesBroadScan = false
    package init() {}
    package mutating func begin(rows: Int) -> Bool {
        self.rows = rows
        layers = 0
        experts = 0
        usesBroadScan = rows >= Self.minimumBroadRows && (learnedCoverage ?? 0) >= Self.minimumCoverage
        return usesBroadScan
    }
    package mutating func observe(experts: Int, domain: Int) {
        guard domain > 0, (0...domain).contains(experts) else { return }
        layers += 1
        self.experts += experts
        // The caller excludes the deliberately pruned terminal layer.
        if Double(experts) / Double(domain) < Self.minimumCoverage { learnedCoverage = nil }
    }
    package mutating func end(completed: Bool, domain: Int) {
        if completed && rows >= Self.minimumBroadRows && layers > 0 && domain > 0 {
            learnedCoverage = Double(experts) / Double(layers * domain)
        }
        rows = 0
        layers = 0
        experts = 0
        usesBroadScan = false
    }
}
