import Foundation
import OrigamiEngine
import OrigamiEngineChecks

/// Deployment gate for both long fused partition counts, including the production 128K context.
/// The engine wrapper reaches the actual paged encoder and asserts its fused dispatch counter.
func longFusedAttentionDeploymentParity() throws {
    try FollowupPagePrefillSelfChecks.longFusedAttention()
}
