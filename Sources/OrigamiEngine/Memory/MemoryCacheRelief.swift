import Foundation

/// Boundary diagnostics, not a claim that a process's physical footprint fell by this amount.
/// Released bytes are expert reservations whose last owner retired after the latest request;
/// unrelated workspace releases are excluded. A new request starts a new observation interval.
public struct MemoryCacheReliefSnapshot: Codable, Sendable, Equatable {
    public let requestCount: UInt64
    public let requestedTargetBytes: UInt64?
    public let cacheBytesAtLastRequest, cacheBytesReleasedSinceLastRequest, cacheOwnedBytes: UInt64
}
