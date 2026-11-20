import Foundation

/// Thrown by a check whose optional input (a Metal device, a real model, a reference capture named by an
/// environment variable) is absent: `engine-checks` reports it as skipped with this reason.
public struct FixtureUnavailable: Error, CustomStringConvertible {
    public let description: String
    public init(description: String) { self.description = description }
}

/// Where `engine-checks` finds its inputs: the small files in `Tests/Fixtures` and a scratch folder in the
/// per-user temporary directory.
package enum CheckFixtures {
    /// A small config or trace in `Tests/Fixtures`, from the package root or the folder above it.
    package static func file(_ relative: String) -> URL {
        let candidates = ["Tests/Fixtures/" + relative, "engine/Tests/Fixtures/" + relative]
        return URL(fileURLWithPath: candidates.first { FileManager.default.fileExists(atPath: $0) } ?? candidates[0])
    }
    /// Speed figures some checks take beside their assertions; `engine-checks` prints them under `measurements`.
    nonisolated(unsafe) package static var measurements: [[String: Any]] = []
    /// A scratch folder for store and checkpoint checks, removed by the caller.
    package static var scratch: URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("origami-checks", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
