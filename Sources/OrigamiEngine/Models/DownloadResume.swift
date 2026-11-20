import Foundation

/// What a download (`origami model add owner/repo`) does with a file already in its folder: kept only when the last
/// finished download recorded it at this revision with this size and digest; else hashed against the listing's SHA-256
/// (weight files are LFS files, listed with one) or fetched again. Size alone proves nothing: a file of an older revision,
/// or a damaged one, can have the same length.
package enum DownloadResume: Equatable {
    case keep, hash, fetch

    package static func decide(name: String, existing: Int64?, size: Int64, sha256: String?, revision: String, record: Data?) -> Self {
        guard existing == size else { return .fetch }
        let fields = record.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        if fields?["revision"] as? String == revision,
            let file = (fields?["files"] as? [String: Any])?[name] as? [String: Any],
            (file["size"] as? NSNumber)?.int64Value == size, file["sha256"] as? String == sha256
        {
            return .keep
        }
        return sha256 == nil ? .fetch : .hash
    }

    /// The record a finished download leaves (`.origami-download.json`): the repo, its revision and every file fetched.
    package static func record(repo: String, revision: String, files: [(name: String, size: Int64, sha256: String?)]) throws -> Data {
        let listed = Dictionary(
            uniqueKeysWithValues: files.map { ($0.name, ["size": $0.size, "sha256": $0.sha256 ?? NSNull()] as [String: Any]) })
        return try JSONSerialization.data(
            withJSONObject: ["repo": repo, "revision": revision, "files": listed], options: [.prettyPrinted, .sortedKeys])
    }
}
