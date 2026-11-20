import Foundation
import OrigamiEngine

/// What `model add owner/repo` and `setup --model owner/repo` refuse before downloading (audit J8), from the
/// repo's reports as the dashboard reads them: GGUF files or several checkpoints (one is downloaded by hand), nothing usable,
/// a refused model.
func downloadRefusalChecks() -> [(String, () throws -> Void)] {
    [
        ("download_refusal_before_any_bytes", downloadRefusalBeforeAnyBytes),
        ("download_resume_trusts_record_not_size", downloadResumeTrustsRecordNotSize),
    ]
}

/// A file already in the folder at the listed size is kept only when the finished download's record names it at this
/// revision, size and digest; a same-size file of another revision (or no record) is hashed, or fetched when unlisted.
private func downloadResumeTrustsRecordNotSize() throws {
    let files: [(name: String, size: Int64, sha256: String?)] = [("model.safetensors", 3, "aa"), ("config.json", 10, nil)]
    let record = try DownloadResume.record(repo: "o/r", revision: "r1", files: files)
    func decide(_ name: String, _ existing: Int64?, _ size: Int64, _ sha: String?, _ revision: String, _ record: Data?) -> DownloadResume {
        DownloadResume.decide(name: name, existing: existing, size: size, sha256: sha, revision: revision, record: record)
    }
    let cases: [(DownloadResume, DownloadResume, String)] = [
        (decide("model.safetensors", 3, 3, "aa", "r1", record), .keep, "recorded at this revision"),
        (decide("config.json", 10, 10, nil, "r1", record), .keep, "recorded small file"),
        (decide("model.safetensors", 3, 3, "bb", "r2", record), .hash, "same size, new revision"),
        (decide("model.safetensors", 3, 3, "bb", "r1", record), .hash, "same revision, other digest"),
        (decide("config.json", 10, 10, nil, "r2", record), .fetch, "unhashable file of an older revision"),
        (decide("model.safetensors", 3, 3, "aa", "r1", nil), .hash, "no record (an interrupted download)"),
        (
            decide("model.safetensors", 3, 3, "aa", "r1", Data(#"{"repo":"o/r","revision":"r1"}"#.utf8)), .hash,
            "an older record without files"
        ),
        (decide("model.safetensors", 2, 3, "aa", "r1", record), .fetch, "a shorter file"),
        (decide("model.safetensors", nil, 3, "aa", "r1", record), .fetch, "no file"),
    ]
    for (got, expected, what) in cases where got != expected {
        throw EngineError("download resume, \(what): \(got), expected \(expected)")
    }
}

private func downloadRefusalBeforeAnyBytes() throws {
    /// A report as `check(hub:)` gives it (decoded: its fields are the engine's own).
    func report(_ format: String, family: String? = "glm-5.3-flash", verdict: String = "runs", source: String = "o/r") throws
        -> CompatibilityReport
    {
        let fields: [String: Any] = [
            "source": source, "format": format, "family": family ?? NSNull(), "architecture": "?", "types": [String: Any](),
            "bytes": [String: Any](), "totalBytes": 1, "verdict": verdict, "reason": "a reason",
        ]
        return try JSONDecoder().decode(CompatibilityReport.self, from: JSONSerialization.data(withJSONObject: fields))
    }
    let cases: [([CompatibilityReport], String?)] = [
        ([try report("gguf"), try report("gguf")], "holds GGUF files"),
        ([try report("gguf")], "holds GGUF files"),
        ([], "holds no safetensors or GGUF model"),
        ([try report("safetensors", verdict: "refused")], "cannot use o/r: a reason"),
        ([try report("safetensors", family: nil)], "cannot use o/r"),
        ([try report("safetensors")], nil),
        ([try report("safetensors", verdict: "converts")], nil),
        (
            [try report("safetensors"), try report("safetensors", source: "o/r/fp8")],
            "holds 2 safetensors checkpoints (the top folder, fp8): download the folder of the one you want"
        ),
    ]
    for (reports, expected) in cases {
        let refusal = ModelCompatibility.downloadRefusal("o/r", reports: reports)
        guard expected.map({ refusal?.contains($0) == true }) ?? (refusal == nil) else {
            throw EngineError("download refusal: \(reports.map(\.format)) gave \(refusal ?? "nil"), expected \(expected ?? "nil")")
        }
    }
}
