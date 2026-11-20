import CryptoKit
import Darwin
import Foundation

/// A local file incarnation, not a portable content digest. Names and sizes alone alias different checkpoints.
/// Resolve symlinks and bind the path, file object and change times: replacing a file or rewriting it while restoring
/// its mtime changes this identity. Copies/moves intentionally lose cache reuse. This does not verify a publisher's
/// checksum or defend against a filesystem that forges/reuses all metadata; verified content receipts are separate.
package enum LocalModelIdentity {
    // An unreadable/non-regular source must never acquire an identity reusable after a process restart.
    private static let unavailableEpoch = UUID().uuidString

    package static func file(_ input: URL) -> Data {
        let url = input.standardizedFileURL.resolvingSymlinksInPath()
        var info = stat()
        guard stat(url.path, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG else {
            return fields(["unavailable-v1", unavailableEpoch, url.path])
        }
        return fields([
            "local-file-incarnation-v1", url.path, String(info.st_dev), String(info.st_ino),
            String(info.st_size), String(info.st_mtimespec.tv_sec), String(info.st_mtimespec.tv_nsec),
            String(info.st_ctimespec.tv_sec), String(info.st_ctimespec.tv_nsec),
            String(info.st_birthtimespec.tv_sec), String(info.st_birthtimespec.tv_nsec),
        ])
    }

    /// Length-prefix fields so arbitrary paths cannot make two sequences have the same serialized form.
    private static func fields(_ values: [String]) -> Data {
        Data(values.map { "\($0.utf8.count):\($0)" }.joined().utf8)
    }

    /// Small support/provenance files also distinguish local checkpoints. A declared digest is only a separator;
    /// it is never treated as evidence that the current weight bytes were verified against it.
    package static func updateSupport(_ hash: inout SHA256, directory: URL) {
        for name in [
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
            "download-provenance.json", ".origami-download.json", "origami-model.json",
        ] {
            let url = directory.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: url.path) {
                hash.update(data: fields([name]))
                hash.update(data: file(url))
            }
        }
    }

    /// The representation actually executed: the GGUF's own experts (the suffix keeps earlier identities).
    package static func familyRepresentation(source: String) -> String { source + "|expert-source" }
}
