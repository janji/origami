import Foundation
import OrigamiEngine

/// Content receipts are explicit offline tooling. No Settings, model loader, server, prompt cache or GPU is used.
enum ContentCommand {
    static func run(_ arguments: [String], flags: [String: [String]], json: Bool) throws {
        guard arguments.count == 2, ["verify", "import", "check"].contains(arguments[0]) else {
            throw CLIError.usage(
                "Use `origami content verify <manifest.json> --root <folder> --receipt <output.json>`, `content import <receipt.json> --root <folder> --receipt <output.json>`, or `content check <receipt.json>`"
            )
        }
        for (key, values) in flags where values.count != 1 { throw CLIError.usage("--\(key) must be specified once") }
        let operation = arguments[0]
        let input = URL(fileURLWithPath: Settings.expand(arguments[1]))
        let receipt: VerifiedContentReceipt.Receipt
        if operation == "check" {
            guard flags.isEmpty else { throw CLIError.usage("content check takes only --json") }
            receipt = try VerifiedContentReceipt.readReceipt(input)
            try VerifiedContentReceipt.check(receipt)
        } else {
            guard let root = flags["root"]?.first, let output = flags["receipt"]?.first else {
                throw CLIError.usage("content \(operation) requires --root and --receipt; it reads every listed byte")
            }
            let destination = URL(fileURLWithPath: Settings.expand(output))
            if operation == "verify",
                input.standardizedFileURL.resolvingSymlinksInPath() == destination.standardizedFileURL.resolvingSymlinksInPath()
            {
                throw CLIError.usage("The receipt output must differ from the input manifest")
            }
            let manifest =
                try operation == "verify"
                ? VerifiedContentReceipt.readManifest(input)
                : VerifiedContentReceipt.readReceipt(input).manifest
            // Import intentionally ignores the old receipt's local bindings and rechecks every expected digest.
            if !json {
                let bytes = manifest.files.reduce(UInt64(0)) { $0 + $1.size }
                print("Verifying \(manifest.files.count) listed files (\(bytes) bytes). No serving or cache identity will change.")
            }
            receipt = try VerifiedContentReceipt.verify(manifest, root: URL(fileURLWithPath: Settings.expand(root)), writingTo: destination)
        }
        let result: [String: Any] = [
            "status": operation == "check" ? "incarnation_bindings_match" : "content_verified",
            "content_identity": receipt.contentIdentity, "files": receipt.manifest.files.count,
            "bytes": receipt.manifest.files.reduce(UInt64(0)) { $0 + $1.size },
            "root": receipt.root, "verified_at": receipt.verifiedAt,
            "scope": "Only files listed in the ordered manifest; publisher authenticity and model completeness are not established",
            "cache_identity_changed": false,
            "payload_rehashed": operation != "check",
        ]
        if json {
            let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
            print(String(decoding: data, as: UTF8.self))
        } else {
            print(
                operation == "check"
                    ? "Local receipt bindings match (payload bytes were not reread)."
                    : "All listed bytes match; the local receipt was published atomically.")
            print("Content identity: \(receipt.contentIdentity)")
            print("Scope: listed files only. Expected hashes must come from a trusted source; model completeness is not inferred.")
        }
    }
}
