import Foundation

/// Console output: one rewritable progress line on a terminal, plain lines otherwise.
enum Terminal {
    static let interactive = isatty(STDOUT_FILENO) == 1 && isatty(STDIN_FILENO) == 1
    nonisolated(unsafe) private static var lastLength = 0
    private static let lock = NSLock()

    /// Shows `line` as the current progress, or clears it when nil.
    static func progress(_ line: String?) {
        lock.lock()
        defer { lock.unlock() }
        if interactive {
            let text = line ?? ""
            FileHandle.standardOutput.write(
                Data(("\r" + text + String(repeating: " ", count: max(0, lastLength - text.count)) + (line == nil ? "\r" : "")).utf8))
            lastLength = line == nil ? 0 : text.count
        } else if let line {
            print(line)
        }
    }

    /// A progress reporter for `done` of `total` steps, at most one update per second.
    static func steps(_ label: String) -> @Sendable (Int, Int) -> Void {
        let state = ThrottleState()
        return { done, total in
            guard state.due(final: done == total) else { return }
            progress(String(format: "  %@ %3.0f%%", label, 100 * Double(done) / Double(max(1, total))))
            if done == total {
                progress(nil)
                print("  \(label) done")
            }
        }
    }
    /// When a `steps` reporter last printed; its callers report from worker threads.
    private final class ThrottleState: @unchecked Sendable {
        private let lock = NSLock()
        private var last = Date.distantPast
        func due(final: Bool) -> Bool {
            lock.lock()
            defer { lock.unlock() }
            guard final || Date().timeIntervalSince(last) >= 1 else { return false }
            last = Date()
            return true
        }
    }

    /// Asks a yes/no question; `yes` (from --yes) answers it. Without a terminal the answer is no.
    static func confirm(_ question: String, yes: Bool) -> Bool {
        if yes { return true }
        guard interactive else {
            print("\(question) (pass --yes to confirm)")
            return false
        }
        print("\(question) [y/N] ", terminator: "")
        return ["y", "yes"].contains(readLine()?.lowercased().trimmingCharacters(in: .whitespaces) ?? "")
    }

    /// One plain line for a file-system error (an unwritable, full or missing folder) instead of Foundation's own text:
    /// the folder and the fix. Nil for any other error.
    static func fileProblem(_ error: Error) -> String? {
        let error = error as NSError
        guard error.domain == NSCocoaErrorDomain || error.domain == NSPOSIXErrorDomain else { return nil }
        let underlying = error.userInfo[NSUnderlyingErrorKey] as? NSError
        guard
            let path = error.userInfo[NSFilePathErrorKey] as? String ?? (error.userInfo[NSURLErrorKey] as? URL)?.path
                ?? underlying?.userInfo[NSFilePathErrorKey] as? String
        else { return nil }
        let posix = error.domain == NSPOSIXErrorDomain ? error.code : underlying?.domain == NSPOSIXErrorDomain ? underlying?.code : nil
        let folder = URL(fileURLWithPath: path).deletingLastPathComponent().path
        let other = "or choose another folder with --data-dir or ORIGAMI_DATA_DIR"
        if [EACCES, EPERM, EROFS].map(Int.init).contains(posix ?? -1)
            || [CocoaError.fileWriteNoPermission, .fileReadNoPermission, .fileWriteVolumeReadOnly].map(\.rawValue).contains(error.code)
        {
            return "Origami cannot write to \(folder): give yourself write access (`chmod u+w \"\(folder)\"`), " + other
        }
        if posix == Int(ENOSPC) || error.code == CocoaError.fileWriteOutOfSpace.rawValue {
            return "the drive holding \(folder) is full: free some space there, " + other
        }
        return nil
    }
}
