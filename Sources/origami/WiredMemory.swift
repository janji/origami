import Darwin
import Foundation
import Metal
import OrigamiEngine

/// The dashboard's explicit manual change. This installs no root helper or boot setting: sysctl lasts until reboot.
/// Apple documents native authorization for `do shell script ... with administrator privileges` here:
/// https://developer.apple.com/library/archive/documentation/AppleScript/Conceptual/AppleScriptLangGuide/reference/ASLR_cmds.html
enum WiredMemory {
    struct Snapshot: Sendable {
        let physicalMemoryBytes: UInt64
        let maximumBudgetBytes: UInt64
        /// Zero means the driver's automatic setting. Missing/negative readings remain unavailable.
        let currentLimitMiB: UInt64?
        let effectiveLimitMiB: UInt64?
        /// A performance recommendation, not the iogpu wired ceiling or a guarantee of free RAM.
        /// https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize
        let metalRecommendedBytes: UInt64?
        let error: String?
    }
    struct Receipt: Sendable {
        let beforeLimitMiB: UInt64
        let afterLimitMiB: UInt64
        let raised: Bool
        let authorizationRequested: Bool
    }
    struct Failure: Error, CustomStringConvertible, Sendable {
        let message: String
        let observedLimitMiB: UInt64?
        var description: String {
            message
                + (observedLimitMiB.map {
                    $0 == 0 ? " macOS currently uses its automatic GPU memory limit." : " macOS currently reports \($0) MiB."
                }
                    ?? " The current macOS ceiling could not be verified.")
                + " Origami settings were not committed by this operation. A system ceiling already raised is not lowered."
        }
    }
    private static let mutationGate = DispatchSemaphore(value: 1)

    static func snapshot() -> Snapshot {
        let physical = ProcessInfo.processInfo.physicalMemory
        let value = Result { try readLimitMiB() }
        let raw = try? value.get()
        let recommended = MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize
        let effective = raw.flatMap { limit -> UInt64? in
            guard limit == 0 else { return limit }
            guard let recommended, recommended >= WiredMemoryPolicy.mebibyte, recommended <= physical else { return nil }
            return recommended / WiredMemoryPolicy.mebibyte
        }
        return Snapshot(
            physicalMemoryBytes: physical,
            maximumBudgetBytes: WiredMemoryPolicy.maximumBudgetBytes(physicalMemoryBytes: physical),
            currentLimitMiB: raw, effectiveLimitMiB: effective.flatMap { $0 > 0 ? $0 : nil }, metalRecommendedBytes: recommended,
            error: {
                if case .failure(let error) = value { return String(describing: error) }
                if raw == 0, effective == nil { return "macOS uses an automatic ceiling but Metal did not report a valid default." }
                return nil
            }())
    }

    /// Reads the kernel directly, without a shell or authorization. Preserve zero as the automatic setting;
    /// negatives, an unsupported sysctl, or an unknown representation are unavailable.
    private static func readLimitMiB() throws -> UInt64 {
        var raw: Int64 = 0
        var size = MemoryLayout<Int64>.size
        let status = sysctlbyname("iogpu.wired_limit_mb", &raw, &size, nil, 0)
        guard status == 0 else {
            throw EngineError("Cannot read iogpu.wired_limit_mb (errno \(errno))", code: "wired_memory_unavailable")
        }
        let signed: Int64
        if size == MemoryLayout<Int32>.size {
            signed = Int64(Int32(truncatingIfNeeded: raw))
        } else if size == MemoryLayout<Int64>.size {
            signed = raw
        } else {
            throw EngineError("macOS returned an unsupported GPU memory limit format", code: "wired_memory_unavailable")
        }
        guard signed >= 0, signed <= Int64(Int32.max) else {
            throw EngineError("macOS returned an unsupported GPU memory limit (\(signed))", code: "wired_memory_unavailable")
        }
        return UInt64(signed)
    }

    /// Synchronous for the settings handler. Validate every setting before calling; save settings only after a
    /// verified receipt. Cancellation/timeout is best effort because an already authorized system change can finish.
    static func ensureCeiling(
        for proposed: WiredMemoryPolicy.Plan, timeout: TimeInterval = 120,
        cancelled: @escaping @Sendable () -> Bool = { false }
    ) throws -> Receipt {
        guard timeout.isFinite, timeout > 0, timeout <= 120 else {
            throw EngineError("Memory limit authorization timeout must be within 120 seconds", code: "invalid_configuration")
        }
        guard mutationGate.wait(timeout: .now()) == .success else {
            throw EngineError("Another memory limit change is already awaiting macOS authorization", code: "wired_memory_busy")
        }
        defer { mutationGate.signal() }
        // Revalidate the host and target immediately before considering a privileged operation. A plan is not
        // authority to reuse a stale ceiling or a physical-memory value supplied by an HTTP client.
        let before = try readLimitMiB()
        let defaultLimit = before == 0 ? MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize : nil
        let plan = try WiredMemoryPolicy.plan(
            requestedBudgetBytes: proposed.requestedBudgetBytes,
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            currentLimitMiB: before, defaultLimitBytes: defaultLimit)
        guard !cancelled() else { throw Failure(message: "Memory limit change was cancelled.", observedLimitMiB: before) }
        guard plan.needsRaise else {
            return Receipt(beforeLimitMiB: before, afterLimitMiB: before, raised: false, authorizationRequested: false)
        }
        let expires = UInt64(Date().timeIntervalSince1970 + timeout)
        let script = administratorScript(
            requiredLimitMiB: plan.requiredLimitMiB, maximumLimitMiB: plan.maximumBudgetBytes / WiredMemoryPolicy.mebibyte,
            defaultLimitMiB: plan.effectiveLimitMiB, startingLimitMiB: before, expires: expires)
        do { try runAuthorization(script, timeout: timeout, cancelled: cancelled) } catch {
            throw Failure(message: String(describing: error), observedLimitMiB: try? readLimitMiB())
        }
        let after: UInt64
        do { after = try readLimitMiB() } catch {
            throw Failure(
                message: "macOS authorization completed, but its GPU memory limit could not be read back.", observedLimitMiB: nil)
        }
        guard !cancelled() else {
            throw Failure(message: "Memory limit change was cancelled after authorization.", observedLimitMiB: after)
        }
        guard after >= plan.requiredLimitMiB, after >= plan.effectiveLimitMiB else {
            throw Failure(
                message: "macOS did not retain the required GPU memory limit; the requested budget was not applied.",
                observedLimitMiB: after)
        }
        return Receipt(beforeLimitMiB: before, afterLimitMiB: after, raised: after > before, authorizationRequested: true)
    }

    /// The only interpolation is a validated unsigned integer. Re-read inside the privileged operation, after
    /// the dialog, so another utility's intervening increase is preserved. There is no arbitrary command input.
    private static func administratorScript(
        requiredLimitMiB: UInt64, maximumLimitMiB: UInt64, defaultLimitMiB: UInt64,
        startingLimitMiB: UInt64, expires: UInt64
    ) -> String {
        precondition(requiredLimitMiB > 0 && requiredLimitMiB <= UInt64(Int32.max))
        precondition(requiredLimitMiB <= maximumLimitMiB && requiredLimitMiB >= defaultLimitMiB)
        let shell = """
            [ "$(/bin/date +%s)" -lt \(expires) ] || exit 74
            [ \(requiredLimitMiB) -le \(maximumLimitMiB) ] || exit 75
            origami_wired_current=$(/usr/sbin/sysctl -n iogpu.wired_limit_mb) || exit 71
            case "$origami_wired_current" in ''|*[!0-9]*) exit 72 ;; esac
            if [ "$origami_wired_current" -eq 0 ]; then
                [ \(startingLimitMiB) -eq 0 ] || exit 76
            fi
            if [ "$origami_wired_current" -eq 0 ] || [ "$origami_wired_current" -lt \(requiredLimitMiB) ]; then
                [ "$(/bin/date +%s)" -lt \(expires) ] || exit 74
                /usr/sbin/sysctl -w iogpu.wired_limit_mb=\(requiredLimitMiB)
            fi
            """
        let quoted = shell.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"").replacingOccurrences(of: "\n", with: "\\n")
        return "do shell script \"\(quoted)\" with administrator privileges"
    }

    private static func runAuthorization(
        _ script: String, timeout: TimeInterval,
        cancelled: @escaping @Sendable () -> Bool
    ) throws {
        let process = Process()
        let errors = Pipe()
        let ended = DispatchSemaphore(value: 0)
        process.executableURL = URL(fileURLWithPath: "/usr/bin/osascript")
        process.arguments = ["-e", script]
        process.environment = ["PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LANG": "en_US.UTF-8"]
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice
        process.standardError = errors
        process.terminationHandler = { _ in ended.signal() }
        defer {
            try? errors.fileHandleForReading.close()
            try? errors.fileHandleForWriting.close()
        }
        guard !cancelled() else { throw EngineError("Memory limit authorization was cancelled.", code: "wired_memory_authorization") }
        try process.run()
        // Close our writer, so normal child exit reaches EOF. The fixed script emits at most a short error.
        try? errors.fileHandleForWriting.close()
        let deadline = DispatchTime.now() + timeout
        var stopped: String?
        while ended.wait(timeout: .now() + 0.1) == .timedOut {
            if cancelled() {
                stopped = "Memory limit authorization was cancelled."
                break
            }
            if DispatchTime.now() >= deadline {
                stopped = "macOS authorization for the memory limit timed out after \(Int(timeout)) seconds."
                break
            }
        }
        if let stopped {
            if process.isRunning { process.terminate() }
            if ended.wait(timeout: .now() + 0.5) == .timedOut, process.isRunning {
                _ = Darwin.kill(process.processIdentifier, SIGKILL)
            }
            // Never block on EOF or waitUntilExit here: an authorization helper may outlive osascript briefly.
            throw EngineError(stopped, code: "wired_memory_authorization")
        }
        // Never wait for EOF, even after osascript exits: an authorization helper could still hold its pipe.
        let fd = errors.fileHandleForReading.fileDescriptor
        let flags = fcntl(fd, F_GETFL)
        var bytes = [UInt8](repeating: 0, count: 4096)
        var count = 0
        if flags >= 0, fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0 {
            let amount = bytes.withUnsafeMutableBytes { Darwin.read(fd, $0.baseAddress, $0.count) }
            if amount > 0 { count = amount }
        }
        let message = String(decoding: bytes.prefix(count), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard process.terminationStatus == 0 else {
            throw EngineError(
                message.isEmpty
                    ? "macOS administrator authorization failed (status \(process.terminationStatus))."
                    : "macOS administrator authorization failed: \(message)",
                code: "wired_memory_authorization")
        }
    }
}
