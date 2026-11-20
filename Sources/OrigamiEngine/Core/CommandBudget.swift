import Foundation

/// The most GPU work one command buffer of a prompt should hold. macOS stops a long buffer when the display or another
/// app needs the GPU (kIOGPUCommandBufferCallbackErrorImpactingInteractivity), failing the request, so prompt work is
/// committed in buffers of at most `limit`, counted in `mm` multiply-adds (an engine weights slower kernels by their
/// speed). The limit starts at `work` (about 17 ms of `mm` on an M2 Pro), is cut to fit `seconds` when a completed
/// buffer's measured GPU time runs longer (a slower GPU, or one another app is using), and is halved when macOS stops a
/// buffer, so the request's retry runs smaller ones; never below `floor`. It doubles back once a prompt has been read
/// (`recover`), not within one, as smaller buffers cost prompt time. Only where buffers end changes, never what they hold.
package final class CommandBudget: @unchecked Sendable {
    package static let work = 3e10
    package static let seconds = 0.025
    package static let floor = work / 64
    private var value = work
    private let lock = NSLock()
    package init() {}

    /// Settable for checks.
    package var limit: Double {
        get {
            lock.lock()
            defer { lock.unlock() }
            return value
        }
        set {
            lock.lock()
            value = newValue
            lock.unlock()
        }
    }
    /// A completed buffer's `work` multiply-adds took `seconds` of GPU time.
    package func fit(work: Double, seconds: Double) {
        guard seconds > Self.seconds else { return }
        lock.lock()
        value = max(Self.floor, min(value, work * Self.seconds / seconds))
        lock.unlock()
    }
    /// macOS stopped a buffer: the retry runs half as much a buffer.
    package func interrupted() {
        lock.lock()
        value = max(Self.floor, value / 2)
        lock.unlock()
    }
    /// A prompt was read through: buffers may grow back.
    package func recover() {
        lock.lock()
        value = min(Self.work, value * 2)
        lock.unlock()
    }
}
