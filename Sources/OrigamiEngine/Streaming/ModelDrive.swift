import CExactIO
import CryptoKit
import Darwin
import Foundation
import Synchronization

/// The drive a model's own files are read from, as their open descriptors see it (both engines: Qwen's
/// checkpoint or GGUF through `PackStore`, GLM's and DeepSeek's GGUF through `FamilyWeights`). A read of one of its
/// files that fails because the device is gone (the descriptor's `fstat` fails, or the read fails with ENXIO, ENODEV,
/// EBADF, ENOENT or ESTALE; a bad block on a present drive is not this) marks the drive away: from then on every read
/// of its files fails at once with `drive_away` (sent to clients as `drive_stalled`), while what is in memory and the
/// hot copies keep serving. While away, reads and a timer (every `retrySeconds`, off the reading thread) try to open
/// each file again by its path. A reopened file is used only if it is the one first opened: the same size and
/// modification time, the same inode when its volume UUID is the same, and the same SHA-256 of a 64 KiB sample taken
/// from the middle of the file at its first open. Then every file's descriptor is replaced at once; a reader still on
/// an old descriptor fails and reads its whole range again once on the new one, and an old descriptor is closed when
/// its last reader leaves. A different file at the path is refused and the drive stays away, saying so.
package final class ModelDrive: @unchecked Sendable {
    /// How often a drive that is away is tried again. Settable for checks.
    nonisolated(unsafe) package static var retrySeconds = 5.0
    private let lock = NSLock()
    private var files: [Weak] = []
    private struct Weak { weak var file: ExactFile? }
    private var awaySince: UInt64?, lastTry: UInt64 = 0, trying = false
    private var cause = "", refusal: String?
    private var timer: DispatchSourceTimer?
    /// Bumped by every swap; a descriptor carries the one it was opened under.
    package var generation: UInt64 { lock.withLock { swaps } }
    private var swaps: UInt64 = 0
    /// Descriptors open now (old ones a reader still holds included), for checks.
    /// Atomic, not under `lock`: a file's last holder may let go while the lock is held (a weak reference being read),
    /// and its descriptor then closes there.
    package var openDescriptors: Int { descriptors.load(ordering: .relaxed) }
    private let descriptors = Atomic<Int>(0)
    /// Reads that failed on a replaced descriptor and read again on the new one, for checks.
    package var retries: Int { lock.withLock { retried } }
    private var retried = 0
    func retriedOnce() { lock.withLock { retried += 1 } }
    /// Checks only: every descriptor opened under a generation up to this one behaves as on a vanished device, and
    /// opening by path fails (`unplug`/`plug`).
    private var deadThrough: UInt64?, unplugged = false
    /// Lock-free answers for every read: whether the drive is away, and whether a check ever unplugged it.
    private let away = Atomic<Bool>(false), simulated = Atomic<Bool>(false)
    /// Checks only: accepts a reopened file whatever it holds (to see that no read mixes two descriptors' bytes).
    nonisolated(unsafe) package var acceptAny = false
    /// Checks only: runs after each read's `pread` ended, before its descriptor is checked (holds a reader in flight).
    nonisolated(unsafe) package var readHook: (@Sendable () -> Void)?
    private let log: @Sendable (String) -> Void

    package init(log: @escaping @Sendable (String) -> Void = { FileHandle.standardError.write(Data("origami: \($0)\n".utf8)) }) {
        self.log = log
        Self.registry.withLock { $0.append(WeakDrive(drive: self)) }
    }
    deinit { timer?.cancel() }

    private struct WeakDrive { weak var drive: ModelDrive? }
    private static let registry = Mutex<[WeakDrive]>([])
    /// A drive of this process that is away now: its words for the status (`degraded`) and its error for a read
    /// failure that lost its cause on the way (one model per serving process).
    package static var away: ModelDrive? {
        registry.withLock { list in
            list.removeAll { $0.drive == nil }
            return list.lazy.compactMap(\.drive).first { $0.isAway }
        }
    }
    package var isAway: Bool { lock.withLock { awaySince != nil } }
    /// What a read fails with while the drive is away.
    package var awayError: EngineError? {
        lock.withLock { awaySince.map { _ in EngineError(noteLocked, code: "drive_away") } }
    }
    /// The status line while away (nil while present).
    package var note: String? { lock.withLock { awaySince == nil ? nil : noteLocked } }
    private var noteLocked: String {
        refusal
            ?? "The model drive is not connected (\(cause)): requests fail until it is back; reading resumes by itself when it is, "
            + "with no reload"
    }

    func register(_ file: ExactFile) {
        lock.withLock {
            files.removeAll { $0.file == nil }
            files.append(Weak(file: file))
        }
    }
    func opened() { descriptors.add(1, ordering: .relaxed) }
    func closed() { descriptors.subtract(1, ordering: .relaxed) }

    /// Throws `drive_away` at once while the drive is away (starting a reopen when one is due).
    func admit() throws {
        guard away.load(ordering: .acquiring) else { return }
        guard let error = awayError else { return }
        tryIfDue()
        throw error
    }

    /// Whether `descriptor` (opened under `generation`) lost its device, after a read or `fstat` failed with `code`.
    func gone(_ descriptor: Int32, generation: UInt64, code: Int32) -> Bool {
        if dead(generation) { return true }
        if [ENXIO, ENODEV, EBADF, ENOENT, ESTALE].contains(code) { return true }
        var raw = origami_file_identity()
        return origami_get_identity(descriptor, &raw) != 0
    }
    /// Checks only: whether a descriptor of `generation` acts as on a vanished device.
    func dead(_ generation: UInt64) -> Bool {
        simulated.load(ordering: .relaxed) && lock.withLock { deadThrough.map { generation <= $0 } ?? false }
    }
    func openFails() -> Bool { simulated.load(ordering: .relaxed) && lock.withLock { unplugged } }

    /// A descriptor of `path` (opened under `generation`) lost its device: the drive is away from now, unless a newer
    /// descriptor already replaced it.
    func lost(_ path: String, generation: UInt64, code: Int32) {
        let first: Bool = lock.withLock {
            guard generation == swaps, awaySince == nil else { return false }
            awaySince = DispatchTime.now().uptimeNanoseconds
            lastTry = 0  // a first reopen at once: the drive may be back already (it left while nothing was read)
            away.store(true, ordering: .releasing)
            cause = "\(URL(fileURLWithPath: path).lastPathComponent) failed with errno \(code)"
            refusal = nil
            let timer = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
            let every = max(Self.retrySeconds, 0.001)
            timer.schedule(deadline: .now() + every, repeating: every)
            timer.setEventHandler { [weak self] in self?.tryIfDue() }
            timer.resume()
            self.timer = timer
            return true
        }
        if first {
            log("the model drive is gone (\(path) failed with errno \(code)); requests fail at once until it is back")
            tryIfDue()
        }
    }

    /// Starts a reopen off the calling thread when the last one is `retrySeconds` old.
    private func tryIfDue() {
        let due: Bool = lock.withLock {
            let now = DispatchTime.now().uptimeNanoseconds
            guard awaySince != nil, !trying, Double(now - lastTry) / 1e9 >= Self.retrySeconds * 0.999 else { return false }
            trying = true
            lastTry = now
            return true
        }
        if due { DispatchQueue.global(qos: .utility).async { [self] in reopen() } }
    }

    /// Opens every file again by its path and, when each is the file first opened, swaps them all in at once.
    private func reopen() {
        let (open, next) = lock.withLock { (files.compactMap(\.file), swaps + 1) }
        var handles: [ExactFile.Handle] = []
        var refused: String?
        for file in open {
            switch file.reopened(generation: next) {
            case .same(let handle): handles.append(handle)
            case .missing: break
            case .different(let reason): refused = refused ?? reason
            }
            if refused != nil { break }
        }
        let back = refused == nil && handles.count == open.count
        // The replaced descriptors, released after the lock (each one's last holder closes it).
        var replaced: [ExactFile.Handle] = []
        var swapped = false
        let (seconds, changed): (Double, Bool) = lock.withLock {
            defer { trying = false }
            let seconds = Double(DispatchTime.now().uptimeNanoseconds - (awaySince ?? 0)) / 1e9
            guard back else {
                let changed = refused != nil && refused != refusal
                if let refused { refusal = refused }
                return (seconds, changed)
            }
            // A file opened since the snapshot has no new descriptor here: this round is dropped, the next covers it.
            guard files.compactMap(\.file).elementsEqual(open, by: { $0 === $1 }) else { return (seconds, false) }
            replaced = zip(open, handles).map { $0.swap($1) }
            swapped = true
            swaps = next
            awaySince = nil
            away.store(false, ordering: .releasing)
            refusal = nil
            timer?.cancel()
            timer = nil
            return (seconds, false)
        }
        replaced.removeAll()
        if swapped {
            log(String(format: "the model drive returned after %.0f s; reads continue (nothing was reloaded)", seconds))
        } else if changed, let refused {
            log(refused)
        }
    }

    /// Checks only: the drive leaves (every open descriptor reads as on a vanished device, opening fails) and returns.
    package func unplug() {
        lock.withLock {
            deadThrough = swaps
            unplugged = true
            simulated.store(true, ordering: .relaxed)
        }
    }
    package func plug() { lock.withLock { unplugged = false } }

    /// The sample a reopened file must match: 64 KiB (less in a smaller file) from the middle, on a 16 KiB boundary.
    package static func sample(size: UInt64) -> (offset: Int64, length: Int) {
        let length = Int(min(size, 64 * 1024))
        return (Int64((size - UInt64(length)) / 2 / 16_384 * 16_384), length)
    }
    static func digest(_ descriptor: Int32, size: UInt64) -> SHA256.Digest? {
        let (offset, length) = sample(size: size)
        var bytes = [UInt8](repeating: 0, count: length)
        let result = bytes.withUnsafeMutableBytes { origami_pread_exact(descriptor, $0.baseAddress, length, offset) }
        return result.error_code == 0 ? SHA256.hash(data: bytes) : nil
    }

    /// The outcome of opening a file again by its path.
    enum Reopened {
        case same(ExactFile.Handle)
        case missing
        case different(String)
    }
}
