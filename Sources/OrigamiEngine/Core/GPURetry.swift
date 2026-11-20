import Foundation
import Metal

extension EngineError {
    /// A command buffer that failed: `gpu_interrupted` when macOS stopped it (`interrupted`), else `gpu_execution_failed`.
    package static func gpu(_ what: String, _ error: Error?) -> EngineError {
        EngineError(
            "\(what): \(error.map { String(describing: $0) } ?? "unknown error")",
            code: interrupted(error) ? "gpu_interrupted" : "gpu_execution_failed")
    }
    /// Whether macOS stopped the command buffer for reasons outside it, so the same work succeeds when run again: IOGPU's
    /// callback errors 14 (Impacting Interactivity: the display or another app needed the GPU), 2 (Caused GPU Timeout
    /// Error) and 5 (Discarded, a victim of another app's GPU recovery), or Metal's timeout. A hang, a page fault, an
    /// out-of-memory or a process macOS ignores for its earlier errors (4) would fail again.
    package static func interrupted(_ error: Error?) -> Bool {
        guard let error = error as NSError? else { return false }
        if let underlying = error.userInfo[NSUnderlyingErrorKey] as? NSError, underlying.domain == "IOGPUCommandQueueErrorDomain" {
            return [2, 5, 14].contains(underlying.code)
        }
        return error.domain == MTLCommandBufferErrorDomain && error.code == MTLCommandBufferError.timeout.rawValue
    }
    /// What macOS's Impacting Interactivity abort looks like, for checks that fake one (`FamilyRunner.interruptWait`).
    package static let fakeInterruption: Error = NSError(
        domain: MTLCommandBufferErrorDomain, code: 1,
        userInfo: [
            NSLocalizedDescriptionKey: "Impacting Interactivity (0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)",
            NSUnderlyingErrorKey: NSError(domain: "IOGPUCommandQueueErrorDomain", code: 14),
        ])
}

/// A request run again when macOS interrupts its GPU work (`gpu_interrupted`), up to `retries` more times. Each attempt
/// starts over from the last saved state (the prompt cache, or the start): a slice that ran halfway has overwritten its
/// rows and running state, so nothing is re-run in place. The reply tokens an interrupted attempt produced are replayed
/// (`replay`: sampled as before, so the sampler's draws stay in step, and kept), and `passing` hands on only text past
/// what the client already has, so a retried reply is bitwise the uninterrupted one. A retry waits `backoff` per attempt
/// first, so whatever needed the GPU gets it; by then the engine has halved its prompt command buffers (`CommandBudget`).
public final class GPURetry: @unchecked Sendable {
    package static let retries = 2
    package static let backoff = 0.25
    /// The client's message once retries run out; Pi retries on "please retry your request".
    package static let message = "macOS interrupted Origami's GPU work (the display or another app needed it); please retry your request"
    package private(set) var attempt = 0
    /// The reply tokens produced so far, in every attempt: an attempt keeps the first `replay.count` and appends after them.
    package var replay: [Int] = []
    private var cause = "", noted = true
    /// Unicode scalars passed on to the client, and produced by this attempt.
    private var sent = 0, produced = 0
    private let log: ((String) -> Void)?
    package init(log: ((String) -> Void)?) { self.log = log }

    /// Also a buffer that outran its bound and then ended (`gpu_stalled`, `GPUWait`): run again like macOS's GPU timeout.
    package static func isInterruption(_ error: Error) -> Bool {
        let code = (error as? EngineError)?.code ?? (error as? ServingFailure)?.code
        return code == "gpu_interrupted" || code == "gpu_stalled"
    }
    package func run<T>(_ body: () throws -> T) throws -> T {
        while true {
            produced = 0
            do { return try body() } catch  where Self.isInterruption(error) {
                guard attempt < Self.retries else {
                    log?("request failed: macOS interrupted the GPU \(attempt + 1) times; the client was asked to retry")
                    throw error
                }
                attempt += 1
                Thread.sleep(forTimeInterval: Self.backoff * Double(attempt))
                noted = false
                // The step, without the NSError's dump.
                cause = String(describing: error).components(separatedBy: ": Error Domain=").first ?? ""
            }
        }
    }
    /// Called by a retried attempt once it knows where it starts: the log line (and dashboard event) of the retry.
    package func note(from token: Int) {
        guard !noted else { return }
        noted = true
        // What happened first, in the user's words; the failed step last, for whoever reads the log.
        log?(
            "prompt: macOS needed the GPU; reading again from " + (token > 0 ? "token \(token) (saved state)" : "the start")
                + (replay.isEmpty ? "" : ", then the \(replay.count) reply tokens already made")
                + ", retry \(attempt) of \(Self.retries) (\(cause))")
    }
    /// The token for reply position `index`: a replayed one (logged when this attempt picked another), else `picked`.
    package func token(_ index: Int, picked: Int) -> Int {
        guard index < replay.count else { return picked }
        if replay[index] != picked {
            log?("retry: reply token \(index) came out \(picked), not \(replay[index]) as before; keeping the one sent")
        }
        return replay[index]
    }
    /// Records a reply token past the replayed ones.
    package func made(_ index: Int, _ token: Int) { if index == replay.count { replay.append(token) } }
    /// `emit` passed only the text past what an earlier attempt already passed (the same text, as the tokens are).
    package func passing(_ emit: @escaping @Sendable (String) throws -> Bool) -> @Sendable (String) throws -> Bool {
        { [self] piece in
            let count = piece.unicodeScalars.count
            let skip = sent - produced
            produced += count
            guard count > skip else { return true }
            sent = produced
            return try emit(skip > 0 ? String(String.UnicodeScalarView(piece.unicodeScalars.dropFirst(skip))) : piece)
        }
    }
}
