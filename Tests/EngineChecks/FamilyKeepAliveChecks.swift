import Foundation
import Metal
import OrigamiEngine

/// A family reply decodes under the request keep-alive (`gpu_keepalive`) and gives it back when it ends, is cancelled or
/// fails, 20 runs each: its flag is charged to the ledger while a token is decoded, never while the prompt is read, and
/// the charge is gone once the request returns. A request's `origami_tune`, the setting false and Low Power Mode under
/// auto leave it off; the setting true holds it in Low Power Mode. Dropping the backend (an unload) frees it.
func familyRequestKeepAliveReleases() throws {
    guard MTLCreateSystemDefaultDevice() != nil else { throw FixtureUnavailable(description: "no Metal device") }
    final class Stub: FamilyModel {
        var position = 0, contextLimit = 4096, routing = QualityDial.exact
        var shouldContinue: () -> Bool = { true }
        var routedExperts: Int { 8 }
        var vocabulary: Int { 261 }
        var expertUses: [Int: Int] { [:] }
        var expertMisses: [Int: Int] { [:] }
        var timing: FamilyRunner.Timing { .init() }
        func statusNotes() -> [String: Any] { [:] }
        let running = MTLCreateSystemDefaultDevice()!.makeBuffer(length: 16, options: .storageModeShared)!
        func stateParts(at position: Int) -> [(buffer: MTLBuffer, bytes: Int)] { [(running, 16)] }
        func reset() throws { position = 0 }
        func restored(_ tokens: [Int]) { position = tokens.count }
        /// Called before each forward; it may throw.
        var step: () throws -> Void = {}
        func forward(_ tokens: [Int]) throws -> [Float] {
            try step()
            position += tokens.count
            var logits = [Float](repeating: 0, count: vocabulary)
            logits[65 + position % 26] = 100
            return logits
        }
    }
    let ledger = try MemoryLedger(ceiling: 2_000_000, transientMargin: 1)
    let tokenizer = try tinyFamilyTokenizer("{% for m in messages %}{{ m.content }}{% endfor %}")
    func request(_ tune: [String: Any]? = nil) throws -> ServingRequest {
        var body: [String: Any] = [
            "model": "family", "max_tokens": 6, "temperature": 0, "reasoning_effort": "none",
            "messages": [["role": "user", "content": "Hello there"]], "prompt_cache_policy": "no_store",
        ]
        if let tune { body["origami_tune"] = tune }
        return try ServingRequest.parse(JSONSerialization.data(withJSONObject: body), contextTier: 4096)
    }
    // outcome: (setting, Low Power Mode, request override, held while decoding)
    let cases: [(String, Bool?, Bool, Bool?, Bool)] = [
        ("finish", nil, false, nil, true), ("cancel", nil, false, nil, true), ("failure", nil, false, nil, true),
        ("request off", nil, false, false, false), ("request on in low power", nil, true, true, true),
        ("setting false", false, false, nil, false), ("low power", nil, true, nil, false),
        ("setting true in low power", true, true, nil, true), ("unload", nil, false, nil, true),
    ]
    for run in 0..<20 {
        for (outcome, setting, lowPower, tune, holds) in cases {
            let stub = Stub()
            var backend: FamilyServingBackend? = FamilyServingBackend(
                model: stub, tokenizer: tokenizer, stops: [259], version: "check", fastAllowed: true)
            var keepAlive: RequestGPUKeepAlive? = try RequestGPUKeepAlive(ledger: ledger)
            weak let freed = keepAlive
            backend?.keepAlive = keepAlive
            keepAlive = nil
            backend?.keepAliveSetting = setting
            backend?.lowPowerMode = { lowPower }
            let control = GenerationControl()
            var decoded = 0
            var wrong: String?
            stub.step = {
                let held = ledger.snapshot().reserved == 16384
                if stub.position == 0 {
                    if held { wrong = "held while the prompt was read" }
                    return
                }
                decoded += 1
                if held != holds { wrong = "\(held ? "held" : "not held") while decoding" }
                if decoded == 2, outcome == "cancel" { control.cancel() }
                if decoded == 2, outcome == "failure" { throw EngineError("Stub failure") }
            }
            // In a pool as the server's request thread runs it: Metal's autoreleased objects hold the flag until it drains.
            let finished = autoreleasepool {
                Result {
                    try backend!.prepare(try request(tune.map { ["gpu_keepalive": $0] }), cancellation: control).run { _ in true }
                }
            }
            if outcome == "unload" { backend = nil }
            if let wrong { throw EngineError("Family keep-alive, \(outcome) run \(run): \(wrong)") }
            switch (outcome, finished) {
            case ("failure", .failure): break
            case ("cancel", .failure) where control.isCancelled: break
            case (_, .success) where outcome != "failure" && outcome != "cancel": break
            default: throw EngineError("Family keep-alive, \(outcome) run \(run): ended as \(finished)")
            }
            guard decoded >= 2, ledger.waitForRelease(until: { reserved, _ in reserved == 0 }) else {
                throw EngineError("Family keep-alive, \(outcome) run \(run): not released (decoded \(decoded))")
            }
            if outcome == "unload", freed != nil { throw EngineError("Family keep-alive, run \(run): outlived its backend") }
        }
    }
}
