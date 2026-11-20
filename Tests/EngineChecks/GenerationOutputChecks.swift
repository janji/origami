import Foundation
import OrigamiEngine

func generatedEOSExcludesOutputAndPresence() throws {
    for eos in TokenizerStack.eos {
        var empty = try GeneratedTokenSequence(maximumOutput: 4)
        guard try !empty.recordSample(eos), empty.ids.isEmpty, empty.generated.isEmpty, empty.terminationTokenID == eos else {
            throw EngineError("Immediate EOS entered output/presence history")
        }
        do {
            try empty.recordSample(49)
            throw EngineError("Post-EOS sample accepted")
        } catch let error as EngineError { if error.description == "Post-EOS sample accepted" { throw error } }
        var sequence = try GeneratedTokenSequence(maximumOutput: 4)
        guard try sequence.recordSample(49), try sequence.recordSample(49), try !sequence.recordSample(eos),
            sequence.ids == [49, 49], sequence.generated == [49: 2], sequence.terminationTokenID == eos
        else {
            throw EngineError("Terminal EOS changed actual outputs or presence history")
        }
    }
}

func generatedEOSPreservesSamplerDraw() throws {
    var parameters = SamplingParameters()
    parameters.temperature = 1
    parameters.topK = 1
    var logits = [Float](repeating: -1000, count: 248_320)
    logits[248_046] = 1000
    var sampler = ReferenceSampler(seed: 93)
    var expected = ReferenceSampler(seed: 93)
    let sampled = try sampler.next(logits: logits, parameters: parameters)
    var output = try GeneratedTokenSequence(maximumOutput: 4)
    guard try !output.recordSample(sampled) else { throw EngineError("Known EOS was emitted") }
    _ = try expected.next(logits: logits, parameters: parameters)
    guard sampler.random.state == expected.random.state, output.terminationTokenID == 248_046, output.generated.isEmpty else {
        throw EngineError("EOS exclusion replayed or discarded the sampler draw")
    }
}

func generatedLengthBound() throws {
    var output = try GeneratedTokenSequence(maximumOutput: 2)
    try output.recordSample(9)
    try output.recordSample(10)
    guard output.ids == [9, 10], output.generated == [9: 1, 10: 1], output.terminationTokenID == nil else {
        throw EngineError("Length-limited outputs changed")
    }
    do {
        try output.recordSample(11)
        throw EngineError("Output allowance exceeded")
    } catch let error as EngineError { if error.description == "Output allowance exceeded" { throw error } }
}

func toolAnchorMarkerBoundaries() throws {
    let marker = "<｜DSML｜ calls>"
    // Every Unicode-scalar boundary can fall between decoded tokens.
    let scalars = Array(marker.unicodeScalars)
    for split in 0...scalars.count {
        var tracker = ToolCallAnchorTracker(openers: [], markers: [marker])
        let first = String(String.UnicodeScalarView(scalars.prefix(split)))
        let second = String(String.UnicodeScalarView(scalars.dropFirst(split)))
        let a = tracker.observe(token: 0, delta: "reasoning " + first)
        let b = tracker.observe(token: 1, delta: second + String(repeating: "arguments", count: 100))
        guard a == (split == scalars.count), b == !a,
            !tracker.observe(token: 2, delta: marker)
        else {
            throw EngineError("Tool anchor lost a split marker or captured twice")
        }
    }
    var partial = ToolCallAnchorTracker(openers: [42], markers: ["<tool_call>", "aab"])
    guard !partial.observe(token: 0, delta: "<tool_calX"),
        !partial.observe(token: 1, delta: "aaaa"), partial.observe(token: 2, delta: "b"),
        !partial.observe(token: 42, delta: "")
    else { throw EngineError("Tool anchor overlap matching failed") }
    var token = ToolCallAnchorTracker(openers: [42], markers: [])
    guard !token.observe(token: 41, delta: ""), token.observe(token: 42, delta: ""),
        !token.observe(token: 42, delta: "")
    else { throw EngineError("Single-token anchor matching failed") }
}
