import Foundation
import Metal
import OrigamiEngine

private func followupRequire(_ value: Bool, _ message: String) throws {
    guard value else { throw EngineError(message) }
}

/// The admitted path must equal the tokenizer's uncached full-render/full-encode
/// path, including after cache hits, earlier edits and changed template context.
/// Tiny byte-level BPE fixtures only: no model weights, Metal or network.
func followupAdmittedPromptCacheParity() throws {
    let ledger = try MemoryLedger(ceiling: 64 << 20, transientMargin: 1 << 20)
    try autoreleasepool {
        let template =
            "Hi {{ messages[-1].content }}|{% for m in messages %}<|user|>{{ m.role }}:{{ m.content }}|"
            + "{% if m.marker is integer %}integer{% elif m.marker is float %}float{% else %}other{% endif %}:{{ m.marker }}|"
            + "{% for c in m.tool_calls or [] %}{{ c.function.name }}={{ c.function.arguments|tojson }}|{% endfor %}{% endfor %}"
            + "{{ tools|default([])|tojson }}|{{ reasoning_effort|default('off') }}|<think></think><|assistant|>"
        let cached = try tinyFamilyTokenizer(template)
        let uncached = try tinyFamilyTokenizer(template)
        cached.admitPromptEncodingCache(ledger: ledger)
        let admitted = ledger.reservedBytes
        try followupRequire(
            admitted > 0 && ledger.snapshot().categories["prompt_encoding_cache"] == admitted,
            "Prompt fixture did not enable the admitted cache path")
        @discardableResult func compare(
            _ messages: [[String: TemplateValue]], tools: [[String: TemplateValue]] = [],
            effort: String = "medium"
        ) throws -> [Int] {
            let expected = try uncached.encodeNormalized(messages, tools: tools, reasoningEffort: effort)
            let actual = try cached.encodeNormalized(messages, tools: tools, reasoningEffort: effort)
            try followupRequire(actual == expected, "Admitted prompt cache changed exact token IDs")
            try followupRequire(ledger.reservedBytes == admitted, "Prompt cache grew its reservation during reuse")
            return actual
        }
        var messages: [[String: TemplateValue]] = [["role": .string("user"), "content": .string("Hi café 🦢\n"), "marker": .integer(1)]]
        let integer = try compare(messages)
        try followupRequire(try compare(messages) == integer, "Identical prompt changed on cache reuse")
        messages[0]["marker"] = .number(1.0)
        let floating = try compare(messages)
        try followupRequire(integer != floating, "Fixture did not distinguish integer and floating template values")
        // Swift String equality equates these spellings; byte-level tokenization must not.
        messages[0]["content"] = .string("Hi cafe\u{0301} 🦢\n")
        let decomposed = try compare(messages)
        try followupRequire(floating != decomposed, "Fixture did not distinguish exact Unicode bytes")
        for zero in [0.0, -0.0] {
            messages[0]["marker"] = .number(zero)
            try compare(messages)
        }

        messages.append([
            "role": .string("assistant"), "content": .string("Hi<think></think>"),
            "tool_calls": .array([
                .object([
                    "id": .string("a"), "type": .string("function"),
                    "function": .object([
                        "name": .string("weather"), "arguments": .string("{\"z\":9007199254740993,\"a\":{\"y\":null,\"x\":true}}"),
                    ]),
                ])
            ]),
        ])
        messages.append(["role": .string("tool"), "tool_call_id": .string("a"), "content": .string("Rome: clear")])
        messages.append(["role": .string("user"), "content": .string("Next Hi")])
        let tools: [[String: TemplateValue]] = [
            [
                "type": .string("function"),
                "function": .object([
                    "name": .string("weather"), "description": .string("Read weather"), "parameters": .object(["type": .string("object")]),
                ]),
            ]
        ]
        try compare(messages, tools: tools)
        try compare(messages, tools: tools)
        messages[0]["content"] = .string("Edited earlier turn")
        try compare(messages, tools: tools)
        messages[messages.count - 1]["content"] = .string("Changed suffix changes the template's leading text too")
        let withTools = try compare(messages, tools: tools)
        let withoutTools = try compare(messages, effort: "none")
        try followupRequire(withTools != withoutTools, "Fixture did not exercise changed tools/reasoning context")
        messages.removeLast(3)
        try compare(messages)

        let deniedLedger = try MemoryLedger(ceiling: 1 << 20, transientMargin: 1)
        let denied = try tinyFamilyTokenizer(template)
        denied.admitPromptEncodingCache(ledger: deniedLedger)
        try followupRequire(deniedLedger.reservedBytes == 0, "Optional prompt cache exceeded admission")
        try followupRequire(
            try denied.encodeNormalized(messages) == uncached.encodeNormalized(messages),
            "Denied optional admission changed the full-encode fallback")
    }
    try followupRequire(ledger.reservedBytes == 0, "Prompt cache reservation outlived its tokenizer")
}

/// Simulated immediate owner retirement makes each policy response observable.
/// Host pressure/counters and time are injected; no real sysctl, timer or waiting.
private struct FollowupMemoryHost {
    static let gb: UInt64 = 1_000_000_000
    let cap: UInt64 = 26_000_000_000
    var usable: UInt64
    var available: UInt64 = 0
    var policy: MemoryGovernorPolicy
    init(usable: UInt64 = 26_000_000_000) {
        self.usable = usable
        policy = MemoryGovernorPolicy(
            mode: .explicit, cap: 26 * Self.gb, floor: 4 * Self.gb,
            minimum: 8 * Self.gb, swapouts: 0)
    }
    mutating func step(_ time: TimeInterval, swap: UInt64 = 0, pressure: MemoryPressureLevel = .normal) -> MemoryGovernorPolicy.Decision? {
        let result = policy.step(
            now: time, available: available, reserved: usable, usable: usable,
            swapouts: swap, pressure: pressure, learned: Self.gb)
        if let result { usable = result.target }
        return result
    }
}

func followupTinySwapAndAdaptiveRecovery() throws {
    let mib: UInt64 = 1 << 20
    let gb = FollowupMemoryHost.gb
    var isolated = FollowupMemoryHost()
    _ = isolated.step(0)
    try followupRequire(
        isolated.step(19, swap: mib) == nil && isolated.usable == isolated.cap,
        "An isolated tiny global swap reduced the explicit target")
    _ = isolated.step(20, swap: mib)
    try followupRequire(isolated.step(39.99, swap: mib) == nil, "Tiny swap failed to restart the quiet interval")
    try followupRequire(
        isolated.step(40, swap: mib)?.reason == .growthRetry,
        "An old tiny swap permanently suppressed target recovery")

    var repeated = FollowupMemoryHost()
    try followupRequire(
        repeated.step(0, swap: mib) == nil && repeated.step(0.25, swap: mib) == nil,
        "Reading the same counter twice manufactured sustained swapping")
    let relief = repeated.step(2, swap: 2 * mib)
    try followupRequire(
        relief?.reason == .swap && relief?.freshSwap == mib && relief?.target == repeated.cap - gb - mib,
        "Distinct small swap increases did not receive relief")
    try followupRequire(repeated.step(2.1, swap: 3 * mib) == nil, "Swap settling failed to throttle callbacks")
    let pending = repeated.step(4, swap: 3 * mib)
    try followupRequire(
        pending?.reason == .swap && pending?.freshSwap == mib,
        "Qualifying swap during settling was lost or counted twice")
    try followupRequire(
        repeated.step(5, swap: 0) == nil && repeated.step(6, swap: mib) == nil,
        "A counter reset retained the previous swap episode")
    var expired = FollowupMemoryHost()
    try followupRequire(
        expired.step(0, swap: mib) == nil && expired.step(10, swap: 2 * mib) == nil,
        "Separate isolated increments outside the window triggered relief")
    for bytes in [64 * mib - 1, 64 * mib] {
        var burst = FollowupMemoryHost()
        let result = burst.step(0, swap: bytes)
        try followupRequire(
            bytes < 64 * mib ? result == nil : result?.reason == .swap,
            "The isolated swap burst threshold is inconsistent")
    }

    var recovery = FollowupMemoryHost(usable: 24 * gb)
    recovery.available = 10 * gb
    _ = recovery.step(0)
    try followupRequire(recovery.step(20)?.target == 25 * gb, "Initial recovery did not use its normal pace")
    try followupRequire(recovery.step(21, pressure: .warning)?.target == 24 * gb, "First pressure reversal was ignored")
    // The cut level held: recovery stops there; the reversal also slows the next raise above it (120 s back-off, then
    // 40 s; memory_held_level_quiet_recovery), which in time returns to the exact immutable target.
    var time: TimeInterval = 22
    while time < 182 {
        let decision = recovery.step(time)
        try followupRequire(
            decision == nil || (decision?.reason == .growthRetry && decision?.target == 24 * gb),
            "Recovery passed the level that held at \(time) s")
        time += 1
    }
    while recovery.usable < recovery.cap, time < 2_000 {
        _ = recovery.step(time)
        time += 1
    }
    try followupRequire(
        recovery.usable == recovery.cap && recovery.step(time + 20)?.reason == .growthRetry,
        "Recovered target remained permanently reduced by backoff or low free estimates")
}

/// No kernels or model weights: bounded shared-buffer ownership, deferred trim
/// completion and a recovery larger than one attachment tranche. The raw Metal
/// reference deliberately outlives its cache slot to check the real ledger owner.
func followupPendingTrimAndBatchedGrowth() throws {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "Metal unavailable") }
    let slab = UInt64(8 * PackStore.expertBytes)
    let recovering = max(2, Int((128 << 20) / slab) + 2)
    let maximum = UInt64(2 + recovering) * slab
    try followupRequire(maximum <= 384 << 20, "Cache lifecycle fixture exceeds its bounded allocation")
    let ledger = try MemoryLedger(ceiling: 512 << 20, transientMargin: 1 << 20)
    let cache = try ExpertCache(
        device: device, ledger: ledger, byteLimit: 4 * slab,
        slotsPerSlab: 8, maximumByteLimit: max(4 * slab, maximum))
    var tickets: [ExpertTicket] = []
    var escaped: MTLBuffer?
    // Raw Objective-C buffer access can extend resource ownership until the
    // enclosing autorelease pool drains; test retirement after that boundary.
    try autoreleasepool {
        for expert in 0..<32 {
            let entry = try cache.acquire(ExpertKey(layer: 3, expert: expert))
            let record = try cache.beginRead(entry.ticket)
            record.bytes[0] = UInt8(expert + 1)
            record.bytes[record.length - 1] = UInt8(expert + 2)
            try followupRequire(cache.completeRead(entry.ticket, verified: true), "Lifecycle record failed verification")
            tickets.append(entry.ticket)
        }
        for index in [0, 8, 16, 24] {
            let held = try cache.pin(tickets[index])
            if index == 24 { escaped = held.buffer }
        }
        try cache.resize(to: 2 * slab)
        try followupRequire(
            cache.snapshot().byteLimit == 2 * slab && cache.snapshot().allocatedBytes == 4 * slab,
            "A trim retired a slab with an active GPU pin")
        cache.completePendingTrim()
        try followupRequire(cache.snapshot().allocatedBytes == 4 * slab, "Pending trim ignored active pins")
        cache.holdSlabGrowth(true)
        for index in [16, 24] { try cache.unpinAfterGPUCompletion(tickets[index]) }
        cache.completePendingTrim()
        try followupRequire(cache.snapshot().allocatedBytes == 4 * slab, "Pending trim changed a held slab layout")
        cache.holdSlabGrowth(false)
        // No second resize/pressure callback: the existing boundary must finish it.
        cache.completePendingTrim()
    }
    try followupRequire(
        cache.snapshot().allocatedBytes == 2 * slab && ledger.reservedBytes == 3 * slab,
        "Pending trim ownership mismatch: allocated \(cache.snapshot().allocatedBytes), charged \(ledger.reservedBytes), slab \(slab)")
    try autoreleasepool {
        try followupRequire(escaped?.contents().load(as: UInt8.self) == 25, "Retirement invalidated the retained raw buffer")
        escaped = nil
    }
    try followupRequire(ledger.reservedBytes == 2 * slab, "Retired storage remained charged after its last owner left")

    try autoreleasepool {
        cache.setGrowthCeiling(maximum)
        let prepared = try cache.prepareGrowth(minimumAvailableBytes: 0, wire: false)
        try followupRequire(
            prepared == recovering && cache.snapshot().allocatedBytes == 2 * slab
                && ledger.reservedBytes == maximum, "Prepared recovery escaped its owner limit or attached before a boundary")
        cache.holdSlabGrowth(true)
        try followupRequire(cache.attachGrowth() == 0, "Recovery attached while the layout was held")
        cache.holdSlabGrowth(false)
        var attached = cache.attachGrowth()
        try followupRequire(attached > 0 && attached < prepared, "A multi-tranche recovery attached all slabs in one boundary")
        try followupRequire(ledger.reservedBytes == maximum, "First attachment lost or duplicated a prepared reservation")
        for _ in 0..<recovering where attached < prepared {
            let added = cache.attachGrowth()
            try followupRequire(added > 0, "Remaining prepared recovery stalled at a later boundary")
            attached += added
            try followupRequire(ledger.reservedBytes == maximum, "Attachment transferred a reservation incorrectly")
        }
        try followupRequire(
            attached == prepared && cache.attachGrowth() == 0
                && cache.snapshot().allocatedBytes == maximum && cache.snapshot().byteLimit == maximum,
            "Batched attachment did not reach the exact admitted cache target")
        for index in 0..<16 {
            let record = try cache.pin(tickets[index])
            try followupRequire(
                record.bytes[0] == UInt8(index + 1) && record.bytes[record.length - 1] == UInt8(index + 2),
                "Trim/recovery changed a retained expert record")
            try cache.unpinAfterGPUCompletion(tickets[index])
        }
        for index in [0, 8] { try cache.unpinAfterGPUCompletion(tickets[index]) }
        try cache.resize(to: 0)
        cache.completePendingTrim()
    }
    try followupRequire(
        cache.snapshot().allocatedBytes == 0 && ledger.reservedBytes == 0,
        "Completed trim/growth lifecycle leaked cache storage or ledger ownership")
}
