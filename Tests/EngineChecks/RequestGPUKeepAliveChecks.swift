import Foundation
import OrigamiEngine

func requestGPUKeepAliveJoinsAndReleases() throws {
    let ledger = try MemoryLedger(ceiling: 2_000_000, transientMargin: 1)
    let helper = try RequestGPUKeepAlive(ledger: ledger)
    for iteration in 0..<8 {
        try autoreleasepool {
            var session: RequestGPUKeepAlive.Session? = try helper.begin()
            guard ledger.snapshot().reserved == 16384 else { throw EngineError("Keepalive shared flag not charged") }
            do {
                _ = try helper.begin()
                throw EngineError("Nested keepalive was admitted")
            } catch let error as EngineError where error.description == "Keepalive already has an active request" {}
            if iteration == 0 { Thread.sleep(forTimeInterval: 0.03) }
            let snapshot = session!.close()
            guard snapshot.joined, snapshot.maximumInFlight <= 2, snapshot.error == nil,
                iteration != 0 || snapshot.submittedCommands > 0
            else { throw EngineError("Keepalive did not bound and join GPU work") }
            guard session!.close().submittedCommands == snapshot.submittedCommands else {
                throw EngineError("Keepalive close was not idempotent")
            }
            // The stopped resource remains retained and charged by the session.
            guard ledger.snapshot().reserved == 16384 else { throw EngineError("Retained keepalive flag lost its charge") }
            session = nil
        }
        // The flag goes with the worker's last reference and Metal's, a moment after the join.
        guard ledger.waitForRelease(until: { reserved, _ in reserved == 0 }) else {
            throw EngineError("Keepalive leaked between requests")
        }
    }
    try autoreleasepool {
        _ = try helper.begin()  // Deinit closes an immediately discarded session.
    }
    guard ledger.waitForRelease(until: { reserved, _ in reserved == 0 }) else {
        throw EngineError("Discarded keepalive did not join and release")
    }
}

func requestGPUKeepAliveAdmissionAndDefault() throws {
    let ledger = try MemoryLedger(ceiling: 16384, transientMargin: 1)
    let helper = try RequestGPUKeepAlive(ledger: ledger)
    do {
        _ = try helper.begin()
        throw EngineError("Uncharged keepalive flag admitted")
    } catch let error as EngineError where error.code == "insufficient_memory" {}
    guard ledger.snapshot().reserved == 0, !RuntimeConfiguration().gpuKeepAlive else {
        throw EngineError("Keepalive denial changed accounting or default behavior")
    }
    // The release plan keeps it on (the 256K file tier, without register QSA, with the original long poll), and the
    // `gpu_keepalive` setting's override turns it off.
    let folder = FileManager.default.temporaryDirectory
    for (context, fileBacked) in [(8192, false), (262_144, true)] {
        let execution =
            try OrigamiPlan.baseObject(
                context: context, fileBacked: fileBacked, storage: .init(), promptCache: folder,
                allowFast: true, bfloatMatrices: true, overrides: [:])["execution"] as! [String: Any]
        guard execution["gpu_keepalive"] as? Bool == true, execution["register_qsa_decode"] as? Bool == !fileBacked else {
            throw EngineError("Release plan keepalive or its poll policy changed at \(context)")
        }
    }
    for value in [true, false] {
        var object = try OrigamiPlan.baseObject(
            context: 8192, storage: .init(), promptCache: folder, allowFast: true, bfloatMatrices: true,
            overrides: ["gpu_keepalive": value, "expert_cache_slots": 6312, "prefill_expert_cache_slots": 4840])
        object["process_ceiling_bytes"] = UInt64(25_000_000_000)
        object["transient_margin_bytes"] = UInt64(2_000_000_000)
        let configuration = try RuntimeConfiguration.load(object: object)
        // Made either way: the tuner's requests turn it on or off for themselves (`TuneOverrides`); the plan's value is the default.
        guard configuration.gpuKeepAlive == value,
            try configuration.makeKeepAlive(ledger: try MemoryLedger(ceiling: 2_000_000, transientMargin: 1))?.policy
                == .shortRegisterAttention
        else {
            throw EngineError("gpu_keepalive \(value) was not honoured by the plan")
        }
    }
}

func requestGPUKeepAliveShortPolicy() throws {
    let ledger = try MemoryLedger(ceiling: 2_000_000, transientMargin: 1)
    var configuration = RuntimeConfiguration()
    configuration.gpuKeepAlive = true
    configuration.registerQSADecode = true
    guard try configuration.makeKeepAlive(ledger: ledger)?.policy == .shortRegisterAttention else {
        throw EngineError("Configured register QSA lost short keepalive policy")
    }
    configuration.registerQSADecode = false
    guard try configuration.makeKeepAlive(ledger: ledger)?.policy == .original else {
        throw EngineError("Control keepalive policy changed")
    }
    let helper = try RequestGPUKeepAlive(ledger: ledger, policy: .shortRegisterAttention)
    for iteration in 0..<8 {
        try autoreleasepool {
            var session: RequestGPUKeepAlive.Session? = try helper.begin()
            guard ledger.snapshot().reserved == 16384 else { throw EngineError("Keepalive shared flag not charged") }
            do {
                _ = try helper.begin()
                throw EngineError("Nested keepalive was admitted")
            } catch let error as EngineError where error.description == "Keepalive already has an active request" {}
            if iteration == 0 { Thread.sleep(forTimeInterval: 0.03) }
            let snapshot = session!.close()
            guard snapshot.policy == .shortRegisterAttention, snapshot.pollIterations == 1024, snapshot.joined,
                snapshot.maximumInFlight <= 2, snapshot.error == nil,
                iteration != 0 || snapshot.submittedCommands > 0
            else { throw EngineError("Keepalive did not bound and join GPU work") }
            guard session!.close().submittedCommands == snapshot.submittedCommands else {
                throw EngineError("Keepalive close was not idempotent")
            }
            // The stopped resource remains retained and charged by the session.
            guard ledger.snapshot().reserved == 16384 else { throw EngineError("Retained keepalive flag lost its charge") }
            session = nil
        }
        // The flag goes with the worker's last reference and Metal's, a moment after the join.
        guard ledger.waitForRelease(until: { reserved, _ in reserved == 0 }) else {
            throw EngineError("Keepalive leaked between requests")
        }
    }
    try autoreleasepool {
        _ = try helper.begin()  // Deinit closes an immediately discarded session.
    }
    guard ledger.waitForRelease(until: { reserved, _ in reserved == 0 }) else {
        throw EngineError("Discarded keepalive did not join and release")
    }
}
