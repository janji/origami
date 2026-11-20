import CDeviceProbe
import CExactIO
import Darwin
import Foundation
import Metal
import OrigamiEngine
import OrigamiEngineChecks

private let failures = FailureLog()
/// Every failed expectation of the run, in order; checks may record from their own threads.
private final class FailureLog: @unchecked Sendable {
    let lock = NSLock()
    var messages: [String] = []
    func add(_ message: String) {
        lock.lock()
        defer { lock.unlock() }
        messages.append(message)
    }
    var snapshot: [String] {
        lock.lock()
        defer { lock.unlock() }
        return messages
    }
    /// Removes the messages recorded after `count` and returns them (a check turned into a skip).
    func take(after count: Int) -> [String] {
        lock.lock()
        defer { lock.unlock() }
        let taken = Array(messages[count...])
        messages.removeSubrange(count...)
        return taken
    }
}
private func expect(_ condition: @autoclosure () throws -> Bool, file: String = #fileID, line: Int = #line) {
    do { if try !condition() { failures.add("\(file):\(line): expectation failed") } } catch {
        failures.add("\(file):\(line): unexpected error: \(error)")
    }
}
private func expectThrows(_ operation: () throws -> Any, file: String = #fileID, line: Int = #line) {
    do {
        _ = try operation()
        failures.add("\(file):\(line): expected EngineError")
    } catch is EngineError {} catch { failures.add("\(file):\(line): wrong error: \(error)") }
}

func exactReadsRespectOffsetsAndEOFFails() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: url) }
    let bytes = Data((0..<257).map { UInt8($0 % 256) })
    try bytes.write(to: url)
    let file = try ExactFile(url: url)
    expect(try file.read(offset: 117, length: 99) == bytes.subdata(in: 117..<216))
    expect(try file.read(offset: 257, length: 0).isEmpty)
    expectThrows { try file.read(offset: 250, length: 8) }
    expectThrows { try file.read(offset: -1, length: 1) }
    expectThrows { try file.read(offset: 0, length: -1) }
    expectThrows { try file.read(offset: 0, length: Int.max) }
    try file.validateIdentity()
    let handle = try FileHandle(forWritingTo: url)
    try handle.truncate(atOffset: 64)
    try handle.close()
    expectThrows { try file.validateIdentity() }
    expectThrows { try file.read(offset: 0, length: 257) }
}

/// The script `injectedReader` follows: EINTR on the first call, then 3-byte reads, with EOF or ENXIO on the third.
private struct InjectedRead {
    var calls: Int = 0
    var eof: Bool = false
    var hardFailure: Bool = false
}
nonisolated(unsafe) private let injectedReader: origami_read_fn = { _, buffer, count, offset, context in
    let state = context!.assumingMemoryBound(to: InjectedRead.self)
    state.pointee.calls += 1
    if state.pointee.calls == 1 {
        errno = EINTR
        return -1
    }
    if state.pointee.hardFailure && state.pointee.calls == 3 {
        errno = ENXIO
        return -1
    }
    if state.pointee.eof && state.pointee.calls == 3 { return 0 }
    let amount = min(count, 3)
    for i in 0..<amount { buffer!.storeBytes(of: UInt8((Int(offset) + i) % 256), toByteOffset: i, as: UInt8.self) }
    return amount
}

func shortReadAndEINTRRetryWithoutLosingOffset() {
    var state = InjectedRead()
    var bytes = [UInt8](repeating: 0, count: 13)
    let result = bytes.withUnsafeMutableBytes { output in
        withUnsafeMutablePointer(to: &state) { context in
            origami_pread_exact_with(-1, output.baseAddress, output.count, 7, injectedReader, context)
        }
    }
    expect(result.error_code == 0)
    expect(result.bytes_read == 13)
    expect(state.calls == 6)
    expect(bytes == Array(UInt8(7)...UInt8(19)))
}

func partialReadNeverSucceeds() {
    for hardFailure in [false, true] {
        var state = InjectedRead(eof: !hardFailure, hardFailure: hardFailure)
        var bytes = [UInt8](repeating: 0, count: 13)
        let result = bytes.withUnsafeMutableBytes { output in
            withUnsafeMutablePointer(to: &state) { context in
                origami_pread_exact_with(-1, output.baseAddress, output.count, 7, injectedReader, context)
            }
        }
        expect(result.error_code == (hardFailure ? ENXIO : EIO))
        expect(result.bytes_read == 3)
    }
}

func invalidReadsDoNotInvokeIO() {
    var state = InjectedRead()
    var bytes: UInt8 = 0
    withUnsafeMutablePointer(to: &state) { context in
        expect(origami_pread_exact_with(-1, nil, 1, 0, injectedReader, context).error_code == EINVAL)
        withUnsafeMutablePointer(to: &bytes) {
            expect(origami_pread_exact_with(-1, $0, 1, -1, injectedReader, context).error_code == EINVAL)
            expect(origami_pread_exact_with(-1, $0, 2, Int64.max, injectedReader, context).error_code == EINVAL)
        }
    }
    expect(state.calls == 0)
}

func memoryCeilingReservationsAndReleaseAreExact() throws {
    let ledger = try MemoryLedger(ceiling: 1_000, transientMargin: 100)
    let first = try ledger.reserve(bytes: 600, category: "shared")
    let second = try ledger.reserve(bytes: 300, category: "shared")
    expect(ledger.snapshot().reserved == 900)
    expect(ledger.snapshot().categories == ["shared": 900])
    expectThrows { try ledger.reserve(bytes: 1, category: "overflow") }
    expectThrows { try ledger.reserve(bytes: UInt64.max, category: "overflow") }
    try ledger.release(first)
    expect(ledger.snapshot().reserved == 300)
    expect(ledger.snapshot().reservationHighWater == 900)
    expectThrows { try ledger.release(first) }
    // `waitForRelease` wakes at a release made on another thread, and gives up at its bound.
    DispatchQueue.global().asyncAfter(deadline: .now() + 0.05) { try? ledger.release(second) }
    expect(ledger.waitForRelease { reserved, categories in reserved == 0 && categories.isEmpty })
    expect(!ledger.waitForRelease(seconds: 0.05) { reserved, _ in reserved == 1 })
    expectThrows { try MemoryLedger(ceiling: 100, transientMargin: 100) }
}

func sequenceAdmissionIncludesContinuationAndRejectsOverflow() throws {
    expect(try MemoryLedger.sequenceStateBytes(inputTokens: 32_000, outputTokens: 768) == 943_718_400)
    expectThrows { try MemoryLedger.sequenceStateBytes(inputTokens: 262_144, outputTokens: 1) }
    expectThrows { try MemoryLedger.sequenceStateBytes(inputTokens: Int.max, outputTokens: Int.max) }
    expectThrows { try MemoryLedger.sequenceStateBytes(inputTokens: -1, outputTokens: 0) }
}

func phaseWorkspacePreservesAdmissionAcrossFailureAndReuse() throws {
    let ledger = try MemoryLedger(ceiling: 2_100_000_000, transientMargin: 100_000_000)
    var owner: ReferencePhaseWorkspace? = try ReferencePhaseWorkspace(ledger: ledger)
    expect(ledger.snapshot().reserved == 2_000_000_000)
    try owner!.requirePrefill()  // Idempotent; no duplicate reservation.
    try owner!.retirePrefill()
    expect(ledger.snapshot().reserved == 536_870_912)
    let slabs = try ledger.reserve(bytes: 1_415_577_600, category: "test_decode_slabs")
    expectThrows { try owner!.requirePrefill() }
    expect(ledger.snapshot().reserved == 536_870_912 + 1_415_577_600)
    expect(ledger.snapshot().categories["reference_prefill_workspace_bound"] == nil)
    try owner!.retirePrefill()  // A failed larger scope cannot retire the base.
    try ledger.release(slabs)
    try owner!.requirePrefill()
    expect(ledger.snapshot().reserved == 2_000_000_000)
    owner = nil
    expect(ledger.snapshot().reserved == 0)
    let blocked = try ledger.reserve(bytes: 100_000_000, category: "test_admission_block")
    expectThrows { try ReferencePhaseWorkspace(ledger: ledger) }
    expect(ledger.snapshot().reserved == 100_000_000)
    try ledger.release(blocked)
}

func bf16TiesExceptionsAndAllFiniteRoundtrips() {
    expect(BF16.encode(Float(bitPattern: 0x3f80_8000)) == 0x3f80)
    expect(BF16.encode(Float(bitPattern: 0x3f81_8000)) == 0x3f82)
    expect(BF16.encode(-Float.zero) == 0x8000)
    expect(BF16.encode(Float(bitPattern: 0x7f80_0001)) & 0x7fc0 == 0x7fc0)
    for bits in UInt32(0)...UInt32(UInt16.max) {
        let value = BF16.decode(UInt16(bits))
        if !value.isNaN { expect(BF16.encode(value) == UInt16(bits)) }
    }
}

func metalReferenceExactIdentities() throws {
    let kernels = try ReferenceKernels()
    let x = [Float](repeating: 1, count: 2_560)
    // Every row stores nibble sequence 0...7, with exact dyadic scales and biases.
    let words = [UInt32](repeating: 0x7654_3210, count: 2 * 2_560 / 8)
    let scales = [UInt16](repeating: BF16.encode(0.25), count: 2 * 2_560 / 64)
    let biases = [UInt16](repeating: BF16.encode(-0.5), count: scales.count)
    expect(try kernels.qmatvec(x: x, words: words, scales: scales, biases: biases) == [960, 960])
    expect(try kernels.rmsnorm(x: x, weights: x) == x)
    expect(try kernels.rmsnorm(x: x, weights: x, groupSize: 128) == x)
    let ropeInput = (0..<256).map { Float($0 - 128) / 16 }
    expect(try kernels.rope(x: ropeInput, width: 256, position: 0) == ropeInput)
    expect(try kernels.silu(x: [0, -0]) == [0, -0])
    // Signed zero and subnormal values tie by original index; NaNs rank after -inf.
    let scores: [Float] = [.nan, -.infinity, -0.0, 0.0, Float.leastNonzeroMagnitude, 5, 5, .infinity]
    expect(try kernels.topIndices(scores: scores, count: 8) == [7, 5, 6, 2, 3, 4, 1, 0])
    // Full supported block domain, with many ties and an exact global-index oracle.
    let fullDomain = (0..<65_536).map { Float($0 % 17) }
    expect(try kernels.topIndices(scores: fullDomain, count: 512) == (0..<512).map { UInt32(16 + 17 * $0) })
    expectThrows { try kernels.rope(x: ropeInput, width: 256, position: 262_144) }
    expectThrows { try kernels.topIndices(scores: scores, count: 9) }
    expectThrows { try kernels.qmatvec(x: [1], words: [0], scales: [0], biases: [0]) }
}

func gdnContinuationMasksAndFP32State() throws {
    let kernels = try ReferenceKernels()
    let qRow = 16 * 128
    let vRow = 48 * 128
    let stateCount = 48 * 128 * 128
    let q = (0..<(qRow * 3)).map { Float($0 % 13 - 6) / 128 }
    let k = (0..<(qRow * 3)).map { Float($0 % 11 - 5) / 128 }
    let v = (0..<(vRow * 3)).map { Float($0 % 17 - 8) / 32 }
    let decay = [Float](repeating: 0.875, count: 3 * 48)
    let beta = [Float](repeating: 0.5, count: 3 * 48)
    let initial = (0..<stateCount).map { Float($0 % 7 - 3) / 1024 }
    let full = try kernels.gdn(q: q, k: k, v: v, decay: decay, beta: beta, state: initial, mask: [1, 0, 1])
    let first = try kernels.gdn(
        q: Array(q[..<qRow]), k: Array(k[..<qRow]), v: Array(v[..<vRow]),
        decay: Array(decay[..<48]), beta: Array(beta[..<48]), state: initial)
    let suffix = try kernels.gdn(
        q: Array(q[qRow...]), k: Array(k[qRow...]), v: Array(v[vRow...]),
        decay: Array(decay[48...]), beta: Array(beta[48...]), state: first.state, mask: [0, 1])
    expect(full.output.map(\.bitPattern) == (first.output + suffix.output).map(\.bitPattern))
    expect(full.state.map(\.bitPattern) == suffix.state.map(\.bitPattern))
    expect(full.output[vRow..<(2 * vRow)].allSatisfy { $0 == 0 })
    expect(full.state.contains { BF16.decode(BF16.encode($0)).bitPattern != $0.bitPattern })
    let masked = try kernels.gdn(
        q: Array(q[..<qRow]), k: Array(k[..<qRow]), v: Array(v[..<vRow]),
        decay: Array(decay[..<48]), beta: Array(beta[..<48]), state: initial, mask: [0])
    expect(masked.state.map(\.bitPattern) == initial.map(\.bitPattern))
}

func causalIndexerAndAttentionBoundaries() throws {
    let kernels = try ReferenceKernels()
    for n in [3, 4, 5, 2048, 2049, 2051] {
        let scores = [Float](repeating: 1, count: n / 4)
        let selected = try kernels.selectedPositions(scores: scores, position: n - 1, kvLength: n)
        expect(selected == (0..<n).map(UInt32.init))
    }
    let n = 2052
    let selected = try kernels.selectedPositions(scores: [Float](repeating: 1, count: n / 4), position: n - 1, kvLength: n)
    expect(selected == (0..<2048).map(UInt32.init))
    // At an earlier position all visible NaNs rank behind invisible -infinity.
    let nanScores = [Float](repeating: .nan, count: 1024)
    expect(try kernels.selectedPositions(scores: nanScores, position: 3, kvLength: 4096).isEmpty)
    expect(try kernels.selectedPositions(scores: nanScores, position: 4, kvLength: 4096) == [4])
    let query = [Float](repeating: 0, count: 24 * 256)
    let keys = [Float](repeating: 0, count: 2 * 4 * 256)
    let values = (0..<(2 * 4 * 256)).map { Float(($0 / 256) % 4) }
    expect(
        try kernels.attention(query: query, keys: keys, values: values, kvLength: 4, position: 3)
            == [Float](repeating: 1.5, count: query.count))
    expect(
        try kernels.attention(query: query, keys: keys, values: values, kvLength: 4, position: 3, selectedPositions: [1, 3])
            == [Float](repeating: 2, count: query.count))
    expect(
        try kernels.attention(query: query, keys: keys, values: values, kvLength: 4, position: 3, selectedPositions: [])
            == [Float](repeating: 1.5, count: query.count))
    let idxQuery = [Float](repeating: 0.125, count: 4 * 128)
    let idxKeys = [Float](repeating: 0.25, count: 3 * 128)
    let expected = Float(16) / sqrt(Float(128))
    expect(try kernels.indexerScores(query: idxQuery, pooledKeys: idxKeys) == [Float](repeating: expected, count: 3))
}

func pinnedQuantizedDispatchAndRounding() throws {
    let schedule = try QuantizedSchedule(rows: 24, width: 10240, outputs: 320)
    expect(schedule.kind == "qmm_splitk" && schedule.splitK == 40 && schedule.partitionSize == 256)
    expect(try QuantizedSchedule(rows: 5, width: 10240, outputs: 320).kind == "qmv_fast")
    expect(try QuantizedSchedule(rows: 24, width: 2560, outputs: 10240).kind == "qmm")
    let kernels = try ReferenceKernels()
    let words = [UInt32](repeating: 0x7654_3210, count: 8 * 2560 / 8)
    let scales = [UInt16](repeating: BF16.encode(0.25), count: 8 * 2560 / 64)
    let biases = [UInt16](repeating: BF16.encode(-0.5), count: scales.count)
    for rows in [1, 24] {
        let result = try kernels.qlinearPinnedBF16(
            x: [Float](repeating: 1, count: rows * 2560), width: 2560,
            words: words, scales: scales, biases: biases)
        expect(result.output == [Float](repeating: 960, count: rows * 8))
    }
}

/// A failed check is reported as skipped for the GPU only on a host whose GPU cannot run Origami and only if it used Metal.
func skipsForGPU(gpuProblem: String?, failed: Bool, usedGPU: Bool) -> Bool { gpuProblem != nil && failed && usedGPU }

/// The runner sees device requests from the checks and from inside the engine, and skips only checks that made one.
func gpuSkipsOnlyChecksThatUseMetal() throws {
    let start = origami_metal_device_requests()
    _ = MTLCreateSystemDefaultDevice()
    expect(origami_metal_device_requests() == start + 1)
    _ = try? OrigamiHost.current()
    expect(origami_metal_device_requests() > start + 1)
    expect(skipsForGPU(gpuProblem: "no Metal device", failed: true, usedGPU: true))
    expect(!skipsForGPU(gpuProblem: "no Metal device", failed: true, usedGPU: false))
    expect(!skipsForGPU(gpuProblem: "no Metal device", failed: false, usedGPU: true))
    expect(!skipsForGPU(gpuProblem: nil, failed: true, usedGPU: true))
}

/// Whether a check failed because macOS aborted its GPU work (what `GPURetry` retries for requests): thrown, or in a message.
func gpuInterrupted(thrown: Error?, messages: ArraySlice<String>) -> Bool {
    thrown.map { EngineError.interrupted($0) || GPURetry.isInterruption($0) } ?? false
        || messages.contains { $0.contains("ImpactingInteractivity") || $0.contains("gpu_interrupted") }
}

func gpuInterruptionsAreRecognized() throws {
    expect(gpuInterrupted(thrown: EngineError.fakeInterruption, messages: []))
    expect(gpuInterrupted(thrown: EngineError.gpu("step", EngineError.fakeInterruption), messages: []))
    expect(gpuInterrupted(thrown: nil, messages: ["x: \(EngineError.fakeInterruption)"]))
    expect(!gpuInterrupted(thrown: EngineError("other"), messages: ["file:1: expectation failed"]))
}

let tests: [(String, () throws -> Void)] =
    [
        ("gpu_skips_only_checks_that_use_metal", gpuSkipsOnlyChecksThatUseMetal),
        ("gpu_interruptions_are_recognized", gpuInterruptionsAreRecognized),
        ("conversation_pages_boundaries_quota", PersistentConversationSelfChecks.pages),
        ("conversation_pending_and_schedules", PersistentConversationSelfChecks.pendingAndSchedules),
        ("conversation_intra_scope_assembly", PersistentConversationSelfChecks.scopeAssembly),
        ("conversation_file_scope_assembly", PersistentConversationSelfChecks.scopeAssemblyFiles),
        ("conversation_dense_metadata_pinned_quota", PersistentConversationSelfChecks.denseMetadataAndPinnedQuota),
        ("conversation_file_ownership_faults", ConversationFrontierSelfChecks.ownershipAndFaults),
        ("conversation_file_write_window_stall", ConversationFrontierSelfChecks.stagingWindowAbsorbsWriterStall),
        ("conversation_file_write_window_configuration", frontierWriteWindowConfiguration),
        ("qsa_shared_metal_page_lifetime", QSAPagedAttentionSelfChecks.lifetime),
        ("qsa_shared_metal_attention_parity", QSAPagedAttentionSelfChecks.attentionParity),
        ("qsa_register_decode_actual_paged_encoder", QSAPagedAttentionSelfChecks.registerDecodeParity),
        ("long_fused_attention_deployment_parity", longFusedAttentionDeploymentParity),
        ("followup_resident_digest_boundaries", followupResidentDigestBoundaries),
        ("followup_short_suffix_scope_rows", followupShortSuffixScopeRows),
        ("followup_admitted_prompt_cache_parity", followupAdmittedPromptCacheParity),
        ("followup_tiny_swap_adaptive_recovery", followupTinySwapAndAdaptiveRecovery),
        ("followup_pending_trim_batched_growth", followupPendingTrimAndBatchedGrowth),
        ("conversation_actual_shape_8k", PersistentConversationSelfChecks.shape8K),
        ("conversation_actual_shape_32k", PersistentConversationSelfChecks.shape32K),
        ("conversation_file_actual_shape_8k", PersistentConversationSelfChecks.shape8KFiles),
        ("conversation_file_actual_shape_32k", PersistentConversationSelfChecks.shape32KFiles),
        ("gdn_preparation_bf16_joined", gdnPreparationPreservesBoundariesAndJoinedResults),
        ("gdn_preparation_geometry_admission", gdnPreparationRejectsInvalidAndUnadmittedWork),
        ("pinned_ple_dot_bf16_boundaries", pinnedPLEDotPreservesBF16ProductLocalAndSIMDBoundaries),
        ("pinned_ple_dot_geometry_admission", pinnedPLEDotRejectsInvalidOrUnadmittedWork),
        ("batched_rope_position_head_bits", batchedRoPEPreservesScalarPositionsAndHeads),
        ("batched_rope_geometry_default", batchedRoPERejectsGeometryAndDefaultsOff),
        ("pinned_attention_dispatch_boundaries", pinnedAttentionDispatchBoundaries),
        ("pinned_attention_batch_plan", pinnedAttentionBatchPlanBounds),
        ("compact_prefill_reuse_layout", CompactPrefillReuseSelfChecks.layout),
        ("compact_prefill_reuse_maximum_union", CompactPrefillReuseSelfChecks.maximumUnion),
        ("compact_prefill_reuse_actual_shapes", CompactPrefillReuseSelfChecks.actualShapes),
        ("compact_prefill_reuse_exceptional_semantics", CompactPrefillReuseSelfChecks.exceptionalRepair),
        ("compact_prefill_reuse_grouped_pv_actual_shapes", { try CompactPrefillReuseSelfChecks.actualShapes(grouped: true) }),
        ("compact_prefill_reuse_grouped_pv_exceptional_semantics", { try CompactPrefillReuseSelfChecks.exceptionalRepair(grouped: true) }),
        (
            "compact_prefill_reuse_exact_kernels_actual_shapes",
            { try CompactPrefillReuseSelfChecks.actualShapes(grouped: true, exact: true) }
        ),
        (
            "compact_prefill_reuse_exact_kernels_exceptional_semantics",
            { try CompactPrefillReuseSelfChecks.exceptionalRepair(grouped: true, exact: true) }
        ),
        ("qsa_exact_kernels_configuration", qsaExactKernelsConfiguration),
        ("qsa_exact_kernels_compact_qk_pv_bytes", QSAExactKernelSelfChecks.compactExactness),
        ("qsa_exact_kernels_stripe_pv_bytes", QSAExactKernelSelfChecks.stripeExactness),
        ("qsa_exact_kernels_paged_library_bytes", QSAExactKernelSelfChecks.pagedExactness),
        ("compact_attention_layout", CompactAttentionSelfChecks.layoutBounds),
        ("compact_attention_pair_semantics", CompactAttentionSelfChecks.originalPairSemantics),
        ("prefill_qk_pv_reuse_original_and_paged_parity", PrefillAttentionReuseSelfChecks.parity),
        ("prefill_qk_pv_reuse_exact_stripe_original_and_paged_parity", { try PrefillAttentionReuseSelfChecks.parity(exact: true) }),
        ("pinned_rope_configuration_geometry", pinnedRoPEConfigurationAndGeometry),
        ("pinned_indexer_config_bounds", pinnedIndexerConfigurationAndBounds),
        ("pinned_indexer_known_answer_retirement", pinnedIndexerKnownAnswerAndRetirement),
        ("qsa_pooled_index_paged_scores", QSAPooledPageTableSelfChecks.pagedScoresMatchCopy),
        ("pinned_rope_rounding_cache_retirement", pinnedRoPERoundingCacheAndRetirement),
        ("pinned_attention_batch_parity_retirement", pinnedAttentionBatchParityAndRetirement),
        ("pinned_attention_batch_denial", pinnedAttentionBatchScratchDenialRetiresBuffers),
        ("pinned_attention_grouping_mask_retirement", pinnedAttentionGroupingMaskAndRetirement),
        ("pinned_attention_geometry_admission", pinnedAttentionRejectsUnadmittedAndInvalidWork),
        ("pinned_attention_partial_key_reduction_order", pinnedAttentionPartialKeyReductionOrder),
        ("expert_command_batch_configuration", expertCommandBatchConfigurationBounds),
        ("expert_command_batch_mixed_schedules", expertCommandBatchPreservesMixedSchedulesAndTiles),
        ("expert_command_batch_order_refill", expertCommandBatchWindowOrderBoundsAndRefill),
        ("expert_command_batch_cancellation_callbacks", expertCommandBatchCancellationAndCallbackFailureRetirePins),
        ("expert_command_batch_failure_admission", expertCommandBatchPartialJoinAndAdmissionFailureCleanup),
        ("expert_command_batch_invalid_jobs", expertCommandBatchInvalidJobsAllocateNothing),
        ("fused_gdn_config_admission", fusedGDNConfigurationAndAdmission),
        ("fused_gdn_values_exceptions_cancel", fusedGDNFixedValuesExceptionsAndCancellation),
        ("fused_gdn_layer_state", fusedGDNLayerParityThresholdAndTransactionalState),
        ("fused_hc_decode_configuration", fusedHCDecodeConfiguration),
        ("gdn_input_composition_configuration", gdnInputCompositionConfiguration),
        ("gdn_device_block_configuration", gdnDeviceBlockConfiguration),
        ("gdn_device_block_composition_ownership", { _ = try GDNDeviceBlockSelfChecks.composed() }),
        ("device_ple_exact_against_reference_arithmetic", DevicePLESelfChecks.exactness),
        ("host_stage_prefetch_exact_rows_and_bounded_dense", HostStagePrefetchSelfChecks.exactness),
        ("prefill_double_buffer_configuration", prefillDoubleBufferConfiguration),
        ("prefill_double_buffer_pipeline_order_deferred_failures", PrefillDoubleBufferSelfChecks.pipeline),
        ("prefill_double_buffer_gdn_chain_exact_bytes", PrefillDoubleBufferSelfChecks.exactness),
        ("prefill_double_buffer_replay_restores_successor_input", PrefillDoubleBufferSelfChecks.replayRestore),
        ("span_read_pool_disjoint_joined_errors", SpanReadPoolSelfChecks.exactness),
        ("prefill_layer_command_waves8_exact_actual_scopes", { try PrefillLayerCommandSelfChecks.exactActualScope(waves: 8) }),
        (
            "prefill_layer_command_waves8_exact_mixed_tails",
            { try PrefillLayerCommandSelfChecks.exactActualScope(mixedOnly: true, waves: 8) }
        ),
        ("quantized_mma_bf16_configuration", quantizedMMAConfiguration),
        ("grouped_mma_bf16_boundaries_counters", groupedMMABoundariesAndCounters),
        ("gdn_exact_kernels_configuration", gdnExactKernelsConfiguration),
        ("hc_exact_fusion_configuration", hcExactFusionConfiguration),
        ("hc_exact_fusion_split_reduce_bit_identical_counted_fallbacks", HCExactFusionSelfChecks.splitReduceExactness),
        ("hc_exact_fusion_fused_entry_bit_identical_tile", HCExactFusionSelfChecks.fusedEntryExactness),
        ("hc_decode_fusion_configuration", hcDecodeFusionConfiguration),
        ("hc_decode_fusion_row_bit_identical_entry_epilogue_injection", HCDecodeFusionSelfChecks.exactness),
        ("decode_projection_tile8_actual_encoder", DecodeProjectionSelfChecks.actualEncoderParity),
        ("decode_frontier_shared_activation_exceptions", DecodeFrontierSelfChecks.exceptions),
        ("whole_schedule_qsa_buffer_interfaces", WholeExecutionSelfChecks.bufferInterfaces),
        ("decode_frontier_routing_shared_gate", DecodeFrontierSelfChecks.routing),
        ("gdn_buffer_region_configuration", gdnBufferRegionConfiguration),
        ("gathered_expert_configuration_startup", gatheredExpertConfigurationAndStartup),
        ("gathered_expert_failed_joins_admission", gatheredExpertFailureAndAdmissionPinOwnership),
        ("gathered_shared_failure_joins_producers", gatheredSharedFailureBeforeConsumeJoinsProducers),
        ("decode_publication_checkpoint_ownership", DecodeEarlyDemandSelfChecks.publicationOwnership),
        ("decode_router_event_joined_ownership", DecodeEarlyDemandSelfChecks.eventOwnership),
        ("decode_early_demand_staged_request_retirement", DecodeEarlyDemandSelfChecks.stagedRequests),
        ("decode_prerouting_predictor_exact_against_reference_router", DecodePreroutingSelfChecks.predictorExactness),
        ("decode_prerouting_adopts_in_flight_read_without_duplication", DecodePreroutingSelfChecks.adoptionWithoutDuplication),
        ("decode_prerouting_drops_without_spare_slot", DecodePreroutingSelfChecks.dropWithoutSpareSlot),
        ("decode_prerouting_counters_absent_when_off", DecodePreroutingSelfChecks.countersAbsentWhenOff),
        ("decode_prerouting_slot_override_evicts_only_cold_records", DecodePreroutingSlotOverrideSelfChecks.evictsOnlyColdRecords),
        ("decode_prerouting_slot_override_width_and_adoption", DecodePreroutingSlotOverrideSelfChecks.widthAndAdoption),
        ("decode_prerouting_read_hygiene_cancel_releases_exact_admission", DecodePreroutingReadHygieneSelfChecks.cancelReleasesExactly),
        ("decode_prerouting_read_hygiene_adoption_cancels_queued_unrouted", DecodePreroutingSelfChecks.readHygieneAdoption),
        ("decode_prerouting_read_hygiene_pool_priority_order", DecodePreroutingReadHygieneSelfChecks.poolPriorityOrder),
        (
            "decode_prerouting_read_hygiene_scheduler_hold_and_cancellation",
            DecodePreroutingReadHygieneSelfChecks.schedulerHoldAndCancellation
        ),
        ("decode_prerouting_read_hygiene_off_state_and_configuration", DecodePreroutingReadHygieneSelfChecks.offStateAndConfiguration),
        ("decode_ple_internal_race_either_source_wins_exactly", DecodePLERaceSelfChecks.eitherSourceWinsExactly),
        ("decode_ple_internal_race_failures_cancellation_default", DecodePLERaceSelfChecks.failuresCancellationAndDefault),
        ("decode_ple_internal_race_configuration_and_report", DecodePLERaceSelfChecks.configurationAndReport),
        ("fast_late_miss_skip_decision", FastLateMissSkipSelfChecks.decision),
        ("fast_late_miss_skip_window_late_only", FastLateMissSkipSelfChecks.windowSkipsOnlyLateReads),
        ("fast_late_miss_skip_read_reserve", FastLateMissSkipSelfChecks.readReserveBoundsSkippedReads),
        ("fast_late_miss_skip_configuration_and_report", FastLateMissSkipSelfChecks.configurationAndReport),
        ("fast_opt_in_request_mode_and_report", FastLateMissSkipSelfChecks.requestModeSelection),
        ("fast_opt_in_conversation_timeline_rule", FastLateMissSkipSelfChecks.conversationTimelineRule),
        ("prefill_ple_overlap_scope_race_exact_and_disjoint", PrefillPLEOverlapSelfChecks.scopeRaceExactAndDisjoint),
        ("prefill_ple_overlap_join_poll_and_tail_rows", PrefillPLEOverlapSelfChecks.joinPollAndTailRows),
        ("prefill_ple_overlap_configuration_and_report", PrefillPLEOverlapSelfChecks.configurationAndReport),
        ("decode_prerouting_slot_override_counters_and_configuration", DecodePreroutingSlotOverrideSelfChecks.countersAndConfiguration),
        ("resident_first_demand_joined_retirement", residentFirstDemandJoinsAndRetires),
        ("gdn_buffer_region_numerics_ownership", GDNBufferRegionSelfChecks.bufferRegionNumericsAdmissionAndCancellation),
        ("gdn_buffer_region_layer_transaction", GDNBufferRegionSelfChecks.bufferRegionLayerParityAndPublication),
        ("gdn_buffer_region_checkpoint_lifetime", GDNBufferRegionSelfChecks.bufferRegionCheckpointOwnership),
        ("gdn_input_composition_actual_rows_joined_ownership", gdnInputCompositionActualRowsAndJoinedOwnership),
        ("fused_hc_prefill_configuration", fusedHCPrefillConfiguration),
        ("fused_hc_prefill_actual_rows_parity", fusedHCPrefillActualRowSchedulesAndParity),
        ("fused_hc_prefill_admission_cancellation_retirement", fusedHCPrefillAdmissionCancellationAndRetirement),
        ("fused_hc_decode_exhaustive_handoffs", fusedHCDecodeExhaustiveHandoffs),
        ("fused_hc_decode_intermediates_layer_parity", fusedHCDecodeAllIntermediatesAndLayerParity),
        ("fused_hc_decode_admission_cancellation_retirement", fusedHCDecodeAdmissionCancellationAndRetirement),
        ("qsa_decode_storage_parity", QSADecodeStorageSelfChecks.parity),
        ("qsa_decode_storage_frontier_failure", QSADecodeStorageSelfChecks.frontierAndFailure),
        ("qsa_decode_storage_escape_denial", QSADecodeStorageSelfChecks.escapedBufferAndDenial),
        ("qsa_encoded_append_ownership", QSADecodeStorageSelfChecks.encodedAppendOwnership),
        ("whole_workspace_two_readers_ownership", WholeExecutionSelfChecks.workspaceAndReaders),
        ("file_kv_integrity_quota_cow_retirement", QSAFilePageSelfChecks.integrityAndLifetime),
        ("file_kv_tail_appends_in_place", QSAFilePageSelfChecks.appendsInPlace),
        ("file_kv_maximum_cache_reader_ownership", QSAFileStagingSelfChecks.maximumCacheAndReaders),
        ("file_kv_attention_maximum_exception_bits", QSAPagedAttentionSelfChecks.fileAttentionParity),
        ("paged_checkpoint_maximum_format", PagedCheckpointSelfChecks.maximumFormat),
        ("paged_checkpoint_clone_delta_rollback", PagedCheckpointSelfChecks.cloneDeltaAndRollback),
        ("file_kv_configuration_simultaneous_owners", fileConversationConfigurationAndOwners),
        ("file_kv_maximum_rotation_ownership", fileConversationMaximumRotationOwnership),
        ("whole_prefill_scope_escrow_adoption", WholeExecutionSelfChecks.scopeEscrowAdoption),
        ("whole_schedule_configuration", wholeScheduleConfigurationIsBoundedAndDefaultOff),
        ("generation_progress_failure_bounds", generationProgressWriterIsBoundedAndFailureVisible),
        ("tool_anchor_marker_boundaries", toolAnchorMarkerBoundaries),
        ("tool_anchor_state_ownership", PersistentConversationSelfChecks.toolAnchorOwnership),
        ("family_tool_anchors", familyToolAnchors),
        ("family_pool_allocation", familyPoolAllocation),
        ("family_adaptive_expert_pools", familyAdaptiveExpertPools),
        ("whole_schedule_joined_command_ownership", wholeScheduleCommandsJoinBeforeRetirement),
        ("whole_token_plan_reader_admission", WholeTokenPlanSelfChecks.readerAdmission),
        ("whole_token_plan_construction_boundaries", WholeTokenPlanSelfChecks.constructionBoundaries),
        ("whole_token_plan_convolution_tail_bytes", WholeTokenPlanSelfChecks.convolutionTailBytes),
        ("whole_token_plan_configuration", wholeTokenPlanConfiguration),
        ("whole_shared_gate_actual_row_bytes", WholeExecutionSelfChecks.sharedGateActualRows),
        ("whole_qsa_dense_index_actual_row_bytes", WholeExecutionSelfChecks.denseIndexActualRows),
        ("fused_hc_config_geometry", fusedHCConfigurationAndGeometry),
        ("fused_hc_rounding_capture_retirement", fusedHCRoundingCaptureAndRetirement),
        ("fused_hc_cancellation_denial", fusedHCCancellationAndAllocationDenial),
        ("fused_hc_layer_threshold_capture", fusedHCLayerThresholdAndCapture),
        ("quantized_tiled_split_fallbacks", quantizedTiledMatchesSplitKAndLeavesOtherSchedulesAlone),
        ("quantized_tiled_offsets_geometry", quantizedTiledBufferOffsetsAndInvalidSpansStayExact),
        ("quantized_bm32_configuration", quantizedBM32Configuration),
        ("quantized_bm32_boundary_parity", quantizedBM32BoundaryParity),
        ("quantized_tiled_configuration", quantizedTiledConfigurationIsDefaultOffAndRequiresSIMD),
        ("device_moe_configuration", PrefillMoESelfChecks.configuration),
        ("device_moe_transfer_bits", PrefillMoESelfChecks.transferBitsAndCanonicalReduction),
        ("device_moe_contributions_pass_on", PrefillMoESelfChecks.contributionsPassToNextLayer),
        ("device_moe_expert_parity", PrefillMoESelfChecks.expertParityAcrossTiles),
        ("device_moe_lifetime", PrefillMoESelfChecks.admissionCancellationAndEscape),
        (
            "device_moe_charge_waits_for_retirement",
            {
                let late = try PrefillMoESelfChecks.chargeWaitsForRetirement()
                FileHandle.standardError.write(
                    Data("device_moe_charge_waits_for_retirement: \(late) of 500 joined commands still held their buffer\n".utf8))
            }
        ),
        ("expert_scratch_configuration", expertScratchConfiguration),
        ("expert_scratch_reuse_retirement", expertScratchReuseAndRetirement),
        ("expert_scratch_cancel_admission", expertScratchCancellationAndAdmission),
        ("grouped_bm32_configuration", groupedBM32Configuration),
        ("grouped_bm32_boundaries_retirement", groupedBM32BoundariesAndRetirement),
        ("grouped_tiled_configuration", groupedExpertTiledConfigurationRequiresItsExecutionPath),
        ("grouped_tiled_tail_batch_parity", groupedExpertTiledPreservesEveryBoundaryAcrossTailAndBatchShapes),
        ("grouped_tiled_cancel_admission", groupedExpertTiledCancellationAndAdmissionRetireBeforeReuse),
        ("fused_expert_projection_schedules", fusedExpertMatchesCachedAndGroupedProjectionSchedules),
        ("fused_expert_unary_boundaries", fusedExpertPreservesIntermediateBF16AndUnaryBoundaries),
        ("fused_expert_cancel_pin_retirement", fusedExpertCancellationJoinsBeforeConsumerRetirement),
        ("fused_expert_geometry_admission", fusedExpertGeometryAndAllocationFailureAreBounded),
        ("expert_phase_configuration_bounds", expertPhaseConfigurationDefaultsAndBounds),
        ("decode_residency_complete_envelope", decodeResidencyCompleteEnvelope),
        ("decode_residency_joined_retirement", decodeResidencyJoinedLoanRetirement),
        ("decode_residency_arena_loan", decodeResidencyArenaLoan),
        ("expert_phase_lazy_growth_hot_shrink", expertPhaseLazyGrowthColdShrinkAndResourceCharge),
        ("expert_phase_busy_atomicity", expertPhaseBusyChangesAreAtomic),
        ("expert_phase_admission_failure", expertPhaseAdmissionFailureIsNotEvictionFallback),
        ("expert_phase_execution_lifecycle", expertPhaseExecutionLifecycleAndBusyBoundary),
        ("expert_phase_cancellation_stability", expertPhaseCancellationJoinsAndKeepsStableTarget),
        ("prefill_read_ahead_maximum_rolling_ownership", prefillReadAheadMaximumRollingOwnership),
        ("prefill_read_ahead_follows_shrunk_cache", prefillReadAheadFollowsShrunkCache),
        ("prefill_read_ahead_after_cut", prefillReadAheadAfterCut),
        ("prefill_read_ahead_completion_refill_late_retirement", prefillReadAheadCompletionRefillAndLateRetirement),
        ("decode_slab_preparation_configuration_admission", decodeSlabPreparationConfigurationAndAdmission),
        ("decode_slab_preparation_preserving_cancellation", decodeSlabPreparationPreservesDemandAndCancellation),
        ("decode_slab_wiring_configuration_reporting", decodeSlabWiringConfigurationAndReporting),
        ("decode_slab_wiring_preserving_counts", decodeSlabWiringPreservesBytesAndCounts),
        ("prefill_read_ahead_exact_fallback", prefillReadAheadExactFallback),
        ("prefill_read_ahead_cancel_phase_guard", prefillReadAheadCancellationAndPhaseGuard),
        ("prefill_read_ahead_maximum_admission", prefillReadAheadMaximumAdmission),
        ("prefill_layer_command_exact_actual_scope", { try PrefillLayerCommandSelfChecks.exactActualScope() }),
        ("prefill_layer_command_mixed_tail_arithmetic", { try PrefillLayerCommandSelfChecks.exactActualScope(mixedOnly: true) }),
        ("prefill_layer_command_cancellation_publication", PrefillLayerCommandSelfChecks.cancellationAndPublication),
        ("prefill_layer_command_complete_admission", PrefillLayerCommandSelfChecks.completeAdmission),
        ("prefill_layer_command_last_layer_row_skip", PrefillLayerCommandSelfChecks.lastLayerRowSkip),
        ("prefill_layer_command_buffers_bounded", PrefillLayerCommandSelfChecks.commandBuffers),
        ("prefill_layer_command_gate_up_fusion_bitwise", PrefillLayerCommandSelfChecks.gateUpFusion),
        ("mixed_tail_scope_schedule_identity", MixedTailSelfChecks.schedulesAndIdentity),
        ("mixed_tail_scope_complete_memory", MixedTailSelfChecks.completeMemoryAndConfiguration),
        ("expert_execution_bounded_window_reuse", expertExecutionWindowIsBoundedAndReusesRecords),
        ("expert_execution_fallback_fault_cleanup", expertExecutionPackedFallbackAndExceptionalConsumerCleanup),
        ("expert_execution_cancel_join", expertExecutionCancellationJoinsBeforeReuse),
        ("expert_execution_buffer_qmv_parity", expertExecutionBufferNativeMatchesIndividualPinnedQMV),
        ("serving_memory_failure_mapping", servingMemoryFailureMappingPreservesAdmissionReason),
        ("serving_stop_scalar_boundaries", stopSequenceHoldbackMatchesAcrossEveryScalarBoundary),
        ("serving_stop_flush_reasoning", stopSequenceFlushPreservesUnmatchedSuffixAndReasoning),
        ("exact_reads_offsets_mutation_eof", exactReadsRespectOffsetsAndEOFFails),
        ("hot_expert_copy_identity_bytes_retirement", hotExpertCopyIdentityBytesAndRetirement),
        ("hot_copy_qwen_keeper_builds_while_serving", hotCopyQwenKeeperBuildsWhileServing),
        ("hot_copy_keeper_yields_to_arriving_request", hotCopyKeeperYieldsToArrivingRequest),
        ("family_hot_copy_keeper_yields_to_arriving_request", familyHotCopyKeeperYieldsToArrivingRequest),
        ("hot_expert_placement_queue_selection", hotExpertPlacementOverridesCalibrationAndKeepsExplicitSource),
        ("short_reads_eintr", shortReadAndEINTRRetryWithoutLosingOffset),
        ("partial_read_failure", partialReadNeverSucceeds),
        ("invalid_reads", invalidReadsDoNotInvokeIO),
        ("memory_reservations", memoryCeilingReservationsAndReleaseAreExact),
        ("phase_workspace_admission_failure_reuse", phaseWorkspacePreservesAdmissionAcrossFailureAndReuse),
        ("continuation_admission", sequenceAdmissionIncludesContinuationAndRejectsOverflow),
        ("bf16_ties_exceptions_exhaustive", bf16TiesExceptionsAndAllFiniteRoundtrips),
        ("metal_reference_exact_identities", metalReferenceExactIdentities),
        ("gdn_continuation_masks_fp32_state", gdnContinuationMasksAndFP32State),
        ("causal_indexer_attention_boundaries", causalIndexerAndAttentionBoundaries),
        ("pinned_quantized_dispatch_rounding", pinnedQuantizedDispatchAndRounding),
        ("hyper_connection_pinned_column_mean", hyperConnectionMeanUsesPinnedColumnReduction),
        ("ordinary_rmsnorm_bf16_before_weight", ordinaryRMSNormRoundsBeforeWeight),
        ("cache_cancel_pin_late_completion_lifetime", cacheCancellationPinsAndLateCompletions),
        ("cache_clock_pins", cacheClockEvictsOnlyUnpinnedRecords),
        ("cache_prefill_scan_eviction", cachePrefillScanEvictsPassedRecords),
        ("cache_reuse_actual_capacity_lifetime", ExpertReuseSelfChecks.actualCapacityAndLifetime),
        ("cache_reuse_cancel_metadata", ExpertReuseSelfChecks.cancellationAndMetadata),
        ("cache_victim_index", ExpertReuseSelfChecks.victimIndex),
        ("slot_core_size_classes", slotCoreSizeClasses),
        ("original_span_fallback_mutation", originalSpanFallbackAndMutation),
        ("pack_copies_spread_and_fall_back", packCopiesSpreadAndFallBack),
        ("hot_copy_set_reads_several_stores", hotCopySetReadsSeveralStores),
        ("pack_store_reads_mirror_beside_new_stores", packStoreReadsMirrorBesideNewStores),
        ("hot_copy_keepers_build_two_stores", hotCopyKeepersBuildTwoStores),
        ("hot_copy_places_split_by_speed", hotCopyPlacesSplitBySpeed),
        ("qwen_hot_copies_on_two_drives", qwenHotCopiesOnTwoDrives),
        ("generated_eos_excluded_from_output_and_presence", generatedEOSExcludesOutputAndPresence),
        ("generated_eos_preserves_sampler_draw", generatedEOSPreservesSamplerDraw),
        ("generated_length_bound", generatedLengthBound),
        ("sampler_pinned_golden_256_draws", samplerGoldenMatchesPinnedBaseline),
        ("sampler_ties_and_frontier_replay", samplerThresholdTiesAndCheckpointReplay),
        ("sampler_greedy_matches_full_path", samplerGreedyMatchesFullPath),
        ("speculative_correction_probability_conservation", speculativeCorrectionConservesTargetDistribution),
        ("state_append_restore_transaction", layerStateAppendAndRestoreAreTransactional),
        ("pinned_bf16_unary_promotion", pinnedUnaryNativePromotionMatchesExplicit),
        ("pinned_unary_tables_domain_lifetime", pinnedUnaryTablesPreserveDomainAndOwnership),
        ("resident_weight_integration", referenceWeightsResidencyPreservesFallbackAndRejectsMutation),
        ("small_tensor_resident_bits_lifetime", smallTensorResidencyPreservesBitsChargesAndMutation),
        ("small_tensor_resident_config", smallTensorResidencyConfigurationIsBoundedAndBoolean),
        ("resident_bytes_lru_group_admission", residentTensorsPreserveBytesAndBoundLRUGroupAdmission),
        ("resident_escaped_buffer_accounting", residentTensorRawBufferEscapeRetainsPhysicalBudget),
        ("resident_cache_wiring", residentCacheWiringFollowsEntriesAndKeepsBytes),
        ("resident_eligibility_geometry", residentTensorEligibilityAndGeometryAreExplicit),
        ("resident_mutation_rejection", residentTensorSourceMutationNeverPublishesOrHits),
        ("dense_resident_bf16_float_bits", denseResidentPreservesAllBF16BitsAndFloatPayloads),
        ("dense_resident_caps_resource_lifetime", denseResidentCapsAndEscapedBuffersRemainAccounted),
        ("dense_resident_geometry_identity_ledger", denseResidentGeometryIdentityAndLedgerFailuresAreFatal),
        ("dense_resident_blas_layer_fallback", denseReferenceWeightsPreservesBLASAcrossLayersAndFallback),
        ("request_keepalive_join_release", requestGPUKeepAliveJoinsAndReleases),
        ("request_keepalive_short_register_policy", requestGPUKeepAliveShortPolicy),
        ("request_keepalive_admission_default", requestGPUKeepAliveAdmissionAndDefault),
        ("pinned_dense_source_dispatch", pinnedDenseDispatchUsesSourceSchedules),
        ("pinned_dense_scalar_dot_original_binary", pinnedDenseScalarDotMatchesOriginalBinary),
        ("pinned_dense_inputs_ledger_bounds", pinnedDenseInputsAndLedgerStayBounded),
        ("checkpoint_frontier_segmentation", checkpointFrontierAndSegmentationChecks),
        ("checkpoint_capture_restore", checkpointCaptureAndRestoreChecks),
        ("batched_indexer_causal_exception_domain", batchedIndexerSelectionPreservesCausalExceptionalDomain),
        ("batched_selector_reference_sort", batchedTopIndicesMatchesScalarAndFullSort),
        ("batched_selector_exception_ties", batchedTopIndicesPreservesExceptionalAndGlobalTieOrder),
        ("batched_selector_full_domain", batchedTopIndicesCoversMaximumBatchAndDomain),
        ("batched_selector_admission", batchedTopIndicesRejectsGeometryAndRetiresDeniedAllocations),
        ("pinned_reductions_source_order", pinnedReductionsPreserveShapesAndSourceOrder),
        ("pinned_reductions_admission", pinnedReductionsRejectInvalidOrUnadmittedWork),
        ("prompt_cache_save_queue", PrefixReuseSelfChecks.saveQueueOrderAndBarrier),
        ("prompt_cache_save_wait_bounded", PrefixReuseSelfChecks.saveDrainBounded),
        ("prefill_read_scope_arithmetic_boundaries", prefillReadScopesPreserveArithmeticTilesAndTails),
        ("pinned_rms_resident_buffer_geometry", pinnedRMSGeometryAndResidentQuantizedBuffers),
        ("assembled_layer_continuation", assembledLayersPreserveContinuation),
        ("ple_wrapping_eos_hash", pleWrappingAndEOSHash),
        ("ple_known_fifo_rows_joined_failure_cancellation_admission", checkKnownPLERows),
        ("tokenizer_tool_message_semantics", tokenizerNormalizationPreservesToolSemantics),
        ("expert_loader_coalescing_pin_handoff", expertLoaderCoalescesAndRetainsConsumerPins),
        ("expert_loader_cancellation_join", expertLoaderCancellationWaitsForProducerAndRejectsLateData),
        ("expert_loader_drive_fallback", expertLoaderFallbackRunsOnOriginalQueueAndNeverSubstitutes),
        ("qwen_demand_read_stall_bounded", qwenDemandReadStallBounded),
        ("qwen_read_teardown_stall_bounded", qwenReadTeardownStallBounded),
        ("expert_source_calibration_defaults", expertSourceCalibrationIsBoundedAndDefaultsAreOff),
        ("drive_speed_measurement", driveSpeedMeasurement),
        ("drive_speed_saved_reused_stale", driveSpeedSavedReusedStale),
        ("drive_speed_neutral_default", driveSpeedNeutralDefaultAndMeasuredChoice),
        ("drive_slowdown_shifts_reads", driveSlowdownShiftsReads),
        ("split_read_planner", splitReadPlanner),
        ("split_read_execution", splitReadExecution),
        ("split_read_family_fetch", splitReadFamilyFetch),
        ("split_read_qwen_loader", splitReadQwenLoader),
        ("split_read_trains_estimates", splitReadTrainsEstimates),
        ("split_read_family_hedge_wins", splitReadFamilyHedgeWins),
        ("split_read_pack_store", splitReadPackStore),
        ("split_read_swapped_key", splitReadSwappedKey),
        ("split_read_qwen_places", splitReadQwenPlaces),
        ("split_read_family_places", splitReadFamilyPlaces),
        ("split_read_queued_part_cancel", splitReadQueuedPartCancel),
        ("split_read_first_part_in_place", splitReadFirstPartInPlace),
        ("expert_source_runtime_legacy_config", expertSourceRuntimeConfigurationAndLegacyDecoding),
        ("expert_source_queued_bytes_ties", expertSourceQueuedBytesAndStableTiesChooseOneSource),
        ("expert_source_atomic_admission", expertSourceConcurrentSelectionAndAdmissionAreAtomic),
        ("expert_source_complete_byte_parity", expertSourceBothPlacementsPublishIdenticalCompleteBytes),
        ("expert_source_adaptive_cancel_join", expertSourceAdaptiveCancellationJoinsActualWriter),
        ("expert_source_fallback_absence", expertSourceAdaptiveFallbackAndDriveAbsenceStayExact),
        ("expert_source_phase_policy", expertSourcePoliciesAreAppliedAtPhaseBoundaries),
        ("expert_source_online_bounds_convergence", expertSourceOnlineEstimatorConvergesWithinBounds),
        ("expert_source_online_hedge_selection_hand_values", expertSourceOnlineHedgeAndSelectionMatchHandValues),
        ("expert_source_online_runtime_config_loader_wiring", expertSourceOnlineRuntimeConfigurationAndLoaderWiring),
        ("expert_replica_first_winner_joined_loser", expertReplicaPublishesBeforeBlockedLoserAndJoinsRetirement),
        ("expert_replica_cancel_joins_writers", expertReplicaCancellationRetainsBothWriters),
        ("expert_replica_dual_failure_arena_bound", expertReplicaDualFailuresAndArenaBoundAreExplicit),
        ("expert_replica_direct_and_escaped_metal", expertReplicaDirectCacheAndEscapedMetalOwnership),
        ("expert_replica_failed_prediction_retires_and_rereads", expertReplicaFailedPredictionRetiresAndDemandRereads),
        ("expert_reuse_demand_rescans_drained_victims", expertReuseDemandRescansDrainedVictims),
        ("expert_reuse_demand_skips_gpu_held_victims", expertReuseDemandSkipsGPUHeldVictims),
        ("expert_replica_win_cancels_queued_primary", expertReplicaWinCancelsQueuedPrimary),
        ("http_bounds_fragmentation", servingHTTPBoundsAndFragmentation),
        ("http_request_lifecycle", httpRequestLifecycleChecks),
        ("http_lifecycle_control_capacity", servingLifecycleAndControlCapacity),
        ("generation_attempt_metrics", generationAttemptMetrics),
        ("family_host_phase_timing", familyHostPhaseTimingChecks),
        ("content_receipt_round_trip", contentReceiptRoundTrip),
        ("content_receipt_changed_incarnations", contentReceiptRejectsChangedIncarnations),
        ("content_receipt_racing_mutation", contentReceiptRejectsRacingMutation),
        ("content_receipt_interrupted_publication", contentReceiptInterruptedPublication),
        ("content_receipt_malformed_input", contentReceiptRejectsMalformedInput),
        ("family_state_chunk_cancellation", familyStateChunkCancellation),
        ("family_state_replacement_preserves_prior", familyStateReplacementPreservesPrior),
        ("family_state_rejects_malformed_sizes", familyStateRejectsMalformedSizes),
        ("family_state_orphan_recovery", familyStateOrphanRecovery),
        ("family_route_trace_bounds_and_ownership", familyRouteTraceBoundsAndOwnership),
        ("http_body_reservation_unwind", servingBodyReservationUntilUnwind),
        ("http_hard_admission_cap", servingHardAdmissionCap),
        ("family_serving_thinking_budget", familyServingThinkingBudgets),
        ("model_incarnation_identity", modelIncarnationsSeparateSameShapeWeights),
        ("earlier_version_identities", earlierVersionIdentities),
        ("family_hot_fallback_source", familyHotFallbackCountsActualSource),
        ("family_masked_index_preserves_scores_and_selection", familyMaskedIndexPreservesScoresAndSelection),
        ("family_lazy_startup_workspace", familyLazyStartupWorkspace),
        ("deepseek_decode_bench_evidence", deepSeekDecodeBenchRejectsInvalidEvidence),
        ("request_validation_tool_history", servingRequestValidationAndToolHistory),
        ("reasoning_tool_stream_fragments", servingReasoningAndNativeToolFragments),
        ("tool_call_grammars_parse", toolCallGrammarsParse),
        ("family_tool_template_round_trip", familyToolTemplateRoundTrip),
        ("template_continuation_round_trip", templateContinuationRoundTrip),
        ("stream_output_budget_usage", servingOutputBudgetAndWireUsage),
        ("server_request_admission_lifetime", servingAdmissionDelayedRetirement),
        ("server_health_ledger", servingHealthAndLedgerLifetime),
        ("loopback_sse_busy_cancel_recovery", servingLoopbackSSEBusyCancellationAndRecovery),
        ("json_nesting_bound_worker_stack", servingJSONNestingBoundOnWorkerStack),
        ("client_errors_nulls_default_budget", servingClientErrorsNullFieldsAndDefaultBudget),
        ("fast_model_id_degraded_status", servingFastModelIDAndDegradedStatus),
        ("keepalive_queue_order_budget_names", servingKeepAliveQueueOrderAndBudgetNames),
        ("reasoning_aliases_server_default", servingReasoningAliasesAndServerDefault),
        ("serving_sampling_defaults", servingSamplingDefaults),
        ("reclamation_wait_bounded_cancellable", reclamationWaitIsBoundedAndCancellable),
        ("thinking_budget_answer_room", servingThinkingBudgetAndAnswerRoom),
        ("status_report_ring_buffer", servingStatusReport),
        ("system_prompt_snapshot_planning", servingSystemPromptSnapshotPlanning),
        ("typed_chat_markers_stay_text", servingTypedChatMarkersStayText),
        ("chat_commands", servingChatCommands),
        ("status_messages_progress", servingStatusMessagesAndProgress),
        ("dashboard_page_route", servingDashboardPage),
        ("plan_mac_sizes", planFitsMacSizes),
        ("plan_small_macs", planSmallMacs),
        ("plan_256k_file_backed", planFits256K),
        ("plan_context_any_length", planContextAnyLength),
        ("plan_expert_memory_and_ceiling", planExpertMemoryAndCeiling),
        ("plan_explicit_memory_target", planExplicitMemoryTarget),
        ("plan_prompt_cache_follows_rows", planPromptCacheFollowsRows),
        ("gathered_expert_matches_fused", gatheredExpertMatchesFusedExperts),
        ("gathered_deferred_group_parity", gatheredDeferredGroupParity),
        ("group32_scheme_record_layout", group32SchemeAndRecordLayout),
        ("zero_centred_norms_read_as_one_plus_w", zeroCentredNormsReadAsOnePlusW),
        ("group32_float_matrices_dequantized", group32QuantizedFloatMatrices),
        ("group32_quantized_tiled_split_fallbacks", { try withGroup32(quantizedTiledMatchesSplitKAndLeavesOtherSchedulesAlone) }),
        ("group32_quantized_bm32_boundary_parity", { try withGroup32(quantizedBM32BoundaryParity) }),
        ("group32_grouped_tiled_tail_batch_parity", { try withGroup32(groupedExpertTiledPreservesEveryBoundaryAcrossTailAndBatchShapes) }),
        ("group32_grouped_bm32_boundaries", { try withGroup32(groupedBM32BoundariesAndRetirement) }),
        ("group32_grouped_mma_bf16_boundaries", { try withGroup32(groupedMMABoundariesAndCounters) }),
        ("group32_fused_expert_projection_schedules", { try withGroup32(fusedExpertMatchesCachedAndGroupedProjectionSchedules) }),
        ("group32_gathered_expert_matches_fused", { try withGroup32(gatheredExpertMatchesFusedExperts) }),
        ("group32_hc_decode_fusion_bit_identical", { try withGroup32(HCDecodeFusionSelfChecks.exactness) }),
        ("nvfp4_expert_kernels_match_reference", nvfp4ExpertKernelsMatchReference),
        ("demand_waits_for_busy_slots", demandWaitsForBusySlots),
        ("slot_wait_counts_only_idle_time", slotWaitCountsOnlyIdleTime),
        ("tensor_ops_prefill_matches_mma", tensorOpsPrefillMatchesMMA),
        ("speculative_verify_and_controller", speculativeVerifyAndController),
        ("two_row_projection_matches_one_row", twoRowProjectionMatchesOneRow),
        ("vision_encoder_reference", visionEncoderMatchesReference),
        ("mrope_positions_tables", mropePositionsAndTables),
        ("prompt_image_preprocessing", promptImagePreprocessing),
        ("image_parts_markers", imagePartsBecomeMarkers),
        ("prompt_image_link", promptImageLink),
        ("prompt_image_identity", promptImageIdentity),
        ("converter_quantizer_matches_mlx", converterQuantizerMatchesMLX),
        ("converter_layout", converterLayout),
        ("converter_fp8_experts", converterFP8Experts),
        ("converter_mlx_affine", converterMLXAffine),
        ("gguf_dequantize", ggufDequantize),
        ("model_family_geometry", modelFamilyGeometry),
        ("family_template_thinking", familyTemplateThinking),
        ("family_serving_dial", familyServingDial),
        ("deepseek_prompt_tail", deepSeekPromptTail),
        ("family_expert_pools", familyExpertPools),
        ("family_victim_choice", familyVictimChoice),
        ("family_adopted_guess_cancelled", familyAdoptedGuessCancelled),
        ("family_stale_ticket_rejected", familyStaleTicketRejected),
        ("family_failed_read_never_resident", familyFailedReadNeverResident),
        ("family_pin_lifetime", familyPinLifetime),
        ("family_hedged_read_fires_on_stall", familyHedgedReadFiresOnStall),
        ("family_hedged_read_primary_wins", familyHedgedReadPrimaryWins),
        ("family_hedged_read_both_fail", familyHedgedReadBothFail),
        ("family_hedged_read_exclusions", familyHedgedReadExclusions),
        ("family_hedged_read_arena_exhausted", familyHedgedReadArenaExhausted),
        ("family_hedged_read_prompt_wave", familyHedgedReadPromptWave),
        ("family_hedged_read_estimates", familyHedgedReadEstimates),
        ("family_hedged_read_slow_period", familyHedgedReadSlowPeriod),
        ("family_expert_prefetch", familyExpertPrefetch),
        ("family_spin_wait", familySpinWait),
        ("family_expert_wave", familyExpertWave),
        ("family_wave_admission", familyWaveAdmission),
        ("family_prompt_fallback_distinct_owners", familyPromptFallbackDistinctOwners),
        ("family_prefetch_allocated_fallback", familyPrefetchAllocatedFallback),
        ("family_prefetch_self_cancelled_landing", familyPrefetchSelfCancelledLanding),
        ("family_prefetch_queued_cancel_rereads", familyPrefetchQueuedCancelRereads),
        ("family_expert_shapes", familyExpertShapes),
        ("family_decode_step_finishes_on_cancel", familyDecodeStepFinishesOnCancel),
        ("family_prompt_cancel_keeps_conversation", familyPromptCancelKeepsConversation),
        ("family_request_keepalive_releases", familyRequestKeepAliveReleases),
        ("family_decode_overlap", familyDecodeOverlap),
        ("family_resident_first", familyResidentFirst),
        ("family_read_completion_overlap", familyReadCompletionsOverlap),
        ("family_demand_join_publication", familyDemandJoinPublication),
        ("demand_read_lane_progress", demandReadLaneProgress),
        ("demand_read_lane_mixed_sources", demandReadLaneMixedSources),
        ("family_arrival_errors_join_recover", familyArrivalErrorsJoinAndRecover),
        ("family_early_delayed_read_ownership", familyEarlyDelayedReadOwnership),
        ("family_bounded_pooling", familyBoundedPooling),
        ("family_pool_checkpoints", familyPoolCheckpoints),
        ("deepseek_block_logits", deepSeekBlockLogits),
        ("deepseek_generation_block_cache_phase", deepSeekGenerationBlockCachePhase),
        ("family_working_set_phases", familyWorkingSetPhases),
        ("family_scratch_admission", familyScratchAdmission),
        ("family_scratch_pressure_recovery", familyScratchPressureRecovery),
        ("family_scratch_pressure_physical_denial", familyScratchPressurePhysicalDenial),
        ("family_scratch_retirement_recheck", familyScratchRetirementRecheck),
        ("family_scratch_replacement_reuse", familyScratchReplacementReuse),
        ("family_scratch_plan_bounds", familyScratchPlanBounds),
        ("pair_residency_probe_backoff", pairResidencyProbeBacksOffAndRecovers),
        ("pair_union_admission", pairUnionAdmissionRequiresCompleteReserve),
        ("deepseek_engram_bounded_hashes", deepSeekEngramHashesMatchFullHistory),
        ("deepseek_engram_read_bytes_failure", deepSeekEngramReadBytesAndFailure),
        ("family_prompt_read_ahead", familyPromptReadAhead),
        ("family_command_split", familyCommandSplit),
        ("family_hot_copy", familyHotCopy),
        ("family_read_counters_after_grow", familyReadCountersAfterGrow),
        ("family_status_counters_during_request", familyStatusCountersDuringRequest),
        ("family_copies_on_two_drives", familyCopiesOnTwoDrives),
        ("copy_places_follow_speeds", copyPlacesFollowSpeeds),
        ("family_drive_split_outstanding", familyDriveSplitOutstanding),
        ("family_hot_copy_keeper", familyHotCopyKeeper),
        ("family_hot_copy_repair", familyHotCopyRepair),
        ("hot_copy_curve", hotCopyCurve),
        ("usage_counts_decay", usageCountsDecay),
        ("slow_reason_rules", slowReasonRules),
        ("hot_copy_automatic_size", hotCopyAutomaticSize),
        ("family_warm_start", familyWarmStart),
        ("family_resident_pack", familyResidentPackIsExplicitBoundedAndVerified),
        ("family_resident_pack_plans_glm_trunk", familyResidentPackPlansGLMTrunk),
        ("family_context_file", familyContextFile),
        ("family_state_loan", familyStateLoan),
        ("family_state_resize_admission", familyStateResizeAdmission),
        ("family_start_minimum", familyStartMinimum),
        ("family_second_conversation_after_long_one", familySecondConversationAfterLongOne),
        ("family_glm_serving_restore_matches_live", { try familyServingRestoreMatchesLive(.glm) }),
        ("family_deepseek_serving_restore_matches_live", { try familyServingRestoreMatchesLive(.deepseek) }),
        ("family_glm_exact_never_restores_fast_state", { try familyExactNeverRestoresFastState(.glm) }),
        ("family_deepseek_exact_never_restores_fast_state", { try familyExactNeverRestoresFastState(.deepseek) }),
        ("family_startup_admission_boundaries", familyStartupAdmissionBoundaries),
        ("family_expert_slab_admission", familyExpertSlabAdmission),
        ("family_expert_slab_waits_for_reclaim", familyExpertSlabWaitsForReclaim),
        ("family_memory_governor", familyMemoryGovernor),
        ("gguf_quantize_round_trip", ggufQuantizeRoundTrip),
        ("family_converter_real", familyConverterReal),
        ("ggml_matrix_kernels_match_dequantized", ggmlMatrixKernelsMatchDequantized),
        ("ggml_matrix_batch_matches_single", ggmlMatrixBatchMatchesSingle),
        ("family_expert_runs_match_single", familyExpertRunsMatchSingle),
        ("family_kernels_match_reference", familyKernelsMatchReference),
        ("family_index_select_matches_previous", familyIndexSelectMatchesPrevious),
        ("family_index_scores_matches_previous", familyIndexScoresMatchesPrevious),
        ("family_attention_heads8_matches_heads", familyAttentionHeads8MatchesHeads),
        ("family_hc_prepare_matches_previous", familyHCPrepareMatchesPrevious),
        ("family_routing_dial", familyRoutingDial),
        ("family_routing_partial_top_k", familyRoutingPartialTopK),
        ("family_reference_logits", familyReferenceLogits),
        ("converter_gguf", converterGGUF),
        ("gguf_experts_match_reference", ggufExpertsMatchReference),
        ("gguf_expert_manifest_bounds", ggufExpertManifestBounds),
        ("gguf_experts_variant_pairing", ggufExpertsVariantPairing),
        ("gguf_queued_decode_matches_joined", ggufQueuedDecodeMatchesJoined),
        ("gguf_pruned_router_never_selects_padding", GGUFExpertSelfChecks.prunedRouting),
        ("routing_profile_ranking", routingProfileRanking),
        ("routing_profile_miss_ranking", routingProfileMissRanking),
        ("serving_image_parts_structure_bound", servingImagePartsAndStructureBound),
        ("serving_wire_translations", servingWireTranslations),
        ("serving_wire_endpoints", servingWireEndpoints),
        ("serving_request_records", servingRequestRecords),
        ("serving_keep_sessions", servingKeepSessions),
        ("scheduler_demand_reserves", schedulerDemandHasReservedExecutionAndAdmission),
        ("scheduler_cancellation_join_generations", schedulerCancellationJoinsWritersAndInvalidatesGenerations),
        ("scheduler_drive_failure_accounting", schedulerDriveIsolationFailureAndMetadataAccounting),
        ("scheduler_promotion_byte_limits", schedulerPromotionAndByteLimits),
        ("fused_residual_stripes_bf16_domains", fusedResidualInjectionPreservesStripedRowsAndBF16Domains),
        ("fused_residual_cancel_admission", fusedResidualInjectionDenialsAndCancellationJoin),
        ("decode_prerouting_adopted_read_joined_at_phase_change", DecodePreroutingSelfChecks.adoptedReadJoinedAtPhaseChange),
        ("residency_set_keeps_slabs_until_emptied", residencySetKeepsSlabsUntilEmptied),
        ("slab_preparation_keeps_minimum_cache", slabPreparationKeepsMinimumCache),
        ("slab_preparation_answers_status", slabPreparationAnswersStatus),
        ("expert_cache_grows_at_token_boundary", expertCacheGrowsAtTokenBoundary),
        ("read_ahead_window_fits_minimum_cache", readAheadWindowFitsMinimumCache),
        ("fast_refusal_names_user_settings", fastRefusalNamesUserSettings),
        ("malformed_safetensors_headers", malformedSafetensorsHeaders),
        ("malformed_gguf_headers", malformedGGUFHeaders),
        ("malformed_mtp_head", malformedMTPHead),
        ("converter_sources_and_destination", converterSourcesAndDestination),
        ("tokenizer_template_fallback", tokenizerTemplateFallback),
        ("group32_decode_projection_tile8_actual_encoder", { try withGroup32(DecodeProjectionSelfChecks.actualEncoderParity) }),
        ("group32_two_row_projection_matches_one_row", { try withGroup32(twoRowProjectionMatchesOneRow) }),
        ("group32_resident_guards_accept_checkpoint_shapes", group32ResidentGuardsAcceptCheckpointShapes),
        ("conversation_unrelated_prompt_releases_protection", PersistentConversationSelfChecks.unrelatedPromptReleasesProtection),
        ("prompt_cache_restore_keeps_entry_on_quota", PrefixReuseSelfChecks.restoreKeepsEntryOnQuota),
        ("prompt_cache_exact_request_skips_fast_image", PrefixReuseSelfChecks.exactRequestSkipsFastImage),
        ("prompt_cache_disk_rules", PrefixReuseSelfChecks.promptCacheDiskRules),
        ("prompt_cache_start_rules", PrefixReuseSelfChecks.promptCacheStartRules),
        ("prompt_cache_unreadable_index_qwen_slot", PrefixReuseSelfChecks.unreadableIndexFromQwenSlot),
        ("prompt_cache_unreadable_sibling_index_kept", PrefixReuseSelfChecks.unreadableSiblingIndexKept),
        ("prompt_cache_hit_index_write_time", PrefixReuseSelfChecks.hitIndexWriteTime),
        ("prompt_cache_kept_sessions_advance", PrefixReuseSelfChecks.keptSessionsAdvance),
        ("prompt_cache_clone_base_saves", PrefixReuseSelfChecks.cloneBaseSaves),
        ("prompt_cache_failed_save_keeps_previous", PrefixReuseSelfChecks.failedSaveKeepsPrevious),
        ("prompt_cache_saves_pause_on_low_disk", PrefixReuseSelfChecks.savesPauseOnLowDisk),
        ("disk_budget_hot_copy_gives_back_for_saves", PrefixReuseSelfChecks.diskBudgetGiveBack),
        ("prompt_cache_restore_reasons", PrefixReuseSelfChecks.restoreReasons),
        ("prompt_cache_dashboard_edits", PrefixReuseSelfChecks.promptCacheEdits),
        ("reply_text_matches_whole_decode", replyTextMatchesWholeDecode),
        ("paged_checkpoint_disk_bytes_of_written_pages", PagedCheckpointSelfChecks.diskBytesOfWrittenPages),
        ("file_kv_leaves_no_folder_and_names_errors", QSAFilePageSelfChecks.leavesNoFolderAndNamesErrors),
        ("plan_data_folder_paths", planDataFolderPaths),
        ("sampler_matches_full_sort_reference", samplerMatchesFullSortReference),
        ("sampler_fallback_penalty_and_nan", samplerFallbackKeepsPenaltyAndRefusesNaN),
        ("sampler_frequency_penalty_by_hand", samplerFrequencyPenaltyByHand),
        ("serving_stop_literal_match", stopSequenceMatchesLiterally),
        ("vision_npy_header_versions", visionNpyHeaderVersions),
        ("vision_encoder_tensor_table", visionEncoderRefusesMismatchedTensors),
        ("vision_encoder_small_and_large_images", visionEncoderRunsSmallAndLargeImages),
        ("expert_replica_late_adopter_repins", expertReplicaLateAdopterRepinsPublishedRecord),
        ("prompt_image_formats_size_redirects", promptImageFormatsSizeAndRedirects),
        ("prompt_image_links_refuse_internal_addresses", promptImageLinksRefuseInternalAddresses),
        ("serving_cross_origin_requests_refused", servingCrossOriginRequestsRefused),
        ("serving_api_key_required", servingAPIKeyRequired),
        ("serving_structure_bound_on_every_wire", servingStructureBoundOnEveryWire),
        ("serving_connection_cap_answers_503", servingConnectionCapAnswers503),
        ("serving_forget_waits_for_the_slot", servingForgetWaitsForTheSlot),
        ("serving_cancelled_request_recorded", servingCancelledRequestRecorded),
        ("http_relay_stalled_reader_cancels", httpRelayStalledReaderCancels),
        ("prompt_cache_ignored_ownership_saves_off", promptCacheIgnoredOwnershipSavesOff),
        ("serving_wire_error_shapes_and_responses_echo", servingWireErrorShapesAndResponsesEcho),
        ("serving_tool_call_text_without_declared_tools", servingToolCallTextWithoutDeclaredTools),
        ("serving_request_limits_named", servingRequestLimitsNamed),
        ("serving_failure_codes", servingFailureCodes),
        ("serving_gpu_interrupted_answers_503", servingGPUInterruptedAnswers503),
        ("serving_many_requests_queue", servingManyRequestsQueue),
        ("prompt_cache_unreadable_index_second_server", promptCacheUnreadableIndexAndSecondServer),
        ("prompt_cache_removes_other_model_folders", promptCacheRemovesOtherModelFolders),
        ("store_header_rejects_malformed_tables", storeHeaderRejectsMalformedTables),
        ("store_builder_no_partials_verified_records", storeBuilderLeavesNoPartialsAndCopiesOnlyVerifiedRecords),
        ("pack_copy_revives_in_background", packCopyRevivesInTheBackground),
        ("expert_replica_no_staging_retries_model_drive", expertReplicaWithoutStagingRetriesOnModelDrive),
        ("wired_memory_policy_limits", wiredMemoryPolicyLimits),
        ("wired_memory_policy_validation", wiredMemoryPolicyValidation),
        ("memory_explicit_target_policy", memoryExplicitTargetPolicy),
        ("memory_growth_pressure_recovery", memoryGrowthPressureRecovery),
        ("memory_growth_pressure_owners", memoryGrowthPressureOwners),
        ("memory_live_pressure_reads", memoryLivePressureReads),
        ("physical_admission_lazy_owners", physicalAdmissionCountsLazyOwnersAndAllowances),
        ("physical_admission_concurrent_recovery", physicalAdmissionConcurrentReservationsAndRecovery),
        ("selective_prefill_coverage_policy", selectivePrefillCoveragePolicy),
        ("selective_prefill_routing_cleanup", selectivePrefillRoutingCleanup),
        ("selective_prefill_exact_groups", selectivePrefillExactGroups),
        ("selective_prefill_demand_bypass", selectivePrefillDemandBypass),
        ("family_adaptive_prefetch_policy", familyAdaptivePrefetchPolicy),
        ("family_slab_wiring_lifetime", familySlabWiringLifetime),
        ("family_pressure_prefetch_floor", familyPressurePrefetchAndFloor),
        ("family_actual_workspace_ownership", familyActualWorkspaceOwnership),
        ("family_queued_prefetch_owner_teardown", familyQueuedPrefetchOwnerTeardown),
        ("memory_pressure_recovery_policy", memoryPressureRecoveryPolicy),
        ("memory_swap_recovery_policy", memorySwapRecoveryPolicy),
        ("memory_hard_footprint_policy", memoryHardFootprintPolicy),
        ("expert_prepared_growth_recovers_reduced_start", expertPreparedGrowthRecoversReducedStart),
        ("expert_lazy_growth_recovers_reduced_start", expertLazyGrowthRecoversReducedStart),
        ("memory_automatic_ledger_policy", memoryAutomaticAndLedgerPolicy),
        ("memory_explicit_held_level_policy", memoryExplicitHeldLevelPolicy),
        ("memory_held_level_quiet_recovery", memoryHeldLevelQuietRecovery),
        ("memory_limit_advice", memoryLimitAdvice),
        ("memory_deficit_against_learned_target", memoryDeficitAgainstLearnedTarget),
        ("available_memory_counts_speculative_once", availableMemoryCountsSpeculativePagesOnce),
        ("available_memory_invalid_counters", availableMemoryRejectsInvalidCounters),
        ("self_tuning_decision_rule", selfTuningDecisionRule),
        ("self_tuning_time_plan", selfTuningTimePlan),
        ("self_tuning_staleness_precedence", selfTuningStalenessAndPrecedence),
        ("self_tuning_memory_calibration", selfTuningMemoryCalibration),
        ("self_tuning_memory_staleness", selfTuningMemoryStaleness),
        ("memory_budget_learned_from_history", memoryBudgetLearnedFromHistory),
        ("family_scratch_compaction_ownership", familyScratchCompactionOwnership),
        ("family_status_boundary_snapshots", familyStatusBoundarySnapshots),
        ("family_scratch_compaction_parity", familyScratchCompactionParity),
        ("qwen_scratch_compaction", qwenScratchCompaction),
    ] + qualityDialChecks() + modelSchedulerChecks() + teacherQualityChecks() + upgradePolicyChecks() + delayedExpertChecks()
    + httpIngressPolicyChecks() + familyShrinkChecks() + hotCopyRefreshChecks() + hotCopyShareChecks() + hotCopyInPlaceChecks()
    + httpMetricsShutdownChecks()
    + familyPrefetchObservationChecks() + syntheticModelChecks() + familyExpertVariantChecks() + boundedWaitChecks() + modelDriveChecks()
    + jsonModeChecks()
    + modelRefreshChecks() + familyOverlapChecks()
    + slowDriveReadsChecks() + diskBudgetCapChecks() + downloadRefusalChecks()
    + fuzzReplayChecks() + sessionWireChecks()
// With no arguments every check runs. `--only a,b` runs the named checks for a quick loop; its report says
// "scope": "selected", so it never stands in for the full run.
var arguments = Array(CommandLine.arguments.dropFirst())
// `--no-gpu` (first, before `--only`): the run sees no Metal device, as a GPU-less CI runner does.
if arguments.first == "--no-gpu" {
    origami_hide_metal_device()
    arguments.removeFirst()
}
var selectedTests = tests
if arguments.first == "--encode-request" {
    do { try await encodeBenchmarkRequest(Array(arguments.dropFirst())) } catch {
        FileHandle.standardError.write(Data("\(error)\n".utf8))
        exit(1)
    }
    exit(0)
}
if arguments.first == "--bench-deepseek-decode" {
    do { try deepSeekDecodeBench(Array(arguments.dropFirst())) } catch {
        FileHandle.standardError.write(Data("\(error)\n".utf8))
        exit(1)
    }
    exit(0)
}
if arguments.first == "--bench-qwen" {
    do { try QwenBench.run(Array(arguments.dropFirst())) } catch {
        FileHandle.standardError.write(Data("\(error)\n".utf8))
        exit(1)
    }
    exit(0)
}
if arguments.first == "--bench-expert-types" {
    do { try expertTypeBench(Array(arguments.dropFirst())) } catch {
        FileHandle.standardError.write(Data("\(error)\n".utf8))
        exit(1)
    }
    exit(0)
}
if arguments.first == "--bench-gpu-gap" {
    do { try gpuGapBench(Array(arguments.dropFirst())) } catch {
        FileHandle.standardError.write(Data("\(error)\n".utf8))
        exit(1)
    }
    exit(0)
}
if arguments.first == "--bench-family-prefill" {
    do { try familyPrefillBench(Array(arguments.dropFirst())) } catch {
        FileHandle.standardError.write(Data("\(error)\n".utf8))
        exit(1)
    }
    exit(0)
}
if arguments.first == "--bench-resident-precision" {
    do { try residentPrecisionBenchmark(Array(arguments.dropFirst())) } catch {
        FileHandle.standardError.write(Data("\(error)\n".utf8))
        exit(1)
    }
    exit(0)
}
if arguments.first == "--build-resident-precision" {
    do { try buildResidentPrecisionPack(Array(arguments.dropFirst())) } catch {
        FileHandle.standardError.write(Data("\(error)\n".utf8))
        exit(1)
    }
    exit(0)
}
if arguments.first == "--fuzz" { fuzzMain(Array(arguments.dropFirst())) }
if arguments == ["--list"] {
    print(tests.map(\.0).joined(separator: "\n"))
    exit(0)
}
if arguments.first == "--bench-family-masked-index" {
    do { try familyMaskedIndexBenchmark() } catch {
        FileHandle.standardError.write(Data("\(error)\n".utf8))
        exit(1)
    }
    exit(0)
}
if !arguments.isEmpty {
    guard arguments.count == 2, arguments[0] == "--only" else {
        FileHandle.standardError.write(
            Data(
                "Usage: engine-checks [--no-gpu] [--only name,name,...] | --list | --fuzz <target> ... | --bench-deepseek-decode <shard1.gguf> [contexts] [read µs and mode] [options] [read file]\n"
                    .utf8))
        exit(64)
    }
    let names = arguments[1].split(separator: ",", omittingEmptySubsequences: false).map(String.init)
    let available = Set(tests.map { $0.0 })
    guard !names.isEmpty, Set(names).count == names.count, names.allSatisfy({ available.contains($0) }) else {
        FileHandle.standardError.write(Data("Unknown, empty or duplicate standalone check name\n".utf8))
        exit(64)
    }
    selectedTests = tests.filter { names.contains($0.0) }
}
// On a host whose GPU cannot run Origami's kernels (no Metal device, or not Apple7 with unified memory, as on CI
// virtual machines) a check that fails after asking for a Metal device is reported as skipped with the reason and its
// error. A check that never asked for one fails as on any other host.
let gpuProblem: String? = {
    guard let device = MTLCreateSystemDefaultDevice() else { return "no Metal device" }
    do {
        _ = try ModelContract.validate(device: device)
        return nil
    } catch { return "\(error)" }
}()
var results: [[String: Any]] = []
for (name, body) in selectedTests {
    let before = failures.snapshot.count
    let deviceRequests = origami_metal_device_requests()
    let start = DispatchTime.now().uptimeNanoseconds
    var skipped: String?
    var rerun = false
    while true {
        var thrown: Error?
        do { try body() } catch let error as FixtureUnavailable { skipped = error.description } catch {
            thrown = error
            failures.add("\(name): \(error)")
        }
        // macOS can abort a command buffer when the display needs the GPU: run such a check once more.
        guard gpuInterrupted(thrown: thrown, messages: failures.snapshot[before...]), !rerun else { break }
        rerun = true
        FileHandle.standardError.write(Data("\(name): run again after a GPU interruption: \(failures.take(after: before))\n".utf8))
    }
    // Process-wide GPU-wait state (`GPUWait`) is a check's to restore: a lost device or a fake left armed would fail
    // or slow every later check.
    if GPUWait.lost || GPUWait.injectedLeft > 0 || GPUWait.onLost != nil {
        failures.add("\(name): left GPUWait lost, armed or hooked")
        GPUWait.inject([])
        GPUWait.recover()
        GPUWait.onLost = nil
    }
    let usedGPU = origami_metal_device_requests() > deviceRequests
    if skipped == nil, skipsForGPU(gpuProblem: gpuProblem, failed: failures.snapshot.count > before, usedGPU: usedGPU) {
        skipped = "needs an Apple-silicon GPU (\(gpuProblem!)); failed with: " + failures.take(after: before).joined(separator: "; ")
    }
    results.append([
        "name": name, "passed": before == failures.snapshot.count, "skipped": skipped ?? false, "rerunAfterGPUInterruption": rerun,
        "milliseconds": Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000,
    ])
}
let report: [String: Any] = [
    "schema": "origami-engine-checks-v1", "testCount": selectedTests.count,
    "availableTestCount": tests.count, "scope": arguments.isEmpty ? "all" : "selected",
    "failureCount": failures.snapshot.count, "skippedCount": results.filter { $0["skipped"] is String }.count,
    "failures": failures.snapshot, "tests": results, "gpuProblem": gpuProblem ?? NSNull(),
    "compactPrefillReuseShapes": CompactPrefillReuseSelfChecks.reports,
    "qsaExactKernelShapes": QSAExactKernelSelfChecks.reports,
    "conversationModelFreeShapes": PersistentConversationSelfChecks.reports,
    "fileKVStagingShapes": QSAFileStagingSelfChecks.reports,
    "measurements": CheckFixtures.measurements,
    "framework": "standalone Swift (Testing and XCTest unavailable in installed CLT)",
]
FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]))
print()
exit(failures.snapshot.isEmpty ? 0 : 1)
