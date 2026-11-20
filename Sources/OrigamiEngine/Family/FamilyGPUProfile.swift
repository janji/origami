import Foundation
import Metal

/// GPU time by kernel class for the prefill benchmark (`engine-checks --bench-family-prefill`): with a profile set on a
/// runner, each run of dispatches of one class gets its own compute encoder, timed by the GPU at its start and end
/// (counter samples at stage boundaries, which every Apple GPU takes). The commands, their order and their command
/// buffers are unchanged, so the output is too; only encoders are added (`unattributed` counts their gaps).
package final class FamilyGPUProfile {
    /// The class of a dispatch from the runner's step ("layer 3 DSA rows 0+2048"), the kernel ("gather_rows", "mm_12")
    /// and, for a matrix, its tensor name ("blk.3.attn_q_a.weight", "gate").
    package let classify: (_ step: String, _ kernel: String, _ matrix: String?) -> String
    package private(set) var gpu: [String: Double] = [:]
    package private(set) var dispatches: [String: Int] = [:]
    package private(set) var host: [String: Double] = [:]
    /// Input rows of each matrix dispatch, by class: rows -> dispatches.
    package private(set) var rows: [String: [Int: Int]] = [:]
    package private(set) var weightBytes: [String: Int] = [:]
    /// GPU idle time (no timed span running) just before each class's spans begin, within one `collect`.
    package private(set) var idleBefore: [String: Double] = [:]
    private let device: MTLDevice
    private let counters: MTLCounterSet
    private var samples: [MTLCounterSampleBuffer] = []
    private var used = 0
    private var spans: [(String, MTLCounterSampleBuffer, Int)] = []
    private var last: (String, UInt64)?
    var open: String?
    private static let capacity = 4_096  // Metal's most samples (32 KB) a buffer

    package init(device: MTLDevice, classify: @escaping (_ step: String, _ kernel: String, _ matrix: String?) -> String) throws {
        guard device.supportsCounterSampling(.atStageBoundary),
            let set = device.counterSets?.first(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue })
        else { throw EngineError("This GPU takes no timestamps at encoder boundaries") }
        self.device = device
        counters = set
        self.classify = classify
    }

    /// Counts a dispatch and returns the encoder it goes into: `current`, or a new timed one on `command` when the class
    /// changes (`current` is ended).
    func dispatch(
        step: String, kernel: String, matrix: String?, rows count: Int?, bytes: Int, current: MTLComputeCommandEncoder,
        command: MTLCommandBuffer
    ) throws -> MTLComputeCommandEncoder {
        let now = DispatchTime.now().uptimeNanoseconds
        if let (name, then) = last { host[name, default: 0] += Double(now - then) / 1e9 }
        let name = classify(step, kernel, matrix)
        last = (name, now)
        dispatches[name, default: 0] += 1
        if let count { rows[name, default: [:]][count, default: 0] += 1 }
        weightBytes[name, default: 0] += bytes
        guard name != open else { return current }
        if samples.isEmpty || used + 2 > Self.capacity {
            let descriptor = MTLCounterSampleBufferDescriptor()
            descriptor.counterSet = counters
            descriptor.storageMode = .shared
            descriptor.sampleCount = Self.capacity
            samples.append(try device.makeCounterSampleBuffer(descriptor: descriptor))
            used = 0
        }
        let pass = MTLComputePassDescriptor()
        pass.sampleBufferAttachments[0].sampleBuffer = samples.last!
        pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = used
        pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = used + 1
        current.endEncoding()
        spans.append((name, samples.last!, used))
        used += 2
        open = name
        guard let encoder = command.makeComputeCommandEncoder(descriptor: pass) else { throw EngineError("No timed encoder") }
        return encoder
    }

    /// Adds the GPU time of every span since the last call (their buffers must have completed). Timestamps are in the
    /// GPU's ticks, converted with the device's CPU/GPU timestamp pair across the run.
    package func collect(nanosecondsPerTick: Double) throws {
        var end: UInt64?
        for (name, buffer, index) in spans {
            guard let data = try buffer.resolveCounterRange(index..<index + 2) else { continue }
            let t = data.withUnsafeBytes { Array($0.bindMemory(to: UInt64.self)) }
            if t.count == 2, t[1] > t[0], t[0] != MTLCounterErrorValue {
                gpu[name, default: 0] += Double(t[1] - t[0]) * nanosecondsPerTick / 1e9
                if let end, t[0] > end { idleBefore[name, default: 0] += Double(t[0] - end) * nanosecondsPerTick / 1e9 }
                end = max(end ?? 0, t[1])
            }
        }
        spans.removeAll()
        samples.removeAll()
        last = nil
    }
}
