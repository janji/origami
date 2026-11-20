import Foundation
import Metal
import OrigamiEngine

/// `engine-checks --bench-gpu-gap [gaps ms, comma-separated] [options]`: how a GPU idle gap slows the work after it, with
/// no model. Each cycle waits `gap` ms (another thread spins and signals, as a read's completion wakes decode; `spin`:
/// this thread spins), then commits `probes` command buffers of one probe each, alternating an arithmetic one (`alu`,
/// 800 iterations of 32,768 threads: about 80 µs at full clock, its time ~ 1 / clock) and a memory one (`mem`, streams
/// 64 MB, as decode's matrix kernels stream weights), and waits for the last. Prints per gap the median GPU time of the
/// first and last alu and mem probes after the gap (the command buffers' GPU start and end), over cycles 10..<`cycles`.
/// Options (comma-separated): `spin`; `touch` (the gap's thread rewrites the mem probe's 64 MB before signalling, as
/// `pread` writes a record); `qos-ui` (the committing thread at user-interactive QoS); `keepalive` (the request
/// keep-alive); `pulse=<threads>:<µs>:<period µs>` (a second queue runs an arithmetic kernel of `threads` threads and
/// about `µs` at full clock, one every `period` µs from its own thread; period 0: resubmitted from its completion, two in
/// flight; with `poll` the kernel polls a word of memory instead, as the request keep-alive does); `warm=<µs>` (a wide arithmetic kernel of about `µs` committed on the probes' queue `warm` µs before the gap
/// ends); `fill=<threads>:<µs>` (while waiting, this thread keeps one arithmetic kernel of that size queued on the probes'
/// queue, so the probes wait behind at most one); `probes=<n>`, `cycles=<n>`.
func gpuGapBench(_ arguments: [String]) throws {
    let gaps = (arguments.first ?? "0,1,2,4,8,12,17,30,60").split(separator: ",").compactMap { Double($0) }
    let options = arguments.count > 1 ? arguments[1].split(separator: ",").map(String.init) : []
    try GPUGapRig(options: options).run(gaps: gaps)
}

private final class GPUGapRig: @unchecked Sendable {
    let options: [String]
    let device: MTLDevice, queue: MTLCommandQueue, alu: MTLComputePipelineState, mem: MTLComputePipelineState, poll: MTLComputePipelineState
    let input: MTLBuffer, out: MTLBuffer
    let memBytes = 64 << 20, wide = 32_768, probeIters: UInt32 = 800
    let lock = NSLock()
    var stop = false, pulses = 0

    init(options: [String]) throws {
        self.options = options
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw EngineError("No Metal device")
        }
        let source = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void alu(device float *out [[buffer(0)]], constant uint &iters [[buffer(1)]], uint id [[thread_position_in_grid]]) {
                float a = float(id) * 1e-7f, b = 0.999f;
                for (uint i = 0; i < iters; i++) { a = fma(a, b, 0.001f); b = fma(b, 0.9999f, a * 1e-6f); }
                if (a == 12345.0f) out[id] = b;
            }
            kernel void mem(device const float4 *in [[buffer(0)]], device float *out [[buffer(1)]], constant uint &n [[buffer(2)]],
                            uint id [[thread_position_in_grid]], uint count [[threads_per_grid]]) {
                float4 s = 0;
                for (uint i = id; i < n; i += count) s += in[i];
                if (s.x == 12345.0f) out[id] = s.y;
            }
            kernel void poll(device atomic_uint *out [[buffer(0)]], constant uint &iters [[buffer(1)]]) {
                for (uint i = 0; i < iters; i++) { if (atomic_load_explicit(out, memory_order_relaxed) == 12345u) break; }
            }
            """
        let library = try device.makeLibrary(source: source, options: nil)
        alu = try device.makeComputePipelineState(function: library.makeFunction(name: "alu")!)
        poll = try device.makeComputePipelineState(function: library.makeFunction(name: "poll")!)
        mem = try device.makeComputePipelineState(function: library.makeFunction(name: "mem")!)
        guard let input = device.makeBuffer(length: memBytes, options: .storageModeShared),
            let out = device.makeBuffer(length: 1 << 22, options: .storageModeShared)
        else { throw EngineError("No buffers") }
        memset(input.contents(), 1, memBytes)
        (self.device, self.queue, self.input, self.out) = (device, queue, input, out)
    }

    func value(_ key: String) -> String? { options.first { $0.hasPrefix(key + "=") }.map { String($0.dropFirst(key.count + 1)) } }
    func stopping() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return stop
    }
    func counted() {
        lock.lock()
        pulses += 1
        lock.unlock()
    }
    func encodeALU(_ command: MTLCommandBuffer, threads: Int, iters: UInt32, polling: Bool = false) {
        let encoder = command.makeComputeCommandEncoder()!
        var n = iters
        encoder.setComputePipelineState(polling ? poll : alu)
        encoder.setBuffer(out, offset: 0, index: 0)
        encoder.setBytes(&n, length: 4, index: 1)
        encoder.dispatchThreads(
            MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(threads, 256), height: 1, depth: 1))
        encoder.endEncoding()
    }
    func encodeMem(_ command: MTLCommandBuffer) {
        let encoder = command.makeComputeCommandEncoder()!
        var n = UInt32(memBytes / 16)
        encoder.setComputePipelineState(mem)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(out, offset: 0, index: 1)
        encoder.setBytes(&n, length: 4, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: 65_536, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }
    func gpuTime(_ c: MTLCommandBuffer) -> Double { (c.gpuEndTime - c.gpuStartTime) * 1e3 }
    /// Iterations for about `micros` µs at full clock (a kernel no wider than the probe: 800 take about 80 µs).
    func iterations(_ micros: Double) -> UInt32 { UInt32(max(1, micros * 10)) }
    func submitPulse(_ second: MTLCommandQueue, threads: Int, iters: UInt32) {
        guard !stopping() else { return }
        let c = second.makeCommandBuffer()!
        encodeALU(c, threads: threads, iters: iters, polling: options.contains("poll"))
        c.addCompletedHandler { [self] _ in
            counted()
            submitPulse(second, threads: threads, iters: iters)
        }
        c.commit()
    }

    func run(gaps: [Double]) throws {
        // The owner must outlive the session: dropping it closes the session (its deinit).
        let owner =
            options.contains("keepalive")
            ? try RequestGPUKeepAlive(ledger: MemoryLedger(ceiling: 1 << 30, transientMargin: 1), policy: .original) : nil
        let keepAlive = try owner?.begin()
        if let pulse = value("pulse")?.split(separator: ":").compactMap({ Double($0) }), pulse.count == 3 {
            let threads = Int(pulse[0])
            let iters = iterations(pulse[1])
            let period = pulse[2]
            let second = device.makeCommandQueue()!
            if period == 0 {
                submitPulse(second, threads: threads, iters: iters)
                submitPulse(second, threads: threads, iters: iters)
            } else {
                let thread = Thread { [self] in
                    var next = DispatchTime.now().uptimeNanoseconds
                    while !stopping() {
                        let c = second.makeCommandBuffer()!
                        encodeALU(c, threads: threads, iters: iters, polling: options.contains("poll"))
                        c.commit()
                        counted()
                        next += UInt64(period * 1000)
                        while DispatchTime.now().uptimeNanoseconds < next {}
                    }
                }
                thread.qualityOfService = .userInitiated
                thread.start()
            }
            print("pulse: \(threads) threads, \(iters) iterations (~\(pulse[1]) µs at full clock), period \(period) µs")
        }
        print("probes: \(probeIters) iterations of \(wide) threads (alu), 64 MB (mem); options \(options)")
        print("gap ms\tfirst alu\tlast alu\tfirst mem\tlast mem\tbusy/cycle ms\talu by position")
        if options.contains("qos-ui") {
            let done = DispatchSemaphore(value: 0)
            let thread = Thread { [self] in
                cycles(gaps: gaps)
                done.signal()
            }
            thread.qualityOfService = .userInteractive
            thread.start()
            done.wait()
        } else {
            cycles(gaps: gaps)
        }
        lock.lock()
        stop = true
        lock.unlock()
        withExtendedLifetime(owner) {
            if let snapshot = keepAlive?.close() { print("keep-alive: \(snapshot.submittedCommands) commands in \(snapshot.seconds) s") }
        }
        if pulses > 0 { print("pulses: \(pulses)") }
    }

    func cycles(gaps: [Double]) {
        let probes = value("probes").flatMap { Int($0) } ?? 8
        let cycles = value("cycles").flatMap { Int($0) } ?? 40
        let warm = value("warm").flatMap { Double($0) }
        let fill = value("fill")?.split(separator: ":").compactMap { Double($0) }
        let spin = options.contains("spin")
        let touch = options.contains("touch")
        for gap in gaps {
            var firstALU: [Double] = []
            var lastALU: [Double] = []
            var firstMem: [Double] = []
            var lastMem: [Double] = []
            var busy: [Double] = []
            var byPosition = [[Double]](repeating: [], count: probes)
            for cycle in 0..<cycles {
                let until = DispatchTime.now().uptimeNanoseconds + UInt64(gap * 1e6)
                var warmed = false
                var filler: MTLCommandBuffer?
                func meanwhile() {
                    if let fill, filler.map({ $0.status == .completed }) ?? true {
                        let c = queue.makeCommandBuffer()!
                        encodeALU(c, threads: Int(fill[0]), iters: iterations(fill[1]))
                        c.commit()
                        filler = c
                    }
                    guard let warm, !warmed, DispatchTime.now().uptimeNanoseconds + UInt64(warm * 1000) >= until else { return }
                    let c = queue.makeCommandBuffer()!
                    encodeALU(c, threads: wide, iters: iterations(warm))
                    c.commit()
                    warmed = true
                }
                if gap > 0 {
                    if spin {
                        while DispatchTime.now().uptimeNanoseconds < until { meanwhile() }
                    } else {
                        let done = DispatchSemaphore(value: 0)
                        DispatchQueue.global(qos: .userInitiated).async { [self] in
                            if touch { memset(input.contents(), Int32(cycle & 0xFF), memBytes) }
                            while DispatchTime.now().uptimeNanoseconds < until {}
                            done.signal()
                        }
                        if warm != nil || fill != nil {
                            while done.wait(timeout: .now() + .microseconds(50)) == .timedOut { meanwhile() }
                        } else {
                            done.wait()
                        }
                    }
                }
                var commands: [MTLCommandBuffer] = []
                for p in 0..<probes {
                    let c = queue.makeCommandBuffer()!
                    if p % 2 == 0 { encodeALU(c, threads: wide, iters: probeIters) } else { encodeMem(c) }
                    c.commit()
                    commands.append(c)
                }
                commands.last!.waitUntilCompleted()
                guard cycle >= 10 else { continue }
                let times = commands.map(gpuTime)
                firstALU.append(times[0])
                lastALU.append(times[probes - 2])
                firstMem.append(times[1])
                lastMem.append(times[probes - 1])
                busy.append(times.reduce(0, +))
                for (p, t) in times.enumerated() { byPosition[p].append(t) }
            }
            func med(_ x: [Double]) -> Double { x.sorted()[x.count / 2] }
            let positions = stride(from: 0, to: probes, by: 2).map { String(format: "%.3f", med(byPosition[$0])) }.joined(separator: " ")
            print(
                String(
                    format: "%.0f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%@", gap, med(firstALU), med(lastALU), med(firstMem), med(lastMem),
                    med(busy), positions))
        }
    }
}
