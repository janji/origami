import Foundation
import Metal
import OrigamiEngine

func familyOverlapChecks() -> [(String, () throws -> Void)] {
    [
        ("family_drive_activity_partition", familyDriveActivityPartition),
        ("family_overlap_counters_add_up", familyOverlapCountersAddUp),
    ]
}

/// `DriveActivity` on a fake clock: every lane's busy and idle seconds add up to the wall time of each host state,
/// nothing counts outside the window, and bytes go to the host state their read ended in.
func familyDriveActivityPartition() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Drive activity: " + what) } }
    final class Clock: @unchecked Sendable { var now: UInt64 = 0 }
    let clock = Clock()
    let activity = DriveActivity(lanes: 2) { clock.now }
    func at(_ ms: UInt64) { clock.now = ms * 1_000_000 }
    activity.began(0)
    at(5)  // outside the window: not counted
    activity.window(true)
    at(7)
    let other = activity.enter(.reads)
    at(10)
    activity.began(1)
    at(12)
    activity.ended(0, bytes: 100)
    at(15)
    activity.restore(other)
    activity.ended(1, bytes: 50)
    at(16)
    _ = activity.enter(.gpu)
    at(20)
    activity.window(false)
    at(30)
    let t = activity.snapshot
    let ms = { (s: Double) in (s * 1e3).rounded() }
    try require(t.wall.map(ms) == [8, 4, 3], "wall \(t.wall)")
    try require(t.busy[0].map(ms) == [5, 0, 2] && t.idle[0].map(ms) == [3, 4, 1], "model drive \(t.busy[0]) \(t.idle[0])")
    try require(t.busy[1].map(ms) == [5, 0, 0] && t.idle[1].map(ms) == [3, 4, 3], "copy place \(t.busy[1]) \(t.idle[1])")
    try require(t.bytes[0] == [100, 0, 0] && t.bytes[1] == [0, 0, 50], "bytes \(t.bytes)")
    for lane in 0..<2 {
        for host in 0..<3 { try require(abs(t.busy[lane][host] + t.idle[lane][host] - t.wall[host]) < 1e-9, "lane \(lane) host \(host)") }
    }
    let notes = DriveActivity.notes(t)
    try require(
        notes["drive_activity.model_drive.idle_seconds.reads"] == t.idle[0][0] && notes["drive_activity.copy_place_0.bytes.other"] == 50,
        "notes \(notes)")
}

/// The resident-first fixture (`familyResidentFirst`) with a hot copy of half the records: four tokens (A, B, A, B) of
/// two MoE layers, three experts of four a token, each layer guessing the next (every guess admitted). Each record read
/// sleeps `readSeconds` first (a slow drive). The drive-activity window is open over the tokens.
private func overlapFixture(readSeconds: Double = 0.002) throws -> (
    out: [Float], experts: FamilyExperts, wall: Double
) {
    guard let device = MTLCreateSystemDefaultDevice() else { throw FixtureUnavailable(description: "no Metal device") }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-family-overlap-counters-\(getpid())")
    defer { try? FileManager.default.removeItem(at: folder) }
    let gguf = try expertFixtureGGUF(in: folder, used: 3)
    let h = 256
    func q8(_ outputs: Int, _ code: (Int, Int) -> UInt8) -> MTLBuffer {
        let bytes = (0..<outputs).flatMap { o in
            (0..<(h / 32)).flatMap { b in [0x00, code(o, b) == 0 ? 0 : 0x3C] + [UInt8](repeating: code(o, b), count: 32) }
        }
        return device.makeBuffer(bytes: bytes, length: bytes.count, options: .storageModeShared)!
    }
    let router = FamilyWeights.Matrix(buffer: q8(4) { $0 == $1 ? 1 : 0 }, offset: 0, type: 8, width: h, outputs: 4, name: "router")
    let third = FamilyWeights.Matrix(buffer: q8(h) { ($0 + $1) % 3 == 0 ? 1 : 0 }, offset: 0, type: 8, width: h, outputs: h, name: "shared")
    let shared = FamilyRunner.FFN(gate: third, up: third, down: third)
    let tokens = [[1, 2, 3, 4], [4, 1, 3, 2], [1, 2, 3, 4], [4, 1, 3, 2]].map { v in
        (0..<h).map { i -> Float in i / 32 < 4 ? Float(v[i / 32]) / 256 : 0 }
    }
    let weights = try FamilyWeights(gguf: GGUFSource(gguf), device: device)
    let runner = try FamilyRunner(weights: weights, context: 64)
    // Small enough that the second pair of tokens reads again.
    runner.experts = try FamilyExperts(weights: weights, budget: UInt64(6 * (212_992 + 114_688)))
    let experts = runner.experts!
    experts.adaptiveGuesses = false
    let copy = try HotCopy.write(source: experts, keys: [4, 6, 9, 11], folder: folder.appendingPathComponent("hot"), identity: "a")
    experts.attach(copy)
    experts.readProbe = { _, _ in Thread.sleep(forTimeInterval: readSeconds) }
    experts.backgroundProbe = { _ in Thread.sleep(forTimeInterval: readSeconds) }
    _ = try experts.prepareWave(layer: 2, maximumExperts: 3)
    var outs: [MTLBuffer] = []
    runner.poolMaximum = runner.experts.budget
    try runner.prepareScratch(runner.moeScratchPlan(rows: 1, sharedWidth: h))
    let started = Date()
    experts.driveActivity.window(true)
    runner.begin()
    for token in tokens {
        let x = device.makeBuffer(bytes: token, length: h * 4, options: .storageModeShared)!
        for layer in [1, 2] {
            outs.append(device.makeBuffer(length: h * 4, options: .storageModeShared)!)
            try runner.moe(
                layer: layer, router: router, bias: [], shared: shared, input: x, output: outs.last!, rows: 1, probability: { $0 },
                normalize: false, scale: 1, sum: { $0 }, limit: 0, next: layer == 1 ? (2, router, [], nil) : (1, router, [], nil))
        }
    }
    runner.end()
    experts.waitForReads()
    experts.driveActivity.window(false)
    let wall = Date().timeIntervalSince(started)
    return (
        outs.flatMap { Array(UnsafeBufferPointer(start: $0.contents().bindMemory(to: Float.self, capacity: h), count: h)) }, experts, wall
    )
}

/// Counters A, C, D and E on the fixture: per drive busy + idle = the window's wall time in
/// each host state, the windows add up to the measured time, the drives' bytes cover the records read; the guesses per
/// drive add up to the totals, every issued guess has its router time; layers with demand have their router time.
func familyOverlapCountersAddUp() throws {
    func require(_ condition: Bool, _ what: String) throws { if !condition { throw EngineError("Family overlap counters: " + what) } }
    do {
        let (_, experts, wall) = try overlapFixture()
        let t = experts.driveActivity.snapshot
        let total = t.wall.reduce(0, +)
        try require(t.busy.count == 2, "\(t.busy.count) lanes, not the model drive and one copy place")
        try require(abs(total - wall) < max(0.005, 0.05 * wall), "window \(total) s, measured \(wall) s")
        for lane in t.busy.indices {
            for host in t.wall.indices {
                try require(
                    abs(t.busy[lane][host] + t.idle[lane][host] - t.wall[host]) < 1e-6, "lane \(lane) host \(host)")
            }
        }
        try require(t.wall[DriveActivity.Host.reads.rawValue] > 0 && t.wall[DriveActivity.Host.other.rawValue] > 0, "host states \(t.wall)")
        let c = experts.counters
        let bytes = t.bytes.joined().reduce(0, +)
        let read = Double(c.copyBytes + c.modelBytes)
        try require(
            bytes >= read && bytes <= read + Double(c.prefetchBytes) && t.bytes[1].reduce(0, +) > 0,
            "drive bytes \(t.bytes), records read \(read)")
        let g = c.guessDrives
        try require(
            g[0].issued + g[1].issued == c.prefetches && g[0].used + g[1].used == c.prefetchUsed
                && g[0].wastedBytes + g[1].wastedBytes == c.prefetchWastedBytes
                && g[0].refusedConfidence + g[1].refusedConfidence == c.guessRejectedConfidence
                && g[0].refusedQueue + g[1].refusedQueue == c.guessRejectedQueue
                && g[0].refusedDeadline + g[1].refusedDeadline == c.guessRejectedDeadline
                && g[0].refusedPressure + g[1].refusedPressure == c.guessRejectedPressure,
            "guesses per drive \(g) against \(c.prefetches) issued, \(c.prefetchUsed) used")
        try require(
            c.prefetches > 0 && g.allSatisfy { $0.issues == $0.issued && $0.completions <= $0.issued && $0.issueSeconds >= 0 },
            "guess times \(g)")
        try require(
            c.routerToDemandLayers > 0 && c.routerToDemandMaximum <= c.routerToDemandSeconds + 1e-12
                && c.routerToDemandMaximum * Double(c.routerToDemandLayers) >= c.routerToDemandSeconds - 1e-12,
            "router to demand \(c.routerToDemandSeconds) s over \(c.routerToDemandLayers), longest \(c.routerToDemandMaximum)")
        try require(experts.splitPlannerNotes["split.copy_rate_bytes_per_second"].map { $0 > 0 } == true, "split state")
    }
}
