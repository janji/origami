import Foundation
import Metal
import OrigamiEngine

/// A benchmark must fail when no requested context was exercised or when a timed read failed.
/// Tiny disposable record fixtures; no model inference or GPU commands.
func deepSeekDecodeBenchRejectsInvalidEvidence() throws {
    func require(_ value: Bool, _ message: String) throws {
        if !value { throw EngineError("Decode benchmark validation: " + message) }
    }
    for contexts in ["", "garbage", "1,bad", "1,", "-1"] {
        var refused = false
        do { try deepSeekDecodeBench(["/unused", contexts]) } catch let error as EngineError {
            refused = String(describing: error).contains("contexts must")
        }
        try require(refused, "invalid contexts accepted: \(contexts)")
    }
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
        throw FixtureUnavailable(description: "no Metal device")
    }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-bench-read-check-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    let file = try expertFixtureGGUF(in: folder)
    let gguf = try GGUFSource(file)
    let geo = try ModelGeometry(gguf: gguf)
    // The reader needs only the fixture's expert shapes. Its synthetic GPU/token path is never run.
    let bench = try DeepSeekDecodeBench(gguf: gguf, geo: geo, device: device, queue: queue, context: 1)
    bench.readMicros = 1
    bench.readMode = "q"
    let source = folder.appendingPathComponent("records.bin")
    try Data().write(to: source)
    bench.readFile = try ExactFile(url: source)
    var refused = false
    do { try bench.read(layer: 1) } catch let error as EngineError {
        refused = String(describing: error).contains("cannot hold one expert record")
    }
    try require(refused, "too-small file was timed or divided by zero")
    try Data(repeating: 42, count: 1 << 20).write(to: source)
    bench.readFile = try ExactFile(url: source)
    let writer = try FileHandle(forWritingTo: source)
    try writer.truncate(atOffset: 0)
    try writer.close()
    refused = false
    do { try bench.read(layer: 1) } catch let error as EngineError { refused = String(describing: error).contains("pread failed") }
    try require(refused, "failed read was silently counted as a completed benchmark")
}
