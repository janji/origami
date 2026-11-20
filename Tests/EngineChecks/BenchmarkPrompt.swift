import Foundation
import OrigamiEngine

/// Exports the serving path's actual prompt for same-checkpoint external comparisons. Loads tokenizer metadata
/// only; it does not allocate model weights, invoke the GPU, or install benchmark instrumentation in production.
func encodeBenchmarkRequest(_ arguments: [String]) async throws {
    guard arguments.count == 2 || arguments.count == 3 else {
        throw EngineError("Usage: engine-checks --encode-request <model-folder-or-first.gguf> <request.json> [teacher-text-file]")
    }
    let model = URL(fileURLWithPath: arguments[0])
    let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("origami-bench-tokenizer-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: temporary) }
    let folder = model.pathExtension.lowercased() == "gguf" ? try FamilySupport.prepare(model, into: temporary) : model
    let tokenizer = try await TokenizerStack(modelDirectory: folder)
    let request = try ServingRequest.parse(Data(contentsOf: URL(fileURLWithPath: arguments[1])), contextTier: 8192)
    let tokens = try ServingPrompt.encode(request, tokenizer: tokenizer)
    var object: [String: Any] = [
        "prompt_tokens": tokens.count, "tokens": tokens, "text": try tokenizer.decode(tokens),
        "tokenizer_identity": tokenizer.identity, "reasoning_effort": request.reasoningEffort,
    ]
    if arguments.count == 3 {
        let teacher = try String(contentsOfFile: arguments[2], encoding: .utf8)
        object["teacher_tokens"] = tokenizer.encode(teacher)
    }
    let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .prettyPrinted])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([10]))
}
