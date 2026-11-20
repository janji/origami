import Foundation

/// A request's prompt token ids: normalized messages (images become markers), the `tool_choice` instruction appended
/// to the last user turn, then the chat template. Used by request preparation and by `origami save`.
public enum ServingPrompt {
    public static func encode(_ request: ServingRequest, tokenizer: TokenizerStack) throws -> [Int] {
        var images: [String] = []
        return try encode(request, tokenizer: tokenizer, images: &images)
    }
    /// The prompt holds one `<|image_pad|>` per image; `images` receives their data URLs in the same order.
    public static func encode(_ request: ServingRequest, tokenizer: TokenizerStack, images: inout [String]) throws -> [Int] {
        var messages = try TokenizerStack.normalizeMessages(request.messages, images: &images)
        let instruction: String?
        switch request.toolChoice {
        case .required: instruction = "You must call one of the provided tools."
        case .named(let name): instruction = "You must call the provided tool named \(name)."
        default: instruction = nil
        }
        // At the end of the last user turn: everything before it (the system prompt, the tools, earlier turns) is the same
        // with or without a tool choice, so saved state still matches. A trailing system message is refused by Qwen's template.
        if let instruction {
            if let last = messages.lastIndex(where: { $0["role"] == .string("user") }), case .string(let text) = messages[last]["content"] {
                messages[last]["content"] = .string(text + "\n\n" + instruction)
            } else {
                messages.append(["role": .string("user"), "content": .string(instruction)])
            }
        }
        return try tokenizer.encodeNormalized(messages, tools: request.tools, reasoningEffort: request.reasoningEffort)
    }
}

extension NativeServingBackend: ServingPromptCounting {
    public func promptTokens(_ request: ServingRequest) throws -> Int { try ServingPrompt.encode(request, tokenizer: tokenizer).count }
}
