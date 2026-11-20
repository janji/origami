import Foundation

/// The sampling a request gets for the fields it omits, per mode: thinking, not thinking, and with tools. The model's
/// recommendation (`recommended`), or a custom set on top of it (`sampling custom`).
public struct SamplingDefaults: Sendable, Equatable {
    public var thinking, plain, tools: SamplingParameters
    /// Where the values come from: "generation_config.json", "GGUF metadata", "Origami's defaults" or "custom".
    public var source: String

    /// The keys a model's files or the settings may set, with their request ranges.
    public static let keys = ["temperature", "top_p", "top_k", "min_p", "presence_penalty"]
    public static func valid(_ key: String, _ value: Double) -> Bool {
        switch key {
        case "temperature": return (0...100).contains(value)
        case "top_p": return value > 0 && value <= 1
        case "top_k": return value >= 0 && value <= 248_320 && value == value.rounded()
        case "min_p": return (0...1).contains(value)
        default: return (-2...2).contains(value)
        }
    }

    /// Qwen's model-card values, which every model got before 2.3.3 and still gets without its own.
    public static let fallback: Self = {
        func set(_ temperature: Double, _ topP: Double, presence: Double) -> SamplingParameters {
            var p = SamplingParameters()
            p.temperature = Float(temperature)
            p.topP = Float(topP)
            p.topK = 20
            p.minP = 0
            p.presencePenalty = Float(presence)
            return p
        }
        return Self(
            thinking: set(1, 0.95, presence: 0), plain: set(0.7, 0.8, presence: 1.5), tools: set(0.2, 0.9, presence: 0),
            source: "Origami's defaults")
    }()

    public func mode(thinking: Bool, tools: Bool) -> SamplingParameters { tools ? self.tools : thinking ? self.thinking : plain }

    /// These values with `values` (keys of `keys`) set in `modes`.
    public func setting(_ values: [String: Double], modes: [WritableKeyPath<Self, SamplingParameters>]? = nil, source: String) -> Self {
        var out = self
        for mode in modes ?? [\.thinking, \.plain, \.tools] {
            for (key, value) in values where Self.valid(key, value) {
                switch key {
                case "temperature": out[keyPath: mode].temperature = Float(value)
                case "top_p": out[keyPath: mode].topP = Float(value)
                case "top_k": out[keyPath: mode].topK = Int(value)
                case "min_p": out[keyPath: mode].minP = Float(value)
                case "presence_penalty": out[keyPath: mode].presencePenalty = Float(value)
                default: break
                }
            }
        }
        out.source = source
        return out
    }

    /// The model's recommendation: the values its files name over `fallback`. One set for every mode, except where the
    /// recommendation splits modes: Qwen's model card gives thinking and non-thinking their own values, and its
    /// generation_config.json holds the thinking ones, so there a single set from the files sets the thinking mode alone.
    public static func recommended(_ values: [String: Double], source: String, splitModes: Bool) -> Self {
        let named = values.filter { keys.contains($0.key) && valid($0.key, $0.value) }
        guard !named.isEmpty else { return fallback }
        return fallback.setting(named, modes: splitModes ? [\.thinking] : nil, source: source)
    }

    /// generation_config.json's values (transformers' names; a top_k of 0 or less turns it off).
    /// shortcut: repetition_penalty is not read, as the sampler has none; add it with a repetition penalty in `ReferenceSampler`.
    public static func values(generationConfig object: [String: Any]) -> [String: Double] {
        var out: [String: Double] = [:]
        for key in keys {
            guard let number = object[key] as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else { continue }
            out[key] = key == "top_k" ? max(0, number.doubleValue) : number.doubleValue
        }
        return out
    }

    /// llama.cpp's `general.sampling.*` keys (temp, top_k, top_p, min_p, penalty_present), as `keys` names them.
    package static func values(gguf metadata: [String: GGUFSource.Value]) -> [String: Double] {
        let names = ["temp": "temperature", "top_k": "top_k", "top_p": "top_p", "min_p": "min_p", "penalty_present": "presence_penalty"]
        var out: [String: Double] = [:]
        for (name, key) in names {
            switch metadata["general.sampling." + name] {
            case .float(let v)?: out[key] = v
            case .int(let v)?: out[key] = Double(v)
            case .uint(let v)?: out[key] = Double(v)
            default: continue
            }
            if key == "top_k" { out[key] = max(0, out[key]!) }
        }
        return out
    }

    /// The recommendation of a Qwen checkpoint folder (its generation_config.json, else the GGUF its experts are read from).
    public static func recommended(folder: URL, gguf: String? = nil) -> Self {
        if let data = try? Data(contentsOf: folder.appendingPathComponent("generation_config.json")),
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
            case let values = Self.values(generationConfig: object), !values.isEmpty
        {
            return recommended(values, source: "generation_config.json", splitModes: true)
        }
        if let gguf, let source = try? GGUFSource(URL(fileURLWithPath: gguf)) { return recommended(source.metadata, splitModes: true) }
        return fallback
    }
    /// The recommendation in a GGUF's metadata; `splitModes` for a Qwen GGUF (see `recommended(_:source:splitModes:)`).
    package static func recommended(_ metadata: [String: GGUFSource.Value], splitModes: Bool) -> Self {
        recommended(values(gguf: metadata), source: "GGUF metadata", splitModes: splitModes)
    }

    /// The values per mode, for /v1/status and /v1/models.
    public var wire: [String: Any] {
        // Float to the decimal it was written as (0.7, not 0.699999988 or 0.69999999999999996 in the JSON).
        func number(_ value: Float) -> NSNumber { NSDecimalNumber(string: "\(value)") }
        func set(_ p: SamplingParameters) -> [String: Any] {
            [
                "temperature": number(p.temperature), "top_p": number(p.topP), "top_k": p.topK, "min_p": number(p.minP),
                "presence_penalty": number(p.presencePenalty),
            ]
        }
        return ["thinking": set(thinking), "non_thinking": set(plain), "tools": set(tools), "source": source]
    }
}
