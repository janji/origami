import Darwin
import Foundation
import OrigamiEngine

// Seeded, structure-aware mutation fuzzing of the parsers that take untrusted bytes (ROADMAP S3): model files (GGUF,
// safetensors and their config), the HTTP ingress and request JSON, the chat template's request-borne inputs, and the
// server over a socket with clients that disconnect. No model or GPU is needed.
//
//   engine-checks --fuzz <target> [--seconds 60] [--seed 1] [--hang 2] [--out folder]   mutate until the time is up
//   engine-checks --fuzz <target> --replay file...                                     run files once (exit 0: no finding)
//   engine-checks --fuzz <target> --minimize file                                      shrink a finding (child processes)
//
// A finding is a crash (the input is written to <out>/crash-<target>.bin before the signal is re-raised), a hang longer
// than --hang seconds on one input, more than 256 MB of new memory for one input, an error of a type the caller does
// not handle, or an accepted parse that is inconsistent. The `fuzz_replay_<target>` checks replay the seeds, the
// regression inputs in Tests/Fixtures/fuzz/<target>/ and a few hundred fixed mutations in under 2 s.

struct FuzzFinding: Error, CustomStringConvertible { let description: String }

struct FuzzRNG {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func below(_ n: Int) -> Int { n <= 1 ? 0 : Int(next() % UInt64(n)) }
    mutating func chance(_ percent: Int) -> Bool { below(100) < percent }
    mutating func pick<T>(_ values: [T]) -> T { values[below(values.count)] }
}

/// A target: its seeds, its structure-aware mutation (byte mutations are added on top) and one run.
struct FuzzTarget {
    let name: String
    let seeds: () throws -> [Data]
    let mutate: (inout FuzzRNG, Data, [Data]) -> Data
    /// Rejections are fine; a `FuzzFinding` (or a crash, hang or allocation) is not.
    let run: (Data) throws -> Void
}

func fuzzTargets() -> [String: FuzzTarget] {
    Dictionary(
        uniqueKeysWithValues: [ggufFuzzTarget(), safetensorsFuzzTarget(), httpFuzzTarget(), templateFuzzTarget(), socketFuzzTarget()].map {
            ($0.name, $0)
        })
}

/// Interesting integers for any field: boundaries of every width, small values and powers of two.
let fuzzInteresting: [UInt64] = [
    0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 31, 32, 33, 63, 64, 127, 128, 255, 256, 1023, 1024, 4095, 4096, 32767, 32768, 65535, 65536,
    1 << 24, (1 << 24) + 1, 1 << 30, (1 << 31) - 1, 1 << 31, (1 << 32) - 1, 1 << 32, 1 << 40, 1 << 62, (1 << 63) - 1, 1 << 63,
    UInt64.max, UInt64.max - 1, UInt64.max - 31,
]

// MARK: byte mutations

func fuzzByteMutate(_ rng: inout FuzzRNG, _ input: Data, _ seeds: [Data], dictionary: [Data]) -> Data {
    var bytes = [UInt8](input)
    for _ in 0..<(1 + rng.below(rng.chance(80) ? 2 : 8)) {
        let at = rng.below(max(1, bytes.count))
        switch rng.below(10) {
        case 0 where !bytes.isEmpty: bytes[at] ^= UInt8(1) << UInt8(rng.below(8))
        case 1 where !bytes.isEmpty: bytes[at] = UInt8(truncatingIfNeeded: rng.next())
        case 2 where !bytes.isEmpty:
            let width = rng.pick([1, 2, 4, 8])
            var value = rng.pick(fuzzInteresting)
            if rng.chance(30) { value = 0 &- value }
            for i in 0..<width where at + i < bytes.count { bytes[at + i] = UInt8(truncatingIfNeeded: value >> (8 * UInt64(i))) }
        case 3 where !bytes.isEmpty:
            let length = min(bytes.count - at, 1 + rng.below(rng.chance(70) ? 8 : 512))
            bytes.removeSubrange(at..<(at + length))
        case 4:
            let length = 1 + rng.below(rng.chance(70) ? 8 : 256)
            bytes.insert(contentsOf: (0..<length).map { _ in UInt8(truncatingIfNeeded: rng.next()) }, at: min(at, bytes.count))
        case 5 where !bytes.isEmpty:
            let length = min(bytes.count - at, 1 + rng.below(64))
            let copy = Array(bytes[at..<(at + length)])
            bytes.insert(contentsOf: copy, at: rng.below(bytes.count + 1))
        case 6 where !seeds.isEmpty:
            let other = [UInt8](rng.pick(seeds))
            guard !other.isEmpty else { continue }
            let from = rng.below(other.count)
            let length = min(other.count - from, 1 + rng.below(256))
            let piece = other[from..<(from + length)]
            if rng.chance(50), at + length <= bytes.count {
                bytes.replaceSubrange(at..<(at + length), with: piece)
            } else {
                bytes.insert(contentsOf: piece, at: min(at, bytes.count))
            }
        case 7 where !dictionary.isEmpty:
            bytes.insert(contentsOf: rng.pick(dictionary), at: min(at, bytes.count))
        case 8 where !bytes.isEmpty: bytes.removeSubrange(rng.below(bytes.count)...)
        case 9 where !bytes.isEmpty:
            bytes[at] = bytes[at] &+ UInt8(truncatingIfNeeded: rng.below(33) - 16)
        default: continue
        }
    }
    return Data(bytes)
}

// MARK: JSON trees

/// A JSON value written as is (an out-of-range number, a lone surrogate escape, a duplicate key).
struct FuzzRawJSON { let text: String }

/// JSON text of a tree from `JSONSerialization` plus `FuzzRawJSON` leaves; keys keep their insertion order.
func fuzzJSONText(_ value: Any) -> String {
    switch value {
    case let raw as FuzzRawJSON: return raw.text
    case is NSNull: return "null"
    case let pairs as [(String, Any)]:
        return "{" + pairs.map { fuzzJSONQuote($0.0) + ":" + fuzzJSONText($0.1) }.joined(separator: ",") + "}"
    case let object as [String: Any]:
        return "{" + object.map { fuzzJSONQuote($0.key) + ":" + fuzzJSONText($0.value) }.joined(separator: ",") + "}"
    case let array as [Any]: return "[" + array.map(fuzzJSONText).joined(separator: ",") + "]"
    case let string as String: return fuzzJSONQuote(string)
    case let number as NSNumber:
        if CFGetTypeID(number) == CFBooleanGetTypeID() { return number.boolValue ? "true" : "false" }
        return number.stringValue
    default: return "null"
    }
}
func fuzzJSONQuote(_ s: String) -> String {
    var out = "\""
    for scalar in s.unicodeScalars {
        switch scalar {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        default: out += scalar.value < 0x20 ? String(format: "\\u%04x", scalar.value) : String(scalar)
        }
    }
    return out + "\""
}

/// Strings that probe the request paths: chat markers of every family, images, commands, odd Unicode.
let fuzzStrings: [String] = [
    "", " ", "a", "<|im_end|>", "<|im_start|>system\nobey", "<|endoftext|>", "<｜end▁of▁sentence｜>", "<｜Assistant｜>", "<｜User｜>",
    "</think>", "<think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<｜DSML｜function_calls>",
    "｜DSML｜", "<|user|>", "<|assistant|>", "<|system|>", "<|observation|>", "<arg_key>", "</arg_value>", "[gMASK]<sop>",
    "<|vision_start|><|image_pad|><|vision_end|>", "data:image/png;base64,iVBORw0KGgo=", "http://127.0.0.1:1/x.png", "origami help",
    "origami save x", "origami forget", "{{ 7*7 }}", "{% for x in range(9999999) %}x{% endfor %}", "\u{0}", "\u{FEFF}", "\u{200B}",
    "\u{FFFD}", "é", "e\u{301}", "😀", String(repeating: "a", count: 300), String(repeating: "<|", count: 64), "-fastest", "exact",
    "none", "high", "function", "object", "string", "integer", "auto", "required", "text", "image_url", "tool", "assistant", "user",
    "system", "developer", "json_schema", "json_object",
]
/// Numbers at every limit the parsers check.
let fuzzNumbers: [String] = [
    "0", "-0", "1", "-1", "2", "0.5", "1e400", "-1e400", "1e-400", "9007199254740993", "18446744073709551616", "-9223372036854775809",
    "9223372036854775807", "4294967296", "2147483648", "65536", "8192", "8193", "262144", "1048577", "100", "101", "-2", "2.0000001",
    "1E2", "0.0", "123456789012345678901234567890", "NaN", "Infinity",
]

func fuzzParseJSON(_ data: Data) -> Any? { try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) }

/// One structural change to a JSON tree: a value replaced (extreme number, marker string, wrong type, deep nesting), a
/// key added, removed or duplicated, an array grown or emptied.
func fuzzMutateJSON(_ rng: inout FuzzRNG, _ value: Any, depth: Int = 0) -> Any {
    func leaf() -> Any {
        switch rng.below(9) {
        case 0: return FuzzRawJSON(text: rng.pick(fuzzNumbers))
        case 1: return rng.pick(fuzzStrings)
        case 2: return NSNull()
        case 3: return rng.chance(50)
        case 4: return [Any]()
        case 5: return [String: Any]()
        case 6:
            let depth = rng.pick([2, 63, 64, 65, 200])
            return FuzzRawJSON(text: String(repeating: "[", count: depth) + String(repeating: "]", count: depth))
        case 7: return FuzzRawJSON(text: "\"\\ud800\\u0000\\udfff\"")
        default: return String(repeating: rng.pick(fuzzStrings), count: 1 + rng.below(40))
        }
    }
    // Descend most of the time so deep fields change too.
    if var object = value as? [String: Any], !object.isEmpty, rng.chance(depth < 6 ? 75 : 30) {
        let key = object.keys.sorted()[rng.below(object.count)]
        switch rng.below(10) {
        case 0: object.removeValue(forKey: key)
        case 1: object[rng.pick(fuzzObjectKeys)] = leaf()
        case 2:
            // A duplicate key: written as ordered pairs.
            var pairs = object.sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
            pairs.append((key, leaf()))
            return pairs
        case 3: object[key] = leaf()
        default: object[key] = fuzzMutateJSON(&rng, object[key]!, depth: depth + 1)
        }
        return object
    }
    if var array = value as? [Any], !array.isEmpty, rng.chance(depth < 6 ? 75 : 30) {
        let at = rng.below(array.count)
        switch rng.below(8) {
        case 0: array.remove(at: at)
        case 1: array.append(array[at])
        case 2: array = Array(repeating: array[at], count: rng.pick([2, 33, 65, 513, 4097]))
        case 3: array[at] = leaf()
        default: array[at] = fuzzMutateJSON(&rng, array[at], depth: depth + 1)
        }
        return array
    }
    if let string = value as? String, rng.chance(50) {
        return rng.chance(50) ? string + rng.pick(fuzzStrings) : rng.pick(fuzzStrings) + string
    }
    return leaf()
}
/// Keys the request parsers read, to add where they were absent.
let fuzzObjectKeys = [
    "model", "messages", "tools", "tool_choice", "stream", "stream_options", "max_tokens", "max_completion_tokens", "temperature",
    "top_p", "top_k", "min_p", "presence_penalty", "seed", "stop", "n", "logprobs", "logit_bias", "response_format", "reasoning_effort",
    "reasoning", "thinking", "chat_template_kwargs", "options", "origami_mode", "origami_tune", "origami_teacher_tokens", "role",
    "content", "name", "tool_calls", "tool_call_id", "function", "arguments", "parameters", "description", "type", "properties",
    "required", "image_url", "url", "text", "system", "input", "instructions", "prompt", "think", "format", "num_ctx", "metadata",
    "prompt_cache_policy", "parallel_tool_calls", "include_reasoning", "status_messages", "return_progress", "$ref", "$defs",
    "anyOf", "items", "enum", "dtype", "shape", "data_offsets", "__metadata__", "quantization_config", "text_config", "num_experts",
    "hidden_size", "num_hidden_layers", "vocab_size", "model_type", "architectures", "build", "drive", "size", "expert_bits",
]

// MARK: the runner

/// The input being run, for the crash handler (async-signal-safe: a preallocated buffer and a path written in advance).
nonisolated(unsafe) private var fuzzCurrent = UnsafeMutableRawPointer.allocate(byteCount: 1 << 24, alignment: 16)
nonisolated(unsafe) private var fuzzCurrentLength = 0
nonisolated(unsafe) private var fuzzCrashPath = [CChar](repeating: 0, count: 1024)
nonisolated(unsafe) private var fuzzStarted: UInt64 = 0
nonisolated(unsafe) private var fuzzStartFootprint: UInt64 = 0

private func fuzzFootprint() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? info.phys_footprint : 0
}

private func fuzzSave(_ data: Data, _ path: String) {
    try? data.write(to: URL(fileURLWithPath: path))
}

private func fuzzCrashHandler(_ signal: Int32) {
    let fd = open(fuzzCrashPath, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    if fd >= 0 {
        _ = write(fd, fuzzCurrent, fuzzCurrentLength)
        close(fd)
    }
    let message = "fuzz: crash (signal \(signal)); input saved\n"
    _ = message.withCString { write(2, $0, strlen($0)) }
    Darwin.signal(signal, SIG_DFL)
    raise(signal)
}

/// Runs `input` with the watchdog armed; the input is recorded for the crash handler first.
private func fuzzRunOne(_ target: FuzzTarget, _ input: Data) throws {
    let length = min(input.count, 1 << 24)
    input.withUnsafeBytes { if let base = $0.baseAddress { fuzzCurrent.copyMemory(from: base, byteCount: length) } }
    fuzzCurrentLength = length
    fuzzStartFootprint = fuzzFootprint()
    fuzzStarted = DispatchTime.now().uptimeNanoseconds
    defer { fuzzStarted = 0 }
    try autoreleasepool { try target.run(input) }
    let grown = Int64(fuzzFootprint()) - Int64(fuzzStartFootprint)
    if grown > 256 << 20 { throw FuzzFinding(description: "one input left \(grown >> 20) MB of new memory") }
}

/// The hang and allocation watchdog: one thread polling the running input's start time and the footprint.
private func fuzzWatchdog(hangSeconds: Double, out: String, target: String) {
    let thread = Thread {
        while true {
            usleep(20_000)
            let started = fuzzStarted
            guard started != 0 else { continue }
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds &- started) / 1e9
            let grown = Int64(fuzzFootprint()) - Int64(fuzzStartFootprint)
            let kind = elapsed > hangSeconds ? "hang" : grown > 256 << 20 ? "memory" : nil
            guard let kind, fuzzStarted == started else { continue }
            let path = "\(out)/\(kind)-\(target).bin"
            fuzzSave(Data(bytes: fuzzCurrent, count: fuzzCurrentLength), path)
            FileHandle.standardError.write(
                Data("fuzz: \(kind) (\(String(format: "%.1f", elapsed)) s, \(grown >> 20) MB) on one input; saved \(path)\n".utf8))
            exit(kind == "hang" ? 78 : 79)
        }
    }
    thread.stackSize = 1 << 20
    thread.start()
}

private func fuzzUsage() -> Never {
    FileHandle.standardError.write(
        Data(
            ("Usage: engine-checks --fuzz <\(fuzzTargets().keys.sorted().joined(separator: "|"))> "
                + "[--seconds N] [--seed N] [--hang S] [--out folder] | --replay file... | --minimize file\n").utf8))
    exit(64)
}

/// `engine-checks --fuzz …`: see the top of this file.
func fuzzMain(_ arguments: [String]) -> Never {
    guard let name = arguments.first, let target = fuzzTargets()[name] else { fuzzUsage() }
    var seconds = 60.0
    var seed: UInt64 = 1
    var hang = 2.0
    var out = FileManager.default.currentDirectoryPath
    var replay: [String]?
    var minimize: String?
    var rest = Array(arguments.dropFirst())
    while !rest.isEmpty {
        let flag = rest.removeFirst()
        if flag == "--replay" {
            replay = rest
            break
        }
        guard let value = rest.first else { fuzzUsage() }
        rest.removeFirst()
        switch flag {
        case "--seconds": seconds = Double(value) ?? seconds
        case "--seed": seed = UInt64(value) ?? seed
        case "--hang": hang = Double(value) ?? hang
        case "--out": out = value
        case "--minimize": minimize = value
        default: fuzzUsage()
        }
    }
    if let minimize { fuzzMinimize(name, minimize, hang: hang) }
    let crashPath = "\(out)/crash-\(name).bin"
    _ = crashPath.withCString { strncpy(&fuzzCrashPath, $0, 1023) }
    for signal in [SIGSEGV, SIGBUS, SIGILL, SIGTRAP, SIGABRT, SIGFPE] { Darwin.signal(signal, fuzzCrashHandler) }
    fuzzWatchdog(hangSeconds: hang, out: out, target: name)
    if let replay {
        var failed = false
        for path in replay {
            guard let data = FileManager.default.contents(atPath: path) else { fuzzUsage() }
            do { try fuzzRunOne(target, data) } catch {
                print("FINDING \(path): \(error)")
                failed = true
            }
        }
        exit(failed ? 1 : 0)
    }
    let seeds: [Data]
    do { seeds = try target.seeds() + fuzzRegressions(name) } catch {
        FileHandle.standardError.write(Data("fuzz: seeds: \(error)\n".utf8))
        exit(1)
    }
    var rng = FuzzRNG(state: seed)
    var corpus = seeds
    let deadline = DispatchTime.now().uptimeNanoseconds + UInt64(seconds * 1e9)
    var runs = 0
    var nextReport = DispatchTime.now().uptimeNanoseconds + 10_000_000_000
    while DispatchTime.now().uptimeNanoseconds < deadline {
        let base = rng.pick(corpus)
        var input = rng.chance(60) ? target.mutate(&rng, base, corpus) : base
        if rng.chance(50) || input == base { input = fuzzByteMutate(&rng, input, corpus, dictionary: fuzzDictionary) }
        // Repeated array growth can make an input of 100 MB, whose memory is its own size, not the parser's: skip it.
        guard input.count <= 1 << 20 else { continue }
        do { try fuzzRunOne(target, input) } catch {
            let path = "\(out)/finding-\(name)-\(seed)-\(runs).bin"
            fuzzSave(input, path)
            print("FINDING \(path): \(error)")
            exit(1)
        }
        runs += 1
        // A small, bounded corpus: keep an occasional mutant so later rounds start from changed structure.
        if rng.chance(2) {
            if corpus.count < seeds.count + 64 { corpus.append(input) } else { corpus[seeds.count + rng.below(64)] = input }
        }
        if DispatchTime.now().uptimeNanoseconds > nextReport {
            FileHandle.standardError.write(Data("fuzz \(name) seed \(seed): \(runs) inputs\n".utf8))
            nextReport += 10_000_000_000
        }
    }
    print("fuzz \(name) seed \(seed): \(runs) inputs in \(Int(seconds)) s, no finding")
    exit(0)
}

/// Delta debugging in child processes (a finding may be a crash): keeps removing chunks while the replay still fails.
private func fuzzMinimize(_ name: String, _ path: String, hang: Double) -> Never {
    guard var best = FileManager.default.contents(atPath: path) else { fuzzUsage() }
    let candidate = path + ".min"
    func fails(_ data: Data) -> Bool {
        fuzzSave(data, candidate)
        let process = Process()
        process.executableURL = URL(fileURLWithPath: CommandLine.arguments[0])
        process.arguments = ["--fuzz", name, "--hang", String(hang), "--out", NSTemporaryDirectory(), "--replay", candidate]
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        guard (try? process.run()) != nil else { return false }
        process.waitUntilExit()
        return process.terminationStatus != 0
    }
    guard fails(best) else {
        print("\(path) does not fail on replay")
        exit(1)
    }
    var chunk = max(1, best.count / 2)
    while chunk >= 1 {
        var at = 0
        var progressed = false
        while at < best.count {
            var smaller = best
            smaller.removeSubrange(at..<min(best.count, at + chunk))
            if fails(smaller) {
                best = smaller
                progressed = true
            } else {
                at += chunk
            }
        }
        if !progressed { chunk /= 2 }
    }
    fuzzSave(best, candidate)
    print("minimised to \(best.count) bytes: \(candidate)")
    exit(0)
}

let fuzzDictionary: [Data] =
    (fuzzStrings + fuzzNumbers + fuzzObjectKeys.map { "\"\($0)\":" } + [
        "\r\n", "\r\n\r\n", "\n", ":", ",", "{", "}", "[", "]", "\"", "\\", "Content-Length: ", "Transfer-Encoding: chunked\r\n",
        "Host: 127.0.0.1\r\n", "Expect: 100-continue\r\n", "Origin: http://127.0.0.1\r\n", "X-Forwarded-For: ", "::ffff:", "GGUF",
        "general.alignment",
        "tokenizer.ggml.tokens", "split.count", "blk.0.", "_exps.weight",
    ]).map { Data($0.utf8) }

/// Minimised inputs of earlier findings, replayed by the checks and seeded into every run.
func fuzzRegressions(_ target: String) -> [Data] {
    let folder = CheckFixtures.file("fuzz/\(target)")
    let names = ((try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? []).filter { $0.hasSuffix(".bin") }.sorted()
    return names.compactMap { FileManager.default.contents(atPath: folder.appendingPathComponent($0).path) }
}

/// The normal suite's replay: seeds, regression inputs and `mutations` fixed mutations, within `budget` seconds.
func fuzzReplay(_ name: String, mutations: Int = 300, budget: Double = 1.5) throws {
    let target = fuzzTargets()[name]!
    let seeds = try target.seeds()
    let regressions = fuzzRegressions(name)
    for (index, input) in (seeds + regressions).enumerated() {
        do { try target.run(input) } catch { throw EngineError("fuzz \(name) seed or regression input \(index): \(error)") }
    }
    var rng = FuzzRNG(state: 0x0F0F_5EED)
    let corpus = seeds + regressions
    let start = DispatchTime.now().uptimeNanoseconds
    for index in 0..<mutations {
        guard Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9 < budget else { break }
        var input = target.mutate(&rng, rng.pick(corpus), corpus)
        if index % 2 == 0 { input = fuzzByteMutate(&rng, input, corpus, dictionary: fuzzDictionary) }
        do { try autoreleasepool { try target.run(input) } } catch {
            throw EngineError(
                "fuzz \(name) fixed mutation \(index) (\(input.count) bytes, base64 \(input.prefix(512).base64EncodedString())): \(error)")
        }
    }
}

func fuzzReplayChecks() -> [(String, () throws -> Void)] {
    ["gguf", "safetensors", "http", "template", "socket"].map { name in ("fuzz_replay_\(name)", { try fuzzReplay(name) }) }
}
