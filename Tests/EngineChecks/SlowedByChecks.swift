import Foundation
import OrigamiEngine

/// `SlowedBy.list` on each case of `Tests/Fixtures/status/slowed-by.json`, whose lists the dashboard's checks show as
/// its Now tab does, so `origami status`, `/v1/status` and the page say the same; every rule has a case.
func slowedByRules() throws {
    let cases =
        try JSONSerialization.jsonObject(with: Data(contentsOf: CheckFixtures.file("status/slowed-by.json"))) as? [[String: Any]] ?? []
    func json(_ value: Any) -> String {
        String(decoding: (try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])) ?? Data(), as: UTF8.self)
    }
    var seen = Set<String>()
    var wrong: [String] = []
    for item in cases {
        let got = SlowedBy.list(item["status"] as? [String: Any] ?? [:])
        seen.formUnion(got.compactMap { $0["id"] as? String })
        if item["slowed_by"].map(json) != json(got) { wrong.append("\(item["rule"] ?? "?"): \(json(got))") }
    }
    guard cases.count >= 20, wrong.isEmpty else { throw EngineError("slowed by: " + wrong.joined(separator: "\n")) }
    let missing = SlowedBy.ids.filter { !seen.contains($0) }
    guard missing.isEmpty else { throw EngineError("slowed by: no fixture case for \(missing)") }
    // Every field a rule reads is in some case; and in the live case (a 3.1.7 server's status, trimmed and anonymised,
    // with what the front adds), every object a rule reads from holds the field it reads, unless marked optional (`?`).
    let statuses = cases.map { $0["status"] ?? [:] }
    let real = cases.first { ($0["rule"] as? String ?? "").hasPrefix("the live ") }?["status"]
    for path in SlowedBy.reads {
        let parts = path.trimmingCharacters(in: CharacterSet(charactersIn: "?")).split(separator: ".").map(String.init)
        guard statuses.contains(where: { !found(parts, in: $0).isEmpty }) else { throw EngineError("slowed by: no case has \(path)") }
        let parents = found(Array(parts.dropLast()), in: real).filter { $0 is [String: Any] }
        guard path.hasSuffix("?") || parents.isEmpty || !found(parts, in: real).isEmpty else {
            throw EngineError("slowed by: the live status has no \(path)")
        }
    }
    // `since`: a condition keeps the time it was first seen until it clears.
    var memory: [String: String] = [:]
    let one = [["id": "queue", "text": "", "fix": "", "since": NSNull()] as [String: Any]]
    let first = SlowedBy.stamp(one, seen: &memory, now: Date(timeIntervalSince1970: 1_000))[0]["since"] as? String
    let later = SlowedBy.stamp(one, seen: &memory, now: Date(timeIntervalSince1970: 2_000))[0]["since"] as? String
    _ = SlowedBy.stamp([], seen: &memory)
    let again = SlowedBy.stamp(one, seen: &memory, now: Date(timeIntervalSince1970: 3_000))[0]["since"] as? String
    guard first == "1970-01-01T00:16:40Z", later == first, again == "1970-01-01T00:50:00Z" else {
        throw EngineError("slowed by: since \(first ?? "-"), \(later ?? "-"), \(again ?? "-")")
    }
}

/// The values at `path` (`a`, `a[]`: each element of the array at `a`) in `value`, a null included; none when a key is
/// missing.
private func found(_ path: [String], in value: Any?) -> [Any] {
    guard let head = path.first else { return value.map { [$0] } ?? [] }
    let key = head.hasSuffix("[]") ? String(head.dropLast(2)) : head
    guard let object = value as? [String: Any], let next = object[key] else { return [] }
    let rest = Array(path.dropFirst())
    return head.hasSuffix("[]") ? (next as? [Any] ?? []).flatMap { found(rest, in: $0) } : found(rest, in: next)
}

/// `ProcessMemory`: this process is listed with a sane footprint, Origami's own name is left out of `others`, and an
/// app's helpers are named as the app.
func processMemoryNames() throws {
    let all = ProcessMemory.all()
    guard let me = all.first(where: { $0.pid == getpid() }), me.bytes > 1_000_000, me.bytes < ProcessInfo.processInfo.physicalMemory
    else { throw EngineError("process memory: this process is missing or its size is wrong (\(all.count) processes)") }
    guard all.allSatisfy({ $0.bytes < ProcessInfo.processInfo.physicalMemory }) else {
        throw EngineError("process memory: a size over RAM")
    }
    let others = ProcessMemory.others(3, own: me.name)
    guard others.count <= 3, !others.contains(where: { $0.name == me.name }), others.map(\.bytes) == others.map(\.bytes).sorted(by: >)
    else { throw EngineError("process memory: others \(others)") }
    let helper =
        "/Applications/Google Chrome.app/Contents/Frameworks/Google Chrome Framework.framework/Helpers/Google Chrome Helper (Renderer).app/Contents/MacOS/Google Chrome Helper (Renderer)"
    guard ProcessMemory.name(path: helper) == "Google Chrome", ProcessMemory.name(path: "/usr/bin/python3") == "python3",
        ProcessMemory.name(path: "/Users/someone/.local/share/tool/versions/2.1.3") == "tool"
    else {
        throw EngineError("process memory: app names")
    }
}

/// The program macOS asks about is the top of the process tree below launchd: origami itself as a launchd job, the
/// launcher that starts it, or the terminal app a shell runs in.
func permissionSubjectName() throws {
    let trees: [(parents: [pid_t: pid_t], paths: [pid_t: String], expected: String)] = [
        ([10: 1], [10: "/opt/origami/origami"], "origami"),
        ([10: 9, 9: 1], [10: "/e/origami-3.1.5/origami", 9: "/e/origami-launcher/origami-launcher"], "origami-launcher"),
        (
            [10: 8, 8: 7, 7: 1], [10: "/usr/local/bin/origami", 8: "/bin/zsh", 7: "/Applications/Terminal.app/Contents/MacOS/Terminal"],
            "Terminal"
        ),
    ]
    for tree in trees {
        let name = PermissionSubject.name(of: 10, parent: { tree.parents[$0] }, path: { tree.paths[$0] })
        guard name == tree.expected else { throw EngineError("permission subject: \(name), expected \(tree.expected)") }
    }
    guard !PermissionSubject.name.isEmpty else { throw EngineError("permission subject: no name for this process") }
}

/// The server log's line for a finished request names what the dashboard's row does, and leaves out what a failed one lacks.
func requestLogLine() throws {
    let done = LocalHTTPServer.summary([
        "finish": "stop", "seconds": 174.2, "prompt_tokens": 61748, "cached_tokens": 61087, "ttft_s": 22.44, "completion_tokens": 1569,
        "decode_tps": 10.36,
    ])
    let failed = LocalHTTPServer.summary(["finish": "cancelled", "seconds": 9.0, "prompt_tokens": 500, "cached_tokens": 0])
    guard
        done == "request: stop in 174 s: 61748 prompt tokens (61087 restored), first word after 22.4 s, 1569 tokens written at 10.4 tok/s",
        failed == "request: cancelled in 9 s: 500 prompt tokens (0 restored)"
    else { throw EngineError("request log line: \(done) | \(failed)") }
}
