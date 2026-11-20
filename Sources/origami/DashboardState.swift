import CryptoKit
import Foundation
import OrigamiEngine

/// Small, versioned control-plane snapshots. No model loading, directory walks or benchmarks.
enum DashboardState {
    static let mutationLock = NSRecursiveLock()
    static func memoryReloadRequired(_ settings: Settings, state: [String: Any]) -> Bool {
        guard let requested = settings.memoryLimit,
            let active = (state["memory"] as? [String: Any])?["cap_bytes"] as? NSNumber
        else { return false }
        let host = WiredMemory.snapshot()
        let attainable = min(requested, host.maximumBudgetBytes, host.effectiveLimitMiB.map { $0 * WiredMemoryPolicy.mebibyte } ?? 0)
        return attainable > active.uint64Value
    }
    /// Why tuning or the memory search cannot start now ("" when it can), in the words the dashboard shows next to its
    /// buttons: the dashboard state, the start of a tuning job and `origami tune` all say this.
    static func tuneBlock(state: [String: Any], job: [String: Any]?, reloadRequired: Bool) -> String {
        if job?["state"] as? String == "running" {
            return "Another job is running" + ((job?["title"] as? String).map { ": " + $0 } ?? "")
        }
        let status = state["status"] as? String ?? ""
        let queue = state["queue"] as? [String: Any] ?? [:]
        switch status {
        case "stopped", "stopping": return "The model is stopped: Start it first"
        // The header's "No model" when none was found, else its "Failed" with Try again.
        case "failed", "down":
            let failure = state["failure"] as? String ?? ""
            return status == "down" || failure.contains("No model found") || failure.contains("No model at")
                ? "No model is running" : "The model failed to start: Try again first"
        case "loading", "reloading", "unavailable": return "The model is loading"
        default: break
        }
        if reloadRequired { return "Reload to apply saved settings first" }
        if status == "busy" || queue["busy"] as? Bool == true || (queue["queued"] as? Int ?? 0) > 0 || state["pending"] != nil {
            return "A request is running: this starts when the model is idle"
        }
        return status == "ready" ? "" : "The model is not ready"
    }
    /// Live host limits are separate from the saved target and the loaded model's budget.
    static func wiredMemory(_ settings: Settings, state: [String: Any]) -> [String: Any] {
        let host = WiredMemory.snapshot()
        let memory = state["memory"] as? [String: Any] ?? [:]
        let active = memory["cap_bytes"] as? NSNumber
        let required = settings.memoryLimit.map { $0 / WiredMemoryPolicy.mebibyte + ($0 % WiredMemoryPolicy.mebibyte == 0 ? 0 : 1) }
        let needsRaise = required.map { requested in host.effectiveLimitMiB.map { requested > $0 } ?? false } ?? false
        return [
            "physical_memory_bytes": host.physicalMemoryBytes,
            "os_headroom_bytes": host.physicalMemoryBytes - host.maximumBudgetBytes,
            "safe_max_budget_bytes": host.maximumBudgetBytes,
            "system_limit_mib": host.currentLimitMiB.map { $0 as Any } ?? NSNull(),
            "system_limit_bytes": host.currentLimitMiB.map { $0 * WiredMemoryPolicy.mebibyte as Any } ?? NSNull(),
            "system_default": host.currentLimitMiB == 0,
            "effective_limit_bytes": host.effectiveLimitMiB.map { $0 * WiredMemoryPolicy.mebibyte as Any } ?? NSNull(),
            "metal_recommended_bytes": host.metalRecommendedBytes.map { $0 as Any } ?? NSNull(),
            "configured_budget_bytes": settings.memoryLimit.map { $0 as Any } ?? NSNull(),
            "active_budget_bytes": active as Any? ?? NSNull(),
            "usable_budget_bytes": memory["budget_bytes"] ?? NSNull(),
            "physical_footprint_bytes": memory["footprint_bytes"] ?? NSNull(),
            "budget_mode": memory["budget_mode"] ?? NSNull(),
            "temporarily_reduced": memory["temporarily_reduced"] as? Bool ?? false,
            "requires_raise": needsRaise, "supported": host.effectiveLimitMiB != nil && host.metalRecommendedBytes != nil,
            "persistent": false, "error": host.error.map { $0 as Any } ?? NSNull(),
        ]
    }
    /// What differs between the saved settings and what the loaded model runs with, in plain words, and the values
    /// that save the running state again ("Discard the saved change"); none when a difference cannot be saved back.
    static func pending(_ settings: Settings, running: [String: String], runningModel: String, servedModel: String, state: [String: Any])
        -> (changes: [[String: String]], revert: [String: String])
    {
        let labels = [
            "context": "Context", "quality": "Quality", "reasoning_effort": "Thinking", "memory_limit": "Memory",
            "hot_copy": "Hot copy", "hot_copy_dir": "Hot copy folder", "hot_copy_more": "More hot copy places",
            "sampling": "Sampling",
        ]
        let cap = ((state["memory"] as? [String: Any])?["cap_bytes"] as? NSNumber)?.uint64Value
        func memory(_ value: String, cap: UInt64?) -> String {
            value.hasPrefix("(") ? "automatic" + (cap.map { " (\(OrigamiPlan.gigabytes($0)))" } ?? "") : value
        }
        var changes: [[String: String]] = []
        var revert: [String: String] = [:]
        var revertable = runningModel == servedModel
        if !revertable { changes.append(["key": "model", "label": "Model", "saved": servedModel, "running": runningModel]) }
        let saved = configuration(settings)
        // Every key a running model reads once (`Settings.needsReload`); `hot_copy_keep` and the others that apply while serving are not.
        for key in Settings.needsReload where saved[key] != running[key] {
            guard let now = running[key], let then = saved[key] else { continue }
            let isMemory = key == "memory_limit"
            changes.append([
                "key": key, "label": labels[key] ?? key, "saved": isMemory ? memory(then, cap: nil) : then,
                "running": isMemory ? memory(now, cap: cap) : now,
            ])
            // The quality knobs are set by `origami config`, not the page.
            if Settings.modelKeys.contains(key), !(["hot_copy", "hot_copy_dir", "hot_copy_more"] + Settings.qualityKnobs).contains(key) {
                revert[key] = storable(key, now)
            } else {
                revertable = false
            }
        }
        // The same limit saved and loaded, but the model started under a lower ceiling than macOS now allows.
        if changes.isEmpty, let limit = settings.memoryLimit, let cap {
            changes.append([
                "key": "memory_limit", "label": "Memory", "saved": OrigamiPlan.gigabytes(limit), "running": OrigamiPlan.gigabytes(cap),
            ])
            revertable = false
        }
        return (changes, revertable ? revert : [:])
    }
    /// A running value as `config set` takes it: a default shown in brackets is the setting's own "automatic".
    static func storable(_ key: String, _ value: String) -> String {
        value.hasPrefix("(")
            ? (["memory_limit", "hot_copy", "hot_copy_dir"].contains(key)
                ? "auto" : key == "sampling" ? "recommended" : key == "reasoning_effort" ? "none" : "") : value
    }
    static func configuration(_ settings: Settings) -> [String: String] {
        let keys = Settings.needsReload
        return Dictionary(uniqueKeysWithValues: keys.map { ($0, settings.display($0)) })
    }
    /// A hash of the settings files' content: any change, by the page, `origami config` or an editor, makes a new one.
    static func revision(_ settings: Settings) -> String {
        var hash = SHA256()
        for url in [settings.configFile, ModelRegistry.file(settings)] {
            hash.update(data: (try? Data(contentsOf: url)) ?? Data())
        }
        let revision = hash.finalize().prefix(12).map { String(format: "%02x", $0) }.joined()
        remember(revision, settings)
        return revision
    }
    /// Each key's value at the revisions this process has handed out (the last 64), so a write based on an older one can
    /// be told which of its keys changed since. The stored text is compared too, so a new key behind "(set)" counts.
    private nonisolated(unsafe) static var seen: [(revision: String, shown: [String: String], stored: [String: String])] = []
    private static let seenLock = NSLock()
    private static func values(_ settings: Settings) -> (shown: [String: String], stored: [String: String]) {
        let file = (try? Settings.readConfig(settings.configFile)) ?? [:]
        return (
            Dictionary(uniqueKeysWithValues: Settings.keys.map { ($0.0, settings.display($0.0)) }),
            file.mapValues { "\($0)" }
        )
    }
    private static func remember(_ revision: String, _ settings: Settings) {
        seenLock.withLock {
            guard !seen.contains(where: { $0.revision == revision }) else { return }
            let now = values(settings)
            seen.append((revision, now.shown, now.stored))
            if seen.count > 64 { seen.removeFirst() }
        }
    }
    /// A write refused because a key it writes changed after the revision its draft was based on.
    struct Conflict: Error {
        let changes: [[String: String]]
        var body: [String: Any] {
            [
                "error": [
                    "message": "Changed elsewhere since you started editing: "
                        + changes.map { $0["key"] ?? "" }.joined(separator: ", ") + ". Keep yours or take the current values.",
                    "type": "invalid_request_error", "code": "settings_conflict",
                ],
                "changes": changes,
            ]
        }
    }
    /// A write from the page (`If-Match`: the revision its draft was based on) goes through when none of the keys it
    /// writes changed since; otherwise it is refused, naming each with the value it was based on and the value now.
    /// A base this process never handed out (it restarted) counts every written key as changed.
    static func check(base: String?, writing keys: [String], settings: Settings) throws {
        let current = revision(settings)
        guard let base, base != current else { return }
        let now = values(settings)
        let then = seenLock.withLock { seen.first { $0.revision == base } }
        let changed = keys.filter { key in
            guard let then else { return true }
            return then.shown[key] != now.shown[key] || then.stored[key] != now.stored[key]
        }
        guard changed.isEmpty else {
            throw Conflict(
                changes: changed.map { ["key": $0, "base": then?.shown[$0] ?? "", "current": now.shown[$0] ?? ""] })
        }
    }
    static func check(_ expected: String?, settings: Settings) throws {
        if let expected, expected != revision(settings) {
            throw ServingFailure("Settings changed in another window. Review the current values and save again.", status: 409)
        }
    }
    private final class Reply: @unchecked Sendable {
        let lock = NSLock()
        var object: [String: Any]?
    }
    static func status(port: UInt16, key: String?, path: String = "/v1/status") -> [String: Any] {
        let reply = Reply()
        let done = DispatchSemaphore(value: 0)
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)\(path)")!)
        request.timeoutInterval = 3
        if let key { request.setValue("Bearer " + key, forHTTPHeaderField: "Authorization") }
        let task = URLSession.shared.dataTask(with: request) { data, response, _ in
            if let http = response as? HTTPURLResponse {
                let object = data.flatMap { (try? JSONSerialization.jsonObject(with: $0)) as? [String: Any] } ?? [:]
                reply.lock.withLock { reply.object = HTTPIngressPolicy.observedStatus(object, httpStatus: http.statusCode) }
            }
            done.signal()
        }
        task.resume()
        if done.wait(timeout: .now() + 3.5) == .timedOut { task.cancel() }
        return reply.lock.withLock { reply.object } ?? ["status": "unavailable", "failure": "The model did not answer. Reconnecting…"]
    }
    static func lastRun(_ settings: Settings) -> Any {
        guard let data = try? Data(contentsOf: Tuning.historyFile(settings)),
            let line = String(decoding: data.suffix(65536), as: UTF8.self).split(separator: "\n").last,
            let object = try? JSONSerialization.jsonObject(with: Data(line.utf8))
        else { return NSNull() }
        return object
    }
    static func checkpoint(_ settings: Settings, status: [String: Any]) {
        guard status["status"] as? String == "ready", settings.quality.name != "custom",
            let active = status["settings"] as? [String: [String: Any]]
        else { return }
        let keys = Settings.modelKeys.filter {
            !(["hot_copy", "hot_copy_dir", "hot_copy_more", "hot_copy_keep"] + Settings.qualityKnobs).contains($0)
        }
        // Save the running values, never a pending configuration that has not loaded successfully.
        let values = keys.reduce(into: [String: String]()) { out, key in
            if let value = active[key]?["value"] as? String { out[key] = storable(key, value) }
        }
        guard !values.isEmpty else { return }
        let model = settings.model ?? ""
        let url = settings.dataDir.appendingPathComponent("last-working.json")
        var saved = (try? Settings.readConfig(url)) ?? [:]
        if NSDictionary(dictionary: saved[model] as? [String: String] ?? [:]).isEqual(to: values) { return }
        saved[model] = values
        try? Settings.write([model: values], to: url)
    }
}
