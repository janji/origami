import Foundation
import OrigamiEngine

/// The dashboard's presets: named copies of the Setup tab's saved settings (the model, context, quality, thinking, memory
/// limit and sampling), kept in `<data_dir>/presets.json` and saved back by `config set`'s rules when one is used, as
/// Restore last working settings does. Tuning needs no copy: its results are kept per model and context (`Tuning`), so
/// they come back with them. Hot copy sizes and folders are left out: using a preset never builds or deletes a copy.
enum Presets {
    static let keys =
        ["model", "context", "quality", "reasoning_effort", "memory_limit", "sampling", "sampling_override"] + SamplingDefaults.keys
    static func file(_ settings: Settings) -> URL { settings.dataDir.appendingPathComponent("presets.json") }

    /// Every preset by name, as `/v1/presets` and the dashboard state list them.
    static func list(_ settings: Settings) -> [[String: Any]] {
        let all = (try? Settings.readConfig(file(settings))) ?? [:]
        return all.keys.sorted { $0.localizedStandardCompare($1) == .orderedAscending }.compactMap { name in
            (all[name] as? [String: String]).map { ["name": name, "values": $0] }
        }
    }

    /// The saved settings a preset keeps, as text `config set` takes.
    static func capture(_ settings: Settings) throws -> [String: String] {
        guard settings.quality.name != "custom" else {
            throw ServingFailure("Quality is a custom mix set with origami config: choose a quality level first", status: 409)
        }
        var values = Dictionary(uniqueKeysWithValues: keys.map { ($0, DashboardState.storable($0, settings.display($0))) })
        // The exact bytes, as the page saves them, not the rounded figure shown.
        if let bytes = settings.memoryLimit { values["memory_limit"] = String(format: "%.9fGB", Double(bytes) / 1e9) }
        return values
    }

    static func name(_ value: Any?) throws -> String {
        let name = (value as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard (1...64).contains(name.count), !name.unicodeScalars.contains(where: CharacterSet.controlCharacters.contains) else {
            throw ServingFailure("A preset's name is 1 to 64 characters, such as Long chats")
        }
        return name
    }

    /// `POST /v1/presets {"name"}` keeps the saved settings under that name (`"replace": true` updates one); on
    /// `/v1/presets/<name>`, `PATCH {"name"}` renames it, `DELETE` removes it and `POST` uses it: its values are saved
    /// as settings, for the next reload. Each answers with every preset; a use, with what the save said too.
    static func change(_ method: String, _ target: String?, _ body: [String: Any], settings: Settings) throws -> [String: Any] {
        let url = file(settings)
        let all = try Settings.readConfig(url)
        var result: [String: Any] = [:]
        switch (method, target) {
        case ("POST", nil):
            let name = try name(body["name"])
            guard all[name] == nil || body["replace"] as? Bool == true else {
                throw ServingFailure("A preset named \(name) exists: Update it, or choose another name", status: 409)
            }
            try Settings.write([name: try capture(settings)], to: url)
        case ("PATCH", let name?), ("DELETE", let name?), ("POST", let name?):
            guard let values = all[name] as? [String: String] else { throw ServingFailure("No preset is named \(name)", status: 404) }
            if method == "DELETE" {
                try Settings.write([name: nil], to: url)
            } else if method == "PATCH" {
                let new = try Self.name(body["name"])
                guard new == name || all[new] == nil else { throw ServingFailure("A preset named \(new) exists", status: 409) }
                if new != name { try Settings.write([name: nil, new: values], to: url) }
            } else {
                // A name that is no longer added would be read as a folder.
                if let model = values["model"], !model.isEmpty, !model.contains("/"), ModelRegistry.entry(model, settings: settings) == nil
                {
                    throw ServingFailure("\(model), this preset's model, is no longer added: add it under Models first", status: 409)
                }
                // Only a preset's own keys: a hand-edited file never changes who can reach Origami.
                result = try Commands.configWriter(settings)(values.filter { keys.contains($0.key) })
            }
        default:
            throw ServingFailure("Use GET or POST /v1/presets, or PATCH, DELETE or POST /v1/presets/<name>", status: 405)
        }
        result["presets"] = list(settings)
        return result
    }
}
