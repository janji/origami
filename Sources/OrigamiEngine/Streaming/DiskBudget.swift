import Foundation

/// One account of the disk on each volume that holds Origami's files: who holds how much there (each hot copy place, the
/// prompt cache's entries, the conversation frontiers), the volume's free space and one reserve no save goes under. A
/// save the next request needs is worth more than the least-used tail of a hot copy: when a save finds too little room,
/// the hot copies on its volume give back the least-used whole records it lacks (`HotCopyKeeper.giveBack`), one line in
/// the log, and the save goes ahead (a prompt-cache save asks after removing its own model's other builds' folders and
/// before its own oldest entries, `PromptCache.makeRoom`; another model's saved prompts are never taken); their swaps fill those slots again once space returns. Only disk changes: a copy
/// keeps serving every record it still holds with the model's own bytes, so replies are unchanged.
package final class DiskBudget: @unchecked Sendable {
    package static let shared = DiskBudget()
    /// Free space no save goes under, for macOS and other apps (the prompt cache's and the frontier's floor each, until now).
    package static let reserve: UInt64 = 10_000_000_000

    private struct Holder {
        let name: String
        let folder: () -> URL?
        let bytes: () -> UInt64
        let giveBack: ((_ bytes: UInt64, _ reason: String) -> UInt64)?
        /// An idle model's copy (`IdleHotCopy`): its model, when that model last served, what it may give the loaded
        /// model's copy (`take`), and its status fields.
        var model: String? = nil, lastUsed: Date? = nil
        var spare: (() -> UInt64)? = nil, about: (() -> [String: Any])? = nil
        /// Givers in the order they are asked: idle copies first, their model used longest ago first; then the rest.
        static func order(_ a: Holder, _ b: Holder) -> Bool {
            let (x, y) = (a.spare != nil, b.spare != nil)
            guard x == y else { return x }
            return (a.lastUsed ?? .distantPast) < (b.lastUsed ?? .distantPast)
        }
    }
    private struct Account {
        var givenBack: UInt64 = 0, giveBacks = 0, skipped = 0
        var lastGiveBack: [String: Any]?, lastSkip: [String: Any]?
        /// Bytes idle models' copies gave each loaded model's copy, and the last such give.
        var takenFor: [String: UInt64] = [:], lastTake: [String: Any]?
    }
    private let lock = NSLock(), room = NSLock()
    private var holders: [Int: Holder] = [:], next = 0
    private var accounts: [String: Account] = [:]

    package init() {}
    /// Model-free checks set a drive's free bytes here; production measures the drive (`PromptCache.freeDiskBytes`).
    package var freeForChecks: ((URL) -> UInt64)?
    /// Free bytes on the drive that holds `url`.
    package func free(at url: URL) -> UInt64 { freeForChecks?(url) ?? PromptCache.freeDiskBytes(at: url) }

    /// Counts `bytes` held by `name` in `folder` (nil: none now) on its volume; `giveBack` (hot copies) frees at least the
    /// bytes asked in whole records, for the reason given, and returns what it freed. The returned id removes it.
    /// An idle model's copy (`IdleHotCopy`) also names its `model`, when it last served, its `spare` and its `about`.
    package func add(
        _ name: String, folder: @escaping () -> URL?, bytes: @escaping () -> UInt64, model: String? = nil, lastUsed: Date? = nil,
        spare: (() -> UInt64)? = nil, about: (() -> [String: Any])? = nil,
        giveBack: ((_ bytes: UInt64, _ reason: String) -> UInt64)? = nil
    ) -> Int {
        lock.withLock {
            next += 1
            holders[next] = Holder(
                name: name, folder: folder, bytes: bytes, giveBack: giveBack, model: model, lastUsed: lastUsed, spare: spare, about: about)
            return next
        }
    }

    /// The idle models' copies on the volume of `folder` (none of `model`), in the order they give.
    private func idle(on folder: URL, except model: String) -> [Holder] {
        let volume = Self.volume(folder)
        return lock.withLock { holders.values.filter { $0.spare != nil && $0.model != model } }.filter { holder in
            holder.folder().map { Self.volume($0) == volume } ?? false
        }.sorted(by: Holder.order)
    }
    /// What the idle models' copies on the volume of `folder` could give `model`'s copy now (`IdleHotCopy.spare`): room
    /// its copy may count beyond the drive's free space.
    package func spare(on folder: URL, for model: String) -> UInt64 {
        idle(on: folder, except: model).reduce(0) { $0 + ($1.spare?() ?? 0) }
    }
    /// The loaded `model`'s copy in `folder` grows past the drive's free space: the idle models' copies on its volume
    /// give back at least `bytes` (less when their spare ends first), model used longest ago first, each down to its
    /// floor. The bytes each giver's model freed; one line per giver in `log`.
    package func take(_ bytes: UInt64, in folder: URL, for model: String, log: (String) -> Void = { _ in }) -> [String: UInt64] {
        room.withLock {
            var freed: UInt64 = 0
            var gave: [String: UInt64] = [:]
            let volume = Self.volume(folder)
            for giver in idle(on: folder, except: model) where freed < bytes {
                let given = giver.giveBack!(bytes - freed, model)
                guard given > 0 else { continue }
                freed += given
                gave[giver.model ?? giver.name, default: 0] += given
                lock.withLock {
                    accounts[volume, default: Account()].takenFor[model, default: 0] += given
                    accounts[volume, default: Account()].lastTake = [
                        "bytes": given, "for": model, "from": giver.model ?? giver.name, "at": ISO8601DateFormatter().string(from: Date()),
                    ]
                }
                log(
                    "Hot copy of \(giver.model ?? "?") gave \(OrigamiPlan.gigabytes(given)) (its least-used experts) to \(model)'s copy, "
                        + "which grows past the drive's free space; it grows back when \(giver.model ?? "its model") is loaded.")
            }
            return gave
        }
    }
    package func remove(_ id: Int) { _ = lock.withLock { holders.removeValue(forKey: id) } }

    /// The volume `url` is on (its nearest existing folder's), by its mount point.
    package static func volume(_ url: URL) -> String {
        var probe = url.standardizedFileURL
        // A drive not connected keeps its own name: its path would otherwise fall back to the startup volume's.
        let parts = probe.pathComponents
        if parts.count > 2, parts[1] == "Volumes", !FileManager.default.fileExists(atPath: "/Volumes/" + parts[2]) {
            return "/Volumes/" + parts[2]
        }
        while !FileManager.default.fileExists(atPath: probe.path), probe.pathComponents.count > 1 { probe.deleteLastPathComponent() }
        return (try? probe.resourceValues(forKeys: [.volumeURLKey]))?.volume?.path ?? "/"
    }

    /// Room for a save of `bytes` into `folder` by `who` that keeps `floor` free after it, as `free` measures the drive
    /// (the saver's own measure): when short, the hot copies on that volume give back what is missing, least-used records
    /// first. True when the save fits now. Saves are serialized here, so two never count the same room.
    package func makeRoom(
        _ bytes: UInt64, in folder: URL, for who: String, floor: UInt64 = reserve, free: () -> UInt64
    ) -> Bool {
        room.withLock {
            let need = bytes.addingReportingOverflow(floor)
            guard !need.overflow else { return false }
            let volume = Self.volume(folder)
            let givers = lock.withLock { holders.values.filter { $0.giveBack != nil } }.filter { holder in
                holder.folder().map { Self.volume($0) == volume } ?? false
            }.sorted(by: Holder.order)
            // A giver is asked again while it frees something and the save is still short: the drive may free less than
            // the slots it was given back (local snapshots, shared extents). One save takes at most twice what it lacked at
            // first, so a volume that frees nothing never empties a copy for it.
            let first = free()
            let cap = first < need.partialValue ? 2 * (need.partialValue - first) : 0
            var taken: UInt64 = 0
            for giver in givers {
                while case let left = free(), left < need.partialValue, taken < cap {
                    let short = need.partialValue - left
                    let freed = giver.giveBack!(min(short, cap - taken), who)
                    taken += freed
                    guard freed > 0 else { break }
                    lock.withLock {
                        accounts[volume, default: Account()].givenBack += freed
                        accounts[volume, default: Account()].giveBacks += 1
                        accounts[volume, default: Account()].lastGiveBack = [
                            "bytes": freed, "for": who, "short_bytes": short, "holder": giver.name,
                            "at": ISO8601DateFormatter().string(from: Date()),
                        ]
                    }
                }
            }
            return free() >= need.partialValue
        }
    }

    /// A save by `who` into `folder` that was skipped for space (it must normally never happen while a copy can give back).
    package func skipped(in folder: URL, for who: String, reason: String) {
        let volume = Self.volume(folder)
        lock.withLock {
            accounts[volume, default: Account()].skipped += 1
            accounts[volume, default: Account()].lastSkip = [
                "for": who, "reason": reason, "at": ISO8601DateFormatter().string(from: Date()),
            ]
        }
    }

    /// For `/v1/status` (`disk_budget`): per volume with a holder, who holds how much, the free space, the reserve, what
    /// was given back and for what, and the saves skipped for space (normally 0).
    package var report: [[String: Any]] {
        let all = lock.withLock { Array(holders.values) }
        var held: [String: [[String: Any]]] = [:]
        for holder in all {
            guard let folder = holder.folder() else { continue }
            let row: [String: Any] = ["name": holder.name, "path": folder.path, "bytes": holder.bytes()]
            held[Self.volume(folder), default: []].append(row.merging(holder.about?() ?? [:]) { $1 })
        }
        let accounts = lock.withLock { self.accounts }
        return Set(held.keys).union(accounts.keys).sorted().map { volume in
            let account = accounts[volume] ?? Account()
            return [
                "volume": volume, "free_bytes": free(at: URL(fileURLWithPath: volume)),
                "reserve_bytes": Self.reserve, "holders": held[volume] ?? [], "given_back_bytes": account.givenBack,
                "give_backs": account.giveBacks, "last_give_back": account.lastGiveBack ?? NSNull(),
                "saves_skipped": account.skipped, "last_skip": account.lastSkip ?? NSNull(),
                "taken_for_copies": account.takenFor, "last_take": account.lastTake ?? NSNull(),
            ]
        }
    }

    /// The line `origami status` and `doctor` show when a save was skipped for space or a copy gave space back (nil
    /// otherwise), from a `/v1/status` `disk_budget`.
    package static func line(_ report: Any?) -> String? {
        let volumes = (report as? [[String: Any]]) ?? []
        let parts = volumes.compactMap { volume -> String? in
            let skipped = volume["saves_skipped"] as? Int ?? 0
            let given = (volume["given_back_bytes"] as? NSNumber)?.uint64Value ?? 0
            guard skipped > 0 || given > 0 else { return nil }
            let name = volume["volume"] as? String ?? "?"
            return "\(name): "
                + (given > 0
                    ? "the hot copy gave back \(OrigamiPlan.gigabytes(given)) (least-used records) for saves" : "")
                + (given > 0 && skipped > 0 ? "; " : "")
                + (skipped > 0
                    ? "\(skipped) save\(skipped == 1 ? "" : "s") skipped for space (last: "
                        + "\((volume["last_skip"] as? [String: Any])?["for"] as? String ?? "?"))" : "")
        }
        return parts.isEmpty ? nil : "disk budget: " + parts.joined(separator: "; ")
    }
    /// The line `origami status` shows once a loaded model's copy grew at an idle model's copy's cost (nil otherwise).
    package static func growthLine(_ report: Any?) -> String? {
        let parts = ((report as? [[String: Any]]) ?? []).compactMap { volume -> String? in
            guard let last = volume["last_take"] as? [String: Any], let model = last["for"] as? String else { return nil }
            let total = ((volume["taken_for_copies"] as? [String: Any])?[model] as? NSNumber)?.uint64Value ?? 0
            return
                "\(model)'s hot copy grew by \(OrigamiPlan.gigabytes(total)) on \(volume["volume"] as? String ?? "?") at the cost of idle "
                + "models' copies (last: \(OrigamiPlan.gigabytes((last["bytes"] as? NSNumber)?.uint64Value ?? 0)) from "
                + "\(last["from"] as? String ?? "?") at \(last["at"] as? String ?? "?"))"
        }
        return parts.isEmpty ? nil : "hot copies: " + parts.joined(separator: "; ")
    }
}
