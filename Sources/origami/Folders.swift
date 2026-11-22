import Darwin
import Foundation
import OrigamiEngine

/// What a `Folders.timed` call returns, written by its thread under `Folders.lock` before the signal.
final class TimedBox<T>: @unchecked Sendable {
    var value: T?
    var finished = false
}

enum Folders {
    /// A place to start from: its name as the page shows it and its resolved path.
    struct Root { let name: String, path: String }

    /// What macOS says when an app reads a drive or folder it has not allowed (the front's own words for a drive).
    static func permission(_ path: String) -> String {
        path.hasPrefix("/Volumes/")
            ? "macOS has not let Origami read \(Stores.driveName(path)) yet: allow Removable Volumes for \(PermissionSubject.name) in System Settings > Privacy & Security > Files & Folders"
            : "macOS has not let Origami read \((path as NSString).lastPathComponent) yet: allow it for \(PermissionSubject.name) in System Settings > Privacy & Security > Files & Folders"
    }

    /// The path with symlinks and `..` resolved, or nil when nothing is there.
    static func real(_ path: String) -> String? {
        guard let resolved = realpath(path, nil) else { return nil }
        defer { free(resolved) }
        return String(cString: resolved)
    }

    /// `body` on its own thread, abandoned after `seconds`: a removable drive macOS has not allowed makes any file call
    /// wait on a prompt (which that call is also what shows). While a call on a drive under /Volumes (`on`) is still
    /// stuck, the next ones give up at once: one stuck drive costs one wait, not one per folder on it, and stuck threads
    /// do not pile up. Once that call returns (access allowed, the drive back), the drive is waited for again.
    static func timed<T>(_ seconds: Double, on path: String? = nil, _ body: @escaping @Sendable () -> T) -> T? {
        let drive = path.flatMap { $0.hasPrefix("/Volumes/") ? Stores.driveName($0) : nil }
        if let drive, lock.withLock({ stuck.contains(drive) }) { return nil }
        let box = TimedBox<T>()
        let done = DispatchSemaphore(value: 0)
        Thread.detachNewThread {
            let value = body()
            lock.withLock {
                box.value = value
                box.finished = true
                if let drive { stuck.remove(drive) }
            }
            done.signal()
        }
        if done.wait(timeout: .now() + seconds) == .success { return box.value }
        return lock.withLock { () -> T? in
            if !box.finished, let drive { stuck.insert(drive) }
            return box.finished ? box.value : nil
        }
    }
    nonisolated(unsafe) private static var stuck = Set<String>()
    static let lock = NSLock()

    /// Drives under /Volumes (the boot drive's link left out), the home folder, then the places Origami uses that exist.
    static func roots(_ settings: Settings) -> [Root] {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let volumes = ((try? FileManager.default.contentsOfDirectory(atPath: "/Volumes")) ?? []).sorted()
            .filter { !$0.hasPrefix(".") && !$0.hasPrefix("com.apple.") }.map { ($0, "/Volumes/" + $0) }
        let places =
            [
                ("Home", home), ("Models", home + "/Models"), ("Origami data", settings.dataDir.path),
                ("Saved prompts", settings.promptCacheRoot.path),
            ]
            + settings.stores.map { ("Store", $0.path) } + [settings.hotCopyDir].compactMap { $0.map { ("Hot copy", $0.path) } }
            + settings.modelsDirs.map { ("Model search", $0.path) }
        var roots: [Root] = []
        for (name, path) in volumes + places {
            // A drive that does not answer is listed by name; opening it is what makes macOS ask.
            let resolved = path.hasPrefix("/Volumes/") ? timed(2, on: path, { real(path) }) ?? path : real(path)
            guard let resolved, resolved != "/", !roots.contains(where: { $0.path == resolved }) else { continue }
            roots.append(Root(name: name, path: resolved))
        }
        return roots
    }

    /// The root `path` (resolved) is in: the widest one that holds it, so the path shows (and goes up) from its drive or home.
    /// Nil also below a hidden folder (~/.ssh) inside the nearest root, which the listings leave out.
    static func root(of path: String, _ roots: [Root]) -> Root? {
        let holding = roots.filter { path == $0.path || path.hasPrefix($0.path + "/") }
        guard let nearest = holding.max(by: { $0.path.count < $1.path.count }),
            !path.dropFirst(nearest.path.count).split(separator: "/").contains(where: { $0.hasPrefix(".") })
        else { return nil }
        return holding.min { $0.path.count < $1.path.count }
    }

    /// A folder's drive and free space, or nil when it did not answer.
    static func space(_ path: String) -> Stores.Volume? {
        let url = URL(fileURLWithPath: path, isDirectory: true)
        return path.hasPrefix("/Volumes/") ? timed(2, on: path, { Stores.volume(url) }) : Stores.volume(url)
    }

    /// `GET v1/fs` without a path: every root with its drive's free space, and the folders the settings name.
    static func places(_ settings: Settings) -> [String: Any] {
        func described(_ path: String, name: String? = nil) -> [String: Any] {
            var entry: [String: Any] = ["path": path, "name": name ?? (path as NSString).lastPathComponent]
            if let volume = space(path) {
                entry["drive"] = volume.name
                entry["free_bytes"] = volume.free
                entry["total_bytes"] = volume.total
            } else {
                entry["permission"] = permission(path)
            }
            return entry
        }
        return [
            "roots": roots(settings).map { described($0.path, name: $0.name) },
            "settings": [
                "stores": settings.stores.map { store in
                    described(store.path).merging(["size": store.size.map { $0 as Any } ?? NSNull()]) { $1 }
                },
                "hot_copy_dir": settings.hotCopyDir.map { described($0.path) } ?? NSNull(),
                "context_dir": described(settings.promptCacheRoot.path), "data_dir": settings.dataDir.path,
                "models_dirs": settings.modelsDirs.map { described($0.path) },
            ],
        ]
    }

    /// `GET v1/fs?path=`: the folders in `path` and, with `models`, the .gguf files (a split one by its first part) and
    /// which folders are checkpoints. A folder macOS has not allowed answers with `permission` (the steps) and no entries.
    static func list(_ asked: String, models: Bool, settings: Settings, search: String = "", offset: Int = 0) throws -> [String: Any] {
        let roots = roots(settings)
        let expanded = Settings.expand(asked)
        // A drive that does not answer is judged by the path as written (`..` already gone) and not listed: its links are
        // unresolved, so reading it then could follow one out of the places the browser shows.
        let answered = expanded.hasPrefix("/Volumes/") ? timed(3, on: expanded, { real(expanded) }) : .some(real(expanded))
        guard let path = answered ?? expanded else { throw ServingFailure("No folder at \(expanded)", status: 404) }
        guard let root = root(of: path, roots) else {
            throw ServingFailure("The folder browser shows drives, your home folder and Origami's folders only", status: 403)
        }
        let relative = path.dropFirst(root.path.count).split(separator: "/").map(String.init)
        var crumbs: [[String: Any]] = [["name": root.name, "path": root.path]]
        var walked = root.path
        for name in relative {
            walked += "/" + name
            crumbs.append(["name": name, "path": walked])
        }
        var out: [String: Any] = [
            "path": path, "root": root.path, "crumbs": crumbs,
            "parent": path == root.path ? NSNull() : (path as NSString).deletingLastPathComponent,
            "folders": [Any](), "models": [Any](),
        ]
        guard answered != nil else { return out.merging(["permission": permission(path)]) { $1 } }
        if let volume = space(path) {
            out["drive"] = volume.name
            out["free_bytes"] = volume.free
            out["total_bytes"] = volume.total
        }
        let names: [String]
        do {
            guard
                let listed = try timed(3, on: path, { Result(catching: { try FileManager.default.contentsOfDirectory(atPath: path) }) })?
                    .get()
            else {
                return out.merging(["permission": permission(path)]) { $1 }
            }
            names = listed
        } catch let error as NSError where error.code == NSFileReadNoPermissionError {
            return out.merging(["permission": permission(path)]) { $1 }
        } catch {
            // A file (ENOTDIR comes as an unknown read error), or a folder gone since: never the server's own failure.
            throw ServingFailure("\(path) is not a folder", status: 404)
        }
        out["writable"] = access(path, W_OK) == 0
        if models { out["model"] = Models.isCheckpoint(names) }
        // Which folders hold a model is read only on this folder's drive and outside the home folders macOS guards: a
        // look into Desktop, Documents or Downloads (or another drive) would ask for access before the user opened them.
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let guarded = Set(["Desktop", "Documents", "Downloads"].map { home + "/" + $0 })
        var here = stat()
        _ = stat(path, &here)
        var folders: [[String: Any]] = []
        var files: [[String: Any]] = []
        let visible = names.filter { !$0.hasPrefix(".") && (search.isEmpty || $0.localizedCaseInsensitiveContains(search)) }
            .sorted { $0.localizedStandardCompare($1) == .orderedAscending }
        let begin = max(0, offset)
        let page = Array(visible.dropFirst(begin).prefix(2000))
        if begin + page.count < visible.count { out["next_offset"] = begin + page.count }
        for name in page {
            let child = (path == "/" ? "" : path) + "/" + name
            var info = stat()
            guard lstat(child, &info) == 0 else { continue }
            if (info.st_mode & S_IFMT) == S_IFLNK {
                // A link is followed only into the places the browser shows (a broken one, or one into a drive that does
                // not answer, is left out), as the folder it names would be refused.
                guard let target = timed(1, { real(child) }) ?? nil, Self.root(of: target, roots) != nil, stat(target, &info) == 0 else {
                    continue
                }
            }
            if (info.st_mode & S_IFMT) == S_IFDIR {
                var entry: [String: Any] = ["name": name, "path": child]
                if models, info.st_dev == here.st_dev, !guarded.contains(child),
                    Models.isCheckpoint((try? FileManager.default.contentsOfDirectory(atPath: child)) ?? [])
                {
                    entry["model"] = true
                }
                folders.append(entry)
            } else if models, (info.st_mode & S_IFMT) == S_IFREG, name.lowercased().hasSuffix(".gguf"), !Models.isLaterPart(name) {
                files.append(["name": name, "path": child, "bytes": Int64(info.st_size)])
            }
        }
        out["folders"] = folders
        out["models"] = files
        return out
    }

    /// `POST v1/fs/mkdir {"path"}`: a new empty folder, its parent inside a root, its name a plain visible one (no
    /// `/`, `:`, control characters or leading dot, at most 255 bytes); answers the new folder's listing.
    static func makeFolder(_ asked: String, settings: Settings) throws -> [String: Any] {
        let trimmed = asked.count > 1 && asked.hasSuffix("/") ? String(asked.dropLast()) : asked
        // Already absolute and plain: `~`, `.` and `..` are the page's to resolve, not the server's to guess.
        guard trimmed.hasPrefix("/"), Settings.expand(trimmed) == trimmed else {
            throw ServingFailure("path must be a full path to the new folder")
        }
        let name = (trimmed as NSString).lastPathComponent
        let parent = (trimmed as NSString).deletingLastPathComponent
        guard name.utf8.count <= 255, name.trimmingCharacters(in: .whitespaces) == name,
            name.range(of: "^[^./:\\p{Cc}][^/:\\p{Cc}]*$", options: .regularExpression) != nil
        else {
            throw ServingFailure("A folder name is letters, digits, spaces and punctuation other than / and :, not starting with a dot")
        }
        if parent.hasPrefix("/Volumes/"), !Stores.answers(URL(fileURLWithPath: parent), within: 3) {
            throw ServingFailure(permission(parent), status: 403)
        }
        guard let real = real(parent) else { throw ServingFailure("No folder at \(parent)", status: 404) }
        guard root(of: real, roots(settings)) != nil else {
            throw ServingFailure("New folders go inside a drive, your home folder or Origami's folders", status: 403)
        }
        guard mkdir(real + "/" + name, 0o755) == 0 else {
            switch errno {
            case EEXIST: throw ServingFailure("\(name) already exists there", status: 409)
            case EACCES, EPERM: throw ServingFailure(permission(real), status: 403)
            default: throw ServingFailure("Cannot make \(name) there (\(String(cString: strerror(errno))))")
            }
        }
        return try list(real + "/" + name, models: false, settings: settings)
    }
}
