import Darwin
import Foundation
import SystemConfiguration

/// Which `Host` a request may name (DNS rebinding) and which page may change settings: one rule for the front and the
/// model's server behind it, which sees the original Host through the pipe. Loopback names always; with `listen lan`
/// also this Mac's current interface addresses and its Bonjour name (`<LocalHostName>.local`, or the bare name). No
/// `.local` wildcard: any device on the network can claim a `.local` name.
public enum HostGuard {
    static let loopback: Set<String> = ["127.0.0.1", "localhost", "[::1]"]

    /// A Host value's name: lowercased, without a numeric port or one trailing dot; IPv6 stays in brackets.
    public static func name(_ host: String) -> String {
        var name = Substring(host.lowercased())
        if let colon = name.lastIndex(of: ":"), name[name.index(after: colon)...].utf8.allSatisfy({ (48...57).contains($0) }),
            !name[..<colon].contains(":") || name[..<colon].hasSuffix("]")
        {
            name = name[..<colon]
        }
        if name.hasSuffix(".") { name = name.dropLast() }
        return String(name)
    }

    /// Whether a request naming `host` is for this Mac. `mac` lists the names and addresses allowed besides loopback.
    public static func allows(_ host: String, lan: Bool, mac: Set<String>) -> Bool {
        let name = name(host)
        return loopback.contains(name) || lan && mac.contains(name)
    }

    /// Whether `host` is a loopback name. A request from this Mac naming its network name or address instead counts as
    /// another device's (a proxy forwarding the network, or a rebinding of that name to 127.0.0.1).
    public static func isLoopback(_ host: String) -> Bool { loopback.contains(name(host)) }

    /// The same with this Mac's current names and addresses, read again on a miss (DHCP and Wi-Fi change them).
    public static func allows(_ host: String, lan: Bool) -> Bool {
        guard lan else { return allows(host, lan: false, mac: []) }
        if allows(host, lan: true, mac: cached.withLock { $0 }) { return true }
        let fresh = macNames()
        cached.withLock { $0 = fresh }
        return allows(host, lan: true, mac: fresh)
    }
    private static let cached = Locked<Set<String>>([])

    /// Whether a write (a settings save, a reload, a job) comes from this server's own page: no `Origin` (apps, curl),
    /// or the page's origin is the Host it asked for, over http or https (also behind a proxy, a Tailscale https name
    /// or on the network). A JSON content type, which the callers also require, forces a browser's cross-site preflight.
    public static func ownPage(origin: String?, host: String) -> Bool {
        guard let origin = origin?.lowercased() else { return true }
        let host = host.lowercased()
        return !host.isEmpty && (origin == "http://" + host || origin == "https://" + host)
    }

    /// `<LocalHostName>.local`, the bare name and this Mac's interface addresses (IPv4 as written, IPv6 in brackets,
    /// without a zone), lowercased.
    public static func macNames() -> Set<String> {
        var names = Set(addresses().map { $0.contains(":") ? "[" + $0 + "]" : $0 })
        if let name = localHostName()?.lowercased(), !name.isEmpty { names.formUnion([name, name + ".local"]) }
        return names
    }

    /// This Mac's Bonjour name (System Settings › General › Sharing › Local hostname), such as `MINI`.
    public static func localHostName() -> String? { SCDynamicStoreCopyLocalHostName(nil) as String? }

    /// Every address of an interface that is up, loopback included (the caller filters), in interface order.
    public static func addresses(ipv6: Bool = true) -> [String] {
        var list: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&list) == 0, let first = list else { return [] }
        defer { freeifaddrs(list) }
        var found: [String] = []
        for entry in sequence(first: first, next: { $0.pointee.ifa_next }) {
            guard let address = entry.pointee.ifa_addr, entry.pointee.ifa_flags & UInt32(IFF_UP) != 0 else { continue }
            let family = Int32(address.pointee.sa_family)
            guard family == AF_INET || ipv6 && family == AF_INET6 else { continue }
            var text = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            guard getnameinfo(address, socklen_t(address.pointee.sa_len), &text, socklen_t(text.count), nil, 0, NI_NUMERICHOST) == 0 else {
                continue
            }
            let name =
                String(decoding: text.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self).split(separator: "%").first.map(
                    String.init) ?? ""
            if !name.isEmpty && !found.contains(name) { found.append(name) }
        }
        return found
    }
}

/// A value behind a lock.
final class Locked<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Value
    init(_ value: Value) { self.value = value }
    func withLock<T>(_ body: (inout Value) -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body(&value)
    }
}
