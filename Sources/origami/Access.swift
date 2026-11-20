import CoreImage
import Darwin
import Foundation
import OrigamiEngine

/// `listen lan`: where other devices open the dashboard, its QR code, and the front restarting itself to apply it.
enum Access {
    /// `http://<LocalHostName>.local:<port>/dashboard`, then the same at each network IPv4 address (not loopback or
    /// link-local), in interface order (Ethernet and Wi-Fi first).
    static func urls(port: Int) -> [String] {
        let name = HostGuard.localHostName().map { ["http://\($0).local:\(port)/dashboard"] } ?? []
        let addresses = HostGuard.addresses(ipv6: false).filter { !$0.hasPrefix("127.") && !$0.hasPrefix("169.254.") }
        return name + addresses.map { "http://\($0):\(port)/dashboard" }
    }

    /// A QR code of `text` as a `data:` PNG URL, black modules on white, 8 pixels a module (CoreImage's generator).
    static func qr(_ text: String) -> String? {
        guard let filter = CIFilter(name: "CIQRCodeGenerator") else { return nil }
        filter.setValue(Data(text.utf8), forKey: "inputMessage")
        filter.setValue("M", forKey: "inputCorrectionLevel")
        guard let image = filter.outputImage?.samplingNearest().transformed(by: CGAffineTransform(scaleX: 8, y: 8)),
            let space = CGColorSpace(name: CGColorSpace.sRGB),
            let png = CIContext(options: [.useSoftwareRenderer: true]).pngRepresentation(of: image, format: .RGBA8, colorSpace: space)
        else { return nil }
        return "data:image/png;base64," + png.base64EncodedString()
    }

    /// The launchd job this process is, which `launchctl kickstart -k` restarts: `XPC_SERVICE_NAME`, when launchd says
    /// that job's process is this one (a terminal's own job name is inherited by what it starts, and is not ours).
    static let launchdLabel: String? = {
        guard let label = ProcessInfo.processInfo.environment["XPC_SERVICE_NAME"], !label.isEmpty, label != "0",
            let printed = run("/bin/launchctl", ["print", "gui/\(getuid())/\(label)"]),
            printed.split(separator: "\n").contains(where: { $0.trimmingCharacters(in: .whitespaces) == "pid = \(getpid())" })
        else { return nil }
        return label
    }()

    /// Asks launchd to stop this server (SIGTERM: running requests finish, as on any stop) and start it again.
    static func restart() throws {
        guard let label = launchdLabel else {
            throw ServingFailure("Origami was not started by launchd: restart origami serve to apply it", status: 409)
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = ["kickstart", "-k", "gui/\(getuid())/\(label)"]
        try process.run()
    }

    private static func run(_ path: String, _ arguments: [String]) -> String? {
        let process = Process()
        let out = Pipe()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = arguments
        process.standardOutput = out
        process.standardError = FileHandle.nullDevice
        guard (try? process.run()) != nil else { return nil }
        let data = out.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return process.terminationStatus == 0 ? String(decoding: data, as: UTF8.self) : nil
    }
}
