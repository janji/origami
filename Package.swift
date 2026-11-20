// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Origami",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "OrigamiEngine", targets: ["OrigamiEngine"]),
        .executable(name: "origami", targets: ["origami"]),
        .executable(name: "engine-checks", targets: ["EngineChecks"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers.git", exact: "1.3.3"),
        // swift-transformers' template engine, pinned to the version it resolves to, for Python's dict() (TokenizerStack).
        .package(url: "https://github.com/huggingface/swift-jinja.git", exact: "2.4.2"),
    ],
    targets: [
        .target(name: "CExactIO", publicHeadersPath: "include", cSettings: [.define("ACCELERATE_NEW_LAPACK")]),
        .target(
            name: "OrigamiEngine",
            dependencies: [
                "CExactIO", .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Jinja", package: "swift-jinja"),
            ],
            linkerSettings: [.linkedFramework("Metal")]),
        .executableTarget(name: "origami", dependencies: ["OrigamiEngine"]),
        // Self-checks and check fixtures that need the engine's package-level internals; only engine-checks links them.
        .target(name: "OrigamiEngineChecks", dependencies: ["OrigamiEngine", "CExactIO"]),
        // A plain executable that prints the checks as JSON, so they build with the Command Line Tools alone (no XCTest or Swift Testing there).
        .executableTarget(name: "EngineChecks", dependencies: ["OrigamiEngine", "OrigamiEngineChecks", "CExactIO", "CDeviceProbe"], path: "Tests/EngineChecks"),
        // Counts Metal device requests in engine-checks, so a host without a usable GPU skips only the checks that use it.
        .target(name: "CDeviceProbe", path: "Tests/CDeviceProbe"),
    ]
)
