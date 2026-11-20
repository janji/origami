import CoreGraphics
import Foundation
import ImageIO
import OrigamiEngine

/// A little-endian float32 `.npy` array in C order.
private func npyFloats(_ url: URL) throws -> (shape: [Int], values: [Float]) {
    let data = try Data(contentsOf: url)
    guard data.count > 12, data.prefix(6) == Data([0x93] + Array("NUMPY".utf8)) else { throw EngineError("\(url.path) is not a .npy file") }
    // Version 1 has a 2-byte header length, versions 2 and 3 a 4-byte one.
    let headerLength =
        data[6] == 1 ? Int(data[8]) | Int(data[9]) << 8 : Int(data[8]) | Int(data[9]) << 8 | Int(data[10]) << 16 | Int(data[11]) << 24
    let start = data[6] == 1 ? 10 : 12
    guard data.count >= start + headerLength else { throw EngineError("\(url.path) is shorter than its .npy header") }
    let header = String(decoding: data[start..<(start + headerLength)], as: UTF8.self)
    guard header.contains("'<f4'"), !header.contains("'fortran_order': True"),
        let open = header.range(of: "'shape': ("), let close = header[open.upperBound...].firstIndex(of: ")")
    else {
        throw EngineError("\(url.path): unsupported .npy header \(header)")
    }
    let shape = header[open.upperBound..<close].split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    let payload = data[(start + headerLength)...]
    let values = payload.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    guard values.count == shape.reduce(1, *) else { throw EngineError("\(url.path): \(values.count) values for shape \(shape)") }
    return (shape, values)
}

/// The vision encoder against the transformers reference on a test image: every merged token within float
/// tolerance (cosine at least 0.9999); its memory goes when it does, without an autorelease pool around it.
/// Needs ORIGAMI_VISION_REFERENCE (a folder with the reference's test-image.png, pixel-values.npy,
/// vision-output.npy and reference.json) and ORIGAMI_MODEL (a model folder with the vision tower).
func visionEncoderMatchesReference() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let reference = environment["ORIGAMI_VISION_REFERENCE"], let modelPath = environment["ORIGAMI_MODEL"] else {
        throw FixtureUnavailable(description: "set ORIGAMI_VISION_REFERENCE and ORIGAMI_MODEL to run the vision reference check")
    }
    let folder = URL(fileURLWithPath: reference, isDirectory: true)
    let model = try ModelSource(directory: URL(fileURLWithPath: modelPath, isDirectory: true))
    let ledger = try MemoryLedger(ceiling: 2_000_000_000, transientMargin: 100_000_000)
    let footprint = ledger.snapshot().physicalFootprint
    try visionEncoderRuns(model: model, ledger: ledger, folder: folder)
    Thread.sleep(forTimeInterval: 1)  // Metal returns some pages a moment after release
    let after = ledger.snapshot().physicalFootprint
    FileHandle.standardError.write(
        Data(
            String(
                format: "vision: footprint %.2f GB before, %.2f GB after the encoder is gone\n", Double(footprint) / 1e9,
                Double(after) / 1e9
            ).utf8))
    guard after < footprint + 400_000_000 else { throw EngineError("The vision encoder left \(after - footprint) bytes behind") }
}

private func visionEncoderRuns(model: ModelSource, ledger: MemoryLedger, folder: URL) throws {
    let encoder = try VisionEncoder(model: model, ledger: ledger)
    let pixels = try npyFloats(folder.appendingPathComponent("pixel-values.npy"))
    let expected = try npyFloats(folder.appendingPathComponent("vision-output.npy"))
    let info = try JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent("reference.json"))) as! [String: Any]
    let thw = (info["grid_thw"] as! [[Int]])[0]
    let started = Date()
    let actual = try encoder.encode(patches: pixels.values, rows: thw[1], columns: thw[2])
    let seconds = Date().timeIntervalSince(started)
    guard actual.count == expected.values.count else {
        throw EngineError("Vision output has \(actual.count) values, reference \(expected.values.count)")
    }
    var worst = 1.0
    var largest: Float = 0
    for token in 0..<expected.shape[0] {
        let a = actual[(token * 2560)..<((token + 1) * 2560)]
        let b = expected.values[(token * 2560)..<((token + 1) * 2560)]
        var dot = 0.0
        var na = 0.0
        var nb = 0.0
        for (x, y) in zip(a, b) {
            dot += Double(x) * Double(y)
            na += Double(x) * Double(x)
            nb += Double(y) * Double(y)
            largest = max(largest, abs(x - y))
        }
        worst = min(worst, dot / (na.squareRoot() * nb.squareRoot()))
    }
    FileHandle.standardError.write(
        Data(
            String(
                format: "vision: %d tokens in %.2f s, worst cosine %.6f, largest difference %.4g\n", expected.shape[0], seconds, worst,
                largest
            ).utf8))
    guard worst >= 0.9999 else {
        throw EngineError(String(format: "Vision encoder differs from the reference: worst token cosine %.6f", worst))
    }
    // The largest image Origami accepts (1,048,576 pixels: 4,096 patches), for time and peak memory.
    let capped = Date()
    let large = try encoder.encode(patches: [Float](repeating: 0.1, count: 4096 * VisionEncoder.patchValues), rows: 64, columns: 64)
    let again = Date()
    _ = try encoder.encode(patches: [Float](repeating: 0.2, count: 4096 * VisionEncoder.patchValues), rows: 64, columns: 64)
    FileHandle.standardError.write(
        Data(
            String(
                format: "vision: 4096 patches -> %d tokens in %.2f s (%.2f s with the graph compiled)\n",
                large.count / VisionEncoder.output, again.timeIntervalSince(capped), Date().timeIntervalSince(again)
            ).utf8))
}

/// M-RoPE positions follow the reference's rule (text sequential, an image's grid at (p, p+row, p+column),
/// the text after it at p + max(rows, columns)), and the pinned RoPE tables give every rotary pair the
/// same cos/sin a text token at that section's position gets, bit for bit; without images nothing changes.
func mropePositionsAndTables() throws {
    func require(_ value: Bool, _ message: @autoclosure () -> String) throws { if !value { throw EngineError(message()) } }
    // 3 text tokens, a 2 x 3 image at 3..<9, 2 text tokens, a 4 x 1 image at 11..<15, 1 text token.
    let map = RoPEPositionMap(images: [.init(start: 3, rows: 2, columns: 3), .init(start: 11, rows: 4, columns: 1)])
    var expected: [(Int, Int, Int)] = (0..<3).map { ($0, $0, $0) }
    for row in 0..<2 { for column in 0..<3 { expected.append((3, 3 + row, 3 + column)) } }
    expected += [(6, 6, 6), (7, 7, 7)]  // 3 + max(2, 3)
    for row in 0..<4 { expected.append((8, 8 + row, 8)) }
    expected.append((12, 12, 12))  // 8 + max(4, 1)
    for (index, value) in expected.enumerated() {
        let p = map.position(index)
        try require(p.t == value.0 && p.h == value.1 && p.w == value.2, "Token \(index) at \(p), expected \(value)")
    }
    try require(map.position(100) == (97, 97, 97), "Decode positions after images shift by the images' max(rows, columns) - count")
    let ledger = try MemoryLedger(ceiling: 64_000_000, transientMargin: 1)
    try autoreleasepool {
        let rope = try PinnedRoPE(ledger: ledger)
        let indices = Array(0..<16)
        let plain = try rope.tables(positions: indices)
        rope.positionMap = RoPEPositionMap(images: [])
        let empty = try rope.tables(positions: indices)
        try require(empty.cosines == plain.cosines && empty.sines == plain.sines, "An empty map changed text tables")
        rope.positionMap = map
        let mapped = try rope.tables(positions: indices)
        rope.positionMap = nil
        for index in indices {
            let p = map.position(index)
            let components = [p.t, p.h, p.w]
            for k in 0..<32 {
                let scalar = try rope.tables(positions: [components[k % 3 == 1 && k < 33 ? 1 : (k % 3 == 2 && k < 30 ? 2 : 0)]])
                try require(
                    mapped.cosines[index * 32 + k].bitPattern == scalar.cosines[k].bitPattern
                        && mapped.sines[index * 32 + k].bitPattern == scalar.sines[k].bitPattern, "Token \(index) pair \(k) differs")
            }
        }
    }
    try require(ledger.snapshot().reserved == 0, "RoPE tables leaked")
}

/// Image preprocessing: smart_resize agrees with transformers' on edge cases; a generated PNG decodes to
/// its patch grid upright with channels in order; markers expand to rows/2 x columns/2 tokens placed in the
/// position map. With ORIGAMI_VISION_REFERENCE, the reference test image's patches match the reference
/// preprocessor's pixel values.
func promptImagePreprocessing() throws {
    func require(_ value: Bool, _ message: @autoclosure () -> String) throws { if !value { throw EngineError(message()) } }
    // transformers' smart_resize(h, w, 32, 65536, 1048576)
    let cases: [((Int, Int), (Int, Int))] = [
        ((480, 640), (480, 640)), ((100, 100), (256, 256)), ((4000, 3000), (1152, 864)),
        ((1080, 1920), (768, 1344)), ((33, 47), (224, 320)), ((2000, 20), (2560, 32)), ((720, 1280), (704, 1280)),
        ((1, 150), (32, 3136)), ((1056, 1008), (1024, 992)), ((5000, 40), (4992, 32)),
    ]
    for (input, expected) in cases {
        let actual = try PromptImage.smartResize(height: input.0, width: input.1)
        try require(actual.height == expected.0 && actual.width == expected.1, "smart_resize\(input) = \(actual), expected \(expected)")
    }
    // A 100 x 60 PNG in four colours (red, blue / green, white) becomes 224 x 352 pixels: 14 x 22 patches.
    let context = CGContext(
        data: nil, width: 100, height: 60, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    for (x, y, r, g, b) in [(0, 30, 1.0, 0.0, 0.0), (50, 30, 0.0, 0.0, 1.0), (0, 0, 0.0, 1.0, 0.0), (50, 0, 1.0, 1.0, 1.0)] {
        context.setFillColor(red: r, green: g, blue: b, alpha: 1)
        context.fill(CGRect(x: x, y: y, width: 50, height: 30))
    }
    let png = NSMutableData()
    let destination = CGImageDestinationCreateWithData(png, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(destination, context.makeImage()!, nil)
    try require(CGImageDestinationFinalize(destination), "PNG encoding failed")
    let image = try PromptImage(dataURL: "data:image/png;base64," + (png as Data).base64EncodedString())
    let patches = try image.patches()
    try require(image.rows == 14 && image.columns == 22 && patches.count == 14 * 22 * 1536, "PNG grid \(image.rows) x \(image.columns)")
    let blocks = image.columns / 2
    for (patch, colour, corner) in [
        (0, [1, -1, -1], "top left red"), ((blocks - 1) * 4 + 1, [-1, -1, 1], "top right blue"),
        (((image.rows / 2 - 1) * blocks) * 4 + 2, [-1, 1, -1], "bottom left green"),
        (image.rows * image.columns - 1, [1, 1, 1], "bottom right white"),
    ] {
        for channel in 0..<3 {
            let value = patches[patch * 1536 + channel * 512 + 8 * 16 + 8]
            try require(abs(value - Float(colour[channel])) < 0.02, "\(corner): channel \(channel) is \(value)")
        }
    }
    let pad = 248_056
    let expanded = try PromptImage.expand([1, 2, pad, 3, pad, 4], pad: pad, images: [image, image])
    try require(
        expanded.prompt.count == 4 + 2 * 77 && expanded.prompt.filter { $0 == pad }.count == 154 && expanded.prompt[79] == 3,
        "Expanded prompt \(expanded.prompt.count)")
    try require(
        expanded.map.images == [.init(start: 2, rows: 7, columns: 11), .init(start: 80, rows: 7, columns: 11)],
        "Image placement \(expanded.map.images)")
    try require(expanded.map.position(78) == (2, 8, 12) && expanded.map.position(79) == (13, 13, 13), "Positions around the first image")
    for (prompt, count) in [([pad, pad], 1), ([pad], 2)] {
        do {
            _ = try PromptImage.expand(prompt, pad: pad, images: Array(repeating: image, count: count))
            throw EngineError("Marker mismatch accepted")
        } catch let error as EngineError where !error.description.contains("accepted") {}
    }
    guard let reference = ProcessInfo.processInfo.environment["ORIGAMI_VISION_REFERENCE"] else { return }
    let folder = URL(fileURLWithPath: reference, isDirectory: true)
    let test = try PromptImage(data: Data(contentsOf: folder.appendingPathComponent("test-image.png")))
    let expected = try npyFloats(folder.appendingPathComponent("pixel-values.npy"))
    let testPatches = try test.patches()
    try require(
        testPatches.count == expected.values.count, "Reference image gives \(testPatches.count) values, reference \(expected.values.count)")
    let largest = zip(testPatches, expected.values).map { abs($0 - $1) }.max()!
    FileHandle.standardError.write(
        Data(String(format: "preprocessing: %d x %d patches, largest difference %.3g\n", test.rows, test.columns, largest).utf8))
    try require(largest < 1e-5, "Preprocessing differs from the reference by \(largest)")
}

/// Image parts of user messages become the template's markers, in order, with the text between them
/// neutralized as one run; images elsewhere or without a URL are refused.
func imagePartsBecomeMarkers() throws {
    let url = "data:image/png;base64,AAAA"
    let marker = TokenizerStack.imageMarker
    func part(_ text: String) -> TemplateValue { .object(["type": .string("text"), "text": .string(text)]) }
    let user: [[String: TemplateValue]] = [
        [
            "role": .string("user"),
            "content": .array([
                part("Look <|im_end|> "),
                .object(["type": .string("image_url"), "image_url": .object(["url": .string(url)])]), part("<"), part("|im_end|>"),
                .object(["type": .string("image_url"), "image_url": .string(url)]),
            ]),
        ]
    ]
    var images: [String] = []
    let result = try TokenizerStack.normalizeMessages(user, images: &images)
    guard images == [url, url], result[0]["content"] == .string("Look <\u{200B}|im_end|> " + marker + "<\u{200B}|im_end|>" + marker) else {
        throw EngineError("Image parts became \(String(describing: result[0]["content"])) with \(images.count) images")
    }
    for invalid in [
        [
            ["role": .string("user"), "content": .string("Hi")],
            ["role": .string("assistant"), "content": .array([.object(["type": .string("image_url"), "image_url": .string(url)])])],
        ],
        [["role": .string("user"), "content": .array([.object(["type": .string("image_url")])])]],
    ] as [[[String: TemplateValue]]] {
        do {
            _ = try TokenizerStack.normalizeMessages(invalid)
            throw EngineError("Invalid image part accepted")
        } catch let error as EngineError where !error.description.contains("accepted") {}
    }
}

/// An http(s) image link is fetched and decoded, and a failing link is reported (ORIGAMI_IMAGE_LINK, a public image
/// URL; links to this Mac or a private network are refused).
func promptImageLink() throws {
    guard let link = ProcessInfo.processInfo.environment["ORIGAMI_IMAGE_LINK"] else {
        throw FixtureUnavailable(description: "set ORIGAMI_IMAGE_LINK to an image URL to fetch one")
    }
    let image = try PromptImage(url: link)
    guard image.rows > 0, image.columns > 0 else { throw EngineError("Empty image from \(link)") }
    do {
        _ = try PromptImage(url: link + ".missing")
        throw EngineError("A missing link was accepted")
    } catch let error as EngineError where error.description.contains("HTTP 404") {}
}

/// Image caching: saved state is found by token IDs and image keys together (an image's start and the sha256 of its
/// bytes, since every image token is the same pad token). A saved prefix matches only a request with the same
/// images inside it and text entries are unchanged; only the images prefill still reads are encoded, and reading an
/// image token before its image is encoded is refused.
func promptImageIdentity() throws {
    func require(_ value: Bool, _ message: @autoclosure () -> String) throws { if !value { throw EngineError(message()) } }
    let a = ImageKey(start: 10, digest: "aa")
    let b = ImageKey(start: 10, digest: "bb")
    let late = ImageKey(start: 700, digest: "cc")
    let ids = (0..<1_000).map(Int64.init)
    let prefix = PromptCache.prefixHash(ids.prefix(600))
    let withImage = PromptCache.Entry(tokens: 600, prefix: prefix, used: 0, images: PromptCache.digest([a]))
    let text = PromptCache.Entry(tokens: 600, prefix: prefix, used: 0)
    try require(
        withImage.holds(ids.prefix(600), images: [a, late]) && !withImage.holds(ids.prefix(600), images: [b])
            && !withImage.holds(ids.prefix(600), images: []) && text.holds(ids.prefix(600), images: [late])
            && !text.holds(ids.prefix(600), images: [a]) && PromptCache.digest([]) == nil, "Saved prefixes and image keys")
    var images = try PromptImages(
        map: RoPEPositionMap(images: [.init(start: 2, rows: 1, columns: 2), .init(start: 6, rows: 1, columns: 1)]),
        digests: ["x", "y"])
    try require(
        images.keys == [ImageKey(start: 2, digest: "x"), ImageKey(start: 6, digest: "y")] && images.unencoded(from: 0) == [0, 1]
            && images.unencoded(from: 4) == [1] && images.unencoded(from: 7) == [], "Images still to encode")
    var embeddings = [Float](repeating: 0, count: 8 * 2560)
    do {
        try images.inject(into: &embeddings, start: 0)
        throw EngineError("An image token was read before its image was encoded")
    } catch let error as EngineError where error.description.contains("before it was encoded") {}
    try images.setFeatures([Float](repeating: 2, count: 2560), image: 1)
    embeddings = [Float](repeating: 0, count: 4 * 2560)
    try images.inject(into: &embeddings, start: 4)  // tokens 4..<8 hold only the second image, at 6
    try require(
        embeddings[2 * 2560] == 2 && embeddings[0] == 0 && embeddings[3 * 2560] == 0 && images.unencoded(from: 4) == [],
        "Restored first image, second injected")
    try images.setFeatures((0..<(2 * 2560)).map(Float.init), image: 0)
    embeddings = [Float](repeating: -1, count: 2 * 2560)
    try images.inject(into: &embeddings, start: 3)  // a scope that starts inside the first image takes its second token
    try require(embeddings[0] == 2560 && embeddings[2560] == -1, "A scope starting inside an image")
    func png(_ red: CGFloat) throws -> Data {
        let context = CGContext(
            data: nil, width: 64, height: 64, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(red: red, green: 0.5, blue: 0.5, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: 64, height: 64))
        let data = NSMutableData()
        let destination = CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, context.makeImage()!, nil)
        try require(CGImageDestinationFinalize(destination), "PNG encoding failed")
        return data as Data
    }
    let one = try png(0.2)
    let digests = try [one, one, png(0.8)].map { try PromptImage(data: $0).digest }
    try require(digests[0] == digests[1] && digests[0] != digests[2], "Image digests follow the bytes: \(digests)")
}

/// The .npy reader takes version 1's 2-byte header length and version 2's 4-byte one, including one whose low byte is 1.
func visionNpyHeaderVersions() throws {
    for (version, length) in [(UInt8(1), 118), (2, 65_537)] {
        var header = Array("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }".utf8)
        header += [UInt8](repeating: 0x20, count: length - header.count - 1) + [0x0A]
        var data = Data([0x93] + Array("NUMPY".utf8) + [version, 0])
        data += (0..<(version == 1 ? 2 : 4)).map { UInt8(length >> (8 * $0) & 0xFF) } + header
        data += [Float](arrayLiteral: 1, 2, 3, 4, 5, 6).withUnsafeBytes { Data($0) }
        let url = CheckFixtures.scratch.appendingPathComponent("header-v\(version).npy")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let array = try npyFloats(url)
        guard array.shape == [2, 3], array.values == [1, 2, 3, 4, 5, 6] else {
            throw EngineError("A version \(version) .npy read as \(array)")
        }
    }
}

/// The vision encoder checks every tower tensor's name, dtype, shape and byte length before it reserves or reads
/// anything, and refuses a mismatch as `model_incompatible`; a matching table gets past the check.
func visionEncoderRefusesMismatchedTensors() throws {
    func span(_ shape: [Int], length: Int? = nil) -> [String: Any] {
        ["file": "absent.safetensors", "dtype": "BF16", "offset": 0, "length": length ?? shape.reduce(2, *), "shape": shape]
    }
    var valid: [String: Any] = [:]
    for (name, shape) in VisionEncoder.shapes { valid["vision_tower." + name] = span(shape) }
    var renamed = valid
    var shortTable = valid
    var shortBuffer = valid
    renamed["vision_tower.merger.norm.gamma"] = renamed.removeValue(forKey: "vision_tower.merger.norm.weight")
    shortTable["vision_tower.pos_embed.weight"] = span([1_024, 1_152])
    shortBuffer["vision_tower.blocks.3.attn.qkv.weight"] = span([3_456, 1_152], length: 1_000)
    let ledger = try MemoryLedger(ceiling: 64_000_000, transientMargin: 1)
    for (spans, expectation) in [
        (renamed, "merger.norm.gamma"), (shortTable, "pos_embed.weight"), (shortBuffer, "attn.qkv.weight"), (valid, nil),
    ] {
        let model = try ModelSource(fixture: CheckFixtures.scratch, spans: spans)
        do {
            _ = try VisionEncoder(model: model, ledger: ledger)
            throw EngineError("A vision tower with no weight files loaded")
        } catch let error as EngineError where error.code == "model_incompatible" {
            guard let expectation, error.description.contains(expectation) else { throw EngineError("Vision tensor check: \(error)") }
        } catch let error as EngineError where expectation == nil && !error.description.contains("loaded") {}  // past the check
    }
    guard ledger.snapshot().reserved == 0 else { throw EngineError("A refused vision tower kept its reservation") }
}

/// The vision encoder runs small images (16 x 16 and 14 x 22 patches) and the largest (64 x 64) to finite output: MPSGraph
/// fused the attention into a kernel that failed to compile under 512 patches on macOS 26 and aborted the process. Zero
/// weights from a sparse file: the graph compiles the same whatever the values.
func visionEncoderRunsSmallAndLargeImages() throws {
    var spans: [String: Any] = [:]
    var offset = 0
    for (name, shape) in VisionEncoder.shapes {
        let length = shape.reduce(2, *)
        spans["vision_tower." + name] = ["file": "vision.safetensors", "dtype": "BF16", "offset": offset, "length": length, "shape": shape]
        offset += length
    }
    let url = CheckFixtures.scratch.appendingPathComponent("vision.safetensors")
    guard FileManager.default.createFile(atPath: url.path, contents: nil) else { throw EngineError("Cannot create \(url.path)") }
    defer { try? FileManager.default.removeItem(at: url) }
    let handle = try FileHandle(forWritingTo: url)
    try handle.truncate(atOffset: UInt64(offset))
    try handle.close()
    let encoder = try VisionEncoder(
        model: ModelSource(fixture: CheckFixtures.scratch, spans: spans), ledger: MemoryLedger(ceiling: 2_000_000_000, transientMargin: 1))
    for (rows, columns) in [(16, 16), (14, 22), (64, 64)] {
        let output = try encoder.encode(
            patches: [Float](repeating: 0.1, count: rows * columns * VisionEncoder.patchValues), rows: rows, columns: columns)
        guard output.count == rows * columns / 4 * VisionEncoder.output, output.allSatisfy(\.isFinite) else {
            throw EngineError("A \(rows) x \(columns) image encoded to \(output.count) values, finite: \(output.allSatisfy(\.isFinite))")
        }
    }
}

/// Only PNG, JPEG, HEIC/HEIF, GIF, WebP and TIFF are decoded, and a source of more than 2^26 pixels is refused from its
/// header before any decoding; image links follow redirects only on the same host and never from https to http.
func promptImageFormatsSizeAndRedirects() throws {
    func encoded(_ type: String, width: Int, height: Int) throws -> Data {
        let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)!
        context.setFillColor(gray: 0.5, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let data = NSMutableData()
        let destination = CGImageDestinationCreateWithData(data, type as CFString, 1, nil)!
        CGImageDestinationAddImage(destination, context.makeImage()!, nil)
        guard CGImageDestinationFinalize(destination) else { throw EngineError("\(type) encoding failed") }
        return data as Data
    }
    for type in ["public.png", "public.jpeg", "public.tiff", "com.compuserve.gif"] {
        _ = try PromptImage(data: encoded(type, width: 300, height: 200))
    }
    for (type, width, height, message) in [
        ("com.microsoft.bmp", 300, 200, "Unsupported image format"),
        ("com.adobe.photoshop-image", 300, 200, "Unsupported image format"),
        ("public.png", 8_200, 8_200, "at most 64 megapixels"),
    ] {
        do {
            _ = try PromptImage(data: encoded(type, width: width, height: height))
            throw EngineError("\(type) \(width) x \(height) was accepted")
        } catch let error as EngineError where error.description.contains(message) {}
    }
    let allowed = [
        ("https://a.example/x", "https://a.example/y"), ("http://a.example/x", "https://A.example:8443/y"),
        ("https://github.com/x", "https://raw.githubusercontent.com/y"), ("http://a.example/x", "http://b.example/y"),
    ]
    let refused = [
        ("https://a.example/x", "http://a.example/y"), ("https://a.example/x", "http://b.example/y"),
        ("https://a.example/x", "file:///etc/passwd"), ("http://a.example/x", "ftp://b.example/y"),
    ]
    for (from, to) in allowed + refused {
        guard PromptImage.followsRedirect(from: URL(string: from), to: URL(string: to)) == allowed.contains(where: { $0 == (from, to) })
        else {
            throw EngineError("Redirect from \(from) to \(to) decided wrongly")
        }
    }
}

/// Image links to this Mac and private networks are refused with a coded 400 before anything is fetched, whatever
/// form the address takes (names, IPv6, IPv4 inside IPv6, decimal and hexadecimal IPv4); redirects run the same
/// check; the address a connection reached is classified the same way.
func promptImageLinksRefuseInternalAddresses() throws {
    func require(_ value: Bool, _ message: @autoclosure () -> String) throws { if !value { throw EngineError(message()) } }
    let refusedAddresses = [
        "127.0.0.1", "127.1.2.3", "10.0.0.1", "172.16.0.1", "172.31.255.255", "192.168.1.1", "169.254.169.254", "100.64.0.1",
        "0.0.0.0", "224.0.0.1", "255.255.255.255", "::", "::1", "fe80::1%en0", "fec0::1", "fc00::1", "fd12:3456::1", "ff02::1",
        "::ffff:127.0.0.1", "::ffff:10.0.0.1", "::10.0.0.1", "64:ff9b::a00:1", "2002:c0a8:101::1", "not an address",
    ]
    let external = [
        "8.8.8.8", "1.1.1.1", "172.32.0.1", "192.169.0.1", "100.128.0.1", "2606:4700:4700::1111", "::ffff:8.8.8.8",
        "2002:808:808::1", "64:ff9b::808:808",
    ]
    for address in refusedAddresses { try require(PromptImage.isInternal(address: address), "\(address) was taken as public") }
    for address in external { try require(!PromptImage.isInternal(address: address), "\(address) was taken as internal") }
    for link in [
        "http://127.0.0.1:9/x.png", "http://localhost:9/x.png", "http://[::1]:9/x.png", "https://2130706433/x.png",
        "http://0x7f000001/x.png",
        "http://169.254.169.254/latest/meta-data", "http://[::ffff:127.0.0.1]/x.png", "http://10.1.2.3/x.png", "http://192.168.0.1/x.png",
    ] {
        var failure: ServingFailure?
        do { _ = try PromptImage(url: link) } catch let error as ServingFailure { failure = error }
        try require(
            failure?.code == "image_link_refused" && failure?.status == 400, "\(link) was not refused: \(String(describing: failure))")
        try require(
            failure.map { ServingFailure.promptEncoding($0).code } == "image_link_refused", "The refusal lost its code on the way out")
    }
    do {
        try PromptImage.checkHost(URL(string: "https://127.0.0.1/redirected.png")!)
        throw EngineError("A redirect target on this Mac passed")
    } catch let error as ServingFailure where error.code == "image_link_refused" {}
}
