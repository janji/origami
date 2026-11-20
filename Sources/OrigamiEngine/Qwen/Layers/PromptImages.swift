import CoreGraphics
import CryptoKit
import Darwin
import Foundation
import ImageIO

/// Where a prompt's image tokens sit and the M-RoPE positions they take. The language model's rotary
/// pairs are interleaved temporal/height/width sections ([11, 11, 10]); a text token has all three equal to
/// its position. An image of `rows` x `columns` merged tokens that starts at position p takes (p, p + row,
/// p + column), and the text after it continues at p + max(rows, columns), so positions stop equalling token
/// indices after the first image.
public struct RoPEPositionMap: Sendable, Equatable {
    /// One image's run of merged tokens: `rows` x `columns` tokens in row-major order from token index `start`.
    public struct Image: Sendable, Equatable {
        /// Token index of the image's first token and its merged grid.
        public let start, rows, columns: Int
        public init(start: Int, rows: Int, columns: Int) {
            self.start = start
            self.rows = rows
            self.columns = columns
        }
        public var count: Int { rows * columns }
    }
    public let images: [Image]
    public init(images: [Image]) { self.images = images.sorted { $0.start < $1.start } }

    /// (temporal, height, width) position of token `index`.
    public func position(_ index: Int) -> (t: Int, h: Int, w: Int) {
        var shift = 0  // position minus token index after the images passed so far
        for image in images {
            if index < image.start { break }
            if index < image.start + image.count {
                let base = image.start + shift
                let local = index - image.start
                return (base, base + local / image.columns, base + local % image.columns)
            }
            shift += max(image.rows, image.columns) - image.count
        }
        return (index + shift, index + shift, index + shift)
    }
    /// Which position rotary pair `k` (of 32) takes: 0 temporal, 1 height, 2 width.
    static func component(_ k: Int) -> Int { k % 3 == 1 && k < 33 ? 1 : (k % 3 == 2 && k < 30 ? 2 : 0) }
    /// True when some of `indices` sit at or after the first image, where positions differ from indices.
    func affects(_ indices: [Int]) -> Bool {
        guard let first = images.first?.start else { return false }
        return indices.contains { $0 >= first }
    }
}

/// What saved state must match for an image: where its tokens start and a digest of its bytes. Token IDs
/// alone do not identify an image, since every image token is the same pad token.
package struct ImageKey: Codable, Equatable, Sendable {
    package let start: Int, digest: String
    package init(start: Int, digest: String) {
        self.start = start
        self.digest = digest
    }
    /// The images that start before token `count`: the ones a saved prefix of that length holds.
    package static func below(_ keys: [ImageKey], _ count: Int) -> [ImageKey] { keys.filter { $0.start < count } }
}

/// A prompt's images: their positions, keys, and the vision tower's merged tokens (2560 values each per image).
public struct PromptImages: Sendable {
    public let map: RoPEPositionMap
    package let keys: [ImageKey]
    /// Empty until the image is encoded; an image whose tokens are restored from saved state is never encoded.
    package private(set) var features: [[Float]]
    package init(map: RoPEPositionMap, digests: [String]) throws {
        guard digests.count == map.images.count else { throw EngineError("Image digests do not match the images") }
        self.map = map
        keys = zip(map.images, digests).map { ImageKey(start: $0.start, digest: $1) }
        features = Array(repeating: [], count: map.images.count)
    }
    /// The images with tokens at or after `offset`, which prefill still reads, that are not encoded yet.
    package func unencoded(from offset: Int) -> [Int] {
        map.images.indices.filter { map.images[$0].start + map.images[$0].count > offset && features[$0].isEmpty }
    }
    package mutating func setFeatures(_ values: [Float], image: Int) throws {
        guard values.count == map.images[image].count * 2560 else { throw EngineError("Image features do not match the image tokens") }
        features[image] = values
    }
    /// Replaces the embeddings of the image tokens in a scope of tokens starting at prompt index `start`.
    package func inject(into embeddings: inout [Float], start: Int) throws {
        let end = start + embeddings.count / 2560
        for (index, image) in map.images.enumerated() where image.start < end && image.start + image.count > start {
            guard features[index].count == image.count * 2560 else { throw EngineError("An image was read before it was encoded") }
            for token in max(start, image.start)..<min(end, image.start + image.count) {
                let row = token - start
                let source = (token - image.start) * 2560
                embeddings.replaceSubrange((row * 2560)..<((row + 1) * 2560), with: features[index][source..<(source + 2560)])
            }
        }
    }
}

/// One image of a prompt: its bytes, its sha256 and its patch grid, read from the file's header when the request
/// arrives. `patches()` decodes it and cuts it into the vision tower's patches the way the checkpoint's
/// preprocessor (Qwen2-VL's) does: sides rounded to multiples of 32 within 65,536 to 1,048,576 pixels
/// (Origami's cap; the checkpoint allows 16.8M, far beyond an 8K context), RGB over white scaled to
/// [-1, 1], the frame doubled for the temporal pair, 16 x 16 patches in 2 x 2 merge-block order. Only the images
/// prefill still reads are decoded (those inside restored state are not).
public struct PromptImage: Sendable {
    public static let minimumPixels = 65_536, maximumPixels = 1_048_576, factor = 32
    /// The largest source image decoded (2^26 pixels, so a "64 MP" camera's 9248 x 6936 fits).
    static let maximumSourcePixels = 1 << 26
    /// The formats the server decodes (ImageIO type identifiers); ImageIO's many others stay out of the serving process.
    static let formats: Set<String> = [
        "public.png", "public.jpeg", "public.heic", "public.heif", "com.compuserve.gif",
        "org.webmproject.webp", "public.tiff",
    ]
    /// The patch grid; the language model sees (rows / 2) x (columns / 2) tokens.
    public let rows, columns: Int
    /// The image file as sent or fetched.
    public let data: Data
    /// Images one request may decode (about 800 MB of patches at full size).
    public static let maximumDecoded = 32
    /// The largest image file a link may bring.
    public static let maximumLinkBytes = 24 * 1_024 * 1_024
    /// Refuses decoding more than `maximumDecoded` images. Only the images prefill still reads count: a long chat's
    /// earlier images live in its restored state (code `too_many_images`).
    package static func checkDecoding(_ needed: Int, of total: Int) throws {
        guard needed > maximumDecoded else { return }
        throw ServingFailure(
            "This request needs \(needed) images read" + (total > needed ? " (\(total - needed) more are in its saved state)" : "")
                + "; at most \(maximumDecoded) a request", code: "too_many_images")
    }
    /// sha256 of the image's bytes, so saved state that holds this image can be found again.
    public let digest: String

    /// The URL of an OpenAI `image_url` content part: `{"url": …}` or a bare string.
    static func url(of part: [String: TemplateValue]) -> String? {
        switch part["image_url"] {
        case .string(let url): return url
        case .object(let object):
            if case .string(let url) = object["url"] { return url }
            return nil
        default: return nil
        }
    }

    /// A base64 `data:` URL, or an http(s) link fetched within a minute and 24 MB, following redirects but never from
    /// https to http, and never to this Mac or a private network (`checkHost`). Files are not read: the server would
    /// read them for any local app, sandboxed or not.
    public init(url: String) throws {
        if url.hasPrefix("data:") {
            try self.init(dataURL: url)
            return
        }
        guard let link = URL(string: url), ["http", "https"].contains(link.scheme?.lowercased() ?? "") else {
            throw EngineError("Send images as base64 data: URLs or http(s) links")
        }
        try Self.checkHost(link)
        try self.init(data: Self.fetch(link))
    }

    /// The refusal of a link to this Mac or a private network (code `image_link_refused`).
    static func refusedLink(_ host: String) -> ServingFailure {
        ServingFailure(
            "The image link's host \(host) is this Mac or a private or link-local network address; Origami fetches only public "
                + "links (send a private image as a base64 data: URL)", code: "image_link_refused")
    }
    /// Refuses a link whose host is, or resolves to, a loopback, private, shared, link-local, multicast or unspecified
    /// address (IPv4, IPv6 and IPv4 inside IPv6), or that does not resolve. The address actually connected to is
    /// checked again after the fetch (`fetch`), since a name may resolve differently the second time.
    package static func checkHost(_ link: URL) throws {
        let host = (link.host ?? "").trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        guard !host.isEmpty else { throw EngineError("The image link has no host") }
        var hints = addrinfo()
        var list: UnsafeMutablePointer<addrinfo>?
        hints.ai_family = AF_UNSPEC
        hints.ai_socktype = SOCK_STREAM
        guard getaddrinfo(host, nil, &hints, &list) == 0, let first = list else {
            throw EngineError("Could not fetch the image link: its host \(host) does not resolve")
        }
        defer { freeaddrinfo(first) }
        var entry: UnsafeMutablePointer<addrinfo>? = first
        while let current = entry {
            if let address = current.pointee.ai_addr, Self.isInternal(socketAddress: address) { throw refusedLink(host) }
            entry = current.pointee.ai_next
        }
    }
    static func isInternal(socketAddress address: UnsafePointer<sockaddr>) -> Bool {
        switch Int32(address.pointee.sa_family) {
        case AF_INET:
            return address.withMemoryRebound(to: sockaddr_in.self, capacity: 1) { pointer in
                withUnsafeBytes(of: pointer.pointee.sin_addr) { isInternal(Array($0)) }
            }
        case AF_INET6:
            return address.withMemoryRebound(to: sockaddr_in6.self, capacity: 1) { pointer in
                withUnsafeBytes(of: pointer.pointee.sin6_addr) { isInternal(Array($0)) }
            }
        default: return true
        }
    }
    /// A numeric address as text (an IPv6 zone after `%` ignored); true for anything that does not parse.
    package static func isInternal(address text: String) -> Bool {
        let numeric = String(text.split(separator: "%", maxSplits: 1).first ?? "")
        var v4 = in_addr()
        var v6 = in6_addr()
        if inet_pton(AF_INET, numeric, &v4) == 1 { return withUnsafeBytes(of: v4) { isInternal(Array($0)) } }
        if inet_pton(AF_INET6, numeric, &v6) == 1 { return withUnsafeBytes(of: v6) { isInternal(Array($0)) } }
        return true
    }
    /// An address in network byte order: 4 bytes (IPv4) or 16 (IPv6).
    package static func isInternal(_ bytes: [UInt8]) -> Bool {
        if bytes.count == 4 {
            let (a, b) = (bytes[0], bytes[1])
            // 0/8, 10/8, 127/8, 169.254/16, 172.16/12, 192.168/16, 100.64/10 (shared), 224/4 and above (multicast, reserved)
            return a == 0 || a == 10 || a == 127 || (a == 169 && b == 254) || (a == 172 && (16...31).contains(b))
                || (a == 192 && b == 168) || (a == 100 && (64...127).contains(b)) || a >= 224
        }
        guard bytes.count == 16 else { return true }
        let embedded = Array(bytes[12...])
        if bytes[..<15].allSatisfy({ $0 == 0 }) { return true }  // :: and ::1
        if bytes[..<10].allSatisfy({ $0 == 0 }) && bytes[10] == 0xff && bytes[11] == 0xff { return isInternal(embedded) }  // ::ffff:a.b.c.d
        if bytes[..<12].allSatisfy({ $0 == 0 }) { return isInternal(embedded) }  // ::a.b.c.d
        if bytes[..<4].elementsEqual([0x00, 0x64, 0xff, 0x9b]) && bytes[4..<12].allSatisfy({ $0 == 0 }) { return isInternal(embedded) }  // NAT64
        if bytes[0] == 0x20 && bytes[1] == 0x02 { return isInternal(Array(bytes[2..<6])) }  // 6to4
        // multicast ff00::/8, unique local fc00::/7, link-local fe80::/10, site-local fec0::/10
        return bytes[0] == 0xff || bytes[0] & 0xfe == 0xfc || (bytes[0] == 0xfe && bytes[1] & 0x80 == 0x80)
    }

    private static func fetch(_ link: URL) throws -> Data {
        final class Download: NSObject, URLSessionDataDelegate, @unchecked Sendable {
            var data = Data(), failure: String?, refused: ServingFailure?
            let done = DispatchSemaphore(value: 0)
            func urlSession(
                _ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
            ) {
                guard let status = (response as? HTTPURLResponse)?.statusCode, !(200..<300).contains(status) else {
                    return completionHandler(.allow)
                }
                failure = failure ?? "The image link answered HTTP \(status)"
                completionHandler(.cancel)
            }
            func urlSession(
                _ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void
            ) {
                guard PromptImage.followsRedirect(from: response.url, to: request.url) else {
                    failure = "The image link redirects from https to http; link the image itself"
                    task.cancel()
                    return completionHandler(nil)
                }
                do { try request.url.map(PromptImage.checkHost) } catch {
                    if let error = error as? ServingFailure { refused = error } else { failure = "\(error)" }
                    task.cancel()
                    return completionHandler(nil)
                }
                completionHandler(request)
            }
            func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
                self.data.append(data)
                if self.data.count > PromptImage.maximumLinkBytes {
                    failure = "The image link is larger than 24 MB"
                    dataTask.cancel()
                }
            }
            // Delivered before completion: the address each connection reached (through a proxy, the proxy's).
            func urlSession(_ session: URLSession, task: URLSessionTask, didFinishCollecting metrics: URLSessionTaskMetrics) {
                if let reached = metrics.transactionMetrics.first(where: {
                    !$0.isProxyConnection && $0.remoteAddress.map(PromptImage.isInternal(address:)) ?? false
                }) {
                    refused = PromptImage.refusedLink(reached.remoteAddress ?? "")
                }
            }
            func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
                if failure == nil, let error { failure = "Could not fetch the image link: \(error.localizedDescription)" }
                done.signal()
            }
        }
        let download = Download()
        let session = URLSession(configuration: .ephemeral, delegate: download, delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        session.dataTask(with: URLRequest(url: link, timeoutInterval: 30)).resume()
        guard download.done.wait(timeout: .now() + 60) == .success else { throw EngineError("The image link took more than a minute") }
        if let refused = download.refused { throw refused }
        if let failure = download.failure { throw EngineError(failure) }
        return download.data
    }

    /// A redirect is followed to any http(s) host (links such as GitHub's raw files and image hosts redirect to a CDN;
    /// the link itself may already name any host), but never from https to http.
    package static func followsRedirect(from: URL?, to: URL?) -> Bool {
        guard let target = to?.scheme?.lowercased(), ["http", "https"].contains(target) else { return false }
        return from?.scheme?.lowercased() != "https" || target == "https"
    }

    public init(dataURL: String) throws {
        guard dataURL.hasPrefix("data:"), let comma = dataURL.firstIndex(of: ","), dataURL[..<comma].hasSuffix(";base64"),
            let data = Data(base64Encoded: String(dataURL[dataURL.index(after: comma)...]), options: .ignoreUnknownCharacters)
        else {
            throw EngineError("Images must be base64 data: URLs")
        }
        try self.init(data: data)
    }

    /// Reads the header only: format, size and orientation give the patch grid; the pixels wait for `patches()`.
    public init(data: Data) throws {
        digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        let (_, height, width) = try Self.source(data)
        let (h, w) = try Self.smartResize(height: height, width: width)
        rows = h / 16
        columns = w / 16
        self.data = data
    }

    /// The image source and its upright size, refused unless one of the six formats within `maximumSourcePixels`.
    private static func source(_ data: Data) throws -> (CGImageSource, height: Int, width: Int) {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil), let type = CGImageSourceGetType(source) as String?,
            Self.formats.contains(type)
        else {
            throw EngineError("Unsupported image format; send PNG, JPEG, HEIC, GIF, WebP or TIFF")
        }
        guard CGImageSourceGetCount(source) > 0,
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
            var width = properties[kCGImagePropertyPixelWidth] as? Int, var height = properties[kCGImagePropertyPixelHeight] as? Int
        else {
            throw EngineError("Unreadable image; send PNG, JPEG, HEIC, GIF, WebP or TIFF")
        }
        // Checked before decoding: a small compressed file can declare a bitmap of gigabytes.
        guard Double(width) * Double(height) <= Double(Self.maximumSourcePixels) else {
            throw EngineError(
                "The image is \(width) x \(height) pixels; send one of at most 64 megapixels (it is scaled to about one megapixel anyway)")
        }
        if let orientation = properties[kCGImagePropertyOrientation] as? Int, orientation >= 5 { swap(&width, &height) }  // EXIF 5-8 turn it
        return (source, height, width)
    }

    /// rows x columns x 1536 values (channel, frame, row, column per patch), decoded from `data`.
    public func patches() throws -> [Float] {
        let (source, _, _) = try Self.source(data)
        let h = rows * 16
        let w = columns * 16
        // Decoded near the target size with its orientation applied, then drawn at exactly that size over white.
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: max(h, w),
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary),
            let context = CGContext(
                data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else {
            throw EngineError("Unreadable image; send PNG, JPEG, HEIC, GIF, WebP or TIFF")
        }
        context.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: w, height: h))
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        let pixels = context.data!.assumingMemoryBound(to: UInt8.self)
        let stride = context.bytesPerRow
        var patches = [Float](repeating: 0, count: rows * columns * VisionEncoder.patchValues)
        var index = 0
        for blockRow in 0..<(rows / 2) {
            for blockColumn in 0..<(columns / 2) {
                for innerRow in 0..<2 {
                    for innerColumn in 0..<2 {
                        let top = (blockRow * 2 + innerRow) * 16
                        let left = (blockColumn * 2 + innerColumn) * 16
                        for channel in 0..<3 {
                            for _ in 0..<2 {
                                for y in 0..<16 {
                                    for x in 0..<16 {
                                        patches[index] = (Float(pixels[(top + y) * stride + (left + x) * 4 + channel]) / 255 - 0.5) / 0.5
                                        index += 1
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return patches
    }

    /// transformers' `smart_resize`: each side to the nearest multiple of 32 (half to even), then scaled into the pixel range.
    package static func smartResize(height: Int, width: Int) throws -> (height: Int, width: Int) {
        guard height > 0, width > 0, Double(max(height, width)) / Double(min(height, width)) <= 200 else {
            throw EngineError("Image is empty or more than 200 times longer than wide")
        }
        let f = Double(factor)
        var h = Int((Double(height) / f).rounded(.toNearestOrEven)) * factor
        var w = Int((Double(width) / f).rounded(.toNearestOrEven)) * factor
        if h * w > maximumPixels {
            let beta = (Double(height * width) / Double(maximumPixels)).squareRoot()
            h = max(factor, Int((Double(height) / beta / f).rounded(.down)) * factor)
            w = max(factor, Int((Double(width) / beta / f).rounded(.down)) * factor)
        } else if h * w < minimumPixels {
            let beta = (Double(minimumPixels) / Double(height * width)).squareRoot()
            h = Int((Double(height) * beta / f).rounded(.up)) * factor
            w = Int((Double(width) * beta / f).rounded(.up)) * factor
        }
        return (h, w)
    }

    /// Each image's single `<|image_pad|>` becomes its (rows / 2) x (columns / 2) tokens; the map places them.
    package static func expand(_ prompt: [Int], pad: Int, images: [PromptImage]) throws -> (prompt: [Int], map: RoPEPositionMap) {
        var result: [Int] = []
        var placed: [RoPEPositionMap.Image] = []
        result.reserveCapacity(prompt.count + images.reduce(0) { $0 + $1.rows * $1.columns / 4 })
        for token in prompt {
            guard token == pad else {
                result.append(token)
                continue
            }
            guard placed.count < images.count else { throw EngineError("The prompt has more image markers than images") }
            let image = images[placed.count]
            placed.append(.init(start: result.count, rows: image.rows / 2, columns: image.columns / 2))
            result += repeatElement(pad, count: image.rows * image.columns / 4)
        }
        guard placed.count == images.count else { throw EngineError("The prompt has fewer image markers than images") }
        return (result, RoPEPositionMap(images: placed))
    }
}
