import Darwin
import Foundation

/// The header of one Origami store payload (`pack.json` or `mirror.json`) inside a store folder.
/// A store is any folder the user names; Origami keeps one subfolder per model there
/// (`<store>/<model-name>-<fingerprint prefix>/`). The header binds the payload to the
/// model fingerprint, never to inode, ctime or volume identity, so copying, moving or
/// restoring a store keeps it usable. A header for another model means "rebuild", never
/// "refuse to start".
public struct StoreHeader: Codable, Sendable {
    /// What the payload holds; the raw value is the header file's base name.
    public enum Kind: String, Codable, Sendable {
        /// Every expert record plus the PLE table: one read per expert instead of nine.
        case pack
        /// The most-used expert records only, next to the model for fast single reads.
        case mirror
    }
    /// One expert record in the payload file: its key, its byte offset (index * stride, Int64 so payloads
    /// above 4 GiB address exactly) and the digest of its `recordBytes` (the padding up to `stride` is not hashed).
    public struct Record: Codable, Sendable {
        /// layer * 512 + expert
        public let key: Int
        public let offset: Int64
        /// SHA-256 of the record bytes, checked on a sample of reads and by `origami verify`.
        public let sha256: String
        public init(key: Int, offset: Int64, sha256: String) {
            self.key = key
            self.offset = offset
            self.sha256 = sha256
        }
    }
    public static let format = "origami-store-v1"
    public var format = StoreHeader.format
    public let kind: Kind
    public let modelFingerprint: String
    public let quantization: QuantizationScheme
    public let expertFile: String
    public let recordBytes, stride: Int
    public let records: [Record]
    /// Pack only: the PLE table, `pleRowBytes` per row, rows in checkpoint order.
    public let pleFile: String?
    public let pleRowBytes, pleRowCount: Int?
    public let created: String

    enum CodingKeys: String, CodingKey {
        case format, kind, quantization, records, created, stride
        case modelFingerprint = "model_fingerprint"
        case expertFile = "expert_file"
        case recordBytes = "record_bytes"
        case pleFile = "ple_file"
        case pleRowBytes = "ple_row_bytes"
        case pleRowCount = "ple_row_count"
    }

    public init(
        kind: Kind, modelFingerprint: String, quantization: QuantizationScheme, expertFile: String,
        recordBytes: Int, stride: Int, records: [Record], pleFile: String? = nil, pleRowBytes: Int? = nil,
        pleRowCount: Int? = nil, created: String
    ) {
        self.kind = kind
        self.modelFingerprint = modelFingerprint
        self.quantization = quantization
        self.expertFile = expertFile
        self.recordBytes = recordBytes
        self.stride = stride
        self.records = records
        self.pleFile = pleFile
        self.pleRowBytes = pleRowBytes
        self.pleRowCount = pleRowCount
        self.created = created
    }

    /// `pack.json` or `mirror.json`.
    public static func fileName(_ kind: Kind) -> String {
        kind.rawValue + ".json"
    }

    /// This model's folder in a store: an existing one with the same fingerprint (the model may
    /// have been copied or renamed since it was built), else a new readable name.
    public static func modelFolder(store: URL, model: ModelSource) -> URL {
        let suffix = "-" + model.fingerprint.prefix(12)
        if let existing = (try? FileManager.default.contentsOfDirectory(atPath: store.path))?.sorted().first(where: { $0.hasSuffix(suffix) }
        ) {
            return store.appendingPathComponent(existing, isDirectory: true)
        }
        let name = model.displayName.replacingOccurrences(of: "/", with: "--")
            .filter { $0.isLetter || $0.isNumber || "-_.".contains($0) }
        return store.appendingPathComponent("\(name.isEmpty ? "model" : name)-\(model.fingerprint.prefix(12))", isDirectory: true)
    }

    /// The header for `kind` in `folder`, or nil when absent. Throws when it exists but is
    /// unreadable or belongs to another format; the caller treats that as "rebuild".
    public static func load(_ kind: Kind, in folder: URL) throws -> StoreHeader? {
        let url = folder.appendingPathComponent(fileName(kind))
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        // Bounded before reading: a damaged or planted header (huge, or a link to a device) must not be read whole.
        var info = stat()
        guard stat(url.path, &info) == 0, info.st_mode & S_IFMT == S_IFREG, info.st_size <= 64 * 1024 * 1024 else {
            throw EngineError("\(url.path) is not a store header (too large, or not a regular file)")
        }
        let data = try Data(contentsOf: url)
        let header = try JSONDecoder().decode(StoreHeader.self, from: data)
        guard header.format == format else { throw EngineError("\(url.path) has unknown format \(header.format)") }
        return header
    }

    public func write(to folder: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(self).write(to: folder.appendingPathComponent(Self.fileName(kind)), options: .atomic)
    }

    /// Why this header cannot serve `model`, or nil. Geometry and file lengths are checked here;
    /// payload bytes are checked by sampled digests while serving and fully by `origami verify`.
    public func mismatch(for model: ModelSource, folder: URL) -> String? {
        guard modelFingerprint == model.fingerprint else { return "built for another model" }
        guard quantization == model.quantization, recordBytes == model.quantization.experts.expertRecordBytes,
            stride >= recordBytes, stride <= 1 << 26, !records.isEmpty, records.count <= ModelContract.layers * ModelContract.experts
        else {
            return "record layout differs"
        }
        var seen = Set<Int>()
        for (index, record) in records.enumerated() {
            guard (0..<ModelContract.layers * ModelContract.experts).contains(record.key), seen.insert(record.key).inserted,
                record.offset == Int64(index * stride), record.sha256.count == 64
            else { return "record table is invalid" }
        }
        func length(_ name: String) -> Int64? {
            // stat follows a symlinked payload (a store may point at files kept elsewhere).
            var info = stat()
            guard !name.contains("/"), stat(folder.appendingPathComponent(name).path, &info) == 0 else { return nil }
            return Int64(info.st_size)
        }
        guard let payload = length(expertFile), payload >= Int64(records.count * stride) else { return "\(expertFile) is missing or short" }
        guard kind != .pack || records.count == ModelContract.layers * ModelContract.experts else { return "pack is incomplete" }
        // Only a pack holds the PLE table.
        guard kind == .pack else { return pleFile == nil ? nil : "record layout differs" }
        guard let pleFile, pleRowBytes == 100, pleRowCount == ModelContract.pleRows,
            let ple = length(pleFile), ple == Int64(ModelContract.pleRows * 100)
        else { return "PLE table is missing or short" }
        return nil
    }
}
