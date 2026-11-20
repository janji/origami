import Foundation
import OrigamiEngine

/// The further-copy writer earlier versions' `origami setup` used (`mirror.json` + `mirror.bin`: records in ranking
/// order, a SHA-256 each). `origami setup` now writes `HotCopy` stores; this stays only so the checks can make
/// old-format copies exactly as they exist on users' drives and prove they are still read.
public enum LegacyMirrorFixture {
    /// Writes the `records` most-used experts into `folder` as a mirror; `ranking` most needed first (nil: the shipped
    /// one). Every record is read digest-checked (from `pack` when given, else the checkpoint's spans), so a bad pack
    /// record is never copied under a fresh, valid-looking digest.
    public static func build(model: ModelSource, folder: URL, records count: Int, from pack: URL? = nil, ranking: [Int]? = nil) throws {
        let keys = Array(StoreBuilder.ranked(ranking).prefix(count))
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let store = try PackStore(model: model, pack: pack, integrity: .sha256)
        try store.enableConcurrentOriginalSpans(pool: SpanReadPool(threads: 16))
        let records = try StoreBuilder.writeRecords(
            keys, store: store, to: folder.appendingPathComponent("mirror.bin"),
            stride: model.quantization.expertRecordBytes, header: folder.appendingPathComponent(StoreHeader.fileName(.mirror)),
            progress: nil)
        try StoreHeader(
            kind: .mirror, modelFingerprint: model.fingerprint, quantization: model.quantization, expertFile: "mirror.bin",
            recordBytes: model.quantization.expertRecordBytes, stride: model.quantization.expertRecordBytes,
            records: records, pleFile: nil, pleRowBytes: nil, pleRowCount: nil, created: StoreBuilder.timestamp()
        ).write(to: folder)
    }
}
