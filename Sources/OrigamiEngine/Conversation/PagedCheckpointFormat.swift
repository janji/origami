import CryptoKit
import Darwin
import Foundation

/// Version two has bounded binary metadata and stable maximum-capacity slots.
/// Payload holes are not tensors and are never returned to a reader.
/// File: a 16 MiB manifest (magic "ORGCKP02", JSON/record/root counts, SHA-256 of the manifest, JSON header,
/// 44-byte timeline records, 32-byte page roots), then each tensor at a page-aligned offset sized for the full
/// capacity, so the file is always the capacity's size (sparse) and a longer image only adds pages. A page root
/// is SHA-256(length || SHA-256 of each 8 KiB chunk) of one 256 KiB page.
package enum PagedCheckpointFormat {
    package static let magic = Data("ORGCKP02".utf8)
    package static let manifestBytes = 16 * 1024 * 1024
    package static let maximumJSON = 65536
    package static let segmentBytes = 44
    package static let pageBytes = QSAFilePageStore.pageBytes
    package static let scratchBytes: UInt64 = 128 * 1024 * 1024
    /// A tensor's slot: its offset is fixed by the capacity's maximum layout, `length` is the current size, and its
    /// `pages` page digests start at `rootStart` in the manifest's root table.
    package struct Section: Codable, Equatable {
        package let name, dtype: String
        package let offset, length: UInt64
        package let rootStart, pages: Int
    }
    /// The manifest's JSON part (at most 64 KiB).
    package struct Header: Codable {
        let version, capacity: Int
        let byteOrder: String
        let identity: CheckpointIdentity
        let frontier: CheckpointFrontier
        let hasLogits: Bool
        package let sections: [Section]
    }
    /// A verified manifest: header, binary timeline (44 bytes per segment) and 32-byte page roots.
    package struct Manifest {
        package let header: Header
        let timeline, roots: Data
        let digest: String
        package let bytes: Int
        package func root(section: Int, page: Int) -> Data {
            let offset = (header.sections[section].rootStart + page) * 32
            return roots.subdata(in: offset..<offset + 32)
        }
    }
    package static func layout(capacity: Int, tokens: Int, hasLogits: Bool) throws -> (sections: [Section], bytes: UInt64) {
        guard ConversationGeometry.contexts.contains(capacity), tokens >= 0, tokens <= capacity else {
            throw EngineError("Paged checkpoint capacity/frontier exceeds its geometry")
        }
        var maximum = PersistentCheckpointStore.expectedSections(tokens: capacity, hasLogits: true)
        // A full capacity is divisible by four, but an earlier frontier can
        // retain three raw indexer rows. Its stable slot must not disappear.
        for i in maximum.indices where maximum[i].0.hasPrefix("qsa.") && maximum[i].0.hasSuffix(".tail") {
            maximum[i].2 = 3 * 128 * 2
        }
        var current = PersistentCheckpointStore.expectedSections(tokens: tokens, hasLogits: true)
        if !hasLogits { current[current.count - 1].2 = 0 }
        var sections: [Section] = []
        var offset = UInt64(manifestBytes)
        var roots = 0
        for (maxSpec, spec) in zip(maximum, current) {
            guard maxSpec.0 == spec.0, spec.2 <= maxSpec.2 else { throw EngineError("Paged checkpoint schema changed") }
            let pages = Int((spec.2 + UInt64(pageBytes) - 1) / UInt64(pageBytes))
            sections.append(Section(name: spec.0, dtype: spec.1, offset: offset, length: spec.2, rootStart: roots, pages: pages))
            roots += pages
            offset += (maxSpec.2 + UInt64(pageBytes) - 1) / UInt64(pageBytes) * UInt64(pageBytes)
        }
        guard 64 + maximumJSON + capacity * segmentBytes + roots * 32 <= manifestBytes else {
            throw EngineError("Paged checkpoint manifest exceeds maximum geometry")
        }
        return (sections, offset)
    }
    /// Disk an image of `layout` takes in `directory`, at most: the manifest and the written pages where the volume keeps
    /// sparse files, plus every gap between written pages shorter than 16 MiB, which APFS fills (measured); the whole
    /// capacity-sized file elsewhere (HFS+, exFAT).
    package static func diskBytes(_ layout: (sections: [Section], bytes: UInt64), at directory: URL) -> UInt64 {
        // Asked of the nearest folder that exists: a folder not made yet is on the same volume, and must not count the
        // full (non-sparse) size where a made one counts the written pages.
        var probe = directory.standardizedFileURL
        while !FileManager.default.fileExists(atPath: probe.path), probe.pathComponents.count > 1 { probe.deleteLastPathComponent() }
        guard (try? probe.resourceValues(forKeys: [.volumeSupportsSparseFilesKey]))?.volumeSupportsSparseFiles == true else {
            return layout.bytes
        }
        var bytes = UInt64(manifestBytes)
        var end = UInt64(manifestBytes)
        for section in layout.sections where section.pages > 0 {
            let gap = section.offset - end
            let written = UInt64(section.pages * pageBytes)
            bytes += (gap < 16 << 20 ? gap : 0) + written
            end = section.offset + written
        }
        return bytes
    }
    package static func root(_ bytes: UnsafeRawBufferPointer) -> Data {
        var digest = SHA256()
        var length = UInt64(bytes.count).littleEndian
        withUnsafeBytes(of: &length) { digest.update(bufferPointer: $0) }
        for start in stride(from: 0, to: bytes.count, by: QSAFilePageStore.chunkBytes) {
            var unit = SHA256()
            unit.update(
                bufferPointer: UnsafeRawBufferPointer(rebasing: bytes[start..<min(bytes.count, start + QSAFilePageStore.chunkBytes)]))
            unit.finalize().withUnsafeBytes { digest.update(bufferPointer: $0) }
        }
        return Data(digest.finalize())
    }
    private static func append<T: FixedWidthInteger>(_ value: T, into data: inout Data) {
        var value = value.littleEndian
        withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
    }
    private static func integer<T: FixedWidthInteger>(_ data: Data, at offset: Int, _: T.Type) -> T {
        data.withUnsafeBytes { T(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: T.self)) }
    }
    package static func timeline(_ segments: [CheckpointSegment]) throws -> Data {
        guard segments.count <= ModelContract.maximumContext else { throw EngineError("Checkpoint timeline exceeds context") }
        var output = Data(capacity: segments.count * segmentBytes)
        var end = 0
        for segment in segments {
            try segment.validate()
            guard segment.start == end else { throw EngineError("Checkpoint timeline is not contiguous") }
            append(UInt32(segment.start), into: &output)
            append(UInt16(segment.count), into: &output)
            append(UInt16(segment.projectionRows), into: &output)
            append(UInt16(segment.queryTile), into: &output)
            output.append(segment.phase == .decode ? 1 : 0)
            output.append(0)
            let hex = Array(segment.dispatchSHA256.utf8)
            func nibble(_ c: UInt8) -> UInt8 { c <= 57 ? c - 48 : c - 87 }
            for i in stride(from: 0, to: 64, by: 2) { output.append(nibble(hex[i]) * 16 + nibble(hex[i + 1])) }
            end += segment.count
        }
        return output
    }
    // Validate binary geometry before constructing any per-segment Strings.
    private static func checkTimeline(_ bytes: Data, tokens: Int) throws {
        guard bytes.count % segmentBytes == 0, bytes.count / segmentBytes <= tokens else {
            throw EngineError("Paged checkpoint timeline record count differs")
        }
        var end = 0
        for offset in stride(from: 0, to: bytes.count, by: segmentBytes) {
            let start = Int(integer(bytes, at: offset, UInt32.self))
            let count = Int(integer(bytes, at: offset + 4, UInt16.self))
            let rows = Int(integer(bytes, at: offset + 6, UInt16.self))
            let tile = Int(integer(bytes, at: offset + 8, UInt16.self))
            let phase = bytes[offset + 10]
            guard start == end, (1...256).contains(count), count <= tokens - end, (count...256).contains(rows),
                (1...256).contains(tile), phase <= 1, phase == 0 || count == 1, bytes[offset + 11] == 0
            else {
                throw EngineError("Paged checkpoint timeline has invalid fields or gaps")
            }
            end += count
        }
        guard end == tokens else { throw EngineError("Paged checkpoint timeline does not reach frontier") }
    }
    package static func decodeTimeline(_ data: Data) throws -> [CheckpointSegment] {
        guard data.count % segmentBytes == 0, data.count / segmentBytes <= ModelContract.maximumContext else {
            throw EngineError("Binary checkpoint timeline exceeds bounded records")
        }
        var output: [CheckpointSegment] = []
        output.reserveCapacity(data.count / segmentBytes)
        for offset in stride(from: 0, to: data.count, by: segmentBytes) {
            let start = Int(integer(data, at: offset, UInt32.self))
            let count = Int(integer(data, at: offset + 4, UInt16.self))
            let hash = data[(offset + 12)..<(offset + 44)].map { String(format: "%02x", $0) }.joined()
            output.append(
                try CheckpointSegment(
                    start: start, count: count,
                    projectionRows: Int(integer(data, at: offset + 6, UInt16.self)),
                    queryTile: Int(integer(data, at: offset + 8, UInt16.self)),
                    keyExtent: start + count, phase: data[offset + 10] == 0 ? .prefill : .decode, dispatchSHA256: hash))
        }
        guard UInt64(output.capacity * MemoryLayout<CheckpointSegment>.stride + output.count * 256) <= UInt64(output.count * 1024 + 65536)
        else {
            throw EngineError("Decoded timeline exceeded its retained admission")
        }
        return output
    }
    package static func encode(header: Header, timeline: Data, roots: Data) throws -> (Data, String) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let json = try encoder.encode(header)
        guard json.count <= maximumJSON, timeline.count % segmentBytes == 0, roots.count % 32 == 0,
            64 + json.count + timeline.count + roots.count <= manifestBytes
        else { throw EngineError("Paged manifest encoding exceeds bounds") }
        var prefix = magic
        append(UInt64(json.count), into: &prefix)
        append(UInt64(timeline.count / segmentBytes), into: &prefix)
        append(UInt64(roots.count / 32), into: &prefix)
        var hash = SHA256()
        hash.update(data: prefix)
        hash.update(data: json)
        hash.update(data: timeline)
        hash.update(data: roots)
        let digest = Data(hash.finalize())
        var output = prefix
        output.append(digest)
        output.append(json)
        output.append(timeline)
        output.append(roots)
        return (output, digest.map { String(format: "%02x", $0) }.joined())
    }
    package static func load(_ file: ExactFile, capacity: Int, maximumBytes: UInt64) throws -> Manifest {
        guard file.identity.size >= UInt64(manifestBytes), file.identity.size <= maximumBytes else {
            throw EngineError("Paged checkpoint file length exceeds bounds")
        }
        let prefix = try file.read(offset: 0, length: 64)
        guard prefix.prefix(8) == magic else { throw EngineError("Unsupported paged checkpoint magic") }
        let jsonLength = integer(prefix, at: 8, UInt64.self)
        let records = integer(prefix, at: 16, UInt64.self)
        let roots = integer(prefix, at: 24, UInt64.self)
        let maxLayout = try layout(capacity: capacity, tokens: capacity, hasLogits: true)
        let maxRoots = maxLayout.sections.reduce(0) { $0 + $1.pages } + 12  // partial raw tails
        guard jsonLength > 0, jsonLength <= maximumJSON, records <= UInt64(capacity), roots <= UInt64(maxRoots),
            64 + jsonLength + records * UInt64(segmentBytes) + roots * 32 <= manifestBytes,
            file.identity.size == maxLayout.bytes
        else { throw EngineError("Paged checkpoint manifest bounds differ") }
        let json = try file.read(offset: 64, length: Int(jsonLength))
        let timeline = try file.read(offset: Int64(64 + jsonLength), length: Int(records) * segmentBytes)
        let rootData = try file.read(offset: Int64(64 + jsonLength + records * UInt64(segmentBytes)), length: Int(roots) * 32)
        var hash = SHA256()
        hash.update(data: prefix.prefix(32))
        hash.update(data: json)
        hash.update(data: timeline)
        hash.update(data: rootData)
        let digest = Data(hash.finalize())
        guard digest == prefix.suffix(32) else { throw EngineError("Paged checkpoint manifest checksum mismatch") }
        let header = try JSONDecoder().decode(Header.self, from: json)
        try header.identity.validate()
        try header.frontier.validate()
        guard header.version == 2, header.byteOrder == "little", header.capacity == capacity else {
            throw EngineError("Paged checkpoint version/capacity differs")
        }
        let expected = try layout(capacity: capacity, tokens: header.frontier.consumedTokenCount, hasLogits: header.hasLogits)
        guard header.sections == expected.sections, header.sections.reduce(0, { $0 + $1.pages }) == Int(roots) else {
            throw EngineError("Paged checkpoint tensor slots, lengths or roots differ")
        }
        try checkTimeline(timeline, tokens: header.frontier.consumedTokenCount)
        try file.validateIdentity()
        return Manifest(
            header: header, timeline: timeline, roots: rootData,
            digest: digest.map { String(format: "%02x", $0) }.joined(), bytes: 64 + json.count + timeline.count + rootData.count)
    }
}

/// Exact mmap capacity avoids allocator-dependent page scratch sizes.
package final class CheckpointPageScratch {
    private let memory: CheckpointMemoryLease
    let pointer: UnsafeMutableRawPointer
    package init(ledger: MemoryLedger) throws {
        memory = try CheckpointMemoryLease(
            ledger: ledger, bytes: UInt64(PagedCheckpointFormat.pageBytes), category: "checkpoint_page_scratch")
        let mapped = mmap(nil, PagedCheckpointFormat.pageBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0)
        guard let mapped, mapped != MAP_FAILED else { throw EngineError("Checkpoint scratch mmap failed", code: "insufficient_memory") }
        pointer = mapped
    }
    deinit { munmap(pointer, PagedCheckpointFormat.pageBytes) }
    package func buffer(_ count: Int) -> UnsafeMutableRawBufferPointer {
        precondition((0...PagedCheckpointFormat.pageBytes).contains(count))
        return UnsafeMutableRawBufferPointer(start: pointer, count: count)
    }
}
