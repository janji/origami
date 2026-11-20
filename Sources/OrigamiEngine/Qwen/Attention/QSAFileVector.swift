import Foundation

/// Persistent vector of externally stored 256-token KV pages. The last partial
/// page is copied transactionally; all complete pages retain immutable leases.
package struct QSAFileVector: Sendable {
    package let store: QSAFilePageStore
    package private(set) var pages: [QSAFilePageStore.Page] = []
    package private(set) var count = 0
    package init(store: QSAFilePageStore) { self.store = store }
    package func appending(_ values: [UInt16], shouldContinue: () -> Bool = { true }) throws -> Self {
        guard !values.isEmpty else { return self }
        guard values.count % 512 == 0, values.count <= ModelContract.maximumContext * 512 - count else {
            throw EngineError("File-KV vector append exceeds its token geometry")
        }
        let capacity = QSAFilePageStore.pageBytes / 2
        var result = self
        var cursor = 0
        if let tail = pages.last, tail.count < capacity {
            let amount = min(capacity - tail.count, values.count)
            let next = try values[0..<amount].withUnsafeBytes { try store.make(prefix: tail, suffix: $0, shouldContinue: shouldContinue) }
            result.pages[result.pages.count - 1] = next
            cursor = amount
        }
        while cursor < values.count {
            let end = min(values.count, cursor + capacity)
            let next = try values[cursor..<end].withUnsafeBytes { try store.make(suffix: $0, shouldContinue: shouldContinue) }
            result.pages.append(next)
            cursor = end
        }
        result.count += values.count
        return result
    }
    package func copy(to pointer: UnsafeMutableRawPointer, shouldContinue: () -> Bool = { true }) throws {
        var offset = 0
        for page in pages {
            let bytes = page.count * 2
            try store.read(
                page, range: 0..<bytes, into: UnsafeMutableRawBufferPointer(start: pointer.advanced(by: offset), count: bytes),
                shouldContinue: shouldContinue)
            offset += bytes
        }
    }
    package func array() throws -> [UInt16] {
        try [UInt16](unsafeUninitializedCapacity: count) { buffer, initialized in
            if let pointer = buffer.baseAddress { try copy(to: UnsafeMutableRawPointer(pointer)) }
            initialized = count
        }
    }
    package var pageIdentities: Set<ObjectIdentifier> { Set(pages.map(ObjectIdentifier.init)) }
    package var referenceCapacityBytes: UInt64 { UInt64(pages.capacity * MemoryLayout<QSAFilePageStore.Page>.stride) }
}
