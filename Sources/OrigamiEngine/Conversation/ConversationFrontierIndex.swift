import Foundation

/// Complete immutable images ordered by token offset. Cache the nearest
/// compatible predecessor so an eviction never rescans every pair of prefixes.
/// Weak predecessor links cannot extend a recurrent file's lifetime.
package final class ConversationFrontierIndex {
    private final class Entry {
        let image: ModelCheckpoint
        weak var predecessor: Entry?
        var offset: Int { image.consumedIDs.count }
        init(_ image: ModelCheckpoint) { self.image = image }
    }
    private var entries: [Entry] = []
    package init() {}
    package var images: [ModelCheckpoint] { entries.map(\.image) }
    package var count: Int { entries.count }
    package var isEmpty: Bool { entries.isEmpty }
    package var reservedBytes: UInt64 { entries.reduce(0) { $0 + $1.image.reservedBytes } }
    private func predecessor(of entry: Entry) -> Entry? {
        entries.reversed().first { candidate in
            candidate.offset < entry.offset
                && candidate.image.consumedIDs.elementsEqual(entry.image.consumedIDs.prefix(candidate.offset))
        }
    }
    package func insert(_ image: ModelCheckpoint) {
        let entry = Entry(image)
        entry.predecessor = predecessor(of: entry)
        // Inserting an earlier branch may improve an already retained later
        // image. The common append case has no such candidates.
        for later in entries where later.offset > entry.offset && (later.predecessor?.offset ?? 0) < entry.offset {
            if image.consumedIDs.elementsEqual(later.image.consumedIDs.prefix(entry.offset)) {
                later.predecessor = entry
            }
        }
        entries.insert(entry, at: entries.firstIndex(where: { $0.offset > entry.offset }) ?? entries.endIndex)
    }
    private func remove(_ entry: Entry) {
        entries.removeAll { $0 === entry }
        for child in entries where child.predecessor === entry { child.predecessor = predecessor(of: child) }
    }
    package func removeDuplicates(ids: [Int64], segments: [CheckpointSegment]) {
        for entry in entries.filter({ $0.image.segments == segments && $0.image.consumedIDs == ids }) { remove(entry) }
    }
    package func remove(_ image: ModelCheckpoint) {
        if let entry = entries.first(where: { $0.image === image }) { remove(entry) }
    }
    @discardableResult
    package func evict(protecting protected: ModelCheckpoint?, also other: ModelCheckpoint? = nil) -> Bool {
        guard
            let selected = entries.filter({ $0.image !== protected && $0.image !== other }).min(by: {
                $0.offset - ($0.predecessor?.offset ?? 0) < $1.offset - ($1.predecessor?.offset ?? 0)
            })
        else { return false }
        remove(selected)
        return true
    }
}
