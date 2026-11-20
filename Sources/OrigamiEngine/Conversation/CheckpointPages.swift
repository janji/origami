import Foundation

extension ModelCheckpoint {
    /// The loader is synchronous and valid only during the callback. File KV
    /// exposes immutable digest metadata without reading unchanged payload.
    func withCheckpointPages(
        _ body: (
            String, String, Int, Int, Data,
            (UnsafeMutableRawBufferPointer) throws -> Void
        ) throws -> Void
    ) throws {
        func flat(_ name: String, _ dtype: String, _ bytes: UnsafeRawBufferPointer) throws {
            for offset in stride(from: 0, to: bytes.count, by: PagedCheckpointFormat.pageBytes) {
                let page = UnsafeRawBufferPointer(rebasing: bytes[offset..<min(bytes.count, offset + PagedCheckpointFormat.pageBytes)])
                try body(name, dtype, offset / PagedCheckpointFormat.pageBytes, page.count, PagedCheckpointFormat.root(page)) { target in
                    guard target.count == page.count else { throw EngineError("Checkpoint page loader size differs") }
                    target.copyMemory(from: page)
                }
            }
        }
        func resident(_ name: String, _ vector: QSAPagedVector) throws {
            try vector.withCheckpointPages { index, count, root, load in
                try body(name, "u16", index, count, root, load)
            }
        }
        func keyVector(_ name: String, _ vector: QSAKeyPages) throws {
            switch vector {
            case .memory(let value): try resident(name, value)
            case .file(let value):
                for (index, page) in value.pages.enumerated() {
                    try body(name, "u16", index, page.count * 2, value.store.checkpointRoot(page)) { target in
                        try value.store.read(page, range: 0..<page.count * 2, into: target)
                    }
                }
            }
        }
        try withPrefixTensors { try flat($0, $1, $2) }
        for layer in qsa.keys.sorted() {
            let state = qsa[layer]!
            let name = "qsa.\(layer)."
            if let pages = state.pages {
                try keyVector(name + "keys", pages.keys)
                try keyVector(name + "values", pages.values)
                try resident(name + "pooled", pages.pooled)
            } else {
                try state.keys.withUnsafeBytes { try flat(name + "keys", "u16", $0) }
                try state.values.withUnsafeBytes { try flat(name + "values", "u16", $0) }
                try state.pooledKeys.withUnsafeBytes { try flat(name + "pooled", "u16", $0) }
            }
            try state.rawTail.withUnsafeBytes { try flat(name + "tail", "u16", $0) }
        }
        if let nextLogits { try nextLogits.withUnsafeBytes { try flat("next_logits", "f32", $0) } }
    }
}
