import Foundation

/// A GLM or DeepSeek hot copy holds a routed expert's record as a slot holds it (gate, up and down back to back), read
/// from the GGUF or the pack (`readRecord`); a key is layer * experts + expert.
extension FamilyExperts: HotCopySource {
    package func hotCopyRecordBytes(_ key: Int) -> Int? { bank(key / weights.geometry.experts).map { $0.bytes.reduce(0, +) } }
    package func readHotCopyRecord(_ key: Int, into: UnsafeMutableRawPointer) throws {
        try readRecord(layer: key / weights.geometry.experts, expert: key % weights.geometry.experts, into: into)
    }
    /// A record's bytes per MoE layer as the pools hold them, from the GGUF's headers alone (`origami verify`).
    package static func recordBytes(_ gguf: GGUFSource, geometry: ModelGeometry) -> [Int: Int] {
        var out: [Int: Int] = [:]
        for layer in geometry.moeLayers {
            let tensors = ["gate", "up", "down"].compactMap { gguf.tensors["blk.\(layer).ffn_\($0)_exps.weight"] }
            if tensors.count == 3 { out[layer] = tensors.reduce(0) { $0 + Int($1.length) / geometry.experts } }
        }
        return out
    }
}
