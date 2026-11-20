import Foundation

/// `origami model add <experts.gguf> --base <model>` for GLM-5.3-Flash and DeepSeek-V4.1-Flash: a GGUF of the family
/// engine whose routed-expert banks of some or all layers are read from another GGUF (a calibrated low-bit release, or an
/// experts-only file), everything else from the base as stored (`GGUFSource.withExperts`). Nothing is copied or
/// converted. Replies differ from the base's (NUMERIC), as the banks hold other values.
package enum FamilyExpertVariant {
    /// The pairing: the merged source, the layers whose banks come from `experts`, and what could not be compared (an
    /// experts-only file names no architecture, tokenizer or routers), for the person adding it to acknowledge.
    package struct Pairing {
        package let source: GGUFSource
        package let layers: [Int]
        package let unverified: [String]
    }

    /// Checks that `experts` (some or all parts of a GGUF; `partial`) holds banks of `base`'s checkpoint quantized
    /// another way, from headers and, with `verify`, the tokenizer and a few MB of routers: the same family and geometry
    /// where it names them, every token at the same id, the router and its bias of three covered layers equal value for
    /// value where both store them as floats, and for every layer it covers all three banks in the base's shapes and in
    /// types the expert kernels run. Layers it does not cover keep the base's banks. Throws naming what differs.
    package static func pair(_ experts: GGUFSource, base: GGUFSource, verify: Bool = true) throws -> Pairing {
        let geometry = try ModelGeometry(gguf: base)
        func refuse(_ why: String) -> EngineError {
            EngineError("These experts do not belong with this \(geometry.family.displayName) GGUF: \(why)", code: "model_incompatible")
        }
        var unverified: [String] = []
        if case .string(let architecture)? = experts.metadata["general.architecture"] {
            guard let family = ModelFamily.detect(gguf: experts.metadata), family == geometry.family else {
                throw refuse("it is \(architecture), not \(geometry.family.displayName)")
            }
            // A part without the token table may not give the whole geometry; its banks' shapes are checked below.
            if let theirs = try? ModelGeometry(gguf: experts) {
                for (what, ours, other) in [
                    ("layers", geometry.layers, theirs.layers), ("hidden size", geometry.hidden, theirs.hidden),
                    ("experts", geometry.experts, theirs.experts), ("experts per token", geometry.expertsPerToken, theirs.expertsPerToken),
                    ("expert width", geometry.expertWidth, theirs.expertWidth),
                ] where ours != other {
                    throw refuse("its \(what) is \(other), the base's \(ours)")
                }
            } else {
                unverified.append("does not give the model's whole geometry")
            }
        } else {
            unverified.append("names no architecture")
        }
        var layers: [Int] = []
        for layer in geometry.moeLayers {
            let names = ["gate", "up", "down"].map { "blk.\(layer).ffn_\($0)_exps.weight" }
            let present = names.filter { experts.tensors[$0] != nil }
            if present.isEmpty { continue }
            guard present.count == 3 else {
                throw refuse("layer \(layer) has \(present.joined(separator: ", ")) but not all three expert banks in the parts given")
            }
            for name in names {
                let t = experts.tensors[name]!
                guard let ours = base.tensors[name] else { throw refuse("the base has no \(name)") }
                guard GGMLMatrixKernels.types.contains(t.type), let size = GGUFQuant.sizes[t.type], size.block > 1 else {
                    throw refuse("\(name) is \(GGUFQuant.name(t.type)), which Origami's expert kernels do not run")
                }
                guard t.shape == ours.shape, (t.shape.last ?? 0) % size.block == 0 else {
                    throw refuse("\(name) has shape \(t.shape), the base's \(ours.shape)")
                }
            }
            layers.append(layer)
        }
        guard !layers.isEmpty else { throw refuse("it holds no routed-expert banks (blk.L.ffn_gate_exps.weight) of the base's layers") }
        guard verify else {
            return Pairing(source: base.withExperts(from: experts, layers: layers), layers: layers, unverified: unverified)
        }

        let ours = base.tokenizer["tokenizer.ggml.tokens"] ?? []
        let theirs = experts.tokenizer["tokenizer.ggml.tokens"] ?? []
        if ours.isEmpty || theirs.isEmpty {
            unverified.append("carries no tokenizer to compare")
        } else if let id = ours.indices.first(where: { $0 >= theirs.count || ours[$0] != theirs[$0] })
            ?? (theirs.count > ours.count ? ours.count : nil)
        {
            throw refuse("token \(id) differs from the base's: another tokenizer")
        }
        // Float values of a tensor, at most 64 MB of them.
        func values(_ source: GGUFSource, _ t: GGUFSource.Tensor) throws -> [Float]? {
            let count = t.shape.reduce(1, *)
            guard [0, 1, 30].contains(t.type), count > 0, Int(t.length) <= 64 << 20 else { return nil }
            let handle = try FileHandle(forReadingFrom: source.url(t.file))
            defer { try? handle.close() }
            try handle.seek(toOffset: UInt64(t.offset))
            guard let data = try handle.read(upToCount: Int(t.length)), data.count == Int(t.length) else {
                throw refuse("\(t.name) is cut short")
            }
            var out = [Float](repeating: 0, count: count)
            data.withUnsafeBytes { raw in
                out.withUnsafeMutableBufferPointer {
                    GGUFQuant.dequantize(raw.baseAddress!, type: t.type, count: count, into: $0.baseAddress!)
                }
            }
            return out
        }
        var compared = 0
        for layer in Set([layers.first!, layers[layers.count / 2], layers.last!]).sorted() {
            for name in ["blk.\(layer).ffn_gate_inp.weight", "blk.\(layer).exp_probs_b.bias"] {
                guard let a = base.tensors[name], let b = experts.tensors[name] else { continue }
                // Stored as F32 in one and BF16 or F16 in the other: compared at the narrower one's precision.
                let types = Set([a.type, b.type])
                guard a.shape == b.shape, types.count == 1 || types == [0, 30] || types == [0, 1],
                    let x = try values(base, a), let y = try values(experts, b)
                else { continue }
                let narrow: (Float) -> UInt32 =
                    types.contains(30)
                    ? { UInt32(BF16.encode($0)) } : types.contains(1) ? { UInt32(Float16($0).bitPattern) } : { $0.bitPattern }
                guard x.map(narrow) == y.map(narrow) else {
                    throw refuse(
                        "layer \(layer)'s \(name.hasSuffix("bias") ? "router bias" : "router") differs from the base's: another checkpoint or a fine-tune"
                    )
                }
                compared += 1
            }
        }
        if compared == 0 { unverified.append("carries no routers to compare") }
        return Pairing(source: base.withExperts(from: experts, layers: layers), layers: layers, unverified: unverified)
    }
}
