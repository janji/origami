import Foundation

/// The weight layout of the model this process serves (one model per process): the group size of the
/// quantized matrices and the expert record format. `OrigamiPlan.make` and `OrigamiRuntime.load` set it
/// from the checkpoint before anything is sized or a kernel is compiled, and it never changes after the
/// runtime has loaded; checks set it around a layout they test and restore it.
/// shortcut: process-wide instead of passed to every owner; pass it through the configuration if two
/// layouts must ever run in one process.
public enum ModelLayout {
    /// Group size of the quantized dense matrices and the experts.
    nonisolated(unsafe) public private(set) static var group = 64
    nonisolated(unsafe) public private(set) static var experts = AffineFormat(bits: 4, group: 64)
    nonisolated(unsafe) public private(set) static var expertRecordBytes = AffineFormat(bits: 4, group: 64).expertRecordBytes
    /// The smallest record a cache slot holds: a GGUF's smallest layer's (a smaller size class's slot can end there at a
    /// slab's end), else `expertRecordBytes`.
    nonisolated(unsafe) public private(set) static var smallestExpertRecordBytes = AffineFormat(bits: 4, group: 64).expertRecordBytes
    /// The GGUF expert layout (`gguf-experts`), or nil: its records are what the cache holds, and the expert paths compute
    /// them with ggml's kernels (`GGUFExpertKernels`).
    nonisolated(unsafe) public private(set) static var ggufExperts: GGUFExperts?
    /// Routed experts per layer: 512, or fewer for a pruned GGUF, whose other router columns are never selected.
    public static var routedExperts: Int { ggufExperts?.experts ?? ModelContract.experts }

    /// Set by `OrigamiRuntime.load`: its kernels, caches and record offsets are built for this layout.
    nonisolated(unsafe) private static var serving = false

    /// Global: a server plans only with its own layout. `serving` fixes the layout for the life of the process; changing
    /// it afterwards is a programming error and stops the process (checks that switch layouts run without a runtime).
    public static func use(_ scheme: QuantizationScheme, gguf: GGUFExperts? = nil, serving fix: Bool = false) {
        let format = scheme.experts
        defer { if fix { serving = true } }
        guard group != scheme.group || experts != format || ggufExperts != gguf else { return }
        precondition(!serving, "The model layout cannot change once a runtime has loaded")
        group = scheme.group
        experts = format
        ggufExperts = gguf
        expertRecordBytes = gguf?.recordBytes ?? format.expertRecordBytes
        smallestExpertRecordBytes = gguf?.layers.map(\.bytes).min() ?? expertRecordBytes
    }
    /// The layout of `model` (its GGUF experts included).
    public static func use(_ model: ModelSource, serving fix: Bool = false) {
        use(model.quantization, gguf: model.ggufExperts, serving: fix)
    }

    /// Prepended to Metal sources that read the quantized dense matrices (4-bit, `group`).
    static var metalDefines: String { defines(bits: 4, group: group, fp4: false) }
    /// Prepended to the libraries that read expert records (`experts`: 4-bit or NVFP4).
    static var expertMetalDefines: String { defines(bits: experts.bits, group: experts.group, fp4: experts.fp4) }

    private static func defines(bits: Int, group: Int, fp4: Bool) -> String {
        "#define QBITS \(bits)\n#define QGROUP \(group)\n#define QFP4 \(fp4 ? 1 : 0)\n"
            + "#define QPROJ_CODES \(experts.codeBytes(width: 2560, outputs: 640))\n"
            + "#define QPROJ_META \(experts.scaleBytes(width: 2560, outputs: 640))\n#define QPROJ_BYTES \(experts.bytes(width: 2560, outputs: 640))\n"
                + #"""
                // One quantized weight is QVAL(code) * QSCALE(scales, biases, group index) + QBIAS(biases, group index).
                // NVFP4: E2M1 codes, an E4M3 scale byte per group and the matrix's float32 global scale where the biases
                // start; affine: the code itself and bf16 scales and biases.
                #if QFP4
                // E2M1 code bits placed in a half's exponent/mantissa give the value / 2^14 exactly (subnormals included);
                // no table, whose per-thread indexing serializes in constant memory. The 2^14 is folded into QSCALE.
                #define QVAL(n) float(as_type<half>(ushort((((n) & 8u) << 12) | (((n) & 7u) << 9))))
                // E4M3 likewise: its bits shifted into a half are the value / 2^8; times 2^8 and QVAL's 2^14.
                inline float qfp4_scale(uint b) { return float(as_type<half>(ushort(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7)))) * 4194304.0f; }
                #define QSCALE(s, b, i) (qfp4_scale(((device const uchar *)(s))[(i)]) * *((device const float *)(b)))
                #define QBIAS(b, i) 0.0f
                #else
                #define QVAL(n) float(n)
                #define QSCALE(s, b, i) from_bf16((s)[(i)])
                #define QBIAS(b, i) from_bf16((b)[(i)])
                #endif

                """#
    }
}
