import Foundation

/// CPU bfloat16 conversion. `encode` rounds to nearest, ties to even (the bits Metal's `round_bf16` gives a finite
/// float); a NaN keeps its sign and top payload bits and is made quiet. Every CPU path that writes bf16 (the
/// converters, the MLX-exact quantizer, checks) uses it, so their bytes are deterministic.
public enum BF16 {
    /// Round to nearest, ties to even; preserve sign/payload and ensure a quiet NaN.
    @inlinable public static func encode(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        if bits & 0x7f80_0000 == 0x7f80_0000 && bits & 0x007f_ffff != 0 {
            return UInt16(bits >> 16) | 0x0040
        }
        return UInt16((bits &+ 0x7fff &+ ((bits >> 16) & 1)) >> 16)
    }
    @inlinable public static func decode(_ value: UInt16) -> Float { Float(bitPattern: UInt32(value) << 16) }
}
