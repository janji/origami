import Foundation

/// ggml's reference quantizers (ggml-quants.c, MIT): Q8_0, Q2_K and Q4_K, the target types of the family converter
/// (dense matrices Q8_0, routed experts Q2_K or Q4_K). The arithmetic order and `nearest_int` follow ggml's
/// `quantize_row_*_ref`; the checks verify the round-trip error, not byte identity with llama.cpp (whose bytes also
/// depend on its compiler fusing multiply-adds). Values must be finite (the converter checks).
package enum GGUFQuantize {
    /// ggml's nearest_int: round half to even through the float mantissa trick.
    static func nearest(_ f: Float) -> Int {
        let v = f + 12_582_912
        return Int(Int32(bitPattern: v.bitPattern) & 0x007f_ffff) - 0x0040_0000
    }
    static func half(_ f: Float) -> UInt16 { Float16(f).bitPattern }
    static func fromHalf(_ h: UInt16) -> Float { Float(Float16(bitPattern: h)) }

    /// Bytes of `count` values (whole blocks) of `type`.
    package static func bytes(_ type: Int, count: Int) -> Int {
        let (block, size) = GGUFQuant.sizes[type]!
        return count / block * size
    }

    /// Encodes `count` values (whole blocks) of `x` as `type` (F32 0, Q8_0 8, Q2_K 10, Q4_K 12, BF16 30) into `out`.
    package static func encode(_ x: UnsafePointer<Float>, count: Int, type: Int, into out: UnsafeMutableRawPointer) {
        switch type {
        case 0: memcpy(out, x, count * 4)
        case 30: for i in 0..<count { out.storeBytes(of: BF16.encode(x[i]), toByteOffset: i * 2, as: UInt16.self) }
        case 8: for b in 0..<(count / 32) { q8_0(x + b * 32, out + b * 34) }
        case 10: for b in 0..<(count / 256) { q2K(x + b * 256, out + b * 84) }
        case 12: for b in 0..<(count / 256) { q4K(x + b * 256, out + b * 144) }
        default: preconditionFailure("GGUFQuantize cannot encode type \(type)")
        }
    }

    static func q8_0(_ x: UnsafePointer<Float>, _ y: UnsafeMutableRawPointer) {
        var amax: Float = 0
        for j in 0..<32 { amax = max(amax, abs(x[j])) }
        let d = amax / 127
        let id: Float = d != 0 ? 1 / d : 0
        y.storeBytes(of: half(d), as: UInt16.self)
        for j in 0..<32 { y.storeBytes(of: Int8((x[j] * id).rounded(.toNearestOrAwayFromZero)), toByteOffset: 2 + j, as: Int8.self) }
    }

    /// make_qkx2_quants: the scale and (negated) min of `n` values on `nmax` levels by a weighted least-squares search.
    static func qkx2(
        _ n: Int, _ nmax: Int, _ x: UnsafePointer<Float>, _ weights: UnsafePointer<Float>, _ L: UnsafeMutablePointer<UInt8>,
        _ aux: UnsafeMutablePointer<UInt8>, _ theMin: inout Float, rmin: Float, rdelta: Float, steps: Int, mad: Bool
    ) -> Float {
        var lo = x[0]
        var hi = x[0]
        var sumW = weights[0]
        var sumX = sumW * x[0]
        for i in 1..<n {
            lo = min(lo, x[i])
            hi = max(hi, x[i])
            sumW += weights[i]
            sumX += weights[i] * x[i]
        }
        if lo > 0 { lo = 0 }
        if hi == lo {
            for i in 0..<n { L[i] = 0 }
            theMin = -lo
            return 0
        }
        var iscale = Float(nmax) / (hi - lo)
        var scale = 1 / iscale
        var best: Float = 0
        for i in 0..<n {
            let l = nearest(iscale * (x[i] - lo))
            L[i] = UInt8(max(0, min(nmax, l)))
            var diff = scale * Float(L[i]) + lo - x[i]
            diff = mad ? abs(diff) : diff * diff
            best += weights[i] * diff
        }
        if steps < 1 {
            theMin = -lo
            return scale
        }
        for step in 0...steps {
            iscale = (rmin + rdelta * Float(step) + Float(nmax)) / (hi - lo)
            var sumL: Float = 0
            var sumL2: Float = 0
            var sumXL: Float = 0
            for i in 0..<n {
                let l = max(0, min(nmax, nearest(iscale * (x[i] - lo))))
                aux[i] = UInt8(l)
                let w = weights[i]
                let lf = Float(l)
                sumL += w * lf
                sumL2 += w * lf * lf
                sumXL += w * lf * x[i]
            }
            let D = sumW * sumL2 - sumL * sumL
            if D > 0 {
                var thisScale = (sumW * sumXL - sumX * sumL) / D
                var thisMin = (sumL2 * sumX - sumL * sumXL) / D
                if thisMin > 0 {
                    thisMin = 0
                    thisScale = sumXL / sumL2
                }
                var error: Float = 0
                for i in 0..<n {
                    var diff = thisScale * Float(aux[i]) + thisMin - x[i]
                    diff = mad ? abs(diff) : diff * diff
                    error += weights[i] * diff
                }
                if error < best {
                    for i in 0..<n { L[i] = aux[i] }
                    best = error
                    scale = thisScale
                    lo = thisMin
                }
            }
        }
        theMin = -lo
        return scale
    }

    static func q2K(_ x: UnsafePointer<Float>, _ y: UnsafeMutableRawPointer) {
        withUnsafeTemporaryAllocation(of: UInt8.self, capacity: 256 + 16 + 16) { bytes in
            withUnsafeTemporaryAllocation(of: Float.self, capacity: 16 + 16 + 16) { floats in
                let L = bytes.baseAddress!
                let aux = L + 256
                let sc = L + 272
                let mins = floats.baseAddress!
                let scales = mins + 16
                let w = mins + 32
                var maxScale: Float = 0
                var maxMin: Float = 0
                for j in 0..<16 {
                    for i in 0..<16 { w[i] = abs(x[16 * j + i]) }
                    scales[j] = qkx2(16, 3, x + 16 * j, w, L + 16 * j, aux, &mins[j], rmin: -0.5, rdelta: 0.1, steps: 15, mad: true)
                    maxScale = max(maxScale, scales[j])
                    maxMin = max(maxMin, mins[j])
                }
                for j in 0..<16 { sc[j] = 0 }
                var d: UInt16 = 0
                var dmin: UInt16 = 0
                if maxScale > 0 {
                    let i = 15 / maxScale
                    for j in 0..<16 { sc[j] = UInt8(truncatingIfNeeded: nearest(i * scales[j])) }
                    d = half(maxScale / 15)
                }
                if maxMin > 0 {
                    let i = 15 / maxMin
                    for j in 0..<16 { sc[j] |= UInt8(truncatingIfNeeded: nearest(i * mins[j]) << 4) }
                    dmin = half(maxMin / 15)
                }
                for j in 0..<16 {
                    let dd = fromHalf(d) * Float(sc[j] & 0xF)
                    if dd == 0 { continue }
                    let dm = fromHalf(dmin) * Float(sc[j] >> 4)
                    for i in 0..<16 { L[16 * j + i] = UInt8(max(0, min(3, nearest((x[16 * j + i] + dm) / dd)))) }
                }
                for j in 0..<16 { y.storeBytes(of: sc[j], toByteOffset: j, as: UInt8.self) }
                for j in stride(from: 0, to: 256, by: 128) {
                    for l in 0..<32 {
                        let q = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6)
                        y.storeBytes(of: q, toByteOffset: 16 + j / 4 + l, as: UInt8.self)
                    }
                }
                y.storeBytes(of: d, toByteOffset: 80, as: UInt16.self)
                y.storeBytes(of: dmin, toByteOffset: 82, as: UInt16.self)
            }
        }
    }

    static func q4K(_ x: UnsafePointer<Float>, _ y: UnsafeMutableRawPointer) {
        withUnsafeTemporaryAllocation(of: UInt8.self, capacity: 256 + 32 + 12) { bytes in
            withUnsafeTemporaryAllocation(of: Float.self, capacity: 8 + 8 + 32) { floats in
                let L = bytes.baseAddress!
                let aux = L + 256
                let packed = L + 288
                let mins = floats.baseAddress!
                let scales = mins + 8
                let w = mins + 16
                var maxScale: Float = 0
                var maxMin: Float = 0
                for j in 0..<8 {
                    var sumX2: Float = 0
                    for i in 0..<32 { sumX2 += x[32 * j + i] * x[32 * j + i] }
                    let av = (sumX2 / 32).squareRoot()
                    for i in 0..<32 { w[i] = av + abs(x[32 * j + i]) }
                    scales[j] = qkx2(32, 15, x + 32 * j, w, L + 32 * j, aux, &mins[j], rmin: -1, rdelta: 0.1, steps: 20, mad: false)
                    maxScale = max(maxScale, scales[j])
                    maxMin = max(maxMin, mins[j])
                }
                let invScale: Float = maxScale > 0 ? 63 / maxScale : 0
                let invMin: Float = maxMin > 0 ? 63 / maxMin : 0
                for j in 0..<12 { packed[j] = 0 }
                for j in 0..<8 {
                    let ls = min(63, UInt8(truncatingIfNeeded: nearest(invScale * scales[j])))
                    let lm = min(63, UInt8(truncatingIfNeeded: nearest(invMin * mins[j])))
                    if j < 4 {
                        packed[j] = ls
                        packed[j + 4] = lm
                    } else {
                        packed[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4)
                        packed[j - 4] |= (ls >> 4) << 6
                        packed[j] |= (lm >> 4) << 6
                    }
                }
                let d = half(maxScale / 63)
                let dmin = half(maxMin / 63)
                for j in 0..<8 {
                    let sc = j < 4 ? packed[j] & 63 : (packed[j + 4] & 0xF) | ((packed[j - 4] >> 6) << 4)
                    let m = j < 4 ? packed[j + 4] & 63 : (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4)
                    let dd = fromHalf(d) * Float(sc)
                    if dd == 0 { continue }
                    let dm = fromHalf(dmin) * Float(m)
                    for i in 0..<32 { L[32 * j + i] = UInt8(max(0, min(15, nearest((x[32 * j + i] + dm) / dd)))) }
                }
                y.storeBytes(of: d, toByteOffset: 0, as: UInt16.self)
                y.storeBytes(of: dmin, toByteOffset: 2, as: UInt16.self)
                for j in 0..<12 { y.storeBytes(of: packed[j], toByteOffset: 4 + j, as: UInt8.self) }
                for j in stride(from: 0, to: 256, by: 64) {
                    for l in 0..<32 { y.storeBytes(of: L[j + l] | (L[j + l + 32] << 4), toByteOffset: 16 + j / 2 + l, as: UInt8.self) }
                }
            }
        }
    }
}
