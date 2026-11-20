import Foundation

/// Host reference of the BF16 operation boundaries of MLX's primitives: each step rounds to BF16
/// (round to nearest even, NaN quieted) exactly where MLX's BF16 kernels do. Mathematical FP32
/// sigmoid is separate: the GDN output gate widens its input and does not round the result.
/// These use libm (expf, logf), which need not equal Metal's elementary functions bit for bit;
/// exact serving uses `LayerUnaryArithmetic.pinned`, whose tables hold the GPU's results.
public enum LayerArithmetic {
    @inlinable @inline(__always) public static func round(_ x: Float) -> Float { BF16.decode(BF16.encode(x)) }

    public static func sigmoidFP32(_ x: Float) -> Float {
        let y: Float = 1 / (1 + expf(abs(x)))
        return x < 0 ? y : 1 - y
    }

    public static func sigmoid(_ x: Float) -> Float {
        let input = round(x)
        let exponential = round(expf(abs(input)))
        let denominator = round(1 + exponential)
        let y = round(1 / denominator)
        return input < 0 ? y : round(1 - y)
    }

    public static func silu(_ x: Float) -> Float { round(round(x) * sigmoid(x)) }

    public static func softplus(_ x: Float) -> Float {
        let input = round(x)
        if input.isNaN { return round(input) }
        if input == .infinity { return input }
        if input == -.infinity { return 0 }
        let maximum = max(input, 0)
        let minimum = min(input, 0)
        let exponential = round(expf(round(minimum - maximum)))
        // utils.h log1p(bfloat) widens xp1 and uses Goldberg's correction.
        let xp1: Float = 1 + exponential
        let correction = xp1 == 1 ? exponential : round(exponential * (logf(xp1) / (xp1 - 1)))
        return round(maximum + correction)
    }

    static func l2norm(_ x: [Float], width: Int, scale: Float = 1) throws -> [Float] {
        guard width > 0, !x.isEmpty, x.count % width == 0 else { throw EngineError("Invalid L2 norm geometry") }
        // mlx-swift's MLXArray * Float first converts the scalar to the array's dtype
        // (MLXArray+Ops.swift); q is already BF16 after the L2 norm.
        let typedScale = round(scale)
        var output = x
        for row in stride(from: 0, to: x.count, by: width) {
            var sum: Float = 0
            for column in 0..<width { sum += x[row + column] * x[row + column] }
            let inverse: Float = 1 / sqrtf(sum + 1e-6)
            for column in 0..<width { output[row + column] = round(round(x[row + column] * inverse) * typedScale) }
        }
        return output
    }
}
