import Foundation
import OrigamiEngine

// Gold values recorded from MLX's own scalar-dot kernel (mlx.metallib at the pinned commit) on these seeded
// inputs, not from this kernel or its dispatch transcription: FP32 then BF16 outputs, seeds 1...64.
func pinnedDenseScalarDotMatchesOriginalBinary() throws {
    let expected: [[UInt32]] = [
        [
            3_323_126_903, 3_349_598_132, 1_183_437_362, 3_339_133_606, 1_193_050_420, 1_186_016_989, 3_342_330_412, 3_331_911_590,
            3_323_915_289, 1_132_844_928, 1_193_459_738, 1_200_146_152, 3_336_616_855, 1_186_654_197, 1_176_453_775, 1_193_590_881,
            3_327_283_602, 3_333_964_941, 1_185_089_168, 1_197_685_855, 3_346_691_824, 3_341_773_395, 3_321_160_562, 1_206_764_595,
            1_169_270_868, 3_322_396_920, 1_187_474_594, 1_177_439_499, 3_332_913_801, 1_192_397_358, 3_329_222_474, 1_193_027_936,
            1_172_082_730, 1_199_287_883, 1_201_156_921, 3_349_982_929, 1_185_873_431, 1_189_372_542, 1_186_002_243, 3_331_559_509,
            3_341_431_712, 3_344_399_238, 1_150_747_744, 3_338_132_730, 1_172_209_870, 1_190_957_813, 3_310_737_576, 3_287_299_712,
            1_191_558_090, 1_168_475_552, 1_148_072_384, 1_175_747_453, 1_196_396_876, 1_192_224_445, 1_164_392_624, 3_344_824_574,
            1_182_875_290, 1_187_353_534, 1_180_559_576, 1_194_441_946, 3_307_944_848, 1_194_970_008, 3_346_656_023, 1_192_753_349,
        ],
        [
            3_323_068_416, 3_349_544_960, 1_183_514_624, 3_339_124_736, 1_193_017_344, 1_186_070_528, 3_342_336_000, 3_331_981_312,
            3_323_985_920, 1_134_100_480, 1_193_476_096, 1_200_160_768, 3_336_699_904, 1_186_660_352, 1_176_633_344, 1_193_607_168,
            3_327_262_720, 3_333_881_856, 1_185_087_488, 1_197_735_936, 3_346_661_376, 3_341_811_712, 3_320_840_192, 1_206_779_904,
            1_169_227_776, 3_322_478_592, 1_187_577_856, 1_177_288_704, 3_332_898_816, 1_192_361_984, 3_329_294_336, 1_193_017_344,
            1_171_718_144, 1_199_308_800, 1_201_143_808, 3_350_003_712, 1_185_808_384, 1_189_347_328, 1_186_004_992, 3_331_653_632,
            3_341_418_496, 3_344_433_152, 1_150_877_696, 3_338_141_696, 1_172_242_432, 1_190_920_192, 3_310_223_360, 3_285_778_432,
            1_191_575_552, 1_168_572_416, 1_147_535_360, 1_175_846_912, 1_196_359_680, 1_192_230_912, 1_164_312_576, 3_344_826_368,
            1_182_793_728, 1_187_315_712, 1_180_499_968, 1_194_393_600, 3_307_536_384, 1_194_983_424, 3_346_595_840, 1_192_755_200,
        ],
    ]
    let ledger = try MemoryLedger(ceiling: 32_000_000, transientMargin: 1_000_000)
    let cpu = try ledger.reserve(bytes: 1_000_000, category: "scalar_dot_gold_inputs")
    do {
        let dense = try PinnedDenseLinear(ledger: ledger)
        for (dtype, bf16) in [false, true].enumerated() {
            for seed in 1...64 {
                var state = UInt64(seed)
                func values() -> [Float] {
                    (0..<2560).map { _ in
                        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                        let bits = UInt32(truncatingIfNeeded: state >> 32)
                        let exponent = UInt32(119) + ((bits >> 21) % 15)
                        let value = Float(bitPattern: (bits & 0x807f_ffff) | (exponent << 23))
                        return bf16 ? BF16.decode(BF16.encode(value)) : value
                    }
                }
                let input = values()
                let weights = values()
                let output = try dense.linear(input, weights: weights, width: 2560, outputs: 1, outputBF16: bf16)
                guard output.count == 1, output[0].bitPattern == expected[dtype][seed - 1] else {
                    throw EngineError(
                        "Scalar dot differs from original BASE binary: bf16=\(bf16), seed=\(seed), actual=\(output.map(\.bitPattern))")
                }
                guard ledger.waitForRelease(until: { reserved, _ in reserved == 1_000_000 }) else {
                    throw EngineError("Scalar dot retained transient buffers")
                }
            }
        }
    } catch {
        try? ledger.release(cpu)
        throw error
    }
    try ledger.release(cpu)
    guard ledger.snapshot().reserved == 0 else { throw EngineError("Scalar dot fixture retained memory") }
}
