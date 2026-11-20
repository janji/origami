import Metal

/// Row-one HC weight views shared by the whole schedule and decode pre-routing. Each owner
/// keeps its cache leases alive through its last joined command/reader.
package struct DecodeFrontierWeights {
    /// One hyper-connection's row-one weights: the RMS norm (BF16) and injection (FP32) vectors and
    /// its quantized down (10,240 to 320) and up (320 to 10,240) projections.
    package struct HC {
        let norm, injection: MTLBuffer
        package let down, up: FusedHCDecode.Projection
        package init(norm: MTLBuffer, injection: MTLBuffer, down: FusedHCDecode.Projection, up: FusedHCDecode.Projection) {
            self.norm = norm
            self.injection = injection
            self.down = down
            self.up = up
        }
    }
}
