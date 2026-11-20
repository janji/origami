/// `expert_qmv_rows8`: one-row expert QMV for up to eight input rows per SIMD group (grid: output row x ceil(rows/8),
/// 32 lanes each), with one accumulator per row sharing the packed-weight and scale/bias loads. Reads an expert
/// projection in the expert format (QFP4 macro: affine 4-bit or NVFP4).
/// shape = [width, outputs, rows, group, -, -, perLane]; perLane is 16 or 8 and must lie inside one quantization group.
/// Each row keeps the pinned one-row lane schedule (four-product tree, BF16 bias-sum boundaries, one SIMD sum), so
/// every row is BITWISE equal to `qmv_pinned_reference` on that row alone.
enum ExpertMultirowQMV {
    static let source = #"""
        kernel void expert_qmv_rows8(device const ushort *x [[buffer(0)]], device const uint *words [[buffer(1)]],
            device const ushort *scales [[buffer(2)]], device const ushort *biases [[buffer(3)]],
            constant uint *shape [[buffer(4)]], device float *out [[buffer(5)]],
            uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
            uint width=shape[0], outputs=shape[1], rows=shape[2], group=shape[3], perLane=shape[6];
            uint block=tid/32, row=block%outputs, first=(block/outputs)*8;
            float result[8]={0,0,0,0,0,0,0,0};
            for(uint lo=lane*perLane;lo<width;lo+=32*perLane) {
                uint index=row*(width/group)+lo/group;
                float scale=QSCALE(scales,biases,index), bias=QBIAS(biases,index);
                uint packed[2];
                packed[0]=words[row*(width/8)+lo/8];
                if(perLane==16) packed[1]=words[row*(width/8)+lo/8+1];
                #pragma unroll
                for(uint r=0;r<8;r++) {
                    if(first+r>=rows) continue;
                    float quantSum=0.0f,biasSum=0.0f;
                    for(uint i=0;i<perLane;i+=4) {
                        float values[4],products[4];
                        #pragma unroll
                        for(uint j=0;j<4;j++) {
                            values[j]=from_bf16(x[(first+r)*width+lo+i+j]);
                            uint nibble=(packed[(i+j)/8]>>(((i+j)%8)*4))&15;
                            products[j]=values[j]*QVAL(nibble);
                        }
                        biasSum=biasSum+round_bf16(round_bf16(round_bf16(values[0]+values[1])+values[2])+values[3]);
                        quantSum=quantSum+(((products[0]+products[1])+products[2])+products[3]);
                    }
                    float dot=fma(scale,quantSum,biasSum*bias);
                    result[r]=result[r]+dot;
                }
            }
            #pragma unroll
            for(uint r=0;r<8;r++) {
                float value=simd_sum(result[r]);
                if(lane==0 && first+r<rows) out[(first+r)*outputs+row]=round_bf16(value);
            }
        }
        """#
}
