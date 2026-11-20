// Tile8 form of the gathered decode experts: one SIMD group computes eight output rows of one record, sharing the
// activation expansion and BF16 bias sums; each row keeps the pinned one-row QMV lane tree, so the results are BITWISE
// equal to `expert_gathered_qmv10` plus the SiLU-table product. Gate and up (2560 -> 640) fuse with the SiLU table into
// the BF16 product; down (640 -> 2560) writes FP32 into the canonical-rank span the reduction reads. Grid: x = rows / 8
// SIMD groups, y = record; threadgroups of 128. Expert formats via the QBITS/QFP4 macros.
enum GatheredExpertTiledKernel {
    static let source = #"""
        // Eight rows of one projection at byte offset BASE in the record: PER_LANE consecutive inputs per lane per step,
        // one scale/bias per lane chunk (PER_LANE must lie inside a group), FP32 accumulation, BF16-rounded SIMD sums.
        template<uint WIDTH,uint PER_LANE,uint BASE>
        inline void expert_dot_tile8(device const ushort *x,device const uchar *record,uint row,uint lane,thread float *result) {
         device const uint *words=reinterpret_cast<device const uint*>(record+BASE);
         device const ushort *scales=reinterpret_cast<device const ushort*>(record+BASE+QPROJ_CODES);
         device const ushort *biases=reinterpret_cast<device const ushort*>(record+BASE+QPROJ_CODES+QPROJ_META);
         #pragma unroll
         for(uint r=0;r<8;r++)result[r]=0;
         for(uint lo=lane*PER_LANE;lo<WIDTH;lo+=32*PER_LANE) {
          float values[PER_LANE];float biasSum=0;
          #pragma unroll
          for(uint i=0;i<PER_LANE;i+=4) {
           #pragma unroll
           for(uint j=0;j<4;j++)values[i+j]=from_bf16(x[lo+i+j]);
           biasSum=biasSum+round_bf16(round_bf16(round_bf16(values[i]+values[i+1])+values[i+2])+values[i+3]);
          }
          #pragma unroll
          for(uint r=0;r<8;r++) {
           float quantSum=0;
           #pragma unroll
           for(uint i=0;i<PER_LANE;i+=4) {
            uint word=words[(row+r)*(WIDTH/8)+(lo+i)/8];
            float p0=values[i]*QVAL((word>>((i%8)*4))&15);
            float p1=values[i+1]*QVAL((word>>(((i+1)%8)*4))&15);
            float p2=values[i+2]*QVAL((word>>(((i+2)%8)*4))&15);
            float p3=values[i+3]*QVAL((word>>(((i+3)%8)*4))&15);
            quantSum=quantSum+(((p0+p1)+p2)+p3);
           }
           uint metadata=(row+r)*(WIDTH/QGROUP)+lo/QGROUP;
           result[r]=result[r]+fma(QSCALE(scales,biases,metadata),quantSum,biasSum*QBIAS(biases,metadata));
          }
         }
         #pragma unroll
         for(uint r=0;r<8;r++)result[r]=round_bf16(simd_sum(result[r]));
        }
        // Gate and up for eight rows, then product = round_bf16(SiLU table[bf16(gate)] * up) into the record's rank span.
        kernel void expert_gate_up_product(device const ushort *x [[buffer(0)]],
         constant ExpertRecords10 &records [[buffer(1)]],constant uint *ranks [[buffer(2)]],
         device ushort *product [[buffer(3)]],device const ushort *silu [[buffer(4)]],
         uint2 tid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
         uint row=(tid.x/32)*8;device const uchar *record=records.values[tid.y];
         float gate[8],up[8];
         expert_dot_tile8<2560,16,0>(x,record,row,lane,gate);
         expert_dot_tile8<2560,16,QPROJ_BYTES>(x,record,row,lane,up);
         if(lane==0) {
          #pragma unroll
          for(uint r=0;r<8;r++)product[ranks[tid.y]*640+row+r]=ushort(as_type<uint>(round_bf16(from_bf16(silu[as_type<uint>(gate[r])>>16])*up[r]))>>16);
         }
        }
        // Down for eight rows from the rank's BF16 product row; FP32 (BF16-rounded) into out[rank * 2560 + row].
        kernel void expert_down_tile8(device const ushort *product [[buffer(0)]],
         constant ExpertRecords10 &records [[buffer(1)]],constant uint *ranks [[buffer(2)]],
         device float *out [[buffer(3)]],uint2 tid [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
         uint row=(tid.x/32)*8,rank=ranks[tid.y];float values[8];
         expert_dot_tile8<640,8,2*QPROJ_BYTES>(product+rank*640,records.values[tid.y],row,lane,values);
         if(lane==0) {for(uint r=0;r<8;r++)out[rank*2560+row+r]=values[r];}
        }
        """#
}
