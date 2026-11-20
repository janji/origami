import Metal

extension FamilyRunner {
    /// Projects and pools a consecutive run without retaining keys and gates for completed groups. `key` and
    /// `gate` hold one group of recurrent state: the unfinished group's rows start at zero. The completed pools
    /// remain append-only in `output`. Decode writes directly into that group, so it needs no extra copy dispatch.
    /// A prompt chunk borrows scratch for the carry plus its rows, then retains only its unfinished final group.
    /// The pooling kernel sees the same rows in the same order as a full-history buffer; only its base offsets move.
    @discardableResult
    package func poolRows(
        _ e: MTLComputeCommandEncoder, key: MTLBuffer, gate: MTLBuffer, ape: MTLBuffer,
        output: MTLBuffer, first: Int, rows: Int, dim: Int, pool: Int,
        project: (Ref, Ref) throws -> Void
    ) throws -> (first: Int, count: Int) {
        let carry = first % pool
        let firstPool = first / pool
        let count = (carry + rows) / pool
        let keys: MTLBuffer
        let gates: MTLBuffer
        if rows == 1 {
            keys = key
            gates = gate
        } else {
            keys = try work("pool-key", (carry + rows) * dim)
            gates = try work("pool-gate", (carry + rows) * dim)
            if carry > 0 {
                let (grid, group) = grid2(dim, carry)
                for (source, target) in [(key, keys), (gate, gates)] {
                    try kernels.dispatch(
                        e, "copy_rows", [Ref(source), Ref(target)], ints: [Int32(dim), Int32(carry)], grid: grid, group: group)
                }
            }
        }
        try project(Ref(keys, carry * dim * 4), Ref(gates, carry * dim * 4))
        if count > 0 {
            try kernels.dispatch(
                e, "kpool_pool", [Ref(keys), Ref(gates), Ref(ape), Ref(output, firstPool * dim * 4)],
                ints: [Int32(dim), 0, Int32(count), Int32(pool)],
                grid: MTLSize(width: (dim + 63) / 64, height: count, depth: 1), group: MTLSize(width: 64, height: 1, depth: 1))
        }
        let tail = (carry + rows) % pool
        if rows > 1, tail > 0 {
            let (grid, group) = grid2(dim, tail)
            for (source, target) in [(keys, key), (gates, gate)] {
                try kernels.dispatch(
                    e, "copy_rows", [Ref(source, count * pool * dim * 4), Ref(target)],
                    ints: [Int32(dim), Int32(tail)], grid: grid, group: group)
            }
        }
        return (firstPool, count)
    }
}
