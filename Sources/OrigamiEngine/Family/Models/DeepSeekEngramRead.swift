import Foundation

/// Engram hashes depend only on the current tokens and the preceding ngram-1 tokens, so only that suffix of the
/// conversation is kept for them.
package enum DeepSeekEngramHash {
    package static func rows(
        tokens: [Int], history: [Int], layer: Int, ngram: Int, heads: Int,
        multipliers: [UInt64], primes: [UInt64], offsets: [Int64],
        tokenMap: [Int32], pad: Int64
    ) -> [Int64] {
        let tail = history.suffix(ngram - 1)
        let all = Array(tail) + tokens
        let start = tail.count
        let cols = (ngram - 1) * heads
        var ids = [Int64](repeating: 0, count: tokens.count * cols)
        for i in tokens.indices {
            let pos = start + i
            func cid(_ t: Int) -> Int64? { t >= 0 && t < tokenMap.count ? Int64(tokenMap[t]) : nil }
            var toks = [UInt64](repeating: 0, count: ngram)
            var blocked = false
            for s in 0..<ngram {
                let c: Int64? = pos - s < 0 ? nil : cid(all[pos - s])
                blocked = blocked || c == nil
                toks[s] = UInt64(bitPattern: blocked ? pad : c!)
            }
            var rolling = toks[0] &* multipliers[layer * ngram]
            for order in 1..<ngram {
                rolling ^= toks[order] &* multipliers[layer * ngram + order]
                for head in 0..<heads {
                    let col = (order - 1) * heads + head
                    ids[i * cols + col] = Int64(rolling % primes[layer * cols + col]) + offsets[layer * cols + col]
                }
            }
        }
        return ids
    }
}
