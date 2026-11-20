import OrigamiEngine

func pleWrappingAndEOSHash() throws {
    let multipliers: [Int64] = [(1 << 62) + 1, 3, 5]
    let sizes = (0..<16).map { Int64(97 + $0) }
    let offsets = (0..<16).map { Int64($0 * 1000) }
    let expected = [45, 1003, 2061, 3045, 4018, 5013, 6057, 7077, 8037, 9061, 10060, 11097, 12004, 13097, 14022, 15065]
    let after = try PLEHash.rows(history: [5, 6, 248044, 9], newCount: 1, multipliers: multipliers, sizes: sizes, offsets: offsets)
    let fresh = try PLEHash.rows(history: [9], newCount: 1, multipliers: multipliers, sizes: sizes, offsets: offsets)
    guard after == [expected], fresh == after else { throw EngineError("Signed PLE wrapping or EOS isolation changed") }
    let history: [Int64] = [248044, 248044, 15, 16, 17, 18, 19]
    let full = try PLEHash.rows(history: history, newCount: 5, multipliers: multipliers, sizes: sizes, offsets: offsets)
    let suffix = try PLEHash.rows(history: Array(history.suffix(4)), newCount: 2, multipliers: multipliers, sizes: sizes, offsets: offsets)
    guard Array(full.suffix(2)) == suffix else { throw EngineError("PLE continuation lost hash history") }
}
