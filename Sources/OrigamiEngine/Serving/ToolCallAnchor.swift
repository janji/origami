import Foundation

/// One marker match per reply. Text fallbacks consume only the newly decoded bytes;
/// partial markers survive token boundaries without retaining or rescanning the reply.
package struct ToolCallAnchorTracker {
    private let openers: Set<Int>
    private let patterns: [[UInt8]]
    private let failures: [[Int]]
    private var matched: [Int]
    private var finished = false

    package init(openers: Set<Int>, markers: [String] = TokenizerStack.toolCallMarkers) {
        self.openers = openers
        patterns = markers.map { Array($0.utf8) }.filter { !$0.isEmpty }
        failures = patterns.map { pattern in
            var table = [Int](repeating: 0, count: pattern.count)
            var n = 0
            for i in 1..<pattern.count {
                while n > 0 && pattern[n] != pattern[i] { n = table[n - 1] }
                if pattern[n] == pattern[i] { n += 1 }
                table[i] = n
            }
            return table
        }
        matched = [Int](repeating: 0, count: patterns.count)
    }

    package mutating func observe(token: Int, delta: String) -> Bool {
        guard !finished else { return false }
        if openers.contains(token) {
            finished = true
            return true
        }
        guard !patterns.isEmpty else { return false }
        for byte in delta.utf8 {
            for i in patterns.indices {
                var n = matched[i]
                while n > 0 && patterns[i][n] != byte { n = failures[i][n - 1] }
                if patterns[i][n] == byte { n += 1 }
                if n == patterns[i].count {
                    finished = true
                    return true
                }
                matched[i] = n
            }
        }
        return false
    }
}
