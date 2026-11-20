import CExactIO
import OrigamiEngine

/// Counter fixtures only: no host probes, allocations, model loading, or GPU access.
func availableMemoryCountsSpeculativePagesOnce() throws {
    func require(_ value: Bool, _ message: String) throws { if !value { throw EngineError(message) } }
    // Disjoint physical cohorts: 100 unused + 40 speculative file + 70 other file + 30
    // purgeable pages. XNU reports speculative pages inside both free and external counters.
    for pageSize in [UInt64(4096), 16384] {
        let actual = origami_available_memory_from_pages(140, 40, 30, 110, pageSize)
        try require(actual == 240 * pageSize, "Speculative pages were counted more than once")
    }
    // Disk warming moves unused pages to speculative file cache; it creates no additional
    // physical headroom. The old formula instead invented two pages for every page warmed.
    let pageSize: UInt64 = 16384
    let initiallyUnused: UInt64 = 524288
    let ordinaryFile: UInt64 = 393216
    let purgeable: UInt64 = 16384
    let expectedBytes: UInt64 = 15_300_820_992  // 8 GiB unused + 6 GiB file + 256 MiB purgeable.
    for warmed in [UInt64(0), 1, 41323, 262144, 524288] {
        let actual = origami_available_memory_from_pages(
            initiallyUnused, warmed, purgeable,
            ordinaryFile + warmed, pageSize)
        try require(actual == expectedBytes, "Warming file cache changed physical headroom")
    }
    // All free-reported pages can be speculative. Count them through external exactly once.
    try require(
        origami_available_memory_from_pages(80, 80, 0, 80, 16384) == 1_310_720,
        "A wholly speculative free cohort was dropped or duplicated")
    try require(
        origami_available_memory_from_pages(0, 0, 0, 0, 16384) == 0,
        "An empty snapshot invented available memory")
}

func availableMemoryRejectsInvalidCounters() throws {
    func require(_ value: Bool, _ message: String) throws { if !value { throw EngineError(message) } }
    let invalid: [(UInt64, UInt64, UInt64, UInt64, UInt64)] = [
        (10, 11, 0, 11, 16384),  // speculative subtraction underflow
        (1, 0, 0, 0, 0),  // unavailable page size
        (1, 0, 0, 0, 3),  // invalid page size
        (.max, 0, 0, 1, 1),  // external addition overflow
        (.max - 1, 0, 2, 0, 1),  // purgeable addition overflow
        (.max / 16384 + 1, 0, 0, 0, 16384),  // byte conversion overflow
    ]
    for (free, speculative, purgeable, external, pageSize) in invalid {
        try require(
            origami_available_memory_from_pages(free, speculative, purgeable, external, pageSize) == 0,
            "Invalid counters became an admissible memory budget")
    }
    // The last representable whole-page result remains valid rather than wrapping or rounding up.
    let largestPages = UInt64.max / 16384
    try require(
        origami_available_memory_from_pages(largestPages, 0, 0, 0, 16384) == UInt64.max - 16383,
        "Largest representable page count was converted incorrectly")
    // Actual kernel fields are 32-bit counters; widening must happen before additions.
    let fieldMaximum = UInt64(UInt32.max)
    try require(
        origami_available_memory_from_pages(fieldMaximum, fieldMaximum, fieldMaximum, fieldMaximum, 16384)
            == 140_737_488_322_560, "32-bit VM counters were added before widening")
}
