import Darwin
import Foundation
import OrigamiEngine

/// What one save may take from hot copies (audit J38, J86): at most about twice its shortfall, also on a
/// volume that frees nothing; never a record from a copy whose drive cannot free part of a file; and a path on a drive
/// not connected keeps that drive's volume.
func diskBudgetCapChecks() -> [(String, () throws -> Void)] {
    [
        ("disk_budget_save_takes_at_most_twice_its_shortfall", diskBudgetSaveCapped),
        ("disk_budget_hot_copy_unpunchable_gives_nothing", diskBudgetUnpunchableGivesNothing),
        ("disk_budget_unplugged_volume_keeps_its_name", diskBudgetUnpluggedVolume),
    ]
}

/// A giver that reports every byte asked as freed while the drive's free space never grows (local snapshots): a save
/// short by X takes no more than 2X, and fails.
private func diskBudgetSaveCapped() throws {
    let budget = DiskBudget()
    let folder = FileManager.default.temporaryDirectory
    final class Taken: @unchecked Sendable { var bytes: UInt64 = 0 }
    let taken = Taken()
    let id = budget.add(
        "copy", folder: { folder }, bytes: { 0 },
        giveBack: { bytes, _ in
            taken.bytes += bytes
            return bytes
        })
    defer { budget.remove(id) }
    let short: UInt64 = 10 << 20
    let fits = budget.makeRoom(short, in: folder, for: "check", floor: 0, free: { 0 })
    guard !fits, taken.bytes <= 2 * short, taken.bytes > 0 else {
        throw EngineError("disk budget: a save short by \(short) took \(taken.bytes) (fits: \(fits))")
    }
}

/// A drive that cannot punch holes: the copy's give-back empties no record (each still read from the copy, also after a
/// reopen), and the keeper gives nothing.
private func diskBudgetUnpunchableGivesNothing() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("origami-punch-\(UUID().uuidString)")
    defer {
        HotCopy.punchForChecks = nil
        try? FileManager.default.removeItem(at: folder)
    }
    final class Source: HotCopySource {
        func hotCopyRecordBytes(_ key: Int) -> Int? { (0..<8).contains(key) ? 16384 : nil }
        func readHotCopyRecord(_ key: Int, into: UnsafeMutableRawPointer) throws { memset(into, Int32(key + 1), 16384) }
    }
    let source = Source()
    let copy = try HotCopy.write(source: source, keys: Array(0..<8), folder: folder, identity: "fixture")
    HotCopy.punchForChecks = false
    do {
        _ = try copy.giveBack([0, 1, 2])
        throw EngineError("disk budget: a copy that cannot free space gave records back")
    } catch let error as EngineError where "\(error)".contains("cannot free part of a file") {}
    let keeper = HotCopyKeeper(
        source: source, identity: "fixture", copy: copy, counts: ([:], [:]),
        place: { _ in .folder(folder, drive: "fixture", budget: UInt64(copy.bytes), free: 0) }, attach: { _ in true }, idle: { false },
        event: { _ in })
    defer { keeper.stop(waiting: 3) }
    let given = keeper.giveBack(1 << 20, for: "check")
    let reopened = HotCopy.open(folder: folder, identity: "fixture", source: source)
    guard given == 0, Set(copy.keys) == Set(0..<8), reopened.map({ Set($0.keys) == Set(0..<8) }) == true else {
        throw EngineError("disk budget: an unpunchable copy emptied records (given \(given), kept \(copy.keys.count))")
    }
}

private func diskBudgetUnpluggedVolume() throws {
    let path = URL(fileURLWithPath: "/Volumes/origami-check-absent-\(UUID().uuidString.prefix(8))/hot/experts.bin")
    let volume = DiskBudget.volume(path)
    guard volume == "/Volumes/" + path.pathComponents[2] else {
        throw EngineError("disk budget: an unplugged drive's path maps to \(volume)")
    }
}
