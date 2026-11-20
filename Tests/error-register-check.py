#!/usr/bin/env python3
"""Checks a built arm64 binary for a compiler fault seen with Swift 6.3.3 and 6.4: a rethrowing function specialized
for a Void closure that only copies bytes ends in a tail call to memmove with a temporary still in x21, Swift's error
register, so its `try` caller takes the buffer's address for a thrown error (3.1.0 built with Swift 6.4 crashed on the
first request this way). For every tail call (a jump out of a function right after its frame is put back) the last
thing put in x21 before it, in the order the instructions are laid out, must come from a saved register, zero or the
stack. The fix in the sources is a non-inlined helper with a Bool result (`GDNState.copyConvolutionBytes`).
Usage: Tests/error-register-check.py [.build/release/origami ...]"""
import re, subprocess, sys

reg = re.compile(r"\b[xw]21\b")
keeps = re.compile(r"^(mov\s+x21, (x(19|2[0-8])|#0x0|xzr)|(ldu?r|ldp)\s+[^\[]*\[(sp|x29|x19))")
reads = {"str", "stp", "stur", "cmp", "cmn", "tst", "cbz", "cbnz", "tbz", "tbnz", "bl", "blr"}


def faults(lines, start, end):
    found = []
    for i, line in enumerate(lines):
        fields = line.split("\t")
        if len(fields) < 3 or fields[1].strip() != "b": continue
        try: address = int(fields[2].split(";")[0].strip(), 16)
        except ValueError: continue
        if start <= address < end or not any("ldp" in l and "x29, x30" in l for l in lines[max(0, i - 6):i]): continue
        for back in range(i - 1, -1, -1):
            f = lines[back].split("\t")
            if len(f) < 3: continue
            mnemonic, operands = f[1].strip(), f[2].split(";")[0].strip()
            # Another exit, or the start of an unnamed function: what is before it belongs to another path.
            if mnemonic in ("ret", "b", "br") or (mnemonic == "stp" and "x29, x30, [sp, #-" in operands): break
            parts = [p.strip() for p in operands.split(",")]
            written = mnemonic not in reads and (
                reg.fullmatch(parts[0]) or (mnemonic.startswith("ldp") and len(parts) > 1 and reg.fullmatch(parts[1])))
            if written:
                if not keeps.match(mnemonic + " " + operands): found.append((mnemonic + " " + operands, " ".join(fields[1:]).strip()))
                break
    return found


def check(binary):
    text = subprocess.run(["objdump", "-d", "--no-show-raw-insn", "--macho", binary], capture_output=True, text=True, check=True).stdout
    flagged, name, lines, start = [], None, [], None

    def flush():
        if name and lines and (found := faults(lines, start, int(lines[-1].split(":")[0], 16) + 4)): flagged.append((name, found))

    for line in text.splitlines():
        if line.endswith(":") and "\t" not in line and not line.startswith(" "):
            flush()
            name, lines, start = line[:-1], [], None
        elif "\t" in line and name is not None:
            if start is None: start = int(line.split(":")[0], 16)
            lines.append(line)
    flush()
    shown = subprocess.run(
        ["xcrun", "swift-demangle"], input="\n".join(n for n, _ in flagged), capture_output=True, text=True).stdout.splitlines()
    for (_, found), demangled in zip(flagged, shown):
        print("FAIL  " + demangled.split(" ---> ")[-1])
        for write, jump in found: print("      x21 last set by `%s` before `%s`" % (write, jump))
    return len(flagged)


def self_check():
    good = ["100:\tstp\tx29, x30, [sp, #-0x10]!", "104:\tmov\tx21, x0", "108:\tmov\tx21, x26", "10c:\tldp\tx29, x30, [sp], #0x10", "110:\tb\t0x900 ; _memmove"]
    bad = good[:2] + good[3:]
    assert not faults(good, 0x100, 0x114) and len(faults(bad, 0x100, 0x110)) == 1, "the check no longer tells the two apart"
    # A scratch value in an earlier, unnamed function is not this exit's.
    assert not faults(["f0:\tmov\tx21, x0", "f4:\tret"] + [good[0]] + good[3:], 0xf0, 0x110)


self_check()
binaries = sys.argv[1:] or [".build/release/origami"]
count = sum(check(binary) for binary in binaries)
print("%d function%s with a temporary in the error register at a tail call (%s)" % (count, "" if count == 1 else "s", ", ".join(binaries)))
sys.exit(1 if count else 0)
