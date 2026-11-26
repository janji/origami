#!/usr/bin/env python3
"""Would one expert record read from two drives at once, split by their speeds, finish sooner than a whole read?

Reads page-aligned record-sized ranges at random offsets of two existing files (read-only, uncached: F_NOCACHE), one
per drive, interleaving per round: (a) whole from drive A, (b) whole from drive B, (c) split A|B sized by the drives'
live speeds over the previous rounds, both parts in parallel, (d) split 50/50. With --concurrent N: N reads at once,
each (a) whole from A, (b) whole from B, (c) split, or (e) spread (half whole from A, half whole from B). Prints JSON: median, p90, p99
milliseconds per variant, the live speeds and the bytes read from B. Stops early once B's bytes reach --max-b-gb.

  split_reads.py A_FILE B_FILE --record 2764800 --rounds 300 [--concurrent 4] [--max-b-gb 14]
"""
import argparse, fcntl, json, mmap, os, random, statistics, threading, time
from concurrent.futures import ThreadPoolExecutor

F_NOCACHE, ALIGN = 48, 16384


def opened(path):
    fd = os.open(path, os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    return fd, os.fstat(fd).st_size


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a"), p.add_argument("b")
    p.add_argument("--record", type=int, required=True)
    p.add_argument("--rounds", type=int, default=300)
    p.add_argument("--concurrent", type=int, default=1)
    p.add_argument("--max-b-gb", type=float, default=14)
    args = p.parse_args()
    drives = [opened(args.a), opened(args.b)]
    n, record = args.concurrent, args.record
    buffers = [mmap.mmap(-1, record) for _ in range(n)]  # page-aligned, one per read in flight
    pool = ThreadPoolExecutor(max_workers=4 * n)
    rate = [2.5e9, 2.5e9]  # live bytes/s per drive (moving average over completed parts), seeded equal
    b_bytes, lock = 0, threading.Lock()

    def read(d, view, length):
        nonlocal b_bytes
        fd, size = drives[d]
        offset = random.randrange(0, (size - length) // ALIGN) * ALIGN
        start = time.perf_counter()
        done = os.preadv(fd, [view[:length]], offset)
        seconds = time.perf_counter() - start
        assert done == length, f"short read {done} of {length}"
        with lock:
            rate[d] += 0.1 * (length / seconds - rate[d])
            if d == 1:
                b_bytes += length
        return seconds

    def one(variant, i):
        view = memoryview(buffers[i])
        if variant in "ab":
            return pool.submit(read, "ab".index(variant), view, record)
        share = rate[0] / (rate[0] + rate[1]) if variant == "c" else 0.5
        first = max(ALIGN, min(record - ALIGN, round(record * share / ALIGN) * ALIGN))
        parts = [pool.submit(read, 0, view, first), pool.submit(read, 1, view[first:], record - first)]
        return pool.submit(lambda: [f.result() for f in parts])

    variants = ("a", "b", "c", "d") if n == 1 else ("a", "b", "c", "e")
    wall = {v: [] for v in variants}
    rounds = 0
    for _ in range(args.rounds):
        if b_bytes >= args.max_b_gb * 1e9:
            break
        for v in variants:
            start = time.perf_counter()
            reads = [one(("a" if i < n // 2 else "b") if v == "e" else v, i) for i in range(n)]
            for r in reads:
                r.result()
            wall[v].append((time.perf_counter() - start) * 1e3)  # all N reads done (one decode layer's misses)
        rounds += 1

    def q(xs, f):
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(f * len(xs)))], 3)

    print(json.dumps({
        "record": record, "concurrent": n, "rounds": rounds, "b_gb": round(b_bytes / 1e9, 2),
        "live_gbps": [round(r / 1e9, 2) for r in rate],
        "ms": {v: {"median": round(statistics.median(w), 3), "p90": q(w, 0.9), "p99": q(w, 0.99)} for v, w in wall.items()},
        "raw_ms": wall,
    }))


if __name__ == "__main__":
    main()
