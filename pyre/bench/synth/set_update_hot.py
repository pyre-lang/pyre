# pyre-check: max-pypy-ratio=24
# pyre-check: jitstats-band=guard_failures=15
# dynasm guard_failures reads 200, 206, 215 on one macOS arm64 binary at
# PYPY_GC_NURSERY 8M, 4M, 2M, and 200 vs 208 on two main builds of the same
# source: `counter.py invoke_after_minor_collection` runs
# `decay_all_counters` every 32 minor collections, so the count follows
# nursery traffic. The width is the 200 -> 215 distance.
# N/ITERS are sized to prove the opcode compiles and to drive the compiled
# loop thousands of times, not to race pypy. The ceiling is 15% over the
# widest gated reading (macos dynasm 20.9x).
N = 1440
ITERS = 500


def run(n, base):
    # `{i, *base}` compiles to BUILD_SET + SET_UPDATE in a while-loop body.
    # Before SET_UPDATE was lowered, its abort_permanent marker declined
    # the whole loop.
    total = 0
    i = 0
    while i < n:
        s = {i, *base}
        total += len(s)
        i += 1
    return total


def main():
    base = [10, 20, 30]
    total = 0
    for _ in range(ITERS):
        total += run(N, base)
    print(total)


main()
