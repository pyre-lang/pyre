# pyre-check: max-pypy-ratio=16.4
# N/ITERS are sized to prove the opcode compiles and to drive the compiled
# loop thousands of times, not to race pypy. The ceiling is 15% over the
# widest gated reading (windows dynasm 14.2x).
N = 600
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
