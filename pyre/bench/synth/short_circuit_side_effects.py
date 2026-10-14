# pyre-check: max-pypy-ratio=30
# pyre-check: spec-folds=subscr
# pyre-check: trace-shape=main:absent=CallMayForceI,absent=CallMayForceR
# The and/or chains below present a non-bool truthiness to the walker.
# Short-circuit and/or chains whose operands call side-effecting helpers,
# inside a hot loop whose first operand flips truthiness after warm-up. The
# helpers bump global counters; the computed totals were always correct, but
# the counters over-counted under the JIT because an aborted trace (or a
# guard-failure replay) re-executed iterations whose side effects had already
# committed once. Output is byte-exact vs the interpreter only if every
# helper call runs exactly once per iteration across warm-up, tracing,
# aborts, bridges, and deopts.
counters = [0, 0, 0]


def side_a(v):
    counters[0] = counters[0] + 1
    return v


def side_b(v):
    counters[1] = counters[1] + 1
    return v


def side_c(v):
    counters[2] = counters[2] + 1
    return v


def main():
    total = 0
    i = 0
    N = 40000
    while i < N:
        # Late flip: the and-chain's first operand is truthy until i=12000,
        # then always falsy so the chain short-circuits early.
        if i < 12000:
            x = i % 7 + 1
        else:
            x = 0

        if side_a(x) and side_b(i % 5) and side_c(i % 11 + 1):
            total += 1
        else:
            total -= 2

        if side_a(x) or side_b(i % 4) or side_c(i % 9):
            total += 3
        else:
            total -= 4

        if (side_a(x) and side_b(i % 3)) or side_c(i % 2):
            total += 5

        if i % 4000 == 3999:
            print("ck", i, total, counters[0], counters[1], counters[2])
        i += 1

    print("final", total, counters[0], counters[1], counters[2])


main()
