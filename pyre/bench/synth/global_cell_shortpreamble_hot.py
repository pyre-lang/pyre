# pyre-check: max-pypy-ratio=26
# Tightened 46 -> 26 when `seeded_callee_resume` stopped requiring the
# callee's own exception table: execution-only time here fell 46x against a
# same-day build of the parent commit, so the previous headroom is kept and
# then some -- the ceiling moves by less than the measured gain.
# Before that change the runners read this fixture between 8.0x and 22.7x
# and the ceiling was twice the slowest of those.
# A module-global int accumulator carried through an inlined function that
# branches on the loop variable `i`, where the loop body ALSO reads `i` at the
# tail (a branch / dead guard / mid-loop reassignment). The `LOAD_NAME i` folds
# to a getfield of the runtime-minted mutable name cell; the tail read
# re-populates that cache at the end of the preamble, and the short-preamble
# export must NOT hoist it as a loop invariant — the residual STORE_NAME writes
# the cell every iteration, so the peeled body must re-read it live or the
# inlined callee sees the previous iteration's `i` and picks the wrong branch.
# Output is verified against CPython/PyPy.
N = 240000


def process(x, i):
    if i % 3 == 0:
        return x // 97 + 1
    else:
        return -(x % 50000) + 3


def blend(x, i):
    if i % 7 >= 5:
        return x * 3 + 1
    return x + 2


acc = 5
tot = 5
for i in range(N):
    acc = process(acc, i)
    # tail read of i: a mid-loop reassignment on a rare cold arm.
    if i == N - 100:
        acc = -acc - 1
    tot = blend(tot, i)
    tot %= 1000000
    # tail read of i: a dead never-true guard.
    if i < 0:
        pass

print(acc, tot)
