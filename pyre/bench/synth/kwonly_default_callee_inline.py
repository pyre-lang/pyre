# pyre-check: max-pypy-ratio=3.2
# pyre-check: trace-shape=main:absent=CallMayForceR
# A hot call into a keyword-only callee that supplies nothing for it, so the
# inline has to fill `step` from `__kwdefaults__` itself.
#
# The gate here is `loops_compiled`, not the ratio. Left residual the callee
# gets a trace of its own and the count reads 2; seeded inline it is folded
# into the caller's and reads 1, with `fbw_walks` and `caro_funcentry`
# following it down. That census is host-independent, which the ratio at this
# size is not -- the loop is well under `EXEC_TIME_FLOOR_S`, so the comparison
# marks it and only the ceiling applies.
#
# The result depends on both the positional argument and the keyword-only
# default, and the aggregate is exact, so a seeding that fills the slot with
# the wrong value or the wrong local changes the printed sum rather than the
# timing.
N = 439024391


def g(x, *, step=1):
    return x + step


def main():
    total = 0
    for i in range(N):
        total += g(i)
    print(total)


main()
