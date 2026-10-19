# pyre-check: max-pypy-ratio=4
# The measured reading stays under 4, so the ceiling is 4.
# pyre-check: skip-cpython
# Jitcounter decay scales every JitCounter entry down once per 32 minor
# collections (majit-trace/src/counter.rs), so how far a guard's counter has
# advanced when the workload reaches it depends on how much the process has
# allocated so far -- and `guard_failures` then counts collections during
# warm-up rather than a compile decision.  Left at the default this fixture sits
# on the crossing: a 1MB..64MB nursery sweep read 3781/3716/3702/3819/3701/
# 3818/3819/3818/3818 with `bridges_compiled` flipping 18 <-> 19 alongside it,
# and the same knife edge is what made one host read 3819 and another 3826 at
# the same commit.  With `decay=0` the same nine-point sweep reads one number,
# 3792, with `loops_compiled=6` and `bridges_compiled=18` throughout.  The pin
# trades the marginal nineteenth bridge for a number that does not move.
#
# CPython (the oracle) has no `pypyjit`; PyPy and pyre do.  Guarding the import
# keeps the output identical across all three while the param only binds where a
# JIT exists.  `set_param` rather than an environment variable because the wasm
# guest sees no environment.
# cpython 1.86s vs pyre 0.32s (5.8x), and it is not gated on — only pypy is.
# Inline chain of increasing depth whose argument flips int -> float partway
# through the loop, so the type guard deopts with 2, 3 and 7 frames inlined.
# Each depth keeps its own driver loop and its own chain so the trace shape is
# the same as when the depths lived in separate files.  FLIP_AT stays two
# thirds of the way through, so each driver keeps the same pre- and post-flip
# share of its iterations.
#
# N is sized so pypy spends time the execution floor can be subtracted from:
# at 150000 it ran the whole file in 0.01s, which left the ratio dividing by
# that floor rather than by pypy's own time.
try:
    import pypyjit

    pypyjit.set_param("decay=0")
except ImportError:
    pass

N = 17036098
FLIP_AT = 2000000


def h(x):
    if x < 3:
        return x + 1
    return x * 2


def g(x):
    return h(x) + 1


def f(x):
    return g(x) + 1


def e(x):
    return f(x) + 1


def d(x):
    return e(x) + 1


def c(x):
    return d(x) + 1


def b(x):
    return c(x) + 1


def depth2():
    acc = 0.0
    i = 0
    while i < N:
        v = (i % 5) if i < FLIP_AT else float(i % 5)
        acc += g(v)
        i += 1
    return acc


def depth3():
    acc = 0.0
    i = 0
    while i < N:
        v = (i % 5) if i < FLIP_AT else float(i % 5)
        acc += f(v)
        i += 1
    return acc


def depth7():
    acc = 0.0
    i = 0
    while i < N:
        v = (i % 5) if i < FLIP_AT else float(i % 5)
        acc += b(v)
        i += 1
    return acc


print(depth2())
print(depth3())
print(depth7())
