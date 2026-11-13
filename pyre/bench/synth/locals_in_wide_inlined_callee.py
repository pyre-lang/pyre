# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=probe
# pyre-check: trace-shape=probe:present=Jump
# `locals()` in an inlined callee whose frame is WIDER than the modelled
# expansion's old ceiling.
#
# `try_walker_specialize_builtin_locals_in_callee_expand` unrolls
# `fast2locals`' slot loop one store per bound slot.  Upstream bounds that
# unroll with nothing but `@jit.unroll_safe` on `pyframe.py`
# `PyFrame.fast2locals` and the tracer's own `trace_limit`
# (`pyjitpl.py` `MetaInterp.blackhole_if_trace_too_long`); the walker used to
# carry a second, fixed ceiling of 32 slots on top of that.
#
# In the portal arm a width refusal costs one residual.  In the callee arm it
# costs the callee: the residual's locals read forces this level's published
# frame, `tracing_after_residual_call` reads the cleared token as
# `VableEscapedDuringResidualCall`, and the walker answers that by denying the
# callee for the rest of the thread's tracing.  So the ceiling decided whether
# a callee is inlinable from its LOCAL COUNT alone.
#
# `wide` carries 42 slots -- `x`, `a00`..`a39` and `d` -- and no other fixture
# reaches the expansion with more than a handful, so the ceiling was a shape
# the corpus did not own.  The gate is the `spec-folds` header: measured on
# release dynasm, this file read `consulted=7 fired=0` with the ceiling in
# place and `consulted=1 fired=1` without it, because the wide callee is the
# only shape it offers.  The name set is asserted too, so a fold that takes
# the shape but answers from the wrong frame fails rather than passing on a
# plausible count -- the answer itself is right either way, since the residual
# the refusal falls back to computes it correctly.
import sys

N = 200000

EXPECTED = [
    "a00",
    "a01",
    "a02",
    "a03",
    "a04",
    "a05",
    "a06",
    "a07",
    "a08",
    "a09",
    "a10",
    "a11",
    "a12",
    "a13",
    "a14",
    "a15",
    "a16",
    "a17",
    "a18",
    "a19",
    "a20",
    "a21",
    "a22",
    "a23",
    "a24",
    "a25",
    "a26",
    "a27",
    "a28",
    "a29",
    "a30",
    "a31",
    "a32",
    "a33",
    "a34",
    "a35",
    "a36",
    "a37",
    "a38",
    "a39",
    "x",
]


def wide(x):
    a00 = x + 0
    a01 = x + 1
    a02 = x + 2
    a03 = x + 3
    a04 = x + 4
    a05 = x + 5
    a06 = x + 6
    a07 = x + 7
    a08 = x + 8
    a09 = x + 9
    a10 = x + 10
    a11 = x + 11
    a12 = x + 12
    a13 = x + 13
    a14 = x + 14
    a15 = x + 15
    a16 = x + 16
    a17 = x + 17
    a18 = x + 18
    a19 = x + 19
    a20 = x + 20
    a21 = x + 21
    a22 = x + 22
    a23 = x + 23
    a24 = x + 24
    a25 = x + 25
    a26 = x + 26
    a27 = x + 27
    a28 = x + 28
    a29 = x + 29
    a30 = x + 30
    a31 = x + 31
    a32 = x + 32
    a33 = x + 33
    a34 = x + 34
    a35 = x + 35
    a36 = x + 36
    a37 = x + 37
    a38 = x + 38
    a39 = x + 39
    # `d` is still unbound at this point, so it binds no key of its own.
    d = locals()
    return d


def probe():
    last = None
    for i in range(N):
        last = wide(i)
    return last


def main():
    # The last iteration is a compiled one, so the mapping asserted on here is
    # the one the fold built.
    last = probe()
    names, a39, x = sorted(last), last["a39"], last["x"]
    if names != EXPECTED:
        print(f"FAIL name set: {names}")
        print(f"  expected: {EXPECTED}")
        return 1
    if x != N - 1:
        print(f"FAIL x: {x} (expected {N - 1})")
        return 1
    if a39 != N - 1 + 39:
        print(f"FAIL a39: {a39} (expected {N - 1 + 39})")
        return 1
    print("PASS wide callee locals")
    return 0


sys.exit(main())
