# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=wide
# pyre-check: trace-shape=wide:present=Jump
# `locals()` on the PORTAL frame of a loop whose frame is wider than the
# expansion's old ceiling.
#
# `try_walker_specialize_builtin_locals` unrolls `fast2locals`' slot loop one
# guard plus at most one store per bound slot.  Upstream bounds that unroll
# with nothing but `@jit.unroll_safe` on `pyframe.py` `PyFrame.fast2locals`,
# and asks the size question once over the whole history in `pyjitpl.py`
# `MetaInterp.blackhole_if_trace_too_long`; the walker used to carry a second,
# fixed ceiling of 32 slots and ask it per fold instead.
#
# A refusal here answers correctly -- it falls through to the generic
# residual, which computes the same mapping -- so nothing about the ANSWER
# says whether the fold ran.  What it costs is the loop: the residual's
# locals read forces the portal frame, and the walk cannot carry that.
#
# `wide` carries 44 slots, and the loop that reads them is in `wide` itself,
# so this is the portal arm rather than the callee arm
# (`locals_in_wide_inlined_callee` is that one).  Measured on release dynasm:
#
#   with the ceiling      consulted=5 fired=0   loops_compiled=0 loops_aborted=5
#   without it            consulted=3 fired=2   loops_compiled=1 loops_aborted=0
#
# so the ceiling was not costing one residual per iteration, it was costing
# the loop.  Both halves are gated -- `spec-folds` on the fold and
# `selfcheck-compiles` on the loop -- because either alone reads as a pass on
# a run that answered right and compiled nothing.
#
# The mapping is never bound to a local.  Binding it would put the previous
# iteration's mapping in the next one's, chaining N dicts through the frame
# and measuring the collector rather than the fold.
import sys

N = 20000

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
    "i",
    "n",
    "total",
    "width",
]

WIDTH = 44


def wide(n):
    a00 = 0
    a01 = 1
    a02 = 2
    a03 = 3
    a04 = 4
    a05 = 5
    a06 = 6
    a07 = 7
    a08 = 8
    a09 = 9
    a10 = 10
    a11 = 11
    a12 = 12
    a13 = 13
    a14 = 14
    a15 = 15
    a16 = 16
    a17 = 17
    a18 = 18
    a19 = 19
    a20 = 20
    a21 = 21
    a22 = 22
    a23 = 23
    a24 = 24
    a25 = 25
    a26 = 26
    a27 = 27
    a28 = 28
    a29 = 29
    a30 = 30
    a31 = 31
    a32 = 32
    a33 = 33
    a34 = 34
    a35 = 35
    a36 = 36
    a37 = 37
    a38 = 38
    a39 = 39
    total = 0
    width = 0
    for i in range(n):
        total += locals()["a39"]
        width = len(locals())
    # Read after the loop, on the same frame: `i` is still bound, so the name
    # set is the one every traced iteration saw.  `width` is what pins that,
    # since it is recorded inside the loop on each of them.
    return total, width, sorted(locals())


def main():
    total, width, names = wide(N)
    if width != WIDTH:
        print(f"FAIL width: {width} (expected {WIDTH})")
        return 1
    if names != EXPECTED:
        print(f"FAIL name set: {names}")
        print(f"  expected: {EXPECTED}")
        return 1
    if total != 39 * N:
        print(f"FAIL total: {total} (expected {39 * N})")
        return 1
    print("PASS wide portal locals")
    return 0


sys.exit(main())
