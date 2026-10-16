# Historical `_declined` companion to `getframe_bridge_force_plain`.  An exact
# portal `_getframe(0).f_locals` now stays in the trace: the getter constructs
# the CPython 3.14 write-through proxy around the same red frame and does not
# force it.
#
# PyPy reaches the same optimized shape through `@jit.unroll_safe fast2locals`:
# one loop, one bridge, no forcings, no virtualizable forcings, and no aborts.
# The rare `i % 97 == 0` arm still pins bridge compilation and the live locals
# result, but no longer endorses a single-frame blackhole escape that upstream
# never performs.
import sys

_gf = sys._getframe


def main():
    total = 0
    names = 0
    for i in range(2000000):
        if i % 97 == 0:
            fr = _gf(0)
            names += len(fr.f_locals)
        total += i
    return total, names


print(main())
