# pyre-check: max-pypy-ratio=32
# Historical `_declined` companion to `getframe_residual_callee_own_frame`.
# Exact `_getframe(0).f_locals` now remains traced and owns the callee's exact
# red frame, so it neither forces that frame nor clears the caller's
# virtualizable token.
#
# The recorded shape is one loop (`bump` inlines, as on `pypy3`), no bridge, no forcings, no virtualizable
# forcings, and no aborts.  The observable regression guard remains the
# non-journaled `STORE_ATTR`: frame inspection must execute `bump` exactly once
# per iteration, so `c.n` must stay equal to the loop count.
import sys


class Counter:
    pass


c = Counter()
c.n = 0


def bump(x):
    c.n += 1                  # STORE_ATTR: non-journaled body effect
    frame = sys._getframe(0)
    frame.f_locals  # may-force residual inspecting the callee's own frame
    return x if frame is not None else -1


def main():
    total = 0
    for i in range(100000):
        total += bump(i)
    print(total, c.n)


main()
