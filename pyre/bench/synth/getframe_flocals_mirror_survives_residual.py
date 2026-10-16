# No `max-pypy-ratio`: pypy's exec time sits on the startup-subtraction floor,
# so the printed ratio compares process spawn rather than the mirror. The
# printed mismatch count and the jitstats baseline gate this fixture.
#
# Folding `frame.f_locals` mirrors the virtualizable shadow into the live
# `locals_cells_stack_w` array, because pyre answers the attribute with a 3.14
# `FrameLocalsProxy` that reads that array lazily rather than copying out of it
# at the call. The mirror is an eager walk-time write, so it needs an undo for
# the walk that does not commit — and the undo it is recorded in decides how
# long it survives.
#
# Recording it in the residual force's escape-flush capture is wrong: the tail
# of `try_execute_residual_call_via_executor` restores that capture after EVERY
# non-forcing residual call, so the next call in the same walk reverts the
# mirror and the proxy answers from before the fold. That is what this fixture
# reads — `str(i)` between the fold and the subscript — and the failure is one
# wrong answer on the trace-recording iteration alone, not every iteration, so
# a short loop passes either way.
import sys


def f(n):
    bad = 0
    for i in range(n):
        x = i * 2
        loc = sys._getframe(0).f_locals
        s = str(i)                     # a residual call between fold and read
        if loc['x'] != x:              # must be this iteration's value
            bad += 1
    return bad


print("mismatches:", f(270000))
