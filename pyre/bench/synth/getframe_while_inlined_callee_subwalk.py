# No `max-pypy-ratio`: the only loop here is the two-statement `while` driver,
# so a pypy ratio compares two interpreters' startup rather than any generated
# code, and reads whatever the host's process spawn cost happens to be that
# run. The jitstats baselines gate it.
# Coverage guard for `_getframe(1).f_lasti` from an inlined callee.  The
# specialization lands on the portal red box and folds the CALL coordinate, so
# the loop compiles without a virtualizable force or a multi-frame blackhole
# adopt.
#
# The shape below is load-bearing, not incidental:
#   - `while`, not `for`: the compiled loop is the two-statement driver;
#   - the CALLER's depth, `_gf(1)`, answered from the portal red box;
#   - a read of `f_lasti` off the frame it returns, folded at
#     `inline_caller_py_pc`.
# Changing any of the three can silently stop exercising the path.
#
# The printed total counts one callee entry per iteration, so a resume that
# replays the region or re-delivers an iteration prints something other than
# 6300000.
import sys

_gf = sys._getframe


def leaf(x):
    _ = _gf(1).f_lasti
    return x + 1


def main():
    total = 0
    i = 0
    while i < 6300000:
        total = leaf(total)
        i = i + 1
    return total


print(main())
