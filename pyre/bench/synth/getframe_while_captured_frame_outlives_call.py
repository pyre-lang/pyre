# No `max-pypy-ratio`: the only loop here is the two-statement `while` driver,
# so a pypy ratio compares two interpreters' startup rather than any generated
# code, and reads whatever the host's process spawn cost happens to be that
# run. The jitstats baselines gate it.
# The callee's frame OUTLIVES the call and its f_back is read after the loop.
#
# The callee's frame is captured at depth 0 (`kept = _gf()`) and its `f_back`
# is read after the loop; `_gf(1).f_lasti` now folds on the portal red box
# instead of forcing a multi-frame adopt.  `kept.f_back` must still name
# `main` after the compiled loop finishes.
import sys

_gf = sys._getframe

kept = None


def leaf(x):
    global kept
    kept = _gf()
    _ = _gf(1).f_lasti
    return x + 1


def main():
    total = 0
    i = 0
    while i < 2400000:
        total = leaf(total)
        i = i + 1
    return total


t = main()
print(t, kept.f_back.f_code.co_name, kept.f_code.co_name)
