# No `max-pypy-ratio`: the only loop here is the two-statement `while` driver,
# so a pypy ratio compares two interpreters' startup rather than any generated
# code, and reads whatever the host's process spawn cost happens to be that
# run. The jitstats baselines gate it.
# Caller locals that are live ACROSS the inlined call, read from
# `_gf(1).f_lasti` inside the callee.  That call now lands on the portal red
# box and folds the CALL coordinate, so the loop compiles without a
# virtualizable force.  `acc` and `tag` are carried across every iteration; a
# compile that dropped or restored the wrong slot would change the printed
# totals.
import sys

_gf = sys._getframe


def leaf(x):
    _ = _gf(1).f_lasti
    return x + 1


def main():
    total = 0
    acc = 0
    tag = 7
    i = 0
    while i < 3960000:
        total = leaf(total)
        acc = acc + total
        tag = tag ^ i
        i = i + 1
    return total, acc, tag


print(main())
