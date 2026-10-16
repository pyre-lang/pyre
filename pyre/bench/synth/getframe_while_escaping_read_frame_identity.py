# No `max-pypy-ratio`: the loop this fixture DOES compile -- its jitstats
# record `loops_compiled=2` on every backend -- runs too few iterations for
# the generated code to dominate a whole-process measurement. The run
# finishes in a fraction of a second, so a pypy ratio compares two
# interpreters' startup and reads whatever the host's process spawn cost
# happens to be that run. The jitstats baselines gate it.
# Frame-identity oracles on two compiled loops: `part_a`'s `_gf()` must name
# `leaf`, and `part_b`'s `_gf(1)` must name `main`.  `_gf(1).f_lasti` now
# folds on the portal red box instead of forcing a multi-frame adopt; the
# names still have to come from the red box at each depth, not from a
# collapsed portal slot.
import sys

_gf = sys._getframe

wrong_a = []
wrong_b = []


def leaf_a(x):
    name = _gf().f_code.co_name
    if name != "leaf_a":
        wrong_a.append(name)
    return x + 1


def part_a():
    total = 0
    i = 0
    while i < 1050000:
        total = leaf_a(total)
        i = i + 1
    return total


def leaf_b(x):
    fr = _gf(1)
    _ = fr.f_lasti
    name = fr.f_code.co_name
    if name != "part_b":
        wrong_b.append(name)
    return x + 1


def part_b():
    total = 0
    i = 0
    while i < 1050000:
        total = leaf_b(total)
        i = i + 1
    return total


print(part_a(), part_b(), len(wrong_a), len(wrong_b))
