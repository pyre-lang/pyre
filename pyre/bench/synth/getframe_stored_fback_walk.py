# No `max-pypy-ratio`: the loop this fixture DOES compile -- its jitstats
# record `loops_compiled=1` on every backend -- runs too few iterations for
# the generated code to dominate a whole-process measurement. The run
# finishes in a fraction of a second, so a pypy ratio compares two
# interpreters' startup and reads whatever the host's process spawn cost
# happens to be that run. The jitstats baselines gate it.
# Regression guard: a callee stores its OWN frame (sys._getframe(0)); the loop
# body later walks .f_back.f_locals. That hop is the caller's frame, so it
# folds without forcing. A residual force would hit the traced caller
# mid-expression; the escape flush must commit with the
# operand-stack mirror (the vable shadow's stack region is NULL there) and
# resume forward AT the escaping opcode. Before the latched-stack escape
# flush, the flush declined, the FOR_ITER inflight deliver was refused, and
# the consumed iteration was DROPPED -- total came up short (JIT-only).
import sys

box = [None]


def bump(x):
    box[0] = sys._getframe(0)
    return x


def main():
    total = 0
    stale = 0
    for i in range(400000):
        total += bump(i)
        if box[0].f_back.f_locals['total'] != total:
            stale += 1
    print(total, stale)


main()
