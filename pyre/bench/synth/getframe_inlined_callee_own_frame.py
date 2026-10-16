# No `max-pypy-ratio`: the loop this fixture DOES compile -- its jitstats
# record `loops_compiled=1` on every backend -- runs too few iterations for
# the generated code to dominate a whole-process measurement. The run
# finishes in a fraction of a second, so a pypy ratio compares two
# interpreters' startup and reads whatever the host's process spawn cost
# happens to be that run. The jitstats baselines gate it.
# Regression guard: an INLINED callee that inspects its own frame via
# sys._getframe(0) must still see ITS OWN frame, not its caller's. The walker
# executes residuals concretely while an inline push never runs the
# interpreter's call sequence, so ec.topframeref still names the caller unless
# the inline declines or materialises the callee's frame. Publishing a frame
# for only the inline levels that have one shifts every _getframe up a level
# instead of fixing it, and the shift is invisible to a JIT-vs-JIT comparison
# -- both pyre paths move together. Hence the assert against the literal
# expected chain rather than a self-consistency check.
import sys


def leaf(x):
    frame = sys._getframe(0)
    return (frame.f_code.co_name, frame.f_back.f_code.co_name)


def main():
    seen = set()
    total = 0
    for i in range(300000):
        seen.add(leaf(i))
        total += i
    print(total, sorted(seen))


main()
