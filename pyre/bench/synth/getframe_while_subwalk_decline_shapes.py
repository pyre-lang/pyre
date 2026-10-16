# No `max-pypy-ratio`: the loop this fixture DOES compile -- its jitstats
# record `loops_compiled=2` on every backend -- runs too few iterations for
# the generated code to dominate a whole-process measurement. The run
# finishes in a fraction of a second, so a pypy ratio compares two
# interpreters' startup and reads whatever the host's process spawn cost
# happens to be that run. The jitstats baselines gate it.
# pypy's exec time is pinned to the startup-subtraction floor here, so the
# ratio is not a measurement: the ceiling is twice the slowest ratio the CI
# runners observe (58.2x), rounded up.
# Two sub-walk shapes the multi-frame blackhole build DECLINES, pinned so the
# decline stays a decline rather than silently becoming a wrong answer.
#
# Both reach the vable-escape latch inside an inline sub-walk and both are then
# refused by `capture_inline_parent_blackhole`, because a ref register that is
# live at the caller's post-call coordinate holds the untracked sentinel rather
# than a value. That sentinel is deliberately distinct from a known null (an
# uninitialised local is `Ref(PY_NULL)`), so recording it as null would fabricate
# a parent frame; declining is the correct answer until the caller's concrete
# banks are complete at an inline escape.
#
#   part_a -- the caller has an exception handler around the inlined call, so a
#             live stack ref at the resume coordinate is untracked.
#   part_b -- two nested inlined levels, where the intermediate level's own
#             parent capture hits the same sentinel.
#
# The values printed are what a correct legacy replay produces; a build that
# started accepting either shape without completing the banks would diverge here.
import sys

_gf = sys._getframe


def leaf_a(x):
    _gf()
    if x < 0:
        raise ValueError("never")
    return x + 1


def part_a():
    total = 0
    caught = 0
    i = 0
    while i < 1000000:
        try:
            total = leaf_a(total)
        except ValueError:
            caught = caught + 1
        i = i + 1
    return total, caught


def inner_b(x):
    _gf()
    return x + 1


def outer_b(x):
    return inner_b(x)


def part_b():
    total = 0
    i = 0
    while i < 1000000:
        total = outer_b(total)
        i = i + 1
    return total


print(part_a(), part_b())
