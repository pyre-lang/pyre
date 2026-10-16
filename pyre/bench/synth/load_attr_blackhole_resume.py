# pyre-check: max-pypy-ratio=12
# Guard-failure blackhole resume across a LOAD_ATTR (issue #143 guard-12
# class).  The rare branch fails its guard until the bridge threshold, and
# each failure resumes the rest of the iteration through `c.v` — a jitcode
# position the canonical splice lowers to the `load_attr_fn` residual call.
# An `abort_permanent` there crashes the blackhole with an out-of-bounds
# register read.
class C:
    def __init__(self):
        self.v = 3


def run(c, n):
    total = 0
    i = 0
    while i < n:
        if i % 37 == 36:
            k = 2
        else:
            k = 1
        total += c.v * k
        i += 1
    return total


print(run(C(), 40000000))
