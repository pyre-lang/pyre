# pyre-check: trace-shape=run:absent=CallMayForceR
# The `for` loop lives in `step`, which the walker inlines into `run`'s trace,
# so the `ForIterNext` residual it carries is dispatched from an inline
# sub-walk rather than from the walk's own snapshot root.
#
# `instance_next` used to refuse that position outright. With the fold
# refused, `Outer.__next__` stayed an opaque residual and ran as a real
# frame in the plain interpreter. The fold now inlines the callee from the
# sub-walk the same way a root walk does (`pyjitpl.py _opimpl_recursive_call`).
#
# No `max-pypy-ratio`: at a size that keeps this fixture's own runtime sane,
# pypy's execution-only time clamps to `EXEC_TIME_FLOOR_S` and check.py skips
# the ratio (`~`) on exactly the runs the ceiling would have to gate. The fold
# census is the instrument that separates the two states here, and it reads the
# same on every host -- `instance_next` fires once with the sub-walk route open
# and not at all with it closed.
#
# `foriter_call_body` covers the same fold from the loop-owning function; only
# the callee position exercises the sub-walk route this fixture guards.
N = 4000000


class Shared:
    def __init__(self):
        self.a = 0


class Outer:
    def __init__(self, sh, m):
        self.sh = sh
        self.m = m
        self.j = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.j >= self.m:
            raise StopIteration
        self.sh.a += 1
        v = self.j
        self.j += 1
        return v


def step(go):
    s = 0
    for x in go:
        s += x
        break
    return s


def run(n):
    sh = Shared()
    go = Outer(sh, n * 10)
    acc = 0
    i = 0
    while i < n:
        acc += step(go)
        i += 1
    return acc, sh.a


print(run(N))
