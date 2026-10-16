# pyre-check: max-pypy-ratio=102
# The ceiling is twice the slowest ratio observed, 50.9x on the macos
# runner; the runners read this fixture between 9.0x and 50.9x.
# gh#495 guard: an inlined subwalk whose callee makes an unjournaled mutation
# through a nested residual CALL, in the two miss-handling shapes.
#
# `step_exc` takes the miss through a KeyError caught LOCALLY in the callee
# frame; `step_noexc` reaches the same result with a membership test and no
# exception at all.  The counter must advance exactly once per iteration in
# both: a replayed or dropped mutation shows up in `c.pos`.
N = 360000


class Counter:
    def __init__(self):
        self.pos = 0

    def bump(self):
        self.pos = self.pos + 1


def step_exc(c, d, k):
    c.bump()                 # unjournaled SetfieldGc via nested residual CALL
    if k < 0:                # branch to force branch-bearing (multiframe) shape
        return 0
    try:
        return d[k]          # residual dict-lookup; raises KeyError for misses
    except KeyError:
        return -1


def step_noexc(c, d, k):
    c.bump()
    if k < 0:
        return 0
    if k in d:
        return d[k]
    return -1


def run_exc():
    d = {1: 100, 2: 200, 3: 300}
    acc = 0
    c = Counter()
    i = 0
    while i < N:
        k = i % 5            # 0 and 4 miss -> KeyError caught -> -1
        v = step_exc(c, d, k)
        if v == -1:
            acc -= 1
        else:
            acc += v
        i = i + 1
    return acc, c.pos


def run_noexc():
    d = {1: 100, 2: 200, 3: 300}
    acc = 0
    c = Counter()
    i = 0
    while i < N:
        k = i % 5
        v = step_noexc(c, d, k)
        if v == -1:
            acc -= 1
        else:
            acc += v
        i = i + 1
    return acc, c.pos


print(run_exc())
print(run_noexc())
