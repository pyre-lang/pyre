# pyre-check: max-pypy-ratio=26
# gh#495 guard: nested inlined callees mutate through a branch-bearing abort path.
# 3-level nested branch-bearing mutating callees. outer while -> A -> B -> C, each mutates.
N = 320000


class Counter:
    def __init__(self):
        self.pos = 0

    def bump(self):
        self.pos = self.pos + 1


def levelC(c, d, k):
    c.bump()
    if k < 0:
        return 0
    if k in d:
        return d[k]
    return -1


def levelB(c, d, k):
    c.bump()
    if k == 4:
        return 7
    return levelC(c, d, k)


def levelA(c, d, k):
    c.bump()
    if k < 0:
        return -3
    return levelB(c, d, k)


def run():
    d = {1: 100, 2: 200, 3: 300}
    acc = 0
    c = Counter()
    i = 0
    while i < N:
        k = i % 5
        v = levelA(c, d, k)
        if v == -1:
            acc -= 1
        else:
            acc += v
        i = i + 1
    return acc, c.pos


print(run())
