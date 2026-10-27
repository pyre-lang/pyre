# pyre-check: max-pypy-ratio=16
N = 6000000


class Box:
    def __init__(self, v):
        self.v = v


def pick(a, b, c):
    return a.v or b.v or c.v


def run():
    z = Box(0)
    two = Box(2)
    acc = 0
    i = 0
    while i < N:
        acc += pick(z, Box(i % 3), two)
        i += 1
    return acc


print(run())
