# pyre-check: max-pypy-ratio=11.8
# dynasm 3.5x, cranelift 8.4x.
# The class entry is an ObjectMutableCell before the loop: the first store
# builds the cell (`typeobject.py write_cell`) and the hot `__next__` has to
# promote `ObjectMutableCell.w_value` instead of declining `next_fast_path`.
N = 7114848


class It:
    def __init__(self, n):
        self.i = 0
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
        self.i += 1
        return self.i


def _n(self):
    if self.i >= self.n:
        raise StopIteration
    self.i += 1
    return self.i


# Rebind once so the class entry becomes a cell before the loop compiles.
It.__next__ = _n

total = 0
# 20 steps per iterator, matching cr_next.py's inner length.
steps = 20
for _r in range(N // steps):
    for x in It(steps):
        total += x
print(total)
