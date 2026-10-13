# CPython-suite gap: the suite does not run one hot list comprehension over a
# `range` thousands of times and then feed the same code object a list, a
# tuple, a user iterator, a dict, a generator, and other ranges.
# parity-tests reason: a single-generator comprehension compiles to
# BUILD_LIST_FROM_ARG, which sizes the list from the iterator's length hint.
# The hint differs per receiver kind (exact, absent, raising), and one traced
# code object has to stay correct across all of them.

def g(x):
    return [i for i in x]


class It:
    def __iter__(self):
        return iter((7, 8))


class HintRaise:
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def __length_hint__(self):
        raise ValueError("x")


class Gen:
    def __iter__(self):
        yield 1
        yield 2


r = range(5)
objs = [r] * 4000 + [
    [1, 2, 3],
    (4, 5),
    It(),
    {6: 0},
    r,
    range(2, 6),
    range(0, 9, 3),
    HintRaise(),
    Gen(),
] * 300
s = 0
k = 0
n = len(objs)
while k < n:
    s += len(g(objs[k])) + sum(g(objs[k]))
    k += 1
assert s == 88200, s
print("OK")
