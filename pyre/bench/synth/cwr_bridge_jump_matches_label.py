# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=main,entry-bridge:cwr1,entry-bridge:combinations1
# Bridge close of a generator `while` (the combinations_with_replacement
# pure-Python shape) used to emit a JUMP one arg short of the compiled
# LABEL (`JUMP args (33) != target LABEL args (34)` in dynasm assembler).
#
# `_jump_to_existing_trace` (`unroll.py`) keeps JUMP args equal to the
# target LABEL because `finalize_short_preamble` only appends
# `sb.used_boxes` (`label_op.initarglist`). A loop `cut_trace_from`
# used to append an escaped outer inputarg (`InputArgRef(1)`) as an extra
# start-LABEL slot that `jump_to_preamble` never carries.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

from itertools import (
    accumulate,
    chain,
    combinations,
    combinations_with_replacement,
    groupby,
    islice,
    zip_longest,
)


def batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if not batch:
            return
        yield batch


def cwr1(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    if not n and r:
        return
    indices = [0] * r
    yield tuple(pool[i] for i in indices)
    while 1:
        for i in reversed(range(r)):
            if indices[i] != n - 1:
                break
        else:
            return
        indices[i:] = [indices[i] + 1] * (r - i)
        yield tuple(pool[i] for i in indices)


def combinations1(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while 1:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)


def mutatingtuple(tuple1, f, tuple2):
    def g(value, first=[1]):
        if first:
            del first[:]
            f(next(z))
        return value

    items = list(tuple2)
    items[1:1] = list(tuple1)
    gen = map(g, items)
    z = zip(*[gen] * len(tuple1))
    next(z)


def chain2(*iterables):
    for it in iterables:
        for element in it:
            yield element


def gen1():
    yield 1
    raise AssertionError


class Repeater:
    def __init__(self, o, t, e):
        self.o = o
        self.t = int(t)
        self.e = e

    def __iter__(self):
        return self

    def __next__(self):
        if self.t > 0:
            self.t -= 1
            return self.o
        raise self.e


def heat_groupby():
    def f(n):
        if n == 5:
            list(b)
        return n != 6

    for (k, b) in groupby(range(10), f):
        list(b)


def main():
    first = []
    T = []

    def f(t):
        nonlocal T
        T = t
        first[:] = list(T)

    mutatingtuple((1, 2, 3), f, (4, 5, 6))
    heat_groupby()
    try:
        list(chain(gen1(), [2]))
    except AssertionError:
        pass

    total = 0
    total += sum(accumulate(range(10)))
    data = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    try:
        from itertools import batched as c_batched
    except ImportError:
        c_batched = batched
    for n in range(1, 6):
        for i in range(len(data)):
            batches = list(c_batched(data[:i], n))
            total += len(batches)
    for maker in (chain, chain2):
        total += len(list(maker("abc", "def")))
    r1 = Repeater(1, 3, StopIteration)
    r2 = Repeater(2, 4, StopIteration)
    for i, j in zip_longest(r1, r2, fillvalue=0):
        total += i + j

    for n in range(7):
        values = [5 * x - 12 for x in range(n)]
        for r in range(n + 2):
            result = list(combinations_with_replacement(values, r))
            total += len(result)
            total += len(list(combinations(values, r)))
            total += len(list(combinations1(values, r)))
            total += len(list(cwr1(values, r)))
            for item in result:
                total += len(item)
    return total


print("PASS", main())
