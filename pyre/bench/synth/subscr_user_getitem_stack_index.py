# A user `__getitem__` inlined inside a `for` body, subscripted with an index
# produced on the operand stack rather than read from a local.
#
# The FOR_ITER admission gate must not identify "entered from a CALL" by the
# absence of an `arg_class_guard`: the subscript route also leaves that empty
# even though it enters from `BINARY_OP`, so the inline is accepted wrongly.
# The abort rewind has no CALL coordinate there, so the flush resumed with the
# operand stack one short and the index operand was replaced by an unrelated
# live reference — the iterator, the iterated list, or a bound method.
N = 30000


class Seq:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Seq(self.data[index])
        return self.data[index]


EMPTY = ()


def run(n):
    total = 0
    rows = [Seq([i, i + 1, i + 2]) for i in range(8)]
    for _ in range(n):
        for p in rows:
            total += p[0]                # LOAD_SMALL_INT index
            total += p[len(EMPTY)]       # index from a call result
            total += p[len(EMPTY) + 1]   # index from arithmetic
    return total


print(run(N))
