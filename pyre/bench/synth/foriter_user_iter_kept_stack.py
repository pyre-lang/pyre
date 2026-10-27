# pyre-check: max-pypy-ratio=69
# User-defined iterator FOR_ITER paths enter a user frame for each item while
# the caller keeps a depth > 1 operand stack resident across condexpr and
# short-circuit branch guards.  The kept-stack branch guards must remain
# resident and byte-exact when the mirror carries the FOR_ITER item through the
# ResultToTos boundary, including heap-int (>= 256) merge slots, nested user
# iterators, and a comprehension.

N = 180000


class NextIter:
    """User __next__ iterator: FOR_ITER consume enters a user frame."""

    def __init__(self, n):
        self.i = 0
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
        v = self.i
        self.i += 1
        return v


class GetItemSeq:
    """Legacy __getitem__ protocol: seqiter calls a user frame per item."""

    def __init__(self, n):
        self.n = n

    def __getitem__(self, i):
        if i >= self.n:
            raise IndexError
        return i * 2


def condexpr_over_user_next():
    # depth>1 kept stack: (a+i, b-i) held while the condexpr guard evaluates.
    a, b, s = 3, 7, 0
    for i in NextIter(N):
        t = (a + i, b - i, (a + i) if (i % 3 == 0) else (b - i))
        s += t[0] + t[1] + t[2]
    return s


def shortcircuit_over_user_next():
    # short-circuit merge slot at depth>1 inside `total + (x + (... or ...))`.
    total = 0
    for i in NextIter(N):
        x = i + 100000
        total = total + (x + ((i & 1) or 1000000))
    return total


def heap_int_shortcircuit_over_user_next():
    # heap-int (>=256) merge slot — the boxed-int kept-slot shape.
    acc = 0
    for i in NextIter(N):
        acc = acc + ((i & 1) and 500000) + ((i & 2) or 300000)
    return acc


def condexpr_over_getitem():
    s = 0
    for v in GetItemSeq(N):
        s += (v + 1) if (v % 3 == 0) else (v - 1)
    return s


def nested_user_iter():
    # iterator kept on the stack across an inner loop's guards.
    s = 0
    for i in NextIter(60):
        for j in NextIter(50):
            s += (i * j) if (j & 1) else (i + j)
    return s


def user_next_in_comprehension():
    return sum([(v * 2) if (v & 1) else (v + 3) for v in NextIter(N)])


def main():
    print(
        condexpr_over_user_next(),
        shortcircuit_over_user_next(),
        heap_int_shortcircuit_over_user_next(),
        condexpr_over_getitem(),
        nested_user_iter(),
        user_next_in_comprehension(),
    )


main()
