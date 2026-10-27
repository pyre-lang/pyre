# pyre-check: max-pypy-ratio=110
# Sequence repeat semantics: a count exceeding the signed machine word overflows
# (OverflowError) rather than wrapping to a huge unsigned count that fails as
# MemoryError, and a count of 1 on an immutable exact sequence returns the
# receiver unchanged while a mutable one copies. Output verified against
# CPython/PyPy.
N = 480000
BIG = 2**63


def main():
    over = 0
    ident = 0
    for _ in range(N):
        try:
            [0] * BIG
        except OverflowError:
            over += 1
        t = (1, 2)
        b = b"ab"
        s = "ab"
        lst = [1, 2]
        if (t * 1 is t) and (b * 1 is b) and (s * 1 is s) and (lst * 1 is not lst):
            ident += 1
    print(over, ident)


main()
