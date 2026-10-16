# pyre-check: max-pypy-ratio=11
# #171/#11 negative-index regression: a canonical-tuple / list subscript that
# the JIT specializes on a NON-NEGATIVE observed index must DEOPT (the
# lower-bound guard `0 <= idx`) when a later NEGATIVE index flows through the
# compiled trace, instead of reading the backing array out of range.  Python
# negative indexing (`seq[-1] == seq[len - 1]`) must hold across the
# specialize -> deopt edge.  A WHILE loop is used so the loop actually traces
# (a `for ... in range()` loop is declined to a single frame).


def main():
    t = (10, 20, 30, 40, 50)
    lst = [11, 22, 33, 44, 55]
    s = 0
    k = 0
    n = 0
    while n < 1200000:
        # Subscript first, on the loop-carried index `k`.  The first half runs
        # with k = 0 (specializes the subscript on a non-negative index); after
        # the flip, k = -1 must deopt to Python negative indexing, not OOB-read.
        s += t[k] + lst[k]
        n += 1
        if n == 600000:
            k = -1
    print(s)


main()
