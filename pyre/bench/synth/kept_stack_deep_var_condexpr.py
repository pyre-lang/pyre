# pyre-check: max-pypy-ratio=13
# Deep operand-stack Variables live across an inner condexpr guard.
# The first two tuple elements (a+i, b-i) are computed Variables left deep on
# the value stack while the third element's `... if ... else ...` evaluates its
# branch guard. A guard resume at that point must reconstruct the two deep-stack
# Variables (not constants) from the per-PC resume map.
def f(a, b, n):
    s = 0
    for i in range(n):
        t = (a + i, b - i, (a + i) if (i % 3 == 0) else (b - i))
        s += t[0] + t[1] + t[2]
    return s


print(f(3, 7, 3000000))
