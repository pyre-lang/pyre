# pyre-check: max-pypy-ratio=1416
N = 1200000


# A loop-callee that constructs a user object via a *type* call and reads
# an attribute off it.  When the outer loop is traced, `make` is inlined
# and its body is concretely executed step by step; the `P(i)` CALL runs
# through the inline residual-call path.  That path must support every
# callable kind the normal call path does — here a type (class) — or the
# constructed instance is lost and the subsequent `.a` read dereferences
# null.
class P:
    def __init__(self, a):
        self.a = a


def make(n):
    p = P(n)
    return p.a + 1


def outer(n):
    s = 0
    i = 0
    while i < n:
        s = s + make(i)
        i = i + 1
    return s


print(outer(N))
