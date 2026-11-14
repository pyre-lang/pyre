# pyre-check: max-pypy-ratio=245
# A tuple/frozenset subclass that sets __hash__ = None is unhashable: hash()
# and set insertion both raise TypeError instead of the structural fast path
# hashing by contents. Output verified against CPython/PyPy.
# The `fill` allocation runs before `main()` so the lambdas' code objects
# are already promoted when `kind`'s loop compiles. Otherwise the GuardValue
# on `Function.code` counts against the live pointer (`compile.py
# AbstractResumeGuardDescr.must_compile`) and a minor collection inside the
# eagerness window moves `guard_failures` with the process's startup allocation.
N = 240000


class T(tuple):
    __hash__ = None


class F(frozenset):
    __hash__ = None


def kind(fn):
    try:
        fn()
    except TypeError:
        return "TypeError"
    return "ok"


def main():
    a = T([1, 2])
    b = F([1, 2])
    n = 0
    for _ in range(N):
        if (
            kind(lambda: hash(a)) == "TypeError"
            and kind(lambda: {a}) == "TypeError"
            and kind(lambda: hash(b)) == "TypeError"
        ):
            n += 1
    print(T.__hash__, n)


fill = list(map(str, range(400000)))
del fill
main()
