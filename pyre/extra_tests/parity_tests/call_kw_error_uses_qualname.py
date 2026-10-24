# CPython-suite gap: test_call checks missing, duplicate, and unexpected
# keyword TypeErrors on module-level functions, whose `__qualname__`
# equals `__name__`.
# parity-tests reason: a plain-function CALL_KW TypeError names
# `func.qualname` from `PyCode.funcrun` — a nested function, a method,
# and an assigned `__qualname__`, including a lone surrogate.

"""Hot CALL_KW binding errors report the callable's qualified name."""

N = 3000


class Outer:
    def f(self, a, b):
        return a + b


def outer():
    def f(a, b):
        return a + b

    return f


def renamed(a, b):
    return a + b


renamed.__qualname__ = "pkg.renamed"


def surrogate(a, b):
    return a + b


SUR = "q\udcff"
surrogate.__qualname__ = SUR

nested = outer()
method = Outer.f
obj = Outer()


def check(got, expected):
    if got != expected:
        raise AssertionError(repr(got))
    # A lone surrogate has no UTF-8 spelling on stdout.
    if "\udcff" not in got:
        print(got)


def exercise():
    total = 0
    i = 0
    while i < N:
        if i < N - 3:
            total += nested(a=1, b=2)
            total += method(obj, a=1, b=2)
            total += renamed(a=1, b=2)
            total += surrogate(a=1, b=2)
        elif i == N - 3:
            try:
                nested(a=1)
            except TypeError as exc:
                check(
                    str(exc),
                    "outer.<locals>.f() missing 1 required positional argument: 'b'",
                )
            else:
                raise AssertionError("nested missing")
            try:
                method(obj, a=1)
            except TypeError as exc:
                check(
                    str(exc),
                    "Outer.f() missing 1 required positional argument: 'b'",
                )
            else:
                raise AssertionError("method missing")
            try:
                renamed(a=1)
            except TypeError as exc:
                check(
                    str(exc),
                    "pkg.renamed() missing 1 required positional argument: 'b'",
                )
            else:
                raise AssertionError("renamed missing")
            try:
                surrogate(a=1)
            except TypeError as exc:
                check(
                    str(exc),
                    SUR + "() missing 1 required positional argument: 'b'",
                )
            else:
                raise AssertionError("surrogate missing")
        elif i == N - 2:
            try:
                nested(1, a=2)
            except TypeError as exc:
                check(
                    str(exc),
                    "outer.<locals>.f() got multiple values for argument 'a'",
                )
            else:
                raise AssertionError("nested duplicate")
            try:
                method(obj, 1, a=2)
            except TypeError as exc:
                check(
                    str(exc),
                    "Outer.f() got multiple values for argument 'a'",
                )
            else:
                raise AssertionError("method duplicate")
            try:
                renamed(1, a=2)
            except TypeError as exc:
                check(
                    str(exc),
                    "pkg.renamed() got multiple values for argument 'a'",
                )
            else:
                raise AssertionError("renamed duplicate")
            try:
                surrogate(1, a=2)
            except TypeError as exc:
                check(
                    str(exc),
                    SUR + "() got multiple values for argument 'a'",
                )
            else:
                raise AssertionError("surrogate duplicate")
        else:
            try:
                nested(a=1, b=2, c=3)
            except TypeError as exc:
                check(
                    str(exc),
                    "outer.<locals>.f() got an unexpected keyword argument 'c'",
                )
            else:
                raise AssertionError("nested unexpected")
            try:
                method(obj, a=1, b=2, c=3)
            except TypeError as exc:
                check(
                    str(exc),
                    "Outer.f() got an unexpected keyword argument 'c'",
                )
            else:
                raise AssertionError("method unexpected")
            try:
                renamed(a=1, b=2, c=3)
            except TypeError as exc:
                check(
                    str(exc),
                    "pkg.renamed() got an unexpected keyword argument 'c'",
                )
            else:
                raise AssertionError("renamed unexpected")
            try:
                surrogate(a=1, b=2, c=3)
            except TypeError as exc:
                check(
                    str(exc),
                    SUR + "() got an unexpected keyword argument 'c'",
                )
            else:
                raise AssertionError("surrogate unexpected")
        i += 1
    expect = (N - 3) * 3 * 4
    if total != expect:
        raise AssertionError("calls did not run: %s != %s" % (total, expect))


exercise()
print("OK")
