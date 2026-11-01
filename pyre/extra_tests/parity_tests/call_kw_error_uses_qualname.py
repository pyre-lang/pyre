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


def exercise_scope():
    # Outside the hot loop: keyword-only binding, **kwargs collection, and a
    # bound-method keyword call. `PyCode.funcrun` → `Arguments.parse_into_scope`.
    def kwonly(a, *, k):
        return (a, k)

    def kwdef(*, k=3):
        return k

    def collect(a, **kw):
        return (a, kw["b"], tuple(sorted(kw)))

    class Box:
        def m(self, a, *, k):
            return (self.n, a, k)

    if kwonly(1, k=2) != (1, 2):
        raise AssertionError(repr(kwonly(1, k=2)))
    if kwdef() != 3:
        raise AssertionError("kwdef")
    if kwdef(k=4) != 4:
        raise AssertionError("kwdef override")
    got = collect(1, c=3, b=2)
    if got != (1, 2, ("b", "c")):
        raise AssertionError(repr(got))
    obj = Box()
    obj.n = 7
    bound = obj.m
    if bound(1, k=2) != (7, 1, 2):
        raise AssertionError(repr(bound(1, k=2)))
    if obj.m(1, k=2) != (7, 1, 2):
        raise AssertionError(repr(obj.m(1, k=2)))

    try:
        kwonly(1)
    except TypeError as exc:
        check(
            str(exc),
            "exercise_scope.<locals>.kwonly() missing 1 required keyword-only argument: 'k'",
        )
    else:
        raise AssertionError("kwonly missing")
    try:
        kwonly(1, a=2, k=3)
    except TypeError as exc:
        check(
            str(exc),
            "exercise_scope.<locals>.kwonly() got multiple values for argument 'a'",
        )
    else:
        raise AssertionError("kwonly duplicate")
    try:
        kwonly(1, k=2, z=3)
    except TypeError as exc:
        check(
            str(exc),
            "exercise_scope.<locals>.kwonly() got an unexpected keyword argument 'z'",
        )
    else:
        raise AssertionError("kwonly unexpected")
    try:
        collect()
    except TypeError as exc:
        check(
            str(exc),
            "exercise_scope.<locals>.collect() missing 1 required positional argument: 'a'",
        )
    else:
        raise AssertionError("collect missing")
    try:
        bound(k=2)
    except TypeError as exc:
        check(
            str(exc),
            "exercise_scope.<locals>.Box.m() missing 1 required positional argument: 'a'",
        )
    else:
        raise AssertionError("bound missing")


exercise()
exercise_scope()
print("OK")
