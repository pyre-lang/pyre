# pyre-check: max-pypy-ratio=265
# pypy's exec time is pinned to the startup-subtraction floor here, so the
# ratio is not a measurement: the ceiling is twice the slowest ratio the CI
# runners observe (131.0x), rounded up.
# An instance attribute shadows a same-named non-data-descriptor class
# attribute on builtin-type subclasses (tuple/int/str), while a data descriptor
# still wins over a competing instance-dict entry (injected directly, since the
# data descriptor's no-op __set__ never writes __dict__). Output verified
# against CPython/PyPy.
N = 120000


class T(tuple):
    x = "class"


class I(int):
    x = "class"


class S(str):
    x = "class"


class DataD:
    def __get__(self, o, tp):
        return "data"

    def __set__(self, o, v):
        pass


class TD(tuple):
    d = DataD()


def main():
    n = 0
    for _ in range(N):
        t = T([1, 2])
        t.x = "inst"
        a = I(5)
        a.x = "inst"
        s = S("hi")
        s.x = "inst"
        td = TD([1])
        td.__dict__["d"] = "inst"
        if t.x == "inst" and a.x == "inst" and s.x == "inst" and td.d == "data":
            n += 1
    print(n)


main()
