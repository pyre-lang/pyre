# pyre-check: max-pypy-ratio=34
# A metaclass overriding __getattribute__ / __delattr__ intercepts class
# attribute reads / deletes (`C.x`, `del C.x`), replacing the builtin
# type.__getattribute__ / type.__delattr__.  objspace.py:664 runs the
# __getattribute__ gate on space.type(C) — the metaclass — for a type
# receiver.  A __getattribute__ raising AttributeError falls back to the
# metaclass __getattr__.  Ordinary classes (metaclass=type) are unaffected.
# Output verified against CPython/PyPy.
N = 20000


class Meta1(type):
    def __getattribute__(cls, name):
        if name == "boom":
            return "META[" + name + "]"
        return type.__getattribute__(cls, name)


class C1(metaclass=Meta1):
    x = 1


class Meta2(type):
    def __getattribute__(cls, name):
        if name == "missing":
            raise AttributeError(name)
        return type.__getattribute__(cls, name)

    def __getattr__(cls, name):
        return "GETATTR[" + name + "]"


class C2(metaclass=Meta2):
    y = 2


class Meta3(type):
    def __delattr__(cls, name):
        recorded.append(name)


recorded = []


class C3(metaclass=Meta3):
    pass


def main():
    hits = 0
    for _ in range(N):
        ok = (
            C1.boom == "META[boom]"               # __getattribute__ intercepts read
            and C1.x == 1                         # delegated to type.__getattribute__
            and C2.missing == "GETATTR[missing]"  # AttributeError -> __getattr__
            and C2.y == 2
        )
        if ok:
            hits += 1

    # metaclass __delattr__ intercepts and records without removing the key
    n_before = len(recorded)
    del C3.marker
    if recorded[n_before:] == ["marker"]:
        hits += 1

    # ordinary class: normal read / delete still work
    class Plain:
        a = 10

    if Plain.a == 10:
        del Plain.a
        if not hasattr(Plain, "a"):
            hits += 1

    print(hits)


main()
