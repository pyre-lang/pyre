# CPython-suite gap: test_metaclass and test_descr define no meta-metaclass
# __call__ combined with class-definition keywords, no descriptor __new__ /
# __init__ on a metaclass, and pass no NUL name to __build_class__.
# parity-tests reason: `__build_class__` calls the metaclass with
# `space.call_args(w_meta, args)` (compiling.py `build_class`), so a
# meta-metaclass `__call__` and descriptor `__new__` / `__init__` see the
# class keywords; the default path reaches `_create_new_type`, which rejects
# a NUL in the name.
# A metaclass `__call__` descriptor whose `__get__` raises propagates that error.

"""__build_class__ calls a type metaclass generically and checks the name."""

log = []


class MM(type):
    def __call__(cls, *args, **kwds):
        log.append(("MM.__call__", sorted(kwds)))
        return super().__call__(*args, **kwds)


class M(type, metaclass=MM):
    def __new__(mcls, name, bases, ns, **kwds):
        log.append(("M.__new__", sorted(kwds)))
        return super().__new__(mcls, name, bases, ns)

    def __init__(cls, name, bases, ns, **kwds):
        log.append(("M.__init__", sorted(kwds)))


class C(metaclass=M, x=1):
    pass


assert log == [
    ("MM.__call__", ["x"]),
    ("M.__new__", ["x"]),
    ("M.__init__", ["x"]),
], log

log2 = []


class InitCallable:
    def __get__(self, obj, objtype=None):
        def bound(name, bases, ns, **kwds):
            log2.append(("bound __init__", obj.__name__, sorted(kwds)))

        return bound


class N(type):
    def __new__(mcls, name, bases, ns, **kwds):
        return super().__new__(mcls, name, bases, ns)

    __init__ = InitCallable()


class D(metaclass=N, y=2):
    pass


assert log2 == [("bound __init__", "D", ["y"])], log2

log3 = []


class NewDesc:
    def __get__(self, obj, objtype=None):
        def new(mcls, name, bases, ns, **kwds):
            log3.append(("desc __new__", sorted(kwds)))
            return type.__new__(mcls, name, bases, ns)

        return new


class P(type):
    __new__ = NewDesc()

    def __init__(cls, *a, **k):
        pass


class E(metaclass=P, z=3):
    pass


assert log3 == [("desc __new__", ["z"])], log3
assert type(E) is P, type(E)

for kwds in ({}, {"metaclass": type}):
    try:
        __build_class__(lambda: None, "a\0b", **kwds)
    except ValueError as e:
        assert str(e) == "type name must not contain null characters", str(e)
    else:
        raise AssertionError("NUL in the class name accepted: %r" % (kwds,))


class RaisingCallDesc:
    def __get__(self, obj, objtype=None):
        raise ValueError("from __get__")


class RaisingMM(type):
    __call__ = RaisingCallDesc()


class RaisingM(type, metaclass=RaisingMM):
    pass


for make in (
    lambda: RaisingM("X", (), {}),
    lambda: __build_class__(lambda: None, "Y", metaclass=RaisingM, k=1),
):
    try:
        make()
    except ValueError as e:
        assert str(e) == "from __get__", str(e)
    else:
        raise AssertionError("the __get__ error was not propagated")

print("OK")
