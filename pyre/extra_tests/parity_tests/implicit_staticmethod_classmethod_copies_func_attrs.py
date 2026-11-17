# pyre-check: pypy-diverges: pins that an implicit staticmethod copies __name__/__qualname__/__module__/__annotations__ (pypy3: AttributeError) and that a builtin bound to a subclass names the receiver (`bool.from_bytes`, `D.fromkeys`; pypy3 names the defining class)
# CPython-suite gap: test_descr / test_types check implicit staticmethod
# and classmethod wrapping without pinning that the wrapper copies the
# function's presentation attributes, including a subclass whose
# `__init__` is empty. test_funcattrs.test_builtin__qualname__ pins
# `dict.fromkeys` but not a subclass receiver.
# parity-tests reason: `sm_new` / `cm_new` / `type_new_staticmethod` /
# `type_new_classmethod`: implicit wrappers copy the function's
# presentation attributes, and a subclass whose `__init__` is empty
# still has `__func__` from `__new__`. A bound builtin `__qualname__`
# names the receiver (`meth_get__qualname__`).

"""Implicit staticmethod/classmethod wrappers copy function attributes."""


class Spam:
    def __new__(cls, x, y):
        pass

    def __init_subclass__(cls):
        pass

    def __class_getitem__(cls, x):
        pass


for n, kind in (
    ("__new__", "staticmethod"),
    ("__init_subclass__", "classmethod"),
    ("__class_getitem__", "classmethod"),
):
    o = Spam.__dict__[n]
    if type(o).__name__ != kind:
        raise AssertionError((n, type(o).__name__))
    if o.__name__ != n:
        raise AssertionError((n, o.__name__))
    if o.__qualname__ != "Spam." + n:
        raise AssertionError(o.__qualname__)
    if o.__module__ != "__main__":
        raise AssertionError(o.__module__)
    if o.__annotations__ != {}:
        raise AssertionError(o.__annotations__)


class MySM(staticmethod):
    def __init__(self, *a):
        pass


def f():
    pass


sm = MySM(f)
if sm.__func__ is not f:
    raise AssertionError(sm.__func__)
if not repr(sm).startswith("<staticmethod("):
    raise AssertionError(repr(sm))
if sm.__name__ != "f":
    raise AssertionError(sm.__name__)


def check_qualname(got, expected):
    if got != expected:
        raise AssertionError((got, expected))


check_qualname(bool.from_bytes.__qualname__, "bool.from_bytes")
check_qualname(int.__init_subclass__.__qualname__, "int.__init_subclass__")
check_qualname(type.__subclasshook__.__qualname__, "type.__subclasshook__")


class D(dict):
    pass


check_qualname(D.fromkeys.__qualname__, "D.fromkeys")
check_qualname(D().keys.__qualname__, "D.keys")


class A:
    @classmethod
    def f(cls):
        pass


class B(A):
    pass


check_qualname(B.f.__qualname__, "A.f")

import math

check_qualname(math.sqrt.__qualname__, "sqrt")
check_qualname(len.__qualname__, "len")

print("OK")
