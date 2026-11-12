# pyre-check: pypy-diverges: pins that an implicit staticmethod copies __name__/__qualname__/__module__/__annotations__; pypy3 raises AttributeError "'staticmethod' object has no attribute '__name__'"
# CPython-suite gap: test_descr / test_types check implicit staticmethod
# and classmethod wrapping without pinning that the wrapper copies the
# function's presentation attributes, including a subclass whose
# `__init__` is empty.
# parity-tests reason: `sm_new` / `cm_new` / `type_new_staticmethod` /
# `type_new_classmethod`: implicit wrappers copy the function's
# presentation attributes, and a subclass whose `__init__` is empty
# still has `__func__` from `__new__`.

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

print("OK")
