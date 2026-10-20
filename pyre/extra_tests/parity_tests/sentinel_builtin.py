# pyre-check: pypy-diverges: PyPy 3.11 has no builtin sentinel (PEP 661).
# CPython-suite gap: 3.15 has no Lib/test/test_sentinel.py; Objects/sentinelobject.c
# is the spec.
# parity-tests reason: pin constructor/repr/pickle/copy/unions/subclassing
# against the 3.15.0 oracle.

import copy
import pickle
import sys

assert sentinel.__module__ == "builtins"
assert sentinel.__mro__ == (sentinel, object)
assert sentinel.__bases__ == (object,)

MISSING = sentinel("MISSING")
assert type(MISSING) is sentinel
assert repr(MISSING) == "MISSING"
assert bool(MISSING) is True
assert MISSING.__name__ == "MISSING"
assert MISSING.__module__ == "__main__"
assert MISSING.__reduce__() == "MISSING"
assert copy.copy(MISSING) is MISSING
assert copy.deepcopy(MISSING) is MISSING
assert pickle.loads(pickle.dumps(MISSING)) is MISSING

custom = sentinel("X", repr="CUSTOM")
assert repr(custom) == "CUSTOM"
assert repr(sentinel("Y", repr=None)) == "Y"


def in_func():
    return sentinel("IN_F")


assert in_func().__module__ == "__main__"

MISSING.__module__ = "other"
assert MISSING.__module__ == "other"
MISSING.__module__ = "__main__"

try:
    MISSING.__name__ = "no"
except AttributeError as e:
    assert str(e) == "readonly attribute", e
else:
    raise AssertionError("__name__ was writable")

try:

    class Derived(sentinel):
        pass
except TypeError as e:
    assert str(e) == "type 'sentinel' is not an acceptable base type", e
else:
    raise AssertionError("sentinel accepted as a base")

try:
    sentinel(1)
except TypeError as e:
    assert str(e) == "sentinel() argument 1 must be str, not int", e
else:
    raise AssertionError("non-str name accepted")

try:
    sentinel("x", repr=1)
except TypeError as e:
    assert str(e) == "sentinel() argument 'repr' must be str or None, not int", e
else:
    raise AssertionError("non-str repr accepted")

union = MISSING | int
assert type(union) is type(int | str)
assert repr(union) == "MISSING | int"

print("OK")
