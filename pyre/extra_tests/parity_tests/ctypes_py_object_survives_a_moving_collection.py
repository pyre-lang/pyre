# CPython-suite gap: test_ctypes never reads a py_object back after a
# collection, so nothing there reaches a relocated target.
# parity-tests reason: this guards a pyre/PyPy moving-GC allocation invariant.
# pyre-check: pypy-diverges: its container slot is zero-based, so `repr` of a
# fresh py_object raises IndexError on an empty table and answers the first
# stored object once one exists (measured on PyPy 7.3.20)
# import ctypes builds byte-swapped simple types through a metaclass `__init__` whose `type` argument must survive the collecting `type.__call__`.
# parity-env: PYPY_GC_NURSERY=4096
# parity-env: MAJIT_GC_NURSERY_POISON=1

"""``ctypes.py_object`` must survive the collector relocating its target.

A ``_SimpleCData`` buffer is ``malloc_raw`` memory: the collector neither
traces it nor rewrites it.  Storing the target's address there is therefore
only correct while nothing moves, and a ``list`` header is an ordinary movable
allocation.  ``GlobalPyobjContainer`` is upstream's answer -- the buffer holds
a table slot, and the table is a real object the root walk forwards.
"""

import ctypes
import gc
import weakref

payload = ["alpha", "beta"]
box = ctypes.py_object(payload)

# `gc.collect(0)` is the incminimark minor collection: it evacuates the
# nursery, so the list header moves.  A buffer holding the old address would
# now be pointing into reclaimed nursery memory.
gc.collect(0)

assert box.value is payload, box.value
assert box.value == ["alpha", "beta"]
assert repr(box) == "py_object(['alpha', 'beta'])", repr(box)

# A fresh `py_object` still reads as NULL, which is the reason the stored word
# is one-based rather than upstream's `num = len(self.objs)`: a zero-based slot
# is indistinguishable from the zero an unset buffer holds, which is why
# `repr(py_object())` on PyPy 7.3.20 answers the first stored object, or
# raises IndexError when nothing has been stored yet.
#
# (Reading `.value` here is left out: pyre answers None where `O_get` raises
# `ValueError: PyObject is NULL`, a divergence that predates this file and is
# unchanged by it.)
empty = ctypes.py_object()
assert repr(empty) == "py_object(<NULL>)", repr(empty)

# The `ctypes.util` shape: a POINTER(py_object) laid over the box's own
# buffer, read through after the target moved.
pointer = ctypes.cast(ctypes.addressof(box), ctypes.POINTER(ctypes.py_object))
assert pointer.contents.value is payload
for index in range(200):
    filler = [[step] for step in range(200)]
    assert len(filler) == 200
    pointer.contents.value.append(index)
assert payload[-1] == 199

# An int is not weakrefable; upstream keeps the object itself for that case.
number = ctypes.py_object(10**30)
gc.collect(0)
assert number.value == 10**30, number.value


# CPython's `O_set` hands `PyCData_set` the referent represented by the word
# it stored.  A same-typed `py_object` assignment therefore retains the value
# at assignment time, not the wrapper whose `.value` can subsequently change.
class Payload:
    pass


class ObjectField(ctypes.Structure):
    _fields_ = [("value", ctypes.py_object)]


ObjectArray = ctypes.py_object * 1


def assert_wrapper_copy_keeps_referent(destination, write, read):
    original = Payload()
    original_ref = weakref.ref(original)
    holder = ctypes.py_object(original)
    write(destination, holder)
    holder.value = Payload()
    del original
    gc.collect(0)
    kept = original_ref()
    assert kept is not None
    assert read(destination) is kept


field = ObjectField()
assert_wrapper_copy_keeps_referent(
    field,
    lambda destination, value: setattr(destination, "value", value),
    lambda destination: destination.value,
)

array = ObjectArray()
assert_wrapper_copy_keeps_referent(
    array,
    lambda destination, value: destination.__setitem__(0, value),
    lambda destination: destination[0],
)

pointer_base = ObjectArray()
pointer = ctypes.cast(pointer_base, ctypes.POINTER(ctypes.py_object))
assert_wrapper_copy_keeps_referent(
    pointer,
    lambda destination, value: destination.__setitem__(0, value),
    lambda destination: destination[0],
)

# `None` is already held strongly by PyPy's `GlobalPyobjContainer`, and
# CPython does not expose a redundant keepalive entry for it.
none_field = ObjectField()
none_field.value = None
assert none_field._objects is None

none_array = ObjectArray()
none_array[0] = None
assert none_array._objects is None

none_pointer_base = ObjectArray()
none_pointer = ctypes.cast(none_pointer_base, ctypes.POINTER(ctypes.py_object))
none_pointer[0] = None
assert "0" not in none_pointer._objects

# After CFuncPtr.__del__, a later call still reaches the Python callable.
@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)
def cb(x):
    return x + 1


recorded = cb(1)
d = getattr(cb, "__del__", None)
if d is not None:
    d()
    d()
    assert cb(1, 2) == 2
assert cb(1) == 2

# A CFUNCTYPE wrapping a native address keeps that address after __del__.
@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)
def kept(x):
    return x + 1


native = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)(
    ctypes.cast(kept, ctypes.c_void_p).value
)
d = getattr(native, "__del__", None)
if d is not None:
    d()
assert native(7) == 8

try:
    absfn = ctypes.CDLL(None).abs
    absfn.argtypes = [ctypes.c_int]
    absfn.restype = ctypes.c_int
except (OSError, AttributeError, TypeError):
    absfn = None
if absfn is not None:
    d = getattr(absfn, "__del__", None)
    if d is not None:
        d()
    assert absfn(-9) == 9

_simple_type_codes = ("h", "H", "i", "I", "l", "L", "q", "Q", "f", "d")
_simple_names = ["T%02d" % n for n in range(50)]
gc.collect(0)
for i in range(50):
    T = type(
        _simple_names[i],
        (ctypes._SimpleCData,),
        {"_type_": _simple_type_codes[i % 10]},
    )
    _ = ["%d-%d" % (i, j) for j in range(500)]
    assert T.__ctype_be__._type_ == T._type_
    assert T.__ctype_le__._type_ == T._type_

print("OK")
