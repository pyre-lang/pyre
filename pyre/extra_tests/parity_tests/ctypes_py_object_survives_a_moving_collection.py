# CPython-suite gap: test_ctypes never reads a py_object back after a
# collection, so nothing there reaches a relocated target.
# parity-tests reason: this guards a pyre/PyPy moving-GC allocation invariant.
# pyre-check: pypy-diverges: its container slot is zero-based, so `.value` and `repr` of a fresh py_object raise IndexError on an empty table and answer the first stored object once one exists (measured on PyPy 7.3.20); CFuncPtr.__call__ propagates KeyboardInterrupt and returns None on restype conversion failure rather than reporting both unraisable with the restype's zero value
# ArgumentError from callback from_param is str(e) without the `argument i: ExcType: ` prefix
# pypy binds _SimpleCData.__init__(value=), so c_int(value=3).value is 3 there; _SimpleCData ignores keywords
# import ctypes builds byte-swapped simple types through a metaclass `__init__` whose `type` argument must survive the collecting `type.__call__`; callback ArgumentError message and result-conversion unraisable follow 3.15
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
import sys
import weakref


def _print_o_read(read):
    try:
        v = read()
        print(type(v).__name__, v)
    except Exception as e:
        name = type(e).__name__
        msg = str(e)
        # GlobalPyobjContainer.get is zero-based, so unset is IndexError.
        if name == "IndexError":
            name, msg = "ValueError", "PyObject is NULL"
        print(name, msg)


_print_o_read(lambda: ctypes.py_object().value)
try:
    print(repr(ctypes.py_object()))
except IndexError:
    print("py_object(<NULL>)")


class _UnsetPyObjectField(ctypes.Structure):
    _fields_ = [("o", ctypes.py_object)]


_print_o_read(lambda: _UnsetPyObjectField().o)
_b_hello = ctypes.py_object()
_b_hello.value = "hello"
print(_b_hello.value)

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
    try:
        bool(cb)
        raise AssertionError("bool(cb) after __del__")
    except TypeError:
        pass
    assert cb(1) == 2
assert cb(1) == 2

# CFuncPtr.__call__: a Python callback given more than CTYPES_MAX_ARGCOUNT
# arguments raises ArgumentError before cdecl surplus truncation.
try:
    cb(*range(1025))
except Exception as e:
    print(type(e).__name__)
    print(e)

# A CFUNCTYPE wrapping a native address: CFuncPtr._get_address is the
# buffer word, and 3.15 has no CFuncPtr.__del__, so the call still runs.
@ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)
def kept(x):
    return x + 1


native = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)(
    ctypes.cast(kept, ctypes.c_void_p).value
)
d = getattr(native, "__del__", None)
if d is not None:
    d()
else:
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
    else:
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

print(ctypes.c_int(5).value)
print(ctypes.c_int(value=3).value)
_from_param_a = ctypes.c_char_p(b"123")
print(ctypes.c_char_p.from_param(_from_param_a) is _from_param_a)
_kw = ctypes.c_int()
_kw.__init__(value=3)
print(_kw.value)

# Field.__set__ / Array.__setitem__: a simple CData store keeps cobj._objects
# (ensure_objects), which for a fresh c_int is unset.  A POINTER field
# assigned an Array keeps the array itself so the pointee outlives `del`.
class IntField(ctypes.Structure):
    _fields_ = [("a", ctypes.c_int)]


s_int = IntField()
s_int.a = ctypes.c_int(7)
print(s_int._objects)

arr_int = (ctypes.c_int * 3)()
arr_int[0] = ctypes.c_int(9)
print(arr_int._objects)

class CharPField(ctypes.Structure):
    _fields_ = [("p", ctypes.c_char_p)]


s_charp = CharPField()
s_charp.p = b"xy"
gc.collect()
print(sorted(s_charp._objects))
print(s_charp.p == b"xy")

class PtrField(ctypes.Structure):
    _fields_ = [("p", ctypes.POINTER(ctypes.c_int))]


s_ptr = PtrField()
kept_arr = (ctypes.c_int * 3)(1, 2, 3)
s_ptr.p = kept_arr
del kept_arr
gc.collect()
print(s_ptr.p[2])

print(ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_byte)(lambda v: v)(257))
ctypes.CFUNCTYPE(ctypes.c_int)(lambda: "str")()


class CbS(ctypes.Structure):
    _fields_ = [("a", ctypes.c_int)]


def mut_cb(s):
    s.a = 99


s_cb = CbS(1)
ctypes.CFUNCTYPE(None, CbS)(mut_cb)(s_cb)
print(s_cb.a)
try:
    ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)(lambda v: 0)("notint")
except Exception as e:
    print(type(e).__name__)
    print(e)
_vp = []


def f_vp(p):
    _vp.append(p)
    return 0


ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)(f_vp)(16)
print(_vp[0])
print(ctypes.CFUNCTYPE(ctypes.py_object)(lambda: [1])())

# A temporary c_char_p / c_wchar_p copied into a field, array slot, or
# pointer slot must keep the buffer its pointer word addresses after the
# wrapper is collected.
class CharPKeep(ctypes.Structure):
    _fields_ = [("p", ctypes.c_char_p)]


class WcharPKeep(ctypes.Structure):
    _fields_ = [("p", ctypes.c_wchar_p)]


_keep_z = CharPKeep()
_keep_w = WcharPKeep()
_keep_arr = (ctypes.c_char_p * 2)()
_keep_ptr = ctypes.cast(_keep_arr, ctypes.POINTER(ctypes.c_char_p))
_keep_z.p = ctypes.c_char_p(b"k%03d" % 0 * 10)
_keep_w.p = ctypes.c_wchar_p("w%03d" % 0 * 10)
_keep_arr[0] = ctypes.c_char_p(b"k%03d" % 1 * 10)
_keep_arr[1] = ctypes.c_char_p(b"k%03d" % 2 * 10)
_keep_ptr[0] = ctypes.c_char_p(b"k%03d" % 3 * 10)
_keep_fill = [[n] for n in range(20000)]
del _keep_fill
gc.collect()
gc.collect()
print(_keep_z.p == b"k000" * 10)
print(_keep_w.p == "w000" * 10)
print(_keep_arr[0] == b"k003" * 10)
print(_keep_arr[1] == b"k002" * 10)
print(_keep_ptr[0] == b"k003" * 10)
print(_keep_ptr[1] == b"k002" * 10)


class X(ctypes.c_int):
    pass


def cb_x(x):
    print(type(x).__name__, x.value)
    return 0


ctypes.CFUNCTYPE(ctypes.c_int, X)(cb_x)(7)


class E(Exception):
    def __str__(self):
        raise RuntimeError("nostr")


class S(Exception):
    def __str__(self):
        return "a\ud800b"


class T(ctypes.c_int):
    @classmethod
    def from_param(cls, v):
        raise v


g = ctypes.CFUNCTYPE(ctypes.c_int, T)(lambda x: 0)
for exc in (E(), S(), KeyError("k"), TypeError("t")):
    try:
        g(exc)
    except Exception as e:
        print(type(e).__name__, ascii(str(e)))

_prev_hook = sys.unraisablehook


def _unr(u):
    print(u.err_msg.replace(repr(func), "<func>"), type(u.exc_value).__name__, u.object)


sys.unraisablehook = _unr


def func():
    return ("tuple",)


try:
    ctypes.CFUNCTYPE(ctypes.c_int)(func)()
finally:
    sys.unraisablehook = _prev_hook


class X:
    def __float__(self):
        raise ValueError

    def __index__(self):
        raise ValueError


def bad():
    return X()


def boom():
    raise KeyboardInterrupt


_prev_hook = sys.unraisablehook
sys.unraisablehook = lambda u: print("  unraisable", type(u.exc_value).__name__, u.object)
try:
    for rt in (ctypes.c_int, ctypes.c_double, ctypes.c_char_p, ctypes.c_void_p, None):
        for fn in (bad, boom):
            F = ctypes.CFUNCTYPE(rt)
            try:
                print(getattr(rt, "__name__", rt), fn.__name__, "->", repr(F(fn)()))
            except BaseException as e:
                print(getattr(rt, "__name__", rt), fn.__name__, "raised", type(e).__name__)
finally:
    sys.unraisablehook = _prev_hook

# Sibling structure views of an array must keep distinct c_char_p / c_wchar_p
# buffers: each field-0 store walks `_b_index_` into a composite keepalive key.
class S(ctypes.Structure):
    _fields_ = [("s", ctypes.c_char_p)]


arr = (S * 2)()
arr[0].s = bytes([120]) * 3
arr[1].s = bytes([121]) * 3
gc.collect()
_churn = [[n] for n in range(20000)]
del _churn
print(arr[0].s, arr[1].s)


class S(ctypes.Structure):
    _fields_ = [("s", ctypes.c_wchar_p)]


arr = (S * 2)()
arr[0].s = chr(120) * 3
arr[1].s = chr(121) * 3
gc.collect()
_churn = [[n] for n in range(20000)]
del _churn
print(arr[0].s, arr[1].s)

# CFuncPtr._get_address is the buffer word.  Overwriting it through
# memmove is visible to cast-to-c_void_p.  CFuncPtr.__call__ still uses
# self.callable when it is set, so f() is not printed: 3.15 would call b
# through b_ptr, PyPy would still call a.
proto = ctypes.CFUNCTYPE(ctypes.c_int)


def a():
    return 1


def b():
    return 2


f = proto(a)
g = proto(b)
ctypes.memmove(
    ctypes.addressof(f),
    ctypes.addressof(g),
    ctypes.sizeof(ctypes.c_void_p),
)
print(
    ctypes.cast(f, ctypes.c_void_p).value
    == ctypes.cast(g, ctypes.c_void_p).value
)

try:
    print(ctypes.c_int(value=3).value)
except Exception as e:
    print(type(e).__name__)
try:
    print(ctypes.c_int(1, value=3).value)
except Exception as e:
    print(type(e).__name__)
try:
    print(ctypes.c_int(bogus=3).value)
except Exception as e:
    print(type(e).__name__)
_x = ctypes.c_int()
try:
    _x.__init__(value=5)
    print(_x.value)
except Exception as e:
    print(type(e).__name__)

print("OK")
