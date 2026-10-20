# pyre-check: gate=1
# `from_param` of the non-simple metaclasses: `PyCPointerType_from_param` for
# pointer types and `CDataType_from_param` for structures and arrays.  A
# foreign call converts its arguments without going through these, so only a
# direct call observes them.
import ctypes

from testutils import assert_raises


class S(ctypes.Structure):
    _fields_ = [("a", ctypes.c_int)]


P = ctypes.POINTER(ctypes.c_int)
CArg = type(ctypes.byref(ctypes.c_int()))

# A pointer type: None stays None, a pointee instance goes through byref, and a
# pointer or array of the pointee is the value itself.
assert P.from_param(None) is None
assert type(P.from_param(ctypes.c_int(3))) is CArg
array = (ctypes.c_int * 3)()
assert P.from_param(array) is array
pointer = ctypes.pointer(ctypes.c_int(1))
assert P.from_param(pointer) is pointer
assert P.from_param(ctypes.byref(ctypes.c_int())) is not None
with assert_raises(TypeError) as cm:
    P.from_param(ctypes.c_short(1))
assert str(cm.exception) == f"expected {P.__name__} instance instead of c_short", cm.exception
with assert_raises(TypeError) as cm:
    P.from_param(1)
assert str(cm.exception) == f"expected {P.__name__} instance instead of int", cm.exception

# A structure or array type: an instance of the type, or what `_as_parameter_`
# converts to; anything else is refused, a tuple included.
s = S()
assert S.from_param(s) is s
with assert_raises(TypeError) as cm:
    S.from_param(1)
assert str(cm.exception) == "expected S instance instead of int", cm.exception


class Wrapper:
    _as_parameter_ = s


assert S.from_param(Wrapper()) is s
array_type = ctypes.c_int * 3
assert array_type.from_param(array) is array
with assert_raises(TypeError):
    array_type.from_param((1, 2, 3))


class Loop:
    pass


loop = Loop()
loop._as_parameter_ = loop
with assert_raises(RecursionError):
    S.from_param(loop)
