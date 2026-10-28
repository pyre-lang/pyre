# pyre-check: platforms=linux,darwin
# pyre-check: pypy-diverges: pypy3 implements 3.11, which has no ctypes C-complex simple types
# CPython-suite gap: test_numbers.test_complex / test_complex_round_trip,
# test_byteswap endian float/double complex, and test_c_simple_type_meta
# test_bad_type_message cover these once the classes exist.
# parity-tests reason: 3.15 adds c_float_complex / c_double_complex /
# c_longdouble_complex (`Zf`/`Zd`/`Zg`) on libffi that has FFI_TYPE_COMPLEX.

import array
import ctypes

types = (
    ctypes.c_float_complex,
    ctypes.c_double_complex,
    ctypes.c_longdouble_complex,
)
for T in types:
    name = T.__name__
    print(name, "_type_", T._type_)
    print(name, "sizeof", ctypes.sizeof(T))
    print(name, "alignment", ctypes.alignment(T))
    print(name, "empty", T().value)
    print(name, "complex", T(1 + 2j).value)
    print(name, "real", T(3).value)
    print(name, "signed-zero", repr(T(-0.0 - 0.0j).value))
    print(name, "inf", T(float("inf")).value)
    print(name, "bytes", bytes(T(1 + 2j)).hex())
    print(name, "mv", memoryview(T()).format)
    print(name, "mv-array", memoryview((T * 3)()).format)
    print(name, "has-be", hasattr(T, "__ctype_be__"))
    if name != "c_longdouble_complex":
        be = T.__ctype_be__(1 + 2j)
        print(name, "be-bytes", bytes(be).hex())
        print(name, "be-value", be.value)
    try:
        T("x")
    except Exception as e:
        print(name, "str-err", type(e).__name__, e)
    print(name, "from_param", type(T.from_param(1 + 2j)).__name__)


class S(ctypes.Structure):
    _fields_ = [("z", ctypes.c_double_complex)]


s = S()
s.z = 1 + 2j
print("struct", s.z, ctypes.sizeof(S))

try:

    class X(ctypes._SimpleCData):
        _type_ = "Zq"
except AttributeError as e:
    print("bad-type", e)

print(
    "longdouble-is-double",
    ctypes.c_longdouble_complex is ctypes.c_double_complex,
)


class IndexLike:
    def __index__(self):
        return 2


for T in types:
    print(T(IndexLike()).value)
print(repr(array.array("Zd", [IndexLike()])))
print("OK")
