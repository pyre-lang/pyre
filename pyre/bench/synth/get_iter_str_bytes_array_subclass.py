# pyre-check: selfcheck
# pyre-check: selfcheck-interpreted
# pyre-check: requires-modules=array
# GET_ITER (`pyopcode.py GET_ITER` → `space.iter`) must honour a heap
# subtype's `__iter__` on str / bytes / bytearray / array.array, and
# `__iter__ = None` must raise TypeError (`descroperation.py iter`).
import array


def collect(obj):
    got = []
    for item in obj:
        got.append(item)
    return got


def raises_typeerror(obj):
    try:
        for _ in obj:
            return False
    except TypeError:
        return True
    return False


class S(str):
    def __iter__(self):
        return iter(["s"])


class B(bytes):
    def __iter__(self):
        return iter(["b"])


class BA(bytearray):
    def __iter__(self):
        return iter(["ba"])


class A(array.array):
    def __iter__(self):
        return iter([7])


class SNone(str):
    __iter__ = None


class BNone(bytes):
    __iter__ = None


class BANone(bytearray):
    __iter__ = None


class ANone(array.array):
    __iter__ = None


s = "a\u00e9\udc80"
assert collect(s) == ["a", "\u00e9", "\udc80"], collect(s)
it_name = type(iter(s)).__name__
assert it_name == "str_iterator", it_name

assert collect(S("abc")) == ["s"]
assert collect(B(b"abc")) == ["b"]
assert collect(BA(b"abc")) == ["ba"]
assert collect(A("b", b"\x01\x02")) == [7]

assert raises_typeerror(SNone("abc"))
assert raises_typeerror(BNone(b"abc"))
assert raises_typeerror(BANone(b"abc"))
assert raises_typeerror(ANone("b", b"\x01"))

# descroperation.py DescrOperation.iter binds `__iter__` through
# get_and_call_function, so a non-function descriptor is applied first.
class LStatic(list):
    __iter__ = staticmethod(lambda: iter([7]))


class AStatic(array.array):
    __iter__ = staticmethod(lambda: iter([8]))


class SClass(str):
    __iter__ = classmethod(lambda cls: iter([cls.__name__]))


assert collect(LStatic([1])) == [7]
assert collect(AStatic("i", [1])) == [8]
assert collect(SClass("x")) == ["SClass"]
got_sclass = []
for x in SClass("q"):
    got_sclass.append(x)
assert got_sclass == ["SClass"]

print("PASS")
