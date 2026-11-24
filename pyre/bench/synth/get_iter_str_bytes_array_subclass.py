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

non_ascii = "aé€😀b"
print("non_ascii", collect(non_ascii))
lone = "a\ud800b"
print("lone", collect(lone))
it = iter(non_ascii)
it.__setstate__(2)
print("setstate2", list(it))
it = iter(non_ascii)
it.__setstate__(-3)
print("setstate_neg", list(it))
it = iter(non_ascii)
it.__setstate__(99)
print("setstate99", list(it))

it = iter("aé€😀b")
mix_for = []
for ch in it:
    mix_for.append(ch)
    if ch == "€":
        break
print("mix_for", mix_for)
print("mix_next", next(it))
print("mix_rest", list(it))


class GetitemOnly:
    def __getitem__(self, i):
        if i < 3:
            return i
        raise IndexError


it = iter(GetitemOnly())
try:
    print("1 length_hint", it.__length_hint__())
except Exception as e:
    print("1 length_hint raises", type(e).__name__)


class RaiseOnce:
    n = 0

    def __getitem__(self, i):
        RaiseOnce.n += 1
        if RaiseOnce.n == 2:
            raise ValueError("once")
        if i < 3:
            return i
        raise IndexError


it = iter(RaiseOnce())
print("2a", next(it))
try:
    next(it)
except ValueError:
    print("2b ValueError")
try:
    print("2c", next(it))
except StopIteration:
    print("2c StopIteration")

for name, obj in [
    ("bytes", b"abc"),
    ("bytearray", bytearray(b"abc")),
    ("array", array.array("b", b"abc")),
    ("str", "abc"),
]:
    it = iter(obj)
    it.__setstate__(-5)
    r = []
    try:
        r = list(it)
    except Exception as e:
        r = type(e).__name__
    it2 = iter(obj)
    it2.__setstate__(10)
    r2 = list(it2)
    print("3", name, r, r2, it2.__reduce__()[1:])


class ArrayGetitem(array.array):
    def __getitem__(self, i):
        return 100 + array.array.__getitem__(self, i)


print("4", list(ArrayGetitem("b", [1, 2])))


class ArrayGetitemRaise(array.array):
    def __getitem__(self, i):
        if i == 0:
            raise ValueError
        return array.array.__getitem__(self, i)


it = iter(ArrayGetitemRaise("b", [1, 2]))
try:
    next(it)
except ValueError:
    print("4b ValueError")
try:
    print("4c", next(it))
except StopIteration:
    print("4c StopIteration")

for o in ["a", b"a", bytearray(b"a"), memoryview(b"a"), array.array("b", [1])]:
    it = iter(o)
    print(
        "5",
        type(o).__name__,
        type(it).__name__,
        sorted(
            m
            for m in ("__reduce__", "__setstate__", "__length_hint__")
            if hasattr(it, m)
        ),
    )

class BytesGetitem(bytes):
    def __getitem__(self, i):
        return 100 + bytes.__getitem__(self, i)


class BytearrayGetitem(bytearray):
    def __getitem__(self, i):
        return 100 + bytearray.__getitem__(self, i)


print("6", list(BytesGetitem(b"ab")))
print("6ba", list(BytearrayGetitem(b"ab")))

ba = bytearray(b"ab")
it = iter(ba)
print("7a", next(it))
ba.append(ord("c"))
print("7b", list(it))
ba2 = bytearray(b"xyz")
it2 = iter(ba2)
print("7c", next(it2))
ba2.clear()
try:
    print("7c2", next(it2))
except StopIteration:
    print("7c2 StopIteration")

it = iter(b"abcde")
got = []
for x in it:
    got.append(x)
    if x == ord("c"):
        break
print("8for", got)
print("8next", next(it))
print("8rest", list(it))

print("PASS")
