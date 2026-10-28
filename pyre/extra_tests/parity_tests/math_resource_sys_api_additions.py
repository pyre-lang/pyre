# pyre-check: requires-modules=math,resource
# pyre-check: platforms=linux,darwin
# pyre-check: pypy-diverges: pypy3 implements 3.11, which has none of these 3.15 additions
# CPython-suite gap: test_math, test_resource, test_sys and test_traceback cover
# the pieces separately; this fixture pins the combined 3.15 surface.
# parity-tests reason: 3.15 adds math.isnormal/issubnormal/signbit/fmax/fmin,
# resource.RLIM_SAVED_CUR/MAX and sys.abi_info, drops types.CodeType.co_lnotab,
# and attaches name/obj on a missing-instance delattr.

import math
import resource
import sys
import traceback

inf = float("inf")
nan = float("nan")
nnan = -float("nan")
values = (1.0, 0.0, -0.0, 5e-324, -5e-324, inf, -inf, nan, nnan)
for x in values:
    print(
        "class",
        math.isnormal(x),
        math.issubnormal(x),
        math.signbit(x),
    )
print("isnormal-int", math.isnormal(1))
try:
    math.isnormal("x")
except TypeError as e:
    print("isnormal-type", e)

pairs = (
    (1.0, 2.0),
    (nan, 1.0),
    (1.0, nan),
    (nan, nan),
    (-0.0, 0.0),
    (0.0, -0.0),
    (inf, -inf),
)
for x, y in pairs:
    for name, fn in (("fmax", math.fmax), ("fmin", math.fmin)):
        r = fn(x, y)
        if r == 0.0:
            print(name, repr(r), math.copysign(1, r))
        else:
            print(name, repr(r))
try:
    math.fmax(1)
except TypeError as e:
    print("fmax-type", e)

print(
    "rlim-saved",
    resource.RLIM_SAVED_CUR == resource.RLIM_SAVED_MAX,
    type(resource.RLIM_SAVED_CUR),
    type(resource.RLIM_SAVED_MAX),
)
print("co_lnotab", hasattr((lambda: 0).__code__, "co_lnotab"))
print("abi-vars", sorted(vars(sys.abi_info)))
print("abi-type", type(sys.abi_info).__name__)
print("abi-pointer", sys.abi_info.pointer_bits)
print("abi-byteorder", sys.abi_info.byteorder == sys.byteorder)
print("abi-free-threaded-type", type(sys.abi_info.free_threaded))
print("abi-debug", sys.abi_info.debug)


class A:
    pass


a = A()
a.abcde = 1
try:
    del a.abcdf
except AttributeError as e:
    print("delattr-name", e.name)
    print("delattr-obj", e.obj is a)
    print("delattr-str", str(e))
    print("delattr-tb", traceback.format_exception(e)[-1].rstrip())

print("OK")
