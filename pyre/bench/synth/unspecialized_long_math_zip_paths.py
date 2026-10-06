# pyre-check: spec-folds=zip_two_tuple_iters
# pyre-check: requires-modules=math
# Throughput gate for the hand-written folds this file still fires: `zip`
# over two tuples (positional and `strict=True`). `math.frexp` is the
# descended `__majit_wrap_math_frexp` gateway. `bigint ** int` is the
# descended `long_pow` body and `math.ldexp` the descended `math` gateway.
#
# Which folds a leg fires is a census question, not a reading of the source:
# `PYRE_FBW_SPEC_CENSUS=1` prints `fold=<label> consulted=N fired=N` per label.
# This fixture claimed eight and delivered seven for as long as `builtin_zip`
# read its `call_kw` operands in the wrong order -- consulted once per run and
# declined every time, so the leg written for it gated nothing.  Read the
# census, not the leg names, before trusting the `spec-folds` coverage claim.
#
# Every result is reduced to an integer checksum: the float legs go through
# `int(round(x * 1e6))` so the comparison cannot turn into a float-formatting
# difference between the three backends.

import math

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

# Each leg only needs to compile and make its declared fold fire; the census
# carries the performance-sensitive signal directly.
N_ZIP = 10000
N_ZIP_STRICT = 10000
N_CMP = 100000
N_TRUEDIV = 50000
N_POW = 20000
N_SHIFT = 50000
N_FREXP = 50000
N_LDEXP = 50000


def f_zip(n):
    t1 = (1, 2, 3, 4, 5, 6, 7, 8)
    t2 = (10, 20, 30, 40, 50, 60, 70, 80)
    s = 0
    for _ in range(n):
        for a, b in zip(t1, t2):
            s += a * b
    return s


def f_zip_strict(n):
    t1 = (1, 2, 3, 4, 5, 6, 7, 8)
    t2 = (10, 20, 30, 40, 50, 60, 70, 80)
    s = 0
    i = 0
    while i < n:
        for a, b in zip(t1, t2, strict=True):
            s += a + b
        i += 1
    return s


def f_cmp_long_int(n):
    big = 1 << 200
    s = 0
    i = 0
    while i < n:
        if big > i:
            s += 1
        if i < big:
            s += 2
        i += 1
    return s


def f_truediv_long(n):
    big = 1 << 200
    s = 0.0
    i = 1
    while i <= n:
        s += big / (big + i)
        i += 1
    return s


def f_pow_long_int(n):
    base = 1 << 70
    s = 0
    i = 0
    while i < n:
        s += ((base + i) ** 2) % 1000003
        i += 1
    return s


def check_pow_side_exits(n):
    big = (1 << 200) + 12345
    for i in range(n):
        exponent = 2 if i & 1 else 0
        value = big ** exponent
        assert value == (big * big if exponent else 1)
    assert big ** True == big and big ** False == 1
    assert 0.0 < big ** -1 < 1e-50
    assert pow(big, 3, 1000003) == (big * big * big) % 1000003


def f_shift_long_int(n):
    big = 1 << 200
    s = 0
    i = 0
    while i < n:
        s += ((big + i) >> 3) % 1000003
        s += ((big + i) << 3) % 1000003
        i += 1
    return s


def f_frexp(n):
    s = 0.0
    i = 1
    while i <= n:
        m, e = math.frexp(i * 1.5)
        s += m + e
        i += 1
    return s


def f_ldexp(n):
    s = 0.0
    i = 1
    while i <= n:
        s += math.ldexp(0.5, i % 8)
        i += 1
    return s


print("zip       ", f_zip(N_ZIP))
print("zip_strict", f_zip_strict(N_ZIP_STRICT))
print("cmp_long  ", f_cmp_long_int(N_CMP))
print("truediv   ", int(round(f_truediv_long(N_TRUEDIV) * 1e6)))
print("pow       ", f_pow_long_int(N_POW))
check_pow_side_exits(N_POW)
print("shift     ", f_shift_long_int(N_SHIFT))
print("frexp     ", int(round(f_frexp(N_FREXP) * 1e6)))
print("ldexp     ", int(round(f_ldexp(N_LDEXP) * 1e6)))
