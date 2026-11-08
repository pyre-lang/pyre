# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=f,g
# A variable exponent overflowing a machine int must recover through
# `_pow_ovf2long` under the JIT, not raise OverflowError.


def f(n):
    out = 0
    for i in range(n):
        e = i % 80
        x = 5 ** e
        y = 2 ** e
        out += (x + y) & 0xffff
    return out


def g(n):
    big = 0
    for i in range(n):
        big += (3 ** (i % 70)) >> 60
    return big


r = f(3000)
assert r == 88995234, r
s = g(3000)
assert s == 45593985878823328, s
print("PASS")
