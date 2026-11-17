# pyre-check: gate=1
"""Variable machine-int ** that overflows to long, long enough to compile.

`int_pow_nomod` is `@jit.look_inside_iff(isconstant(exponent))`. A variable
exponent residualizes; overflow must take `_pow_ovf2long` via the helper
`catch_exception` rather than escape as OverflowError from the compiled
GUARD_NO_EXCEPTION.
"""

N = 1500
last = 0
for i in range(N):
    k = 10 + (i % 20)
    last = 10 ** k
print(last)
assert last == 10**29, last


def pow_overflows(n):
    return 2**n


def foriter_around_inlined_pow_overflow():
    # An inlined callee whose `**` overflows to long, after an inner
    # FOR_ITER loop in the same traced iteration: the inner loop's first item
    # must not run twice.
    total = 0
    i = 0
    while i < 2000:
        for v in range(10, 0, -3):
            total += v
        pow_overflows(70)
        i += 1
    return total


total = foriter_around_inlined_pow_overflow()
assert total == 2000 * 22, total
