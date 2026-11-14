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
