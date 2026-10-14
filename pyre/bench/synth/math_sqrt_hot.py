# pyre-check: max-pypy-ratio=1
# pyre-check: requires-modules=math
# The ceiling is pypy's own speed.  With both descents recording this runs
# 0.3x on Windows dynasm; with the `isqrt` fast arm lost it ran 0.8-1.0x on
# the CI runners, inside the ceiling, so this gate does not catch that loss.
# pyre-check: skip-cpython
# cpython 3.46s vs pyre 0.29s (11.9x on the ubuntu runner), and it is not
# gated on — only pypy is.
# Two hot math-builtin loops whose walker specializations this ratio gates.
#
# `math.sqrt(x)`: the builtin-call descent walks the `__majit_wrap_math_sqrt`
# gateway (interp_math.py `math1` + ll_math.py `ll_math_sqrt`) and records a
# pure `CALL_F(sqrt_nonneg_jit)` (`sqrt_nonneg`, EF_ELIDABLE_CANNOT_RAISE)
# plus the boxed float, instead of the opaque `bh_call_fn(sqrt_builtin, NULL,
# x)` residual.  The gateway's `x >= 0` and `isfinite(x)` branches become the
# two guards, so the result `W_FloatObject` virtualizes.  A negative /
# non-finite argument takes the `dont_look_inside` slow path.
#
# `math.isqrt(i)`: `__majit_wrap_math_isqrt` walks `_int_isqrt` for an exact
# positive machine int below `2**53`.  The arm's range test is two
# comparisons: written as `(1..EXACT_FLOAT_INT).contains(&n)` it is a
# `Range::contains` call the walker does not descend, so the gateway walk
# rolls back to the opaque residual call.  Do not drop the loop.
import math

# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than these loops.
N_SQRT = 32645190
N_ISQRT = 20000000


def run_sqrt():
    total = 0.0
    for i in range(N_SQRT):
        total += math.sqrt(float(i))
    return total


def run_isqrt():
    total = 0
    i = 0
    while i < N_ISQRT:
        total += math.isqrt(i)
        i += 1
    return total


print(round(run_sqrt(), 6))
print(run_isqrt())
