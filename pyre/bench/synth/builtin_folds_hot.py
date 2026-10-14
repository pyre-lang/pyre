# pyre-check: max-pypy-ratio=2.9
# pyre-check: trace-shape=run_abs_float:absent=CallMayForceR,run_min_max:absent=CallMayForceR
# A ceiling under 3 derives no floor.
# pyre-check: skip-cpython
# The ceiling was fitted when the `builtin_fold1` / `builtin_fold2` hand
# folds answered these calls, at 4.6x / 4.7x on darwin-arm64, 7.2x / 7.6x on
# ubuntu-24.04 and 9.2x / 10.0x on windows (half the loop counts below).  The
# loop counts are doubled so pypy's execution-only time clears
# FLOOR_GATE_MIN_BASELINE_S on the windows runner, whose CPU accounting
# advances in 1/64s ticks, and so the fixed startup-subtraction error is a
# smaller share of the denominator.
#
# Every builtin here is now an interp2app gateway the generic builtin descent
# walks, the way the tracer walks straight into an RPython builtin body:
#
#   hash_int           `_hash_int` inline on an exact machine int.
#   hash_str           an exact `str` pinned first, then one
#                      `elidable_cannot_raise` leaf the optimizer hoists out
#                      of the loop.
#   ord_str            the same shape, the leaf returning the code point of a
#                      one-code-point exact `str`.
#   abs_int/abs_float  `descr_abs` on an exact int (short of `i64::MIN`) or
#                      float, boxed by a leaf the optimizer keeps virtual.
#   min_max            two exact machine ints or floats compared inline; the
#                      answer is the winning operand's own reference.
#
# Every other operand shape reaches the original builtin body through a
# `dont_look_inside` slow path.  Without the descent each call is a
# `bh_call_fn(builtin, NULL, ...)` residual: the frame force, the argument
# rooting, the execution-context resolution and the gateway signature binding
# all run before the body does, and the ratio is the detector, because the
# residual compiles the same loop and changes no jit-stats counter.
HASH_N = 32000000
ORD_N = 32000000
ABS_N = 32000000
MINMAX_N = 24000000


def run_hash_int():
    # `hash(int)` is the value itself, so this total is the same everywhere.
    total = 0
    x = 1234567
    for _ in range(HASH_N):
        total += hash(x)
    return total


def run_hash_str():
    # A string's hash is seeded per process, so the digest itself cannot be
    # printed.  Count the iterations that agree with the first one instead:
    # the descent still has to produce the digest, and the count is invariant.
    s = "specialize"
    first = hash(s)
    same = 0
    for _ in range(HASH_N):
        if hash(s) == first:
            same += 1
    return same


def run_ord():
    total = 0
    c = "q"
    for _ in range(ORD_N):
        total += ord(c)
    return total


def run_abs_int():
    total = 0
    x = -7
    for _ in range(ABS_N):
        total += abs(x)
    return total


def run_abs_float():
    total = 0.0
    x = -7.5
    for _ in range(ABS_N):
        total += abs(x)
    return total


def run_min_max():
    total = 0
    a = 3
    b = 9
    for _ in range(MINMAX_N):
        total += min(a, b) + max(a, b)
    return total


print(run_hash_int())
print(run_hash_str())
print(run_ord())
print(run_abs_int())
print(round(run_abs_float(), 6))
print(run_min_max())
