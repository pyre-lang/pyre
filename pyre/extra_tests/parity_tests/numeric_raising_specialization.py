# CPython-suite gap: numeric tests do not compare cold and hot raising traces.
# parity-tests reason: this guards pyre's JIT bigint, float, shift, and range
# raising specializations without repeating the same traceback harness.

"""Guarded raising arms for bigint, float, shift, and range operations."""

import sys


ROUNDS = 6000
BIG = (1 << 100) + 123


def signature(exc, expected_type):
    frames = []
    tb = exc.__traceback__
    while tb is not None:
        frames.append((tb.tb_frame.f_code.co_name, tb.tb_lineno))
        tb = tb.tb_next
    return (
        type(exc) is expected_type,
        str(exc),
        exc.args,
        exc.__suppress_context__,
        exc.__context__,
        tuple(frames),
    )


def check_signature(cold, hot, message, expected_names):
    assert cold == hot, (cold, hot)
    exact_type, actual_message, args, suppress, context, frames = hot
    assert exact_type
    assert actual_message == message, actual_message
    assert args == (message,), args
    assert suppress is False, suppress
    assert context is None, context
    assert tuple(name for name, _ in frames) == expected_names, frames
    assert all(lineno > 0 for _, lineno in frames), frames


# Exact-bigint floor division and modulo by exact zero.
if sys.implementation.name == "pypy":
    BIGINT_MESSAGES = ("integer division or modulo by zero", "long division or modulo by zero")
else:
    BIGINT_MESSAGES = ("division by zero", "division by zero")


def bigint_apply(value, divisor, use_mod):
    return value % divisor if use_mod else value // divisor


def bigint_same(rounds, use_mod):
    found = None
    for i in range(rounds):
        try:
            if use_mod:
                (BIG + i) % 0
            else:
                (BIG + i) // 0
        except ZeroDivisionError as exc:
            found = signature(exc, ZeroDivisionError)
    return found


def bigint_one(rounds, use_mod):
    found = None
    for i in range(rounds):
        try:
            bigint_apply(BIG + i, 0, use_mod)
        except ZeroDivisionError as exc:
            found = signature(exc, ZeroDivisionError)
    return found


def bigint_middle_1(value, divisor, use_mod):
    return bigint_apply(value, divisor, use_mod)


def bigint_middle_2(value, divisor, use_mod):
    return bigint_middle_1(value, divisor, use_mod)


def bigint_three(rounds, use_mod):
    found = None
    for i in range(rounds):
        try:
            bigint_middle_2(BIG + i, 0, use_mod)
        except ZeroDivisionError as exc:
            found = signature(exc, ZeroDivisionError)
    return found


def bigint_flip(use_mod):
    raised = completed = checksum = 0
    for i in range(ROUNDS):
        divisor = 0 if i < 2000 or i >= 4000 else 7
        try:
            checksum += bigint_apply(BIG + i, divisor, use_mod)
            completed += 1
        except ZeroDivisionError:
            raised += 1
    return raised, completed, checksum


for bigint_use_mod, bigint_message in zip((False, True), BIGINT_MESSAGES):
    check_signature(
        bigint_same(1, bigint_use_mod),
        bigint_same(ROUNDS, bigint_use_mod),
        bigint_message,
        ("bigint_same",),
    )
    check_signature(
        bigint_one(1, bigint_use_mod),
        bigint_one(ROUNDS, bigint_use_mod),
        bigint_message,
        ("bigint_one", "bigint_apply"),
    )
    check_signature(
        bigint_three(1, bigint_use_mod),
        bigint_three(ROUNDS, bigint_use_mod),
        bigint_message,
        ("bigint_three", "bigint_middle_2", "bigint_middle_1", "bigint_apply"),
    )
    assert bigint_flip(bigint_use_mod)[:2] == (4000, 2000)


# Exact-bigint left and right shifts by a negative exact int.
SHIFT_MESSAGE = "negative shift count"


def shift_apply(value, count, use_left):
    return value << count if use_left else value >> count


def shift_same(rounds, use_left, count):
    found = None
    for i in range(rounds):
        try:
            if use_left:
                (BIG + i) << count
            else:
                (BIG + i) >> count
        except ValueError as exc:
            found = signature(exc, ValueError)
    return found


def shift_one(rounds, use_left):
    found = None
    for i in range(rounds):
        try:
            shift_apply(BIG + i, -1, use_left)
        except ValueError as exc:
            found = signature(exc, ValueError)
    return found


def shift_middle_1(value, count, use_left):
    return shift_apply(value, count, use_left)


def shift_middle_2(value, count, use_left):
    return shift_middle_1(value, count, use_left)


def shift_three(rounds, use_left):
    found = None
    for i in range(rounds):
        try:
            shift_middle_2(BIG + i, -1, use_left)
        except ValueError as exc:
            found = signature(exc, ValueError)
    return found


def shift_flip(use_left):
    raised = completed = checksum = 0
    for i in range(ROUNDS):
        count = -1 if i < 2000 or i >= 4000 else 2
        try:
            checksum += shift_apply(BIG + i, count, use_left)
            completed += 1
        except ValueError:
            raised += 1
    return raised, completed, checksum


for shift_use_left in (True, False):
    for shift_count in (-1, -sys.maxsize - 1):
        check_signature(
            shift_same(1, shift_use_left, shift_count),
            shift_same(ROUNDS, shift_use_left, shift_count),
            SHIFT_MESSAGE,
            ("shift_same",),
        )
    check_signature(
        shift_one(1, shift_use_left),
        shift_one(ROUNDS, shift_use_left),
        SHIFT_MESSAGE,
        ("shift_one", "shift_apply"),
    )
    check_signature(
        shift_three(1, shift_use_left),
        shift_three(ROUNDS, shift_use_left),
        SHIFT_MESSAGE,
        ("shift_three", "shift_middle_2", "shift_middle_1", "shift_apply"),
    )
    assert shift_flip(shift_use_left)[:2] == (4000, 2000)


# Exact-float true division by zero.
FLOAT_MESSAGE = "float division by zero" if sys.implementation.name == "pypy" else "division by zero"


def float_leaf(value, divisor):
    return value / divisor


def float_same(rounds, divisor):
    found = None
    for i in range(rounds):
        try:
            1.5 * i / divisor
        except ZeroDivisionError as exc:
            found = signature(exc, ZeroDivisionError)
    return found


def float_one(rounds):
    found = None
    for i in range(rounds):
        try:
            float_leaf(i + 0.5, 0.0)
        except ZeroDivisionError as exc:
            found = signature(exc, ZeroDivisionError)
    return found


def float_middle_1(value, divisor):
    return float_leaf(value, divisor)


def float_middle_2(value, divisor):
    return float_middle_1(value, divisor)


def float_three(rounds):
    found = None
    for i in range(rounds):
        try:
            float_middle_2(i + 0.5, 0.0)
        except ZeroDivisionError as exc:
            found = signature(exc, ZeroDivisionError)
    return found


def float_flip():
    raised = completed = 0
    total = 0.0
    for i in range(ROUNDS):
        divisor = 0.0 if i < 2000 or i >= 4000 else 1.5
        try:
            total += (i + 0.5) / divisor
            completed += 1
        except ZeroDivisionError:
            raised += 1
    return raised, completed, round(total, 4)


for float_zero in (0.0, -0.0):
    check_signature(
        float_same(1, float_zero),
        float_same(ROUNDS, float_zero),
        FLOAT_MESSAGE,
        ("float_same",),
    )
check_signature(
    float_one(1), float_one(ROUNDS), FLOAT_MESSAGE, ("float_one", "float_leaf")
)
check_signature(
    float_three(1),
    float_three(ROUNDS),
    FLOAT_MESSAGE,
    ("float_three", "float_middle_2", "float_middle_1", "float_leaf"),
)
assert float_flip()[:2] == (4000, 2000)


# Exact range construction with a zero step.
RANGE_MESSAGE = (
    "step argument must not be zero"
    if sys.implementation.name in ("pyre", "pypy")
    else "range() arg 3 must not be zero"
)


def range_leaf(stop, step):
    return range(0, stop, step)


def range_same(rounds, stop):
    found = None
    for _ in range(rounds):
        try:
            range(0, stop, 0)
        except ValueError as exc:
            found = signature(exc, ValueError)
    return found


def range_one(rounds):
    found = None
    for _ in range(rounds):
        try:
            range_leaf(3, 0)
        except ValueError as exc:
            found = signature(exc, ValueError)
    return found


def range_middle_1(stop, step):
    return range_leaf(stop, step)


def range_middle_2(stop, step):
    return range_middle_1(stop, step)


def range_three(rounds):
    found = None
    for _ in range(rounds):
        try:
            range_middle_2(3, 0)
        except ValueError as exc:
            found = signature(exc, ValueError)
    return found


def range_flip():
    raised = completed = checksum = 0
    for i in range(ROUNDS):
        step = 0 if i < 2000 or i >= 4000 else 2
        try:
            checksum += len(range(0, 9, step))
            completed += 1
        except ValueError:
            raised += 1
    return raised, completed, checksum


for range_stop in (3, 0):
    check_signature(
        range_same(1, range_stop),
        range_same(ROUNDS, range_stop),
        RANGE_MESSAGE,
        ("range_same",),
    )
check_signature(
    range_one(1), range_one(ROUNDS), RANGE_MESSAGE, ("range_one", "range_leaf")
)
check_signature(
    range_three(1),
    range_three(ROUNDS),
    RANGE_MESSAGE,
    ("range_three", "range_middle_2", "range_middle_1", "range_leaf"),
)
assert range_flip() == (4000, 2000, 10000)

assert list(range(3, -4, -2)) == [3, 1, -1, -3]
assert list(range(sys.maxsize, sys.maxsize - 3, -1)) == [
    sys.maxsize,
    sys.maxsize - 1,
    sys.maxsize - 2,
]
assert list(range(-sys.maxsize - 1, sys.maxsize, sys.maxsize)) == [
    -sys.maxsize - 1,
    -1,
    sys.maxsize - 1,
]


# Mixed fit-then-overflow `int ** int`: a trace compiled on an i64-fitting
# exponent must promote through long (`descr_pow` except OverflowError /
# `pow_ovf2long`), not raise `OverflowError: integer overflow` at `**`.
def two_fit_then_ovf(rounds):
    n = 0
    for i in range(rounds):
        d = 20 + (i % 80)
        n += (2 ** d).bit_length()
    return n


def ten_fit_then_ovf(rounds):
    n = 0
    for i in range(rounds):
        e = 1 + (i % 60)
        n += (10 ** e).bit_length()
    return n


assert two_fit_then_ovf(1) == (2 ** 20).bit_length()
assert two_fit_then_ovf(2000) == 121000
assert ten_fit_then_ovf(2000) == 202304


# Warmup compiles `2**d` / `2**-d` on |d| < 63 inside a callee of
# `try/except OverflowError`, then |d| >= 63 must still return a long
# (`descr_pow` except OverflowError / `pow_ovf2long`). `two_fit_then_ovf`
# mixes both magnitudes in one loop so the ovf2long bridge is compiled
# while tracing; `check_truediv` warms up on tiny |d| then hits overflow
# under that outer except.
def power_pos(d):
    return 2 ** d


def power_neg_shape(d):
    return 2 ** -d


def guarded_power(fn, d):
    try:
        return fn(d)
    except OverflowError:
        return "overflow"


def compile_small_then_ovf(fn, small_values, ovf_ds, rounds):
    for i in range(rounds):
        got = guarded_power(fn, small_values[i % len(small_values)])
        assert got != "overflow", (fn, small_values[i % len(small_values)], got)
    return [guarded_power(fn, d) for d in ovf_ds]


assert compile_small_then_ovf(power_pos, tuple(range(20)), (63, 80), ROUNDS) == [
    1 << 63,
    1 << 80,
]
assert compile_small_then_ovf(
    power_neg_shape, tuple(range(-20, 0)), (-63, -80), ROUNDS
) == [
    1 << 63,
    1 << 80,
]

print("OK")
