# pyre-check: max-pypy-ratio=50
# Float true-division by zero inside a try in a JIT-hot callee.  The walker
# specializes `float / float` to a bare `FloatTrueDiv` llop (raw IEEE), which
# would compute `inf` for a zero divisor instead of raising ZeroDivisionError.
# A `float_eq(rhs, 0.0) -> guard_false` precondition (the JIT form of
# floatobject.py:519 `_floatdiv`'s zero check) deopts a zero divisor to the
# checked descr_truediv path, which raises and is caught in-frame.
N = 3840000


def helper(i):
    try:
        return 1.5 / (i % 5)      # ZeroDivisionError every 5th iteration
    except ZeroDivisionError:
        return 3.25


def run():
    total = 0.0
    for i in range(N):
        r = helper(i)
        if r > 1.0:               # guard on the returned float
            total += r
        else:
            total -= r
    return total


print(round(run(), 4))
