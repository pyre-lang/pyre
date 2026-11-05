# `range()` with a bound whose type supplies `__index__`, plus the raise
# and force paths a portal `perform_call` from the RANGE helper must
# resume past (`MetaInterp.perform_call` / `capture_resumedata`).
#
# Same constructor shape as `range_ctor_user_index_bound.py`. Every 50th
# `__index__` raises `ValueError`, caught around `range(...)`. On another
# iteration `__index__` calls `sys._getframe(1)` and reads `f_lineno`.

import sys

N = 24000


class Index:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        calls[0] += 1
        n = calls[0]
        if n % 50 == 0:
            raise ValueError(n)
        if n % 50 == 25:
            # Read `f_lineno` so the caller is forced; count the event so
            # the printed line is the same on every implementation.
            sys._getframe(1).f_lineno
            forced[0] += 1
        return self.value


calls = [0]
forced = [0]


def main():
    total = 0
    caught = 0
    stop = Index(0)
    step = Index(2)
    for i in range(N):
        stop.value = i % 7
        try:
            total += len(range(stop))
            total += len(range(0, stop, step))
        except ValueError:
            caught += 1
    print(total, caught, forced[0], calls[0])


main()
# Expected: 110715 1411 1412 70589
