# `range()` with a bound whose type supplies `__index__`.
# `functional.py:461-474 W_Range.descr_new` converts each bound with
# `space.index`, and `descroperation.py _index` carries no JIT hints,
# so the user method is traced into like any other call. The bound is live
# rather than trace-constant, which is what puts the emitted
# `compute_range_length` (functional.py:42-53) on the recorded path: the
# step-sign guard, the emptiness guard and the overflow guards all come from
# the source's own conditionals.
N = 1200000


class Index:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


def main():
    total = 0
    stop = Index(0)
    step = Index(2)
    for i in range(N):
        stop.value = i % 7
        total += len(range(stop))
        total += len(range(0, stop, step))
    print(total)


main()
# Expected: 5657134
