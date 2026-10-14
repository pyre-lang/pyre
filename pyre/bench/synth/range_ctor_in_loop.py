# pyre-check: spec-folds=builtin_range
# pyre-check: trace-shape=entry-bridge:main:absent=CallMayForceR
# Pins virtual range construction for one-, two-, and three-bound calls while
# retaining correct residual behavior for exceptional, subclass, index, and
# escaping-object shapes.
#
# The `try: range(0, 3, 0)` below puts an out-of-line handler after the
# trailing comprehension. A loop region that gates the back edge by spanning
# the gap between them picks up that comprehension's call-bearing `FOR_ITER`
# -- an opcode this loop never reaches -- and `main` then runs interpreted end
# to end. With the region built from the exception table instead, the while
# loop and the three `for` loops compile.  The fold census verifies the range
# construction itself without sizing the arithmetic loop to a timer floor.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 40000


try:
    class RangeSubclass(range):
        pass
except TypeError:
    RangeSubclass = None


class Index:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


def main():
    one = 0
    two = 0
    three = 0
    zero = 0
    subclass = 0
    indexed = 0
    escaped = []
    n = 5
    a = 2
    b = 7
    step = -2
    i = 0
    while i < N:
        for value in range(n):
            one += value
        for value in range(a, b):
            two += value
        for value in range(9, -1, step):
            three += value
        try:
            range(0, 3, 0)
        except ValueError:
            zero += 1
        if RangeSubclass is None:
            subclass += 3
        else:
            subclass += len(RangeSubclass(3))
        indexed += len(range(Index(4)))
        escaped.append(range(i, i + 3))
        i += 1
    print(one, two, three, zero, subclass, indexed)
    print(len(escaped))
    print(
        [
            (item.start, item.stop, item.step, len(item))
            for item in (escaped[0], escaped[10000], escaped[-1])
        ]
    )


main()
