# pyre-check: max-pypy-ratio=23
# An extended slice `seq[a:b:c]` (a step operand) compiles the bounds to
# BUILD_SLICE — the `newslice(start, stop, step)` HLOp — then BINARY_SUBSCR,
# a distinct lowering from the two-bound BINARY_SLICE path. A `newslice`
# residual (`bh_build_slice_fn` → `w_slice_new`) must be emitted so the hot
# body JIT-compiles; without the lowering the assembler aborts on the
# unmapped opname. Both a literal reverse `seq[::-1]` and variable bounds are
# exercised, over list/tuple/str/bytes. None bounds and an `__index__` step
# resolve like plain ints. Output is verified against CPython/PyPy.
# Trip count kept clear of the major-collection threshold check.py pins: at
# the previous 600000 this loop crossed it, and the eval-breaker bailout that
# follows re-enters through a bridge whose guard can then fail once more,
# which moves guard_failures for reasons outside this fixture. Crossing
# resumes around 0.3x of the old count; the gated counters are unchanged.
N = 300000


class Idx:
    def __init__(self, v):
        self.v = v

    def __index__(self):
        return self.v


def show(label, fn):
    try:
        print(label, fn())
    except Exception as e:
        print(label, type(e).__name__)


def main():
    seq = "abcdefghij"
    lst = list(range(10))
    tup = tuple(range(10))
    bts = b"abcdefghij"

    # Hot loop 1: literal reverse slice `[::-1]` (None/None/-1 bounds) over four
    # sequence types every iteration — the canonical BUILD_SLICE panic shape.
    acc = 0
    i = 0
    while i < N:
        acc = acc + len(seq[::-1]) + lst[::-1][0] + tup[::-1][0] + bts[::-1][0]
        i = i + 1
    print("acc", acc)

    # Hot loop 2: variable start/stop/step locals — cannot fold to a const
    # slice, so BUILD_SLICE runs on live operands every iteration.
    acc2 = 0
    j = 0
    a = 8
    b = 1
    c = -2
    while j < N:
        acc2 = acc2 + sum(lst[a:b:c])
        j = j + 1
    print("acc2", acc2)

    # Reverse of each type.
    show("str_rev", lambda: seq[::-1])
    show("list_rev", lambda: lst[::-1])
    show("tuple_rev", lambda: tup[::-1])
    show("bytes_rev", lambda: list(bts[::-1]))

    # Explicit reverse bounds.
    show("str_bounds_rev", lambda: seq[8:2:-1])
    show("list_bounds_rev", lambda: lst[8:2:-1])

    # Positive step.
    show("str_step2", lambda: seq[::2])
    show("list_step3", lambda: lst[1:9:3])

    # None start/stop with a step.
    show("none_bounds_step", lambda: lst[::3])

    # __index__ step resolves like a plain int.
    show("idx_step", lambda: lst[Idx(0):Idx(9):Idx(2)])

    # Zero step is rejected (ValueError) — the residual must propagate it.
    show("zero_step", lambda: lst[::0])


main()
