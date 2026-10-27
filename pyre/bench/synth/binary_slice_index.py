# pyre-check: max-pypy-ratio=370
# pyre-check: trace-shape=main:max=CallMayForceR:6
# Slice bounds are evaluated through __index__ in both lowering paths:
#   * a dynamic slice `seq[a:b]` compiles to BINARY_SLICE, handled by
#     `binary_slice_values`;
#   * an all-constant slice `seq[1.0:4]` is folded to a `slice` constant and
#     compiled to BINARY_SUBSCR (`seq[slice]`), handled by `normalize_slice`.
# Both must run __index__ on each bound, so a custom __index__ and a bool work
# while a float (which has no __index__) raises TypeError.  Only the exception
# type is printed so the line matches across CPython/PyPy/Pyre.
N = 50000


class Idx:
    def __init__(self, v):
        self.v = v

    def __index__(self):
        return self.v


class BadIdx:
    def __index__(self):
        return "not-an-int"


def show(label, fn):
    try:
        print(label, fn())
    except Exception as e:
        print(label, type(e).__name__)


def main():
    seq = "abcdefghij"
    lst = list(range(10))
    tup = tuple(range(10))

    # Hot loop 1 (BINARY_SLICE / binary_slice_values): the JIT-compiled residual
    # must run __index__ on the instance bounds every iteration.  A fast path
    # that read the raw int field of the bound object would mis-slice or crash.
    acc = 0
    i = 0
    while i < N:
        a = Idx(i % 5)
        b = Idx(i % 5 + 3)
        acc = acc + len(seq[a:b]) + len(lst[a:b]) + len(tup[a:b])
        i = i + 1
    print("acc", acc)

    # Hot loop 2 (BINARY_SUBSCR / normalize_slice): a constant-bound slice folds
    # to a `slice` constant, so the residual goes through normalize_slice.  Valid
    # integer constant bounds must keep working under the JIT.
    acc2 = 0
    j = 0
    while j < N:
        acc2 = acc2 + len(seq[1:4]) + sum(lst[2:5]) + len(tup[0:3])
        j = j + 1
    print("acc2", acc2)

    # __index__ bounds resolve to the same slice as plain ints (BINARY_SLICE).
    show("str_idx", lambda: seq[Idx(2):Idx(6)])
    show("list_idx", lambda: sum(lst[Idx(2):Idx(6)]))
    show("tuple_idx", lambda: sum(tup[Idx(2):Idx(6)]))

    # bool is an int subtype; True/False are valid bounds.
    show("bool_bound", lambda: seq[True:4])

    # An __index__ that returns a non-int -> TypeError.
    show("badidx_bound", lambda: seq[BadIdx():4])

    # None bounds still default to 0 / len.
    show("none_start", lambda: seq[:4])
    show("none_stop", lambda: sum(lst[6:]))

    # A float bound has no __index__ -> TypeError, on both lowering paths.
    # Constant-folded slice (BINARY_SUBSCR / normalize_slice):
    show("str_const_float", lambda: seq[1.0:4])
    show("list_const_float", lambda: lst[1.0:4])
    show("tuple_const_float", lambda: tup[1.0:4])
    show("bytes_const_float", lambda: b"abcdefghij"[1.0:4])
    show("const_float_step", lambda: seq[::1.0])
    show("const_float_stop", lambda: seq[:1.0])

    # Dynamic slice (BINARY_SLICE / binary_slice_values):
    fv = 1.0
    nn = 4
    show("dyn_float_start", lambda: seq[fv:nn])
    show("dyn_float_stop", lambda: seq[1:fv])

    # Code-point-boundary slicing over a lone surrogate, with __index__ bounds.
    surr = "a\ud800b"
    show("surrogate_len", lambda: len(surr[Idx(1):Idx(3)]))
    show("surrogate_ord", lambda: ord(surr[Idx(1):Idx(2)]))


main()
