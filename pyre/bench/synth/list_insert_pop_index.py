# pyre-check: max-pypy-ratio=48
# list.insert(index, x) and list.pop(index) coerce the index through
# `__index__` (`getindex_w(index, OverflowError)`): a small custom index
# inserts/pops at the resolved position, an out-of-range pop reports "pop index
# out of range", and a non-index key surfaces the "cannot be interpreted as an
# integer" TypeError.  A plain-int warmup loop exercises the working fast paths
# first.  Only cpython==pypy outputs are asserted (an overflowing index raises
# OverflowError with text that diverges between the oracles).  Deterministic.
class Idx:
    def __init__(self, v):
        self.v = v

    def __index__(self):
        return self.v


def warm(n):
    acc = 0
    data = []
    for i in range(n):
        data.insert(0, i)
        if len(data) > 4:
            acc += data.pop()
    return acc


def m(label, fn):
    try:
        print(label, "->", repr(fn()))
    except BaseException as e:
        print(label, "!!", type(e).__name__, repr(str(e)))


def ins(i):
    l = [0, 1, 2]
    l.insert(i, 9)
    return l


def pop(i):
    l = [10, 11, 12]
    return (l.pop(i), l)


def main():
    print("warm", warm(1875000))
    m("ins_idx", lambda: ins(Idx(1)))
    m("ins_neg", lambda: ins(Idx(-1)))
    m("ins_float", lambda: ins(1.5))
    m("pop_idx", lambda: pop(Idx(1)))
    m("pop_neg", lambda: pop(Idx(-1)))
    m("pop_oob", lambda: pop(Idx(9)))
    m("pop_float", lambda: pop(1.5))


main()
