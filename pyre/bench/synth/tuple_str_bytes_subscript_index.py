# pyre-check: max-pypy-ratio=15
# tuple, str, and bytes subscripts coerce a non-int, non-slice key through
# `__index__` (`getindex_w`): a small custom index reads the resolved position,
# an out-of-range index raises the type's "index out of range", and an index
# too large for a machine word raises IndexError naming the key's real type
# ("cannot fit '<type>' ..."). Plain-int warmup loops exercise the working fast
# paths first. Only cpython==pypy outputs are asserted (the "indices must be"
# TypeError text and the bytes out-of-range wording diverge between the
# oracles). Deterministic.
class Idx:
    def __init__(self, v):
        self.v = v

    def __index__(self):
        return self.v


class Big:
    def __index__(self):
        return 10 ** 40


def warm(n):
    acc = 0
    t = (1, 2, 3, 4, 5)
    s = "abcde"
    b = b"abcde"
    for i in range(n):
        j = i % 5
        acc += t[j] + (s[j] == "c") + b[j]
    return acc


def m(label, fn):
    try:
        print(label, "->", repr(fn()))
    except BaseException as e:
        print(label, "!!", type(e).__name__, repr(str(e)))


def main():
    print("warm", warm(5400000))
    t = (10, 20, 30)
    m("t_idx", lambda: t[Idx(1)])
    m("t_neg", lambda: t[Idx(-1)])
    m("t_oob", lambda: t[Idx(9)])
    m("t_big", lambda: t[Big()])
    s = "abcde"
    m("s_idx", lambda: s[Idx(1)])
    m("s_neg", lambda: s[Idx(-1)])
    m("s_oob", lambda: s[Idx(9)])
    m("s_big", lambda: s[Big()])
    b = b"abcde"
    m("b_idx", lambda: b[Idx(1)])
    m("b_neg", lambda: b[Idx(-1)])
    m("b_big", lambda: b[Big()])


main()
