# pyre-check: spec-folds=subscr
# pyre-check: trace-shape=hot_index:absent=CallMayForceR
# Hot-loop `s[i]` on an exact `str`. The subscript reaches the walker as the
# `BinaryOp` helper's `Subscr` tag; an exact `str` receiver with an exact
# `int` index descends `baseobjspace::getitem_str` (`descr_getitem`'s scalar
# arm) instead of the generic `CallMayForce`, which forces virtualizables and
# clears the heap cache across itself.
#
# NO throughput ceiling, deliberately. The remaining cost is the boxed code
# point itself, which this loop discards immediately and pypy never
# allocates at all -- `len(s[0])` and `ord(s[0])` cost the same, i.e. the
# cost does not depend on the consumer. Virtualizing that allocation is a
# different lever, so no ratio here would be a stable gate. The `spec-folds`
# line above is the gate; a ratio would only measure the allocation.
#
# The other legs are correctness legs for the shapes the descent must REFUSE,
# each written so a wrongly-admitted shape is a wrong value and not a silent
# pass:
#
#   * a `str` SUBCLASS may override `__getitem__`, which `baseobjspace::getitem`
#     honours; admitting one through the payload `ob_type` it shares would drop
#     the prefix its override adds.
#   * a NON-ASCII receiver indexes code points, not bytes; a fixed-stride read
#     would return a fragment of a multi-byte sequence.
#   * a NEGATIVE index counts from the end, and an out-of-range one raises
#     `IndexError`; the descended body keeps its own length test, so the
#     trace must guard it rather than bake this receiver's length in.
#   * an index past `usize::MAX` is out of range on the 32-bit wasm guest for
#     a reason the other backends never see: the machine int is 64-bit
#     everywhere, so `2**32` must still raise rather than wrap to `0`.
#   * a `bool` index shares `int`'s `intval` but carries its own type, and a
#     `__index__` object is not an int at all.
class Prefixed(str):
    def __getitem__(self, index):
        return "!" + str.__getitem__(self, index)


class Ix:
    def __index__(self):
        return 2


def hot_index(n, s):
    acc = 0
    for _ in range(n):
        for i in range(5):
            acc = (acc * 31 + ord(s[i])) & 0xFFFFFFFFFF
    return acc


def declined_shapes(n, plain, wide, sub):
    acc = 0
    for _ in range(n):
        acc = (acc * 31 + ord(wide[0]) + ord(wide[3])) & 0xFFFFFFFFFF
        acc = (acc * 31 + len(sub[0])) & 0xFFFFFFFFFF
        acc = (acc * 31 + ord(plain[-1]) + ord(plain[True])) & 0xFFFFFFFFFF
        acc = (acc * 31 + ord(plain[Ix()])) & 0xFFFFFFFFFF
        try:
            plain[99]
        except IndexError:
            acc = (acc * 31 + 7) & 0xFFFFFFFFFF
        try:
            plain[4294967296]
        except IndexError:
            acc = (acc * 31 + 11) & 0xFFFFFFFFFF
    return acc


PLAIN = "abcde"
WIDE = "aé中𝄞x"
SUB = Prefixed("abcde")

print(hot_index(300000, PLAIN))
print(declined_shapes(20000, PLAIN, WIDE, SUB))
# Code points, not the characters: check.py drops the locale chain from the
# child environment, so a Windows runner resolves its piped stdout to the
# ANSI codepage and printing U+4E2D there raises instead of answering.
print(ord(WIDE[1]), ord(WIDE[2]), ord(WIDE[3]), SUB[0], PLAIN[-2])
