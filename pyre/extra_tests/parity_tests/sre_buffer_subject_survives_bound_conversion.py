# CPython-suite gap: re's tests pass plain ints for `pos`/`endpos`, so nothing
# there runs user code between acquiring a buffer subject and matching on it.
# test.test_enum TestMinimalFloatFunction.test_overridden_str_format does
# ''.join(reversed(self.name)) on a young interned member name while intern
# WEAKDICT / WEAKREF mallocs collect.
# parity-tests reason: this is a pyre/PyPy moving-GC root-liveness regression.
# A buffer subject (a `memoryview`, an `array.array`) is gathered into a fresh
# `bytes` that the subject slice points into and that a match or a scanner then
# stamps into its own fields; converting `pos`/`endpos` runs `__index__`
# between the two, and the gathered object has no other owner.
# unicodeobject.py _str_join_many_items keeps self._utf8 (Ptr(STR) getfield)
# as a livevar; framework.py push_roots / shadowstack.py expand_pop_roots
# reloads the str object and recomputes chars after any collecting call.
# parity-env: PYPY_GC_NURSERY=4096

"""A gathered buffer subject stays live across the pos/endpos conversion."""

import enum
import gc
import re
import sys


class Index:
    """A `pos`/`endpos` that allocates hard on its way to an int."""

    def __init__(self, value):
        self.value = value

    def __index__(self):
        churn = [bytes(64) for _ in range(2000)]
        gc.collect()
        del churn
        return self.value


def churn():
    junk = [bytes(128) for _ in range(4000)]
    gc.collect()
    del junk


DATA = bytearray(b"ab12cd34ef56gh78" * 400)
DIGITS = re.compile(rb"\d+")
QUAD = [b"12", b"34", b"56", b"78"]


# search / match / fullmatch all reach the same subject acquisition.
found = DIGITS.search(memoryview(DATA), Index(0), Index(len(DATA)))
assert found is not None and found.group() == b"12", found

anchored = DIGITS.match(memoryview(DATA), Index(2), Index(len(DATA)))
assert anchored is not None and anchored.group() == b"12", anchored

whole = re.compile(rb"\d+").fullmatch(memoryview(bytearray(b"9081")), Index(0))
assert whole is not None and whole.group() == b"9081", whole

# A match built over the gathered buffer is read back after further churn: its
# spans index storage the match itself is the only owner of.
churn()
assert found.group() == b"12" and anchored.span() == (2, 4), (found, anchored)

# The scanner keeps the same pair for as long as it yields.
scanner = DIGITS.finditer(memoryview(DATA), Index(0), Index(64))
first = next(scanner)
churn()
rest = list(scanner)
assert [first.group()] + [m.group() for m in rest] == QUAD * 4, (first, rest)

# The three siblings that already pinned the buffer, as a control.
assert DIGITS.findall(memoryview(DATA), Index(0), Index(16)) == QUAD
assert DIGITS.split(memoryview(bytearray(b"a1b22c")), Index(0)) == [b"a", b"b", b"c"]
assert DIGITS.sub(b"#", bytearray(b"a1b22c"), Index(0)) == b"a#b#c"


def churn_join_reversed():
    junk = []
    for i in range(40):
        junk.append([object() for _ in range(32)])
        if i % 4 == 0:
            gc.collect()
    return junk


def join_reversed_of_a_young_interned_name_survives_minors():
    for i in range(80):
        name = sys.intern("first%d" % (i % 7))
        churn_join_reversed()
        got = "".join(reversed(name))
        expect = "".join(reversed(list(name)))
        if got != expect:
            raise AssertionError((i, name, got, expect))

    class NewStrFormatEnum(enum.Enum):
        def __format__(self, spec):
            return "".join(reversed(self.name))

        first = enum.auto()

    for i in range(40):
        churn_join_reversed()
        got = format(NewStrFormatEnum.first)
        if got != "tsrif":
            raise AssertionError(("format", i, got))
        got = "{}".format(NewStrFormatEnum.first)
        if got != "tsrif":
            raise AssertionError(("str.format", i, got))


join_reversed_of_a_young_interned_name_survives_minors()
print("OK")
