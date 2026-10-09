# CPython-suite gap: test.test_enum TestMinimalFloatFunction.test_overridden_str_format
# does ''.join(reversed(self.name)) on a young interned member name while
# intern WEAKDICT / WEAKREF mallocs collect.
# parity-tests reason: unicodeobject.py _str_join_many_items keeps self._utf8
# (Ptr(STR) getfield) as a livevar; framework.py push_roots / shadowstack.py
# expand_pop_roots reloads the str object and recomputes chars after any
# collecting call. A derived &Wtf8 / interior of W_UnicodeObject must not
# remain a GC root across sequence_fast or newutf8.
# parity-env: PYPY_GC_NURSERY=4096

"""join(reversed()) of a young interned name survives minor collections."""

import enum
import gc
import sys


def churn():
    junk = []
    for i in range(40):
        junk.append([object() for _ in range(32)])
        if i % 4 == 0:
            gc.collect()
    return junk


for i in range(80):
    name = sys.intern("first%d" % (i % 7))
    churn()
    got = "".join(reversed(name))
    expect = "".join(reversed(list(name)))
    if got != expect:
        raise AssertionError((i, name, got, expect))


class NewStrFormatEnum(enum.Enum):
    def __format__(self, spec):
        return "".join(reversed(self.name))

    first = enum.auto()


for i in range(40):
    churn()
    got = format(NewStrFormatEnum.first)
    if got != "tsrif":
        raise AssertionError(("format", i, got))
    got = "{}".format(NewStrFormatEnum.first)
    if got != "tsrif":
        raise AssertionError(("str.format", i, got))

print("OK")
