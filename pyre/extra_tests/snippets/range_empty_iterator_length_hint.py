# pyre-check: gate=1
# pyre-check: pypy-diverges: pins 3.14 empty range_iterator remaining 0; pypy3 keeps the raw stop, so iter(range(-1)).__length_hint__() is -1 and operator.length_hint raises ValueError
# CPython-suite gap: test_range does not pin length_hint/reduce/setstate of empty one-arg and two-arg range iterators.
import operator
import sys

it = iter(range(-1))
assert type(it).__name__ == "range_iterator"
assert operator.length_hint(it) == 0
assert it.__length_hint__() == 0
assert it.__reduce__() == (iter, (range(0, 0),), None)
it.__setstate__(5)
assert list(it) == []

it = iter(range(3, 1))
assert it.__length_hint__() == 0
assert it.__reduce__() == (iter, (range(3, 3),), None)
it.__setstate__(2)
assert list(it) == []

# `rangeobject.c range_iter` / `range_reverse` type names (3.14).
f15 = range(-9223372036854775807, 9223372036854775807, 9223372036854775807)
assert list(f15) == [-9223372036854775807, 0]
assert type(iter(f15)).__name__ == "longrange_iterator"

f3 = range(-(1 << 62), (1 << 63) - 1, 1 << 62)
assert type(reversed(f3)).__name__ == "range_iterator"
assert list(reversed(f3)) == [1 << 62, 0, -(1 << 62)]

overflow = range(sys.maxsize - 5, sys.maxsize, 10)
assert type(iter(overflow)).__name__ == "longrange_iterator"
assert list(overflow) == [sys.maxsize - 5]

rev_big_stop = range(0, 2**63, 2**62)
assert list(reversed(rev_big_stop)) == [2**62, 0]
assert type(reversed(rev_big_stop)).__name__ == "longrange_iterator"

rev_min = reversed(range(-(2**63), 0, 1))
assert next(rev_min) == -1

assert type(reversed(range(0))).__name__ == "range_iterator"
assert list(reversed(range(0))) == []

print("OK")
