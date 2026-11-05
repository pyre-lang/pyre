# pyre-check: gate=1
# pyre-check: pypy-diverges: pins 3.14 empty range_iterator remaining 0; pypy3 keeps the raw stop, so iter(range(-1)).__length_hint__() is -1 and operator.length_hint raises ValueError
# CPython-suite gap: test_range does not pin length_hint/reduce/setstate of empty one-arg and two-arg range iterators.
import operator

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

print("OK")
