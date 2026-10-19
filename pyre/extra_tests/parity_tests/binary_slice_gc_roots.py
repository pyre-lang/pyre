# CPython-suite gap: the vendored suite slices sequences with plain integer
# bounds, so no test in it runs Python between the receiver's type test and the
# fetch that reads its payload.
# parity-tests reason: this is a pyre/PyPy moving-GC root-liveness regression.

"""``seq[a:b]`` must survive a bound ``__index__`` that collects.

``BINARY_SLICE`` converts each bound through ``__index__`` -- arbitrary
Python, so a collection point -- and only then fetches out of the receiver.
The ``list`` / ``str`` / ``tuple`` arms each read the receiver again after
that conversion, and the arity-2 specialised tuples box their payload on
every fetch, so the receiver moves inside the fetch loop as well.
"""

import gc


def churn():
    garbage = [[index, index + 1] for index in range(4000)]
    assert len(garbage) == 4000
    gc.collect()


class Bound:
    """An ``__index__`` that collects, so every bound conversion moves the receiver."""

    def __init__(self, value):
        self.value = value

    def __index__(self):
        churn()
        return self.value


for step in range(5):
    # Array-backed tuple, freshly built so it is young enough to be moved by
    # the collection the bounds run.  Reading item 2 out of a moved tuple is
    # what classified it as an arity-2 specialisation and crashed.
    wide = (step, step + 1, step + 2, step + 3, step + 4)
    assert wide[Bound(0):Bound(5)] == wide
    assert wide[Bound(1):Bound(4)] == (step + 1, step + 2, step + 3)
    assert wide[Bound(-2):Bound(5)] == (step + 3, step + 4)

    # Arity two with int slots is the specialised representation, whose fetch
    # allocates a box per item and so collects inside the loop itself.
    pair = (step, step + 1)
    assert pair[Bound(0):Bound(2)] == pair
    floats = (float(step), float(step) + 0.5)
    assert floats[Bound(0):Bound(2)] == floats
    objs = ([step], [step + 1])
    assert objs[Bound(0):Bound(2)] == objs

    # The list arm resolves its length after both bounds convert, and the str
    # arm derives a view into the receiver's payload.
    items = [step, step + 1, step + 2, step + 3]
    assert items[Bound(1):Bound(3)] == [step + 1, step + 2]
    text = "%03d-abcde" % step
    assert text[Bound(0):Bound(3)] == "%03d" % step
    assert text[Bound(4):Bound(9)] == "abcde"

    # A bound that is None skips its conversion, so only the other one moves
    # the receiver.
    assert wide[Bound(2):] == (step + 2, step + 3, step + 4)
    assert wide[:Bound(2)] == (step, step + 1)

# Integer-strategy list slices of many lengths, including above the nursery
# large-object threshold, while allocating garbage so a minor collection can
# walk the adopted int-items block.  Every slice's contents must match.
sizes = list(range(0, 64)) + [100, 200, 500, 1000, 2000, 4000, 5000]
for n in sizes:
    src = list(range(n))
    for _ in range(8):
        churn()
        whole = src[:]
        assert whole == src
        if n >= 2:
            inner = src[1 : n - 1]
            assert inner == list(range(1, n - 1))
        empty = src[n:n]
        assert empty == []

print("OK")
