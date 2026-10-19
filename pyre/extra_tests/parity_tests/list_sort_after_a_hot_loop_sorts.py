# CPython-suite gap: test.test_heapq TestHeapPython.test_naive_nbest /
# test_naive_nworst.  After the pure-Python heapq loop, `heap.sort()` leaves
# the list unsorted from the assertion's view (same multiset, heap layout).
#
# parity-tests reason: two ways a traced `list.sort()` can be lost.
#
# 1. Residual `IntegerListStrategy.sort` / `IntSort` used to TimSort the
#    `GcArray` storage through a raw view.  Raw stores are not in the call's
#    write effects (`WriteAnalyzer.analyze_simple_operation`), so the heap
#    cache kept pre-sort `getarrayitem` values across the call and a compiled
#    `list.__eq__` reused them.  `TimSort` now writes `self.list[i] = x` on the
#    storage.  The suite runs on fixed seeds below reach that shape.
#
# 2. `lst.sort()` with no `key=` / `reverse=` reaches `sort_list_in_place_obj`
#    with both keyword slots holding the gateway's NULL holes.  A trace
#    recording that call must still execute it, as `pyjitpl.py`
#    `do_residual_call` executes every residual call it records.  In `t` the
#    recording bridge (its loop exit) then aborts inside a nested call, and the
#    blackhole resumes forward from the walk's state; a call that was recorded
#    but not executed leaves the list unsorted.  `test_heapq` reaches this only
#    on some random data, so `t` builds it deterministically.

import io
import os
import random
import unittest

from test import test_heapq as T


def run_seed(seed):
    random.seed(seed)
    names = ["TestErrorHandlingC", "TestHeapC", "TestHeapPython"]
    suite = unittest.TestSuite(
        unittest.defaultTestLoader.loadTestsFromTestCase(getattr(T, n))
        for n in names
    )
    result = unittest.TextTestRunner(
        verbosity=0, stream=open(os.devnull, "w")
    ).run(suite)
    if not result.wasSuccessful():
        failed = [t.id() for t, _ in result.failures]
        errored = [t.id() for t, _ in result.errors]
        raise AssertionError("seed %s fail %s err %s" % (seed, failed, errored))


def inner():
    s = io.StringIO()
    s.write("x")
    return s.getvalue()


def outer():
    return inner()


def t(data):
    heap = []
    for x in data:
        heap.append(x)
        if len(heap) > 5:
            heap.pop()
    heap.sort()
    outer()
    return heap


run_seed(2)
run_seed(3)

for k in range(300):
    h = t([(j * 37 + k) % 11 for j in range(40)])
    assert h == sorted(h), (k, h)

print("OK")
