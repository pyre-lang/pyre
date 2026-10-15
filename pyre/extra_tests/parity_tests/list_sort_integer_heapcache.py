# CPython-suite gap: test.test_heapq TestHeapPython.test_naive_nbest /
# test_naive_nworst.  After the pure-Python heapq loop, `heap.sort()` leaves
# the list unsorted from the assertion's view (same multiset, heap layout).
# parity-tests reason: residual IntegerListStrategy.sort must record the
# GcArray<i64> write so a compiled list.__eq__ does not reuse pre-sort
# getarrayitem values.

"""Integer-strategy list.sort must invalidate cached getarrayitem.

Residual `IntegerListStrategy.sort` / `IntSort` used to TimSort the `GcArray`
storage through a raw view.  Raw stores are not in the call's write effects
(`WriteAnalyzer.analyze_simple_operation`), so the heap cache kept pre-sort
`getarrayitem` values across the call.  `TimSort` now writes `self.list[i] = x`
on the storage.
"""

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


run_seed(2)
run_seed(3)
print("OK")
