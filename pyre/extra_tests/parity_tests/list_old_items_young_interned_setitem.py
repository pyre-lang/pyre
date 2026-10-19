# CPython-suite gap: test_collections TestNamedTuple.test_large_size stores
# interned field names into an Object-strategy items block that a minor has
# already promoted; the suite only checks namedtuple construction, not that
# those slots stay live.
# parity-tests reason: list_write_barrier_impl Phase L2 barriers the items
# GcArray; rewrite.py handle_write_barrier_setarrayitem / gen_write_barrier
# on that array. The JIT rewrite of prepare_list_ref_store used to emit
# CondCallGcWb on the list only, and list_object_custom_trace does not
# re-scan an already-old block.
# parity-env: PYPY_GC_NURSERY=4096

"""Young interned strs stored into an old 4-slot list survive minors."""

import gc
import sys

try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass

N = 4000


def promote():
    junk = [[i] for i in range(400)]
    del junk
    gc.collect()


def run():
    xs = [None, None, None, None]
    promote()
    for i in range(N):
        xs[i % 4] = sys.intern("k%d-%d" % (i, i))
        if i % 32 == 0:
            promote()
    for i in range(4):
        last = N - 4 + i
        expected = "k%d-%d" % (last, last)
        s = xs[i]
        if s != expected:
            raise AssertionError((i, s, expected))
        if sys.intern(expected) is not s:
            raise AssertionError(("identity", i, s, expected))
    return xs[0]


print(run())
print("OK")
