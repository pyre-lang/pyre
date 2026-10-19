# CPython-suite gap: test.test_collections intermittently crashes in
# minor_old_objects_pointing_to_young: an old type-9 varsize ref array
# (frame locals_cells_stack_w) holds nursery addresses a later JITFRAME
# covers. test_collections JIT-traces namedtuple / ChainMap helpers whose
# snapshot_for_tracing copies those locals.
# parity-tests reason: gctransform/framework.py push_roots / shadowstack.py
# pop_roots around the dest malloc; a native Vec taken before
# alloc_frame_locals_array is not a livevar. setarrayitem_gc /
# incminimark.py write_barrier then stores the reloaded words.
# Fresh object() stays young at the copy; intern + a 4096-byte nursery
# promotes first.
# parity-env: PYPY_GC_NURSERY=4096

"""JIT-traced frame locals that hold young objects survive minor collections."""

import gc
from collections import ChainMap, namedtuple

try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass


def churn():
    junk = [[object() for _ in range(16)] for _ in range(24)]
    del junk
    gc.collect()


def four_locals(a, b, c, d):
    return a, b, c, d


N = 600
kept = []
hold = []
for i in range(N):
    row = four_locals(object(), object(), object(), object())
    kept.append(row)
    if i % 8 == 0:
        churn()
    if i % 11 == 0:
        T = namedtuple("Nt%d" % (i % 5), ["a", "b", "c", "d"])
        hold.append(T(i, i, i, i))
    if i % 13 == 0:
        cm = ChainMap({"a": i}, {"b": i}, {"c": i}, {"d": i})
        cm.maps[1] = {object(): i}
        hold.append(cm)

for i in range(N):
    got = kept[i]
    if any(item is None for item in got):
        raise AssertionError(("dead", i, got))
    for item in got:
        if type(item) is not object:
            raise AssertionError(("type", i, item))

print(type(kept[-1][0]).__name__, len(hold))
print("OK")
