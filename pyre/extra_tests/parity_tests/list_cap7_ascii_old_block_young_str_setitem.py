# CPython-suite gap: test_collections namedtuple field-name lists are
# AsciiListStrategy, grown to rlist_overallocate(4)=7 (BUILD_LIST 3 +
# append), then store young STR payloads into that block. Existing
# Object-strategy fixtures never enter the ascii items block.
# parity-tests reason: rlist.py ll_setitem_fast for AsciiListStrategy is
# setarrayitem_gc on the erased STR GcArray; rewrite.py
# handle_write_barrier_setarrayitem / gen_write_barrier_array emit
# COND_CALL_GC_WB on that array. list_write_barrier_impl Phase L2
# barriers ascii_items.block the same way it barriers Object items
# (incminimark.py write_barrier). The host store is
# gc_hook.try_gc_write_barrier_from_array (dont_look_inside) then the
# store, matching items_block_set_ref. Tracing the store while the
# block is still young, then promoting, is the TRACK_YOUNG_PTRS fold.
# parity-env: PYPY_GC_NURSERY=4096

"""Young ascii strs stored into a cap-7 ascii list survive minors."""

import gc

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


def store(xs, i):
    s = "n%d-%d" % (i, i)
    xs[i % 4] = s
    return s


def run():
    # BUILD_LIST 3 + append → ascii_grow(4) → rlist_overallocate(4)=7.
    xs = ["a", "b", "c"]
    xs.append("d")
    kept = [None, None, None, None]
    for i in range(N):
        kept[i % 4] = store(xs, i)
        if i % 32 == 0:
            promote()
    for i in range(4):
        if xs[i] is not kept[i]:
            raise AssertionError((i, xs[i], kept[i]))
        if xs[i] != "n%d-%d" % (N - 4 + i, N - 4 + i):
            raise AssertionError(("text", i, xs[i]))
    return xs[0]


print(run())
print("OK")
