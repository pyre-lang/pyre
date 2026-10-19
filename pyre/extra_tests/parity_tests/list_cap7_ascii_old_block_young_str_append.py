# CPython-suite gap: test_collections namedtuple / ABC mixin lists are
# AsciiListStrategy, grown to rlist_overallocate(4)=7 (BUILD_LIST 3 +
# append), then JIT-compile ascii append of young unique strs into that
# already-old block. prepare_list_ref_store / list_write_barrier residual
# must keep Phase L2 (ascii_items.block) rather than CondCallGcWb on the
# list header alone: list_object_custom_trace forwards the block pointer
# and does not re-scan an already-old one.
# parity-tests reason: rlist.py ll_append is resize-ge then ll_setitem_fast;
# listobject.py AbstractUnwrappedStrategy.append unwraps after the resize.
# list_write_barrier_impl Phase L2 is incminimark.py write_barrier on the
# current strategy's GcArray. assembler.py _write_barrier_fastpath TESTs
# with no null check, so empty ascii cannot fold to Getfield+CondCallGcWb.
# parity-env: PYPY_GC_NURSERY=4096

"""Young ascii strs appended into an old cap-7 ascii list survive minors."""

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


def add_young(xs, i):
    s = "n%d-%d" % (i, i)
    xs.append(s)
    return s


def run():
    # BUILD_LIST 3 + append → ascii_grow(4) → rlist_overallocate(4)=7.
    xs = ["a", "b", "c"]
    xs.append("d")
    promote()
    kept = xs[0]
    for i in range(N):
        s = add_young(xs, i)
        interned = sys.intern(s)
        if interned is not xs[-1]:
            raise AssertionError(("intern", i, interned, xs[-1]))
        if i % 32 == 0:
            promote()
        popped = xs.pop()
        if popped is not s:
            raise AssertionError(("pop", i, popped, s))
    if len(xs) != 4:
        raise AssertionError(("len", len(xs), xs))
    return kept


print(run())
print("OK")
