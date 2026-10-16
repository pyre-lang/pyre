# CPython-suite gap: test_collections stores young objects into an
# Object-strategy items block grown to rlist_overallocate(4)=7
# (BUILD_LIST 3 + append; AbstractUnwrappedStrategy.setslice grow)
# after a minor has already promoted it. The suite does not check
# those slots stay live.
# parity-tests reason: rlist.py ll_setitem_fast / listobject.py
# AbstractUnwrappedStrategy.setitem is setarrayitem_gc; rewrite.py
# handle_write_barrier_setarrayitem emits COND_CALL_GC_WB on the items
# GcArray. setslice is items += [none]*delta then ll_setitem_fast
# (object_splice / writebarrier_before_move). Fresh object() stays
# young at the store; intern + a 4096-byte nursery promotes first.
# parity-env: PYPY_GC_NURSERY=4096

"""Young objects stored into an old cap-7 list survive minors."""

import gc
from collections import UserList

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
    # BUILD_LIST 3 + append → object_grow(4) → rlist_overallocate(4)=7.
    ys = [None, None, None]
    ys.append(None)
    # UserList of 3 then slice-assign 4 is object_splice grow to cap 7.
    xs = UserList([None, None, None])
    promote()
    kept_ys = None
    kept_xs = None
    for i in range(N):
        a, b, c, d = object(), object(), object(), object()
        ys[0] = a
        ys[1] = b
        ys[2] = c
        ys[3] = d
        xs[:] = [a, b, c, d]
        kept_ys = (a, b, c, d)
        kept_xs = (xs[0], xs[1], xs[2], xs[3])
        if i % 32 == 0:
            promote()
    for i in range(4):
        if ys[i] is not kept_ys[i]:
            raise AssertionError(("ys", i, ys[i], kept_ys[i]))
        if xs[i] is not kept_xs[i]:
            raise AssertionError(("xs", i, xs[i], kept_xs[i]))
    return ys[0]


print(run())
print("OK")
