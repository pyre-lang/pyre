# CPython-suite gap: nothing in the suite walks a traceback's frame locals in a
# loop tight enough for a minor collection to land between the walk and the read.
# test.test_collections ABC mixins JIT-trace ValuesView.__contains__ (nlocals=4,
# stacksize=3, locals array length 7). An old type-9 varsize ref array
# (frame locals_cells_stack_w) holds nursery addresses a later JITFRAME covers;
# snapshot_for_tracing copies those locals.
# parity-tests reason: this is a pyre/PyPy moving-GC regression gate.
# pyframe.py STORE_FAST is setarrayitem_gc on the virtualizable locals GcArray;
# framework.py transform_generic_set emits write_barrier_from_array; rewrite.py
# handle_write_barrier_setarrayitem / gen_write_barrier_array emit
# COND_CALL_GC_WB on the array. gctransform/framework.py push_roots /
# shadowstack.py pop_roots around the dest malloc; setarrayitem_gc /
# incminimark.py write_barrier then stores the reloaded words.
# parity-env: PYPY_GC_NURSERY=4096

"""A traceback walk reading `f_locals` must survive the collections it triggers.

On 2026-08-04 this body aborted 3/3 on cranelift with

    GC BUG: invalid type_id=<varies> at obj_addr=0x...
      (site=minor_varsize_item_target, holder_type_id=Some(9), holder_offset=Some(8))

The holder was a varsize object carrying its length (13) at offset 0 and its
items from offset 8, with only `items[0]` set, sitting above `nursery_top` --
so an old-gen holder pointing into the nursery. The reported `type_id` differed
every run and was pointer-shaped, meaning the target's header was recycled
memory: an unbarriered old-to-young store whose target an earlier minor
collection had already moved. Reached from `gc_alloc_nursery_shim` ->
`alloc_with_type_slow` -> `do_collect_nursery` -> `trace_and_update_object` ->
`copy_nursery_object`.

It is green now, on both backends, and this file is here so it stays that way
rather than because it still reproduces.

Re-measured 2026-08-30 on cranelift: 23 runs clean -- three with no environment,
two each at `PYPY_GC_NURSERY` 64/256/1536/3072/8192, and three each under
`MAJIT_GC_NURSERY_POISON=1` at the default, 1536 and 64. No nursery size is
pinned here because none of them discriminated.

The trigger was allocation layout rather than the loop bounds: the same body
reading `N`/`WHICH` from `sys.argv` was clean, and wrapping the defaults to
defeat constant folding did not change it.
"""

import gc
from collections import ChainMap, namedtuple
from collections.abc import ValuesView

try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass

N = 15000
WHICH = "head"

ERR = ValueError("boom")


def mid(_i):
    raise ValueError("boom")


def locs(tb):
    out = []
    idx = 0
    while tb is not None:
        f = tb.tb_frame
        want = (
            WHICH == "all"
            or (WHICH == "head" and idx == 0)
            or (WHICH == "tail" and tb.tb_next is None)
        )
        if want:
            out.append((f.f_code.co_name, tuple(sorted(f.f_locals))))
        else:
            out.append(f.f_code.co_name)
        tb = tb.tb_next
        idx += 1
    return tuple(out)


def drive():
    seen = set()
    k = 0
    while k < N:
        try:
            mid(k)
        except ValueError as e:
            seen.add(locs(e.__traceback__))
            e.__traceback__ = None
        k += 1
    return sorted(seen)


r = drive()
assert r == [(("drive", ("e", "k", "seen")), "mid")], r


def promote():
    junk = [[object() for _ in range(16)] for _ in range(24)]
    del junk
    gc.collect()


def f4t(a, b, c, d):
    return a, b, c


class Mapping:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)


def old_cap7_frame_locals_that_receive_young_args_survive_minors():
    N = 600
    kept = []
    old_self = object()
    promote()
    for i in range(N):
        row = f4t(old_self, object(), object(), object())
        kept.append(row)
        if i % 8 == 0:
            promote()
    if kept[-1][0] is not old_self:
        raise AssertionError(("f4t", kept[-1][0], old_self))

    data = {}
    for i in range(8):
        data[object()] = object()
    view = ValuesView(Mapping(data))
    promote()
    for i in range(N):
        needle = object()
        if needle in view:
            raise AssertionError(("contains", i, needle))
        if i % 8 == 0:
            promote()


def churn():
    junk = [[object() for _ in range(16)] for _ in range(24)]
    del junk
    gc.collect()


def four_locals(a, b, c, d):
    return a, b, c, d


def jit_traced_frame_locals_that_hold_young_objects_survive_minors():
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


old_cap7_frame_locals_that_receive_young_args_survive_minors()
jit_traced_frame_locals_that_hold_young_objects_survive_minors()
print("OK")
