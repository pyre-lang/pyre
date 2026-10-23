# CPython-suite gap: `test_unpack_ex` and `test_extcall` check what a starred
# display and a `f(x, *args)` call produce, once per shape. None of them runs
# the display often enough to be compiled, and the lost item below only shows
# on the iteration that traces it.
# An empty list can be traced, then grow into an Object-strategy items block
# that a minor has already promoted, and store interned names into that block.
# test_collections stores young objects into an Object-strategy items block
# grown to rlist_overallocate(4)=7 (BUILD_LIST 3 + append;
# AbstractUnwrappedStrategy.setslice grow) after a minor has already promoted
# it. namedtuple field-name lists are AsciiListStrategy, grown the same way,
# then store or append young STR payloads into that already-old block.
#
# parity-tests reason: `[x, *args]` builds a one-item list and extends it from
# the tuple, so the append has to grow the items block
# (`_ll_list_resize_ge` -> `_ll_list_resize_hint_really`) before
# `ll_setitem_fast` stores the item. A tracer that sets the new length but
# leaves the store out of the grown block hands back a list whose second slot
# is empty, and the call built from it reports a missing argument.
# record_list_and_items_cond_call_gc_wb must not skip the array half from a
# tracing-time box_value of 0; rewrite.py handle_write_barrier_setarrayitem /
# gen_write_barrier_array emit COND_CALL_GC_WB on the SETARRAYITEM array
# operand. rlist.py ll_setitem_fast / listobject.py
# AbstractUnwrappedStrategy.setitem is setarrayitem_gc; setslice is items +=
# [none]*delta then ll_setitem_fast (object_splice / writebarrier_before_move).
# AsciiListStrategy ll_setitem_fast / ll_append is setarrayitem_gc on the
# erased STR GcArray; list_write_barrier_impl Phase L2 barriers
# ascii_items.block (incminimark.py write_barrier). assembler.py
# _write_barrier_fastpath TESTs with no null check, so empty ascii cannot fold
# to Getfield+CondCallGcWb.
#
# CPython 3.14 and PyPy agree on every arm below.
# parity-env: PYPY_GC_NURSERY=4096
import gc
import sys
from collections import UserList

try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass

ITEM = ["a", "b"]


def display_list(*args):
    return [len, *args]


def display_tuple(*args):
    return (len, *args)


def forward(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def target(args=None, namespace=None):
    return len(args)


def call_through(*args, **kwargs):
    return forward(target, *args, **kwargs)


def a_starred_list_display_keeps_the_unpacked_item():
    for i in range(3000):
        r = display_list(ITEM)
        assert len(r) == 2 and r[0] is len and r[1] is ITEM, (i, len(r))


def a_starred_tuple_display_keeps_the_unpacked_item():
    for i in range(3000):
        r = display_tuple(ITEM)
        assert len(r) == 2 and r[0] is len and r[1] is ITEM, (i, len(r))


def a_forwarded_star_call_keeps_its_positional_argument():
    total = 0
    for i in range(3000):
        total += call_through(ITEM)
    assert total == 6000, total


a_starred_list_display_keeps_the_unpacked_item()
a_starred_tuple_display_keeps_the_unpacked_item()
a_forwarded_star_call_keeps_its_positional_argument()


def promote():
    junk = [[i] for i in range(400)]
    del junk
    gc.collect()


def a_list_traced_empty_then_filled_keeps_young_interned_strs():
    N = 4000
    xs = []
    promote()
    for i in range(N):
        xs.append((sys.intern("k%d-%d" % (i, i)),))
        if i % 32 == 0:
            promote()
    for i in range(4):
        last = N - 4 + i
        expected = "k%d-%d" % (last, last)
        s = xs[last][0]
        if s != expected:
            raise AssertionError((last, s, expected))
        if sys.intern(expected) is not s:
            raise AssertionError(("identity", last, s, expected))


def young_objects_stored_into_an_old_cap7_list_survive_minors():
    N = 4000
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


def store(xs, i):
    s = "n%d-%d" % (i, i)
    xs[i % 4] = s
    return s


def young_ascii_strs_stored_into_a_cap7_ascii_list_survive_minors():
    N = 4000
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


def add_young(xs, i):
    s = "n%d-%d" % (i, i)
    xs.append(s)
    return s


def young_ascii_strs_appended_into_an_old_cap7_ascii_list_survive_minors():
    N = 4000
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


a_list_traced_empty_then_filled_keeps_young_interned_strs()
young_objects_stored_into_an_old_cap7_list_survive_minors()
young_ascii_strs_stored_into_a_cap7_ascii_list_survive_minors()
young_ascii_strs_appended_into_an_old_cap7_ascii_list_survive_minors()


def _permute_old_list_keeps_young_payloads(kind, make, fresh, is_ok):
    n = 200
    lst = [make(i) for i in range(n)]
    gc.collect()
    gc.collect()
    for k in range(10):
        for j in range(20):
            lst[(j * 9 + k) % n] = fresh(k, j)
        if k % 2:
            lst.reverse()
        else:
            lst.sort(reverse=True)
        junk = [make(i) for i in range(300)]
        del junk
        for x in lst:
            if not is_ok(x):
                raise AssertionError((kind, k, type(x), x))
    out = sorted(lst, reverse=True)
    for x in out:
        if not is_ok(x):
            raise AssertionError((kind, "sorted", type(x), x))
    lst.sort()
    print("%s reverse/sort kept %d" % (kind, n))
    print(lst[:3], lst[-3:])


def old_ascii_str_list_reverse_or_sort_keeps_young_payloads():
    _permute_old_list_keeps_young_payloads(
        "ascii-str",
        lambda i: "s%d" % i,
        lambda k, j: "n%d_%d" % (k, j),
        lambda x: isinstance(x, str) and (x.startswith("s") or x.startswith("n")),
    )


def old_bytes_list_reverse_or_sort_keeps_young_payloads():
    _permute_old_list_keeps_young_payloads(
        "bytes",
        lambda i: b"s%d" % i,
        lambda k, j: b"n%d_%d" % (k, j),
        lambda x: isinstance(x, bytes) and (x.startswith(b"s") or x.startswith(b"n")),
    )


old_ascii_str_list_reverse_or_sort_keeps_young_payloads()
old_bytes_list_reverse_or_sort_keeps_young_payloads()
print("OK")
