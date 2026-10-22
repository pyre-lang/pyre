# pyre-check: max-pypy-ratio=11.7
# pyre-check: ungated-jitstats=bridges_compiled,guard_failures
# The realloc-boundary append leaves one bridge on the margin of the compile
# threshold: re-running the same binary moves bridges_compiled between 4 and 5
# (and guard_failures with it, ~196 per bridge), and the three backends
# disagree within a single run, so neither value is a property of the tree.
# GC stress for the in-place Object-append write barrier the tracer does NOT
# record. `w_list_append`'s in-place arm stores a GC ref into the items block
# and runs `list_write_barrier`; when the block is GC-managed the backend GC
# rewrite already marks that store with COND_CALL_GC_WB_ARRAY, so the walker
# drops the barrier residual instead of leaving a second barrier in the loop
# (rewrite.py:936-944; pyjitpl records no write barrier at all,
# executor.py:446). These cases pin what that suppression must not break: an
# OLD list whose block is already promoted receiving YOUNG elements. If the
# store went unremembered, a minor collection would never scan the slot and the
# element would be freed or read back stale. Run with MAJIT_GC_ITEMSBLOCK=0 too
# — there the block is std::alloc with no GC header, the barrier on the
# W_ListObject is load-bearing, and the walker must keep emitting it.
# Output verified against CPython/PyPy.
N = 40000
CHURN = 300


def churn(k):
    # Allocation pressure to drive minor collections between appends.
    junk = None
    for i in range(k):
        junk = (i, [i], {i: i})
    return junk


def old_list_young_appends():
    r = []
    for i in range(N):
        r.append((i, i))
    churn(CHURN * 20)
    # r and its block are old now; append fresh young tuples.
    for i in range(N):
        r.append((i + N, i + N))
        if i % 200 == 0:
            churn(CHURN)
    assert len(r) == 2 * N, len(r)
    total = 0
    for a, b in r:
        assert a == b, (a, b)
        total += a
    return total


def interleaved_growth():
    lists = [[] for _ in range(16)]
    total = 0
    for i in range(N):
        lists[i % 16].append((i,))
        if i % 100 == 0:
            churn(CHURN)
    for lst in lists:
        for (v,) in lst:
            total += v
    return total, sum(len(x) for x in lists)


def strings_and_dicts():
    # Non-tuple Object-strategy elements: str and dict payloads that the
    # collector must keep reachable through the appended slots.
    r = []
    for i in range(N):
        r.append(str(i))
        if i % 250 == 0:
            churn(CHURN)
    joined = 0
    for i, s in enumerate(r):
        assert s == str(i), (i, s)
        joined += len(s)
    return joined


def none_then_objects():
    r = [None] * 8
    r.clear()
    for i in range(N):
        r.append(None if i % 2 else [i])
        if i % 300 == 0:
            churn(CHURN)
    total = 0
    for i, v in enumerate(r):
        if i % 2:
            assert v is None, (i, v)
        else:
            assert v == [i], (i, v)
            total += v[0]
    return total


def big_live_len_regrow():
    # The resize itself is the case under test: `grow_list_items_block_gc` roots
    # the OLD BLOCK across the (collecting) allocation of the new one and copies
    # the items out of it afterwards, rather than rooting each item separately.
    # A collection landing inside that allocation therefore has to relocate the
    # block AND update every slot in it — so this drives resizes at the largest
    # live lengths it can, with fresh young elements and allocation pressure
    # right up against the growth boundary. Distinct identities per slot catch a
    # copy that reads pre-collection addresses.
    r = []
    for i in range(N * 4):
        r.append([i, str(i)])
        if i % 64 == 0:
            churn(20)
    assert len(r) == N * 4, len(r)
    total = 0
    for i, v in enumerate(r):
        assert v[0] == i and v[1] == str(i), (i, v)
        total += v[0]
    return total


print(old_list_young_appends())
print(interleaved_growth())
print(strings_and_dicts())
print(none_then_objects())
print(big_live_len_regrow())
