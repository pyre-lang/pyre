# pyre-check: max-pypy-ratio=195
# In-place list repeat (`list *= n`) parity.  CPython and PyPy both mutate the
# list in place (listobject descr_inplace_mul), so the list object identity is
# preserved across `*=`; only `*` produces a fresh list.  check.py's oracle is
# PyPy, which agrees with CPython here.


def drive():
    out = []

    # `*=` repeats in place: same object, repeated contents.
    x = [1, 2]
    y = x
    x *= 3
    out.append(("repeat", x, x is y))

    # `*= 0` empties in place.
    z = [9, 8, 7]
    zid = z
    z *= 0
    out.append(("zero", z, z is zid))

    # A negative count empties in place too.
    n = [1, 2, 3]
    nid = n
    n *= -4
    out.append(("negative", n, n is nid))

    # `*= 1` is a no-op that keeps identity and contents.
    o = [5, 6]
    oid = o
    o *= 1
    out.append(("one", o, o is oid))

    # The repeat is observable through an alias.
    a = [0]
    b = a
    a *= 4
    out.append(("alias", b))

    # `__imul__` is exposed on the list type.
    out.append(("has_imul", hasattr([1], "__imul__")))

    # `*= n` accepts any object with __index__, not just int/long.
    class Count:
        def __index__(self):
            return 3

    ix = [7]
    ix_id = ix
    ix *= Count()
    out.append(("index_obj", ix, ix is ix_id))

    # Drive a hot loop so a compiled trace exercises the in-place repeat.
    total = 0
    k = 0
    while k < 480000:
        h = [k, k + 1]
        h *= 3
        total += len(h)
        k += 1
    out.append(("hot", total))

    return out


for row in drive():
    print(row)
