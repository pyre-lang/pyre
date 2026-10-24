# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=first_reads,reread
# pyre-check: trace-shape=reread:absent=CallMayForceR,build:absent=CallMayForceN
# `UnboxedPlainAttribute._direct_write` freezes unboxing for the whole class
# after a type change.  Instances created before the freeze retain an unboxed
# slot until `UnboxedPlainAttribute._direct_read` migrates them.  A folded read
# that performs `_prim_direct_read` alone would skip that migration and leave
# unboxed and boxed instances mixed under one promoted-map guard.
# The old throughput ratio required two million live instances merely to lift
# PyPy's nearly empty optimized loops above the timer floor.  CI consequently
# spent most of this fixture first-touching object pages, not exercising the
# migration or folded reread.  Low JIT thresholds plus explicit compilation
# and fold censuses cover those two paths directly with a bounded live set.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 50000


class C:
    def __init__(self, v):
        self.x = v


def build(n):
    # Created while unboxing is still allowed, so each gets an unboxed int
    # slot, and none of them has been read yet.
    objs = [C(i) for i in range(n)]
    freeze = C(0)
    freeze.x = 1.5
    return objs, freeze.x


def first_reads(objs, n):
    total = 0
    i = 0
    while i < n:
        total += objs[i].x
        i += 1
    return total


def reread(objs, n):
    # Every instance has migrated to boxed storage by now; the same loop must
    # keep reading the same values.
    total = 0
    i = 0
    while i < n:
        total += objs[i].x
        i += 1
    return total


objs, frozen_value = build(N)
expected = N * (N - 1) // 2
first = first_reads(objs, N)
second = reread(objs, N)
if first != expected or second != expected or frozen_value != 1.5:
    raise AssertionError((first, second, frozen_value, expected))
print("PASS mapdict frozen unboxing migration")
