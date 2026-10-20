# pyre-check: jitstats-band=guard_failures=17
# guard_failures reads 849 or 854 on one macOS arm64 dynasm binary as
# PYPY_GC_NURSERY moves between 2M, 4M, 8M and 16M, and ubuntu CI reads 864 on
# dynasm and 866 on cranelift with ten loops and four bridges: `counter.py
# invoke_after_minor_collection` runs `decay_all_counters` every 32 minor
# collections, so one guard's count follows nursery traffic. The width is the
# 849 -> 866 distance.
# A BINARY_OP (`+`/`*`) whose receiver alternates at runtime between an exact
# builtin int and a user-class (or int-subclass) instance at the SAME hot site.
# The raw int specialization bypasses special-method dispatch, so it must only
# admit exact builtin ints/bools; a numeric subclass keeps the builtin int
# layout while its Python-visible class lives in `w_class`, so an unguarded raw
# path silently ran `int.__add__` on the subclass instead of its `__add__`.
# The alternation is driven by a `TO_BOOL` conditional (`i % k == 0`), and the
# three-class block below drives a LOAD_ATTR residual on the guard-failure
# blackhole path — its receiver/result must stay live across the deopt.
# Deterministic, terminating, prints an int checksum; jit == nojit.
M = 1000000007

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass


class C0:
    def __init__(self, x):
        self.x = x % M

    def __add__(self, o):
        return C0(self.x + (o.x if isinstance(o, C0) else o))


class C1:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = x % M

    def __add__(self, o):
        if isinstance(o, C1):
            return C1(self.x + o.x)
        if isinstance(o, int):
            return C1(self.x + o)
        return NotImplemented

    def __radd__(self, o):
        return C1(self.x + o)


class C2:
    def __init__(self, x):
        self.x = x % M

    def __radd__(self, o):
        if isinstance(o, int):
            return C2(self.x + o)
        return NotImplemented

    def __rmul__(self, o):
        if isinstance(o, int):
            return C2(self.x * o)
        return NotImplemented


class MyInt(int):
    def __add__(self, o):
        return int(self) * 1000 + int(o)

    def __radd__(self, o):
        return int(o) * 1000 + int(self)


def fold(acc, r):
    if isinstance(r, int):
        return (acc + (r % M)) % M
    return (acc + r.x) % M


def run():
    acc = 0
    for i in range(30000):
        # Exact int on ~1/113 of iterations, a plain user class otherwise; the
        # `w + w` site dispatches int.__add__ vs C0.__add__.
        w = i if i % 113 == 0 else C0(i)
        try:
            acc = fold(acc, w + w)
        except TypeError:
            acc = (acc + 7) % M

        # int subclass alternating with exact int: MyInt.__add__ differs from
        # int.__add__, so the raw path must decline for the subclass.
        w = i if i % 71 == 0 else MyInt(i)
        try:
            acc = fold(acc, w + w)
        except TypeError:
            acc = (acc + 7) % M

        # __slots__ class alternating with exact int at a third site.
        w = i if i % 997 == 0 else C1(i)
        try:
            acc = fold(acc, w + w)
        except TypeError:
            acc = (acc + 7) % M

        # Reflected NotImplemented dispatch then a LOAD_ATTR (`r.x`) read whose
        # residual receiver/result must stay live on the blackhole path.
        a = C2(i)
        try:
            r = i + a
            r = i * r
            acc = (acc + r.x) % M
        except TypeError:
            acc = (acc + 11) % M
    return acc


print(run())
