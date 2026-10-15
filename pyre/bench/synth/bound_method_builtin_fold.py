# pyre-check: spec-folds=load_bound_method_attr
# pyre-check: trace-shape=escaping_bound_method:absent=CallMayForceR
# `lst.append(x)` on a builtin receiver: the name resolves to a builtin-code
# function on the type, so LOAD_ATTR's method fast path declines (it only
# pushes `[descr, self]` for `flag_method_descriptor` types) and `getattr`
# materialises a bound method. The tracer folds that construction to
# guard_class + guard_value(w_class) + guard_value(version_tag) +
# new_with_vtable(Method) so it virtualizes into the following CALL, instead of
# leaving an opaque getattr residual in the loop. These cases pin the guards
# that fold depends on: a __slots__ subclass whose method is replaced mid-loop
# (version tag), an instance-dict entry shadowing the method name (the fold
# must decline for a dict-bearing receiver), alternating receiver types at one
# call site (class guard), `__`-names that bind through the same shape, and a
# bound method that outlives its CALL (the virtual must materialize). Output
# verified against CPython/PyPy.
# Every leg compiles independently and the fold census names the optimization
# directly.  A larger count only accumulated materialized lists, sets and
# escaping method objects after all guarded shapes had already run.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 10000


def build(n):
    r = []
    for i in range(n):
        r.append(i)
    return r


def dict_and_set(n):
    d = {}
    s = set()
    for i in range(n):
        d.setdefault(i & 7, i)
        s.add(i & 15)
    return len(d), len(s)


class L(list):
    __slots__ = ()


def slots_subclass_override(n):
    lst = L()
    seen = []
    for i in range(n):
        lst.append(i)
        if i == n // 2:
            L.append = lambda self, v, _old=list.append: _old(self, v * 100)
            seen.append("patched")
    total = sum(lst)
    del L.append
    return total, seen


class Shadow(list):
    pass


def shadowed(n):
    obj = Shadow()
    calls = []
    obj.append = calls.append
    for i in range(n):
        obj.append(i)
    return len(obj), len(calls)


def polymorphic(n):
    a, b = [], bytearray()
    for i in range(n):
        target = a if (i & 1) else b
        target.append(i & 127)
    return len(a), len(b)


def dunder_names(n):
    # `__`-names bind through the same `function` + builtin-code shape as
    # `append`, so the fold applies to them too.
    lst = [0] * 8
    total = 0
    for i in range(n):
        total += lst.__len__() + (i,).__len__()
        lst.__setitem__(i & 7, i)
    return total, lst[3]


def escaping_bound_method(n):
    r = []
    keep = []
    for i in range(n):
        m = r.append
        keep.append(m)
        m(i)
    keep[0](-1)
    return len(r), r[-1], len(keep)


print(len(build(N)), sum(build(N // 8)))
print(dict_and_set(N))
print(slots_subclass_override(N))
print(shadowed(N))
print(polymorphic(N))
print(dunder_names(N))
print(escaping_bound_method(N))
