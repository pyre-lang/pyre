# pyre-check: max-pypy-ratio=3
# pyre-check: skip-cpython
# pyre-check: trace-shape=class_function_read:absent=CallMayForceR,instance_method_read:absent=CallMayForceR
# `typeobject.py:822 space.get(w_value, w_None, self)` and the instance twin
# `object.__getattribute__` performs, for the three values whose binding is
# pure: a plain function read off a class (returned unbound by
# `gateway.py descr_function_get`), a `staticmethod` (`w_function?` handed back
# by `function.py descr_staticmethod_get`), and a function read off an instance
# without calling it (a `Method`).
#
# Each of the three is a bind with no Python in it, so it folds to constants
# under the receiver + version pins and the loop body is one add per
# iteration.  Left unfolded the read costs an MRO walk for `__get__` plus an
# interp-level call to the slot wrapper on every iteration -- 82-100ns against
# pypy's 0.8, which is what this file measures.
#
# The three loops are separated so the `spec-folds` header names a fold that
# only one of them can supply: `load_type_attr` covers the two class-receiver
# reads and `load_bound_method_attr` the instance one, and neither fires for
# the other's shape.
#
# N is sized so pypy's execution time is a measurement rather than a clock
# tick: at 1,000,000 the three loops together finish in a few milliseconds,
# under `EXEC_TIME_FLOOR_S`, and check.py prints the ratio with a `~` --
# "ratio is not a measurement, and no ratio gate is applied to it".  cpython
# cannot usefully run 70,000,000 three times over, hence `skip-cpython`; pypy
# stays the oracle the backends' output is compared against.
#
# The ceiling is the number `load_super_attr.py` carries: the unclamped
# measurement is 1.5-1.7x on dynasm, and 3 leaves room for a slower host and
# for the backends that read higher.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 70000000


class Base:
    def val(self, x):
        return x + 1

    @staticmethod
    def sm(x):
        return x + 1


class Child(Base):
    pass


def class_function_read(n):
    """`Base.val` — `descr_function_get` returns the function unbound."""
    acc = 0
    for _ in range(n):
        g = Base.val
        acc = acc + 1
    return acc, g


def class_staticmethod_call(n):
    """`Base.sm(x)` — the wrapper hands back its `w_function?` slot."""
    acc = 0
    for _ in range(n):
        acc = Base.sm(acc)
    return acc


def instance_method_read(n, o):
    """`o.val` with no call after it, so a `Method` really is materialised."""
    acc = 0
    for _ in range(n):
        g = o.val
        acc = acc + 1
    return acc, g


read_acc, read_fn = class_function_read(N)
call_acc = class_staticmethod_call(N)
bound_acc, bound = instance_method_read(N, Child())
print(read_acc, read_fn is Base.__dict__["val"])
print(call_acc)
print(bound_acc, bound.__self__.__class__.__name__, bound(0))
