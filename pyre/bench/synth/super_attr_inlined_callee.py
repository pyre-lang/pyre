# pyre-check: max-pypy-ratio=3
# pyre-check: skip-cpython
# pyre-check: trace-shape=run:absent=CallMayForceR,absent=CallR
# `super(C, self).m(x)` in a callee the FOR_ITER-driven trace inlines.
# `load_super_attr.py` keeps its loop inside the super-bearing method, so the
# callee replay scan never sees the LOAD_SUPER_ATTR residuals; this shape does.
#
# The two-argument oparg form's three folds have to fire from INSIDE an inlined
# callee, where `two_arg_super_call` additionally asks that the caller be paused
# on the framestack with its own parent frame -- which it is here, because the
# call this callee was inlined at is an ordinary CALL.
#
# N is sized so pypy clears `FLOOR_GATE_MIN_BASELINE_S` and the ratio is a
# measurement rather than warmup -- see `zero_arg_super_attr.py` for the
# reasoning and the numbers.  pypy runs ~0.69ns per iteration here, so
# 250,000,000 lands its execution time near 0.17s.  cpython cannot usefully
# run that many, hence `skip-cpython`.
#
# The ceiling is fitted to what that measurement reads: 1.7x on dynasm and
# 1.9x on cranelift, carried with room for a slower host.
# `load_super_attr.py` carries the same 3 for the same family.
#
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 250000000


class Base:
    def val(self, x):
        return x + 1


class Child(Base):
    def val(self, x):
        return super(Child, self).val(x)


def run(o, n):
    acc = 0
    for _ in range(n):
        acc = o.val(acc)
    return acc


print(run(Child(), N))
