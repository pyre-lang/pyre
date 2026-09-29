# pyre-check: max-pypy-ratio=3
# pyre-check: skip-cpython
# pyre-check: trace-shape=run:absent=CallMayForceR
# `super(C, self)` bound to a name, then read — the spelling LOAD_SUPER_ATTR
# does not cover, because the name binding splits the proxy construction and
# the attribute load into two opcodes the compiler never fuses.
#
# The construction arrives as an opaque may-force `bh_call_fn`, which costs
# more than the four-word allocation it performs: the walk publishes a vref
# for the executing frame ahead of it and re-checks it after, nine ops around
# one.  The read arrives as an opaque MRO walk that also wipes the trace's
# heap-field cache.  Folded, the proxy is `New` + `SetfieldGc` whose reads the
# optimizer answers from the emission itself, so the allocation dies and this
# shape costs what `super(C, self).val(x)` costs.
#
# N is sized so pypy clears `FLOOR_GATE_MIN_BASELINE_S` and the ratio is a
# measurement rather than warmup -- see `zero_arg_super_attr.py` for the
# reasoning and the numbers.  pypy runs ~0.65ns per iteration here, so
# 250,000,000 lands its execution time near 0.17s.  cpython cannot usefully
# run that many, hence `skip-cpython`.
#
# The ceiling is fitted to what that measurement reads: 1.7x on dynasm and
# 1.8x on cranelift, carried with room for a slower host.
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
        su = super(Child, self)
        return su.val(x)


def run(o, n):
    acc = 0
    for _ in range(n):
        acc = o.val(acc)
    return acc


print(run(Child(), N))
