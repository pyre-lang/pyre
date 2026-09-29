# pyre-check: max-pypy-ratio=3
# pyre-check: skip-cpython
# pyre-check: trace-shape=loop:absent=CallMayForceR
# Zero-argument `super()` bound to a name, with the loop inside the
# super-bearing method.
#
# Which channel the walk reads is not visible in the Python: a callee the trace
# inlines owns a slot shadow the inline seeded, while a method that carries its
# own loop is the frame the portal traces and its slots come out of the
# standard virtualizable.  This fixture is the portal-frame side; the callee
# side is `super_attr_inlined_callee.py`, and `bare_super_frame_escape.py`
# pins both channels' answers.  A fold reading only the callee channel leaves
# this spelling on the re-routed may-force residual, which is where it was: ~62ns per iteration
# at 2,000,000 iterations, against ~3 for the same body with no `super()` in
# it at all.
#
# N is sized so pypy clears `FLOOR_GATE_MIN_BASELINE_S` and the ratio is a
# measurement rather than warmup -- see `zero_arg_super_attr.py` for the
# reasoning and the numbers.  pypy runs ~0.61ns per iteration here, so
# 250,000,000 lands its execution time near 0.17s.  cpython cannot usefully
# run that many, hence `skip-cpython`.
#
# The ceiling is fitted to what that measurement reads: 1.6x on dynasm and
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
    def loop(self, n):
        acc = 0
        for _ in range(n):
            su = super()
            acc = su.val(acc)
        return acc


print(Child().loop(N))
