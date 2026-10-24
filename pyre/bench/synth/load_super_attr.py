# pyre-check: max-pypy-ratio=3
# pyre-check: skip-cpython
# pyre-check: trace-shape=run:absent=CallMayForceR,absent=CallR
# The loop folds to one add per iteration on both JITs, so N has to be in
# the hundreds of millions before pypy's execution time is a measurement
# rather than a clock tick — and at that size cpython is minutes behind.
# `super(C, self).val()` with the loop INSIDE the super-bearing method, in the
# `while` spelling.  `zero_arg_super_attr.py` is the `for` twin.
#
# `spec-folds` is the invariant here: those three folds fire or the fixture
# fails, on every host and backend.  The two-argument oparg form lowers to the
# two ops it is -- the call, then the attribute load -- so the proxy is emitted
# virtual (`two_arg_super_call`), the read is answered out of that emission
# (`load_attr_on_super`), and the method form's two runtime unwraps stay pure
# (`super_attr_unwrap`).  The fused `load_super_attr` residual is what the
# ZERO-argument form reaches instead; `zero_arg_super_attr.py` gates that one.
# The ratio is 3 rather than 2 because super
# is no longer what it measures — the identical loop calling a plain
# `self.plain()`, no super anywhere, reads the same user CPU time as this one
# to the centisecond (0.36s against pypy's 0.22s, best of three, both files).
# What is left is pyre's
# generic `while` scaffolding (a loop-invariant class re-check on the bound, an
# unused ForceToken, the `ec.w_tracefunc` null check and the eval-breaker
# poll), and a ceiling of 2 here would gate on that instead of on super.  The
# `for`-form super fixtures beside this one do carry 2.
#
# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
N = 500000000


class Base:
    def val(self):
        return 1


class Child(Base):
    def run(self, n):
        acc = 0
        i = 0
        while i < n:
            acc = acc + super(Child, self).val()
            i = i + 1
        return acc


print(Child().run(N))
