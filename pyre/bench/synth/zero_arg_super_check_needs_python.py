# pyre-check: max-pypy-ratio=3
# pyre-check: spec-folds=bare_super_virtual,load_attr_on_super
# pyre-check: skip-cpython
# pyre-check: trace-shape=loop:absent=CallMayForceR
# The apparent-class twin of `zero_arg_name_bound_super_attr.py`: `type(self)`
# is unrelated to `Child`, while the ordinary class attribute
# `self.__class__` names it.  This reaches PyPy's third `_super_check` arm.
#
# PyPy traces that `space.getattr` like any other attribute read.  For this
# non-descriptor class attribute, mapdict promotes the receiver type/version
# and map, then returns the installed `Child` constant; `_super_check` promotes
# that type for its subtype test and stores it as `W_Super.w_objtype`.
# `bare_super_virtual` must reproduce exactly those guards.  In particular it
# must keep the receiver's real `w_class == Proxy` and must not replace that
# proof with `w_class == Child`, which is the normal `_super_check` arm and is
# false by construction here.
#
# Before that attribute trace was connected to super construction, the
# frame-explicit `bare_super_call` residual preserved correctness and kept the
# loop compilable, but paid `_super_check` once per iteration: dynasm was about
# 0.17s here against 0.06s for a real `Child` receiver.  The virtual fold now
# emits the proxy directly, and `load_attr_on_super` consumes its stored
# apparent type for `s.val` without re-deriving it from the receiver.
#
# Keep this as a benchmark rather than only a selfcheck: it holds both the
# zero-abort snapshot and the disappearance of the residual cost.
#
# N is sized to be that unclamped PyPy timing.  At 1,000,000 iterations PyPy
# finished in a few milliseconds, under `EXEC_TIME_FLOOR_S`, so check.py
# printed the ratio with a `~` -- "ratio is not a measurement, and no ratio
# gate is applied to it" -- and the large numbers in that column were warmup
# rather than steady state: this file reads 12.6ns per iteration at 1,000,000,
# 1.5ns at 20,000,000 and 0.9ns at 80,000,000, against PyPy's 0.6ns.  cpython
# cannot usefully run 200,000,000, hence `skip-cpython`; PyPy stays the oracle
# the backends' output is compared against.
#
# So the ceiling is re-fitted, which is what the paragraph this replaced asked
# for.  150 was a level target read off the clamped column; the unclamped
# measurement is 1.7x on dynasm and 1.8x on cranelift, so 3 is what the file
# now carries -- the same number `load_super_attr.py` carries for the same
# family, with room for a slower host.
N = 200000000


class Base:
    def val(self, x):
        return x + 1


class Child(Base):
    def loop(self, n):
        acc = 0
        for _ in range(n):
            s = super()
            acc = s.val(acc)
        return acc


class Proxy:
    """`type(self)` is not a `Child`.  Only `self.__class__` says so."""

    __class__ = Child


print(Child.loop(Proxy(), N))
