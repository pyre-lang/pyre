# pyre-check: requires-modules=gc
# pyre-check: max-pypy-ratio=100
# pyre-check: trace-shape=main:absent=CallMayForceR
# The ceiling is twice the slowest ratio the CI runners observe (49.3x on the
# linux runner's cranelift), rounded up. It read 218 before a later tightening
# fitted it to a single run's numbers.
# Reassigning obj.__class__ inside a hot loop must actually re-root the
# instance's type, so method lookup and type() follow the new class.  The
# STORE_ATTR mapdict inline cache (store_attr_slowpath) must not classify
# `__class__` as an ordinary instance-dict attribute (its data-descriptor role
# is modelled by object_setattr's special-case, not a getset, so it never
# surfaces for classify_attr): storing it into the instance dict leaves the
# real type unchanged and obj.kind() keeps dispatching through the old class.  The
# exact aggregate over the loop makes that silent miscompile observable.
import gc

N = 800000


class A:
    def kind(self):
        return 1


class B:
    def kind(self):
        return 2


def main():
    total = 0
    obj = A()
    # `must_compile` keys a failing ref GUARD_VALUE by the object's address
    # (compile.py `lltype.cast_ptr_to_int`). Promote A, B and obj before the
    # loop warms up, so a minor collection does not move them mid-warm-up and
    # restart their counters at a point set by the nursery size: without it,
    # guard_failures reads 406/454/581/401 at a 128K/256K/512K/1M+ nursery,
    # and 544 on the windows runner.
    gc.collect()
    for _ in range(N):
        total += obj.kind()
        obj.__class__ = B if obj.__class__ is A else A
    print(total)


main()
