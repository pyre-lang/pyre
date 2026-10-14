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
N = 400000


class A:
    def kind(self):
        return 1


class B:
    def kind(self):
        return 2


def main():
    total = 0
    obj = A()
    for _ in range(N):
        total += obj.kind()
        obj.__class__ = B if obj.__class__ is A else A
    print(total)


main()
