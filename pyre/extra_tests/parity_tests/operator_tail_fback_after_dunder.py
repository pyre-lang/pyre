# CPython-suite gap: no suite test inspects f_back from __index__ after an
# inlined __len__ whose result later fails the operator-tail guard.
# parity-tests reason: a blackholed `__len__` that returns a non-int must run
# the Len tail's `space.index` with the dunder's caller as `f_back`, not the
# finished dunder frame. A mid-trace abort below the dunder must convert the
# continuation in place the same way `operator_tail_abort_below_dunder.py`
# does.

"""f_back after a blackholed operator dunder is the dunder's caller."""

try:
    import pypyjit
except ImportError:
    pypyjit = None

if pypyjit is not None:
    pypyjit.set_param("threshold=1,function_threshold=1")

import sys

N = 4000
RARE = 997
SWITCH = N // 2

names = []


class Idx:
    def __index__(self):
        fr = sys._getframe(0)
        assert fr.f_back is not None
        names.append(sys._getframe(1).f_code.co_name)
        return 1


class Obj:
    calls = 0

    def __init__(self):
        self.n = 1
        self.fail = False

    def __len__(self):
        Obj.calls += 1
        if self.fail:
            raise ValueError("x")
        return self.n


def driver():
    obj = Obj()
    raised = 0
    for i in range(N):
        obj.fail = i % RARE == RARE - 1
        if i == SWITCH:
            obj.n = Idx()
        try:
            len(obj)
        except ValueError:
            raised += 1
    return raised


raised = driver()
print(sorted(set(names)))
print(Obj.calls, raised, len(names))
print("OK")
