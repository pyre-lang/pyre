# pyre-check: max-pypy-ratio=3
# pyre-check: skip-cpython
# pyre-check: spec-folds=sys_exc_info
# pyre-check: trace-shape=type_hot:absent=CallMayForceR,slice_hot:absent=CallMayForceR
# PyPy's `vm.py exc_info_direct` lets `function.py funccall_valuestack`
# omit the traceback when bytecode look-ahead proves that only slots 0/1 can
# escape.  A residual `sys.exc_info()` call used to allocate its three-tuple
# and repeatedly force the trace in each of these loops.  Keep the workload
# large enough that PyPy's execution-only time is measurable after startup
# subtraction; before the direct-call fold the pyre run takes tens of seconds.

import sys

N = 400_000_000


def type_hot(n):
    try:
        1 // 0
    except ZeroDivisionError:
        value = None
        i = 0
        while i < n:
            value = sys.exc_info()[0]
            i += 1
        return value is ZeroDivisionError


def value_hot(n):
    try:
        1 // 0
    except ZeroDivisionError as caught:
        value = None
        i = 0
        while i < n:
            value = sys.exc_info()[1]
            i += 1
        return value is caught


def slice_hot(n):
    try:
        1 // 0
    except ZeroDivisionError as caught:
        value = None
        i = 0
        while i < n:
            value = sys.exc_info()[:2]
            i += 1
        return value[0] is ZeroDivisionError and value[1] is caught


print(type_hot(N), value_hot(N), slice_hot(N))
