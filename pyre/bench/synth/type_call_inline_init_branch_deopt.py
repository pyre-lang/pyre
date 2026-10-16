# pyre-check: max-pypy-ratio=105
# pypy's exec time is pinned to the startup-subtraction floor here, so the
# ratio is not a measurement: the ceiling is twice the slowest ratio the CI
# runners observe (51.5x), rounded up.
# A guard failing inside an inlined `__init__` must hand the caller the
# instance, not `__init__`'s return value.  `typeobject.py descr_call` discards
# `__init__`'s result and returns the instance; the flattened constructor
# inline has no such frame at a mid-`__init__` deopt.  The debug flag flips
# AFTER the call site is hot, so the branch guard inside `__init__` fails on a
# warmed trace and the deopt must still produce the instance.
class Handle:
    def __init__(self, callback, args, loop, context):
        self._context = context
        self._loop = loop
        self._callback = callback
        self._args = args
        self._cancelled = False
        if loop.get_debug():
            self._source_traceback = 1
        else:
            self._source_traceback = None


class Loop:
    def __init__(self):
        self._debug = False

    def get_debug(self):
        return self._debug

    def _call_soon(self, callback, args, context):
        handle = Handle(callback, args, self, context)
        if handle._source_traceback:
            del handle._source_traceback
        return handle


loop = Loop()
none_handles = 0
for i in range(100000):
    h = loop._call_soon(int, (), None)
    if h is None:
        none_handles += 1
loop._debug = True
traced = 0
for i in range(200):
    h = loop._call_soon(int, (), None)
    if h is None:
        none_handles += 1
    elif not hasattr(h, "_source_traceback"):
        traced += 1
print("none_handles =", none_handles)
print("traced =", traced)
