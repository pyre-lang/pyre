# pyre-check: max-pypy-ratio=32
#
# A guard-failure resume inside an inlined callee must close that callee's
# execution-context scope before its blackhole advances to the caller
# (`executioncontext.py` leave). The blackhole run loop transfers the
# callee's return value and releases its interpreter, but releasing a
# BlackholeInterpreter does not restore `topframeref` on its own, so without
# the leave transition the completed callee stays the current frame.
#
# Two things then go wrong, and this fixture asserts both because either can
# hold while the other breaks:
#   1. a later `sys._getframe(1)` chains behind the stale callee and reads that
#      callee's sparse locals image, so `f_locals['base']` raises KeyError;
#   2. the caller's own `locals()` selects the stale frame and answers with the
#      callee's parameter set — no `sys._getframe` involved at the read.
#
# The first triggering call is correct either way: the damage is only
# observable once the resumed callee should have left. A single triggering
# call therefore passes with or without the fix.
import sys


def inner(k):
    if k > 863997:                       # two triggering calls, not one
        return sys._getframe(1).f_locals['base']
    return k


def outer(n):
    base = 11
    acc = 0
    for i in range(n):
        acc += inner(i) & 7
    return acc, sorted(locals().keys())


print(outer(864000))
