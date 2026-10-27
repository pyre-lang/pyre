# pyre-check: max-pypy-ratio=130
# A conditional raise two call levels below a `try` must reach the enclosing
# `except`, including once the frame holding the `try` runs as its own compiled
# trace.
#
# The exposed shape is a CALL that sits inside a try-block and whose callee is
# inlined.  A guard inside the inlined callee has no frame of its own to resume
# into, so it collapses onto the caller's CALL boundary -- the CALL's pre-call
# `-live-`, which precedes that CALL's own `catch_exception`.  A post-call
# GUARD_NO_EXCEPTION failing there hands the blackhole a pending exception at a
# coordinate from which the handler search cannot reach the catch, and the
# raise leaves the frame past its matching `except`.
#
# The dict lookup in the loop is load-bearing: it keeps the module-level loop
# from completing a trace of its own, so `shape` is compiled as a func-entry
# trace instead of being inlined into the caller.  The raise must also be
# conditional, so the exception edge stays off the recorded path.
N = 4320000


def leaf(i):
    if i % 7 == 0:
        raise ValueError(i)
    return i


def mid(i):
    return leaf(i)


def shape(i):
    try:
        return mid(i)
    except ValueError:
        return -1


seen = {}
caught = 0
for i in range(N):
    caught += shape(i) == -1
    seen.get(0, 0)
print("caught =", caught)
