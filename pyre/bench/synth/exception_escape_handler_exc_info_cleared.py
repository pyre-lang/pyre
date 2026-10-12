# Regression oracle: an exception raised inside an `except` handler leaves
# the callee, and the caller's handler for it then exits.  POP_EXCEPT must
# leave `sys.exc_info()` empty once both handlers are gone.
#
# The walker stores the handled exception into the EC `sys_exc_value` slot at
# handler entry.  When the walk of the inlined callee then aborted and the
# CALL ran again from its start, that store stayed applied, the re-run callee
# saved it as its `prev`, and every later iteration read the first traced
# iteration's `ValueError` back out of `sys.exc_info()`.  `escape_direct`
# reaches the abort from the loop body, `escape` through one more inlined
# frame.  Expected: no leak in either, and the sum of `i & 7` twice.
import sys

N = 4000


def escape_inner(i):
    try:
        raise ValueError(i)
    except ValueError:
        raise IndexError(i & 7)


def escape(i):
    try:
        escape_inner(i)
    except IndexError as e:
        r = e.args[0]
    return r


def run_direct(n):
    leaked = 0
    acc = 0
    i = 0
    while i < n:
        try:
            escape_inner(i)
        except IndexError as e:
            acc += e.args[0]
        if sys.exc_info()[1] is not None:
            leaked += 1
        i += 1
    return acc, leaked


def run_nested(n):
    leaked = 0
    acc = 0
    i = 0
    while i < n:
        acc += escape(i)
        if sys.exc_info()[1] is not None:
            leaked += 1
        i += 1
    return acc, leaked


print(run_direct(N))
print(run_nested(N))
