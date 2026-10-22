# pyre-check: max-pypy-ratio=6
# ubuntu cranelift 5.5x at pypy exec 0.08s (wall 0.47s). 6 is 9% over that reading.
# A residual callee raises for the extremes of a hot descending loop and
# returns in the middle, so the loop trace's exception guard chronically fails
# WITHOUT a pending exception and a no-exception bridge gets compiled.  That
# bridge's entry GUARD_NO_EXCEPTION is what makes the rare pending-exception
# entry deopt instead of running the no-exception continuation on a NULL
# raised-call result.
#
# The `str(i) + str(i)` ahead of the try is the point of the shape: it lands a
# removable call in the bridge's resume data, right before the entry guard.
# `optimize_GUARD_NO_EXCEPTION` (rewrite.py, pure.py) drops a
# GUARD_NO_EXCEPTION whose predecessor was removed, so without the
# SAVE_EXC_CLASS / SAVE_EXCEPTION + RESTORE_EXCEPTION pair around it the entry
# guard is deleted and the final IndexError escapes uncaught -- upstream issue
# #2132, pinned there by test_guard_no_exception_incorrectly_removed_from_bridge.
R = 272728
START = 14


def do(n):
    if n > 7:
        raise ValueError(n)
    if n > 1:
        return n
    raise IndexError(n)


def one_pass(start):
    caught_value = 0
    caught_index = 0
    i = start
    while i > 0:
        s = str(i) + str(i)
        try:
            do(i)
        except ValueError:
            caught_value += 1
        except IndexError:
            caught_index += 1
        if len(s) == 0:
            return -1, -1
        i -= 1
    return caught_value, caught_index


def run():
    total_value = 0
    total_index = 0
    for _ in range(R):
        v, x = one_pass(START)
        total_value += v
        total_index += x
    return total_value, total_index


print(run())
