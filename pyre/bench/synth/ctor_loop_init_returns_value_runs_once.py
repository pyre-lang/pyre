# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot,entry-bridge:__init__,loop:__init__
# A loop-bearing `__init__` that returns a non-None value must raise
# TypeError without re-running the constructor body.
#
# After the loop is cut into CALL_ASSEMBLER, residual-executing the portal
# already ran `__init__`. Aborting the inline into an entry replay re-enters
# the constructor, so a visible side effect (append) happens twice before
# `descr_call` raises TypeError. `W_TypeObject.descr_call` checks the result
# after `get_and_call_args` and never re-enters `__init__`.
import sys

try:
    import pypyjit

    pypyjit.set_param("threshold=100,function_threshold=100")
except ImportError:
    pass

WARM = 400
N = 20000
shared = []


class C:
    def __init__(self):
        shared.append(1)
        i = 0
        while i < 5:
            i += 1
        return 1


def hot(n):
    for _ in range(n):
        try:
            C()
        except TypeError:
            pass


def main():
    hot(WARM)
    hot(N)
    got = len(shared)
    owed = WARM + N
    if got != owed:
        print('FAIL constructor ran %d times, owed %d' % (got, owed))
        return 1
    print('PASS constructor that returns non-None ran once per call')
    return 0


sys.exit(main())
