# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot,entry-bridge:__init__,loop:__init__
# An inlined `__init__` with code before its first loop header must not
# re-run that prefix when the loop is cut into CALL_ASSEMBLER.
#
# Residualizing the original CALL after the sub-walk already executed the
# prefix applies the side effect twice. The cut continues from the merge
# point (`ctor_continuation` plays `descr_call`'s tail).
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


def hot(n):
    for _ in range(n):
        C()


def main():
    hot(WARM)
    hot(N)
    got = len(shared)
    if got != WARM + N:
        print('FAIL constructor prefix ran %d times, owed %d' % (got, WARM + N))
        return 1
    print('PASS constructor prefix before a cut loop ran once per call')
    return 0


sys.exit(main())
