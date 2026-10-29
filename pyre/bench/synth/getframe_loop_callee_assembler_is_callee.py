# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=callee,hot
# An inlined callee whose own loop is cut into CALL_ASSEMBLER must still
# report itself from `sys._getframe()` inside that loop.
#
# `opimpl_jit_merge_point` keeps the callee's vref / `topframeref` current
# until `finishframe` -> `do_recursive_call(assembler_call=True)` returns.
# Leaving the EC before the assembler call makes the compiled remainder name
# the caller.
import sys

try:
    import pypyjit

    pypyjit.set_param("threshold=100,function_threshold=100")
except ImportError:
    pass

N = 20000
INNER = 8


def callee(n):
    names = []
    i = 0
    while i < n:
        names.append(sys._getframe().f_code.co_name)
        i += 1
    return names


def hot():
    seen = set()
    for _ in range(N):
        for name in callee(INNER):
            seen.add(name)
    return seen


def main():
    callee(INNER)
    seen = hot()
    if seen != {'callee'}:
        print(
            'FAIL sys._getframe() inside the cut callee loop named %r, owed {\'callee\'}'
            % (sorted(seen),)
        )
        return 1
    print('PASS sys._getframe() inside a cut callee loop names the callee')
    return 0


sys.exit(main())
