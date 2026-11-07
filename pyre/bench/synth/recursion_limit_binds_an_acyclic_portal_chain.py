# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot,level0,level1,level2,level3,level4,level5,entry-bridge:level0,entry-bridge:level2
# Every Python activation consumes recursion budget, not only an activation
# whose code object is already present in the live call chain.  The six
# distinct callees below each carry a loop so a warmed caller compiles each
# of them as its own portal loop.
#
# `_opimpl_recursive_call` (`pyjitpl.py`) inlines a callee whose greenkey is
# not already on the framestack, so an acyclic chain inlines the next pycode
# and residualizes after that. Only `level0` and `level2` therefore mint
# function-entry bridges; `level1`/`level3`/`level4`/`level5` compile as
# loops. pypy3 `jit-log-opt` on this file is the same two entry-bridges.
#
# The two limits pin the conservative lower-level check used by PyPy's
# approximative recursion limit. On this six-callee compiled shape, `base + 5`
# admits the chain while `base + 4` refuses its last aggregate activation.
# Charging recursive code objects only would let the second case return
# normally.
#
# `pypy/module/sys/vm.py setrecursionlimit` is `@jit.dont_look_inside` and
# explicitly documents this limit as approximative and checked at a lower
# level. This fixture therefore pins pyre's conservative compiled seam rather
# than claiming CPython's exact frame-count boundary.
import sys

WARM = 3000
INNER = 20


def level5():
    total = 0
    for i in range(INNER):
        total += i
    return total


def level4():
    total = 0
    for i in range(INNER):
        total += i
    return total + level5()


def level3():
    total = 0
    for i in range(INNER):
        total += i
    return total + level4()


def level2():
    total = 0
    for i in range(INNER):
        total += i
    return total + level3()


def level1():
    total = 0
    for i in range(INNER):
        total += i
    return total + level2()


def level0():
    total = 0
    for i in range(INNER):
        total += i
    return total + level1()


def hot(n):
    result = 0
    for _ in range(n):
        result = level0()
    return result


def frame_depth():
    frame = sys._getframe()
    depth = 0
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


def main():
    expected = 6 * sum(range(INNER))
    assert hot(WARM) == expected

    base = frame_depth()
    saved = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(base + 5)
        exact = level0()
        sys.setrecursionlimit(base + 4)
        try:
            level0()
        except RecursionError:
            over = "raised"
        else:
            over = "returned"
    finally:
        sys.setrecursionlimit(saved)

    assert exact == expected, exact
    assert over == "raised", over
    print("PASS")


main()
