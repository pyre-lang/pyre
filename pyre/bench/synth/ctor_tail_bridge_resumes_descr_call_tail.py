# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot,entry-bridge:__init__,loop:__init__
# A loop-bearing `__init__` cut to CALL_ASSEMBLER must resume through
# `descr_call`'s tail when a later call returns non-None often enough to
# compile a BRIDGE: `C()` is the instance while `__init__` returns None, and
# a non-None result raises TypeError. `resume.py rebuild_from_resumedata`
# rebuilds that tail as an ordinary frame (`W_TypeObject.descr_call` after
# `get_and_call_args`).
import sys

try:
    import pypyjit

    pypyjit.set_param("threshold=100,function_threshold=100")
except ImportError:
    pass

WARM = 400
N = 20000
FAILS = 800
flag = [0]


class C:
    def __init__(self):
        i = 0
        while i < 5:
            i += 1
        if flag[0]:
            return 1


def hot(n):
    raised = 0
    none = 0
    ok = 0
    for _ in range(n):
        try:
            obj = C()
            if obj is None:
                none += 1
            elif type(obj) is C:
                ok += 1
        except TypeError:
            raised += 1
    return raised, none, ok


def expect_init_typeerror():
    try:
        C()
        return None
    except TypeError as e:
        return str(e)


def main():
    expected = "__init__() should return None, not 'int'"
    raised, none, ok = hot(WARM)
    if raised or none or ok != WARM:
        print(
            "FAIL warmup: raised=%d none=%d ok=%d (owed %d instances)"
            % (raised, none, ok, WARM)
        )
        return 1
    raised, none, ok = hot(N)
    if raised or none or ok != N:
        print(
            "FAIL compiled None path: raised=%d none=%d ok=%d (owed %d instances)"
            % (raised, none, ok, N)
        )
        return 1
    flag[0] = 1
    raised, none, ok = hot(FAILS)
    if none or ok or raised != FAILS:
        print(
            "FAIL later non-None: raised=%d none=%d ok=%d (owed %d TypeError)"
            % (raised, none, ok, FAILS)
        )
        return 1
    got = expect_init_typeerror()
    if got != expected:
        print("FAIL expected TypeError %r, got %r" % (expected, got))
        return 1
    print("PASS descr_call tail bridge still returns the instance and raises TypeError")
    return 0


sys.exit(main())
