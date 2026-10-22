# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot,entry-bridge:__init__,loop:__init__
# A loop-bearing `__init__` traced while it returned None must still raise
# TypeError when a later call returns a non-None value.
#
# After the loop is cut into CALL_ASSEMBLER, the compiled constructor path
# used to record `descr_call`'s None check only when the tracing-time
# `__init__` result was already non-None. A later call that returns a value
# then produced the instance. `W_TypeObject.descr_call` checks every
# successful `__init__` result after `get_and_call_args`.
import sys

try:
    import pypyjit

    pypyjit.set_param("threshold=100,function_threshold=100")
except ImportError:
    pass

WARM = 400
N = 20000
flag = [0]
force = [0]


class C:
    def __init__(self):
        i = 0
        while i < 5:
            i += 1
        if flag[0]:
            return 1


class Forced:
    def __init__(self):
        i = 0
        while i < 5:
            i += 1
        if force[0]:
            sys._getframe(1).f_locals
            return 1


def hot(n):
    for _ in range(n):
        C()


def hot_forced(n):
    for _ in range(n):
        Forced()


def expect_init_typeerror(fn):
    try:
        fn()
        return None
    except TypeError as e:
        return str(e)


def main():
    expected = "__init__() should return None, not 'int'"
    hot(WARM)
    hot(N)
    flag[0] = 1
    got = expect_init_typeerror(lambda: hot(1))
    if got != expected:
        print("FAIL expected TypeError from __init__ returning non-None, got %r" % (got,))
        return 1
    # The loop is already cut into CALL_ASSEMBLER. Forcing that call
    # (`sys._getframe(1).f_locals` after warm-up) must still run
    # `descr_call`'s None check: GUARD_NOT_FORCED used to resume in the
    # caller with the instance and skip it.
    hot_forced(WARM)
    hot_forced(N)
    force[0] = 1
    got = expect_init_typeerror(lambda: hot_forced(1))
    if got != expected:
        print(
            "FAIL expected TypeError from forced CALL_ASSEMBLER __init__ "
            "returning non-None, got %r" % (got,)
        )
        return 1
    print("PASS TypeError: %s" % got)
    return 0


sys.exit(main())
