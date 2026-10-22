# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=entry-bridge:__init__,loop:__init__,direct_for,wrapped_for
# A constructor whose `__init__` carries a loop of its own evaluated to
# `None` instead of the instance.
#
# `__init__`'s loop reaches its own back-edge threshold first and compiles, so
# when the enclosing loop traces `C()` the walker's `__init__` sub-walk stops
# at that loop header and reports `SubLoopCalleeCallAssembler`: the callee is
# entered through `CALL_ASSEMBLER` into the token it already has, and the
# fold writes the assembler's result into the CALL's destination.  That result
# is `__init__`'s own return -- `None` -- where the CALL has to evaluate to the
# instance.  The recorded trace then reads `C() is None` off the wrong value,
# and every iteration deopts on the guard that says it was False.
#
# `type_descr_call_impl` is what discards `__init__`'s result and answers
# `w_newobject`, and the fold has no resume coordinate to play that tail at,
# so it residualizes the instantiation instead.
#
# Measured before the fix on dynasm at the DEFAULT thresholds: the `while`
# shapes reported 97 of 300 constructor calls evaluating to `None`, the `for`
# shapes 5 of 20000, and `PYRE_NO_JIT=1` reported none.
import sys

N = 20000


class WhileInit:
    def __init__(self):
        j = 0
        while j < 5:
            j += 1


class ForInit:
    def __init__(self):
        for _ in range(5):
            pass


def make_while():
    return WhileInit()


def make_for():
    return ForInit()


def direct_while():
    missing = 0
    for _ in range(N):
        if WhileInit() is None:
            missing += 1
    return missing


def wrapped_while():
    missing = 0
    for _ in range(N):
        if make_while() is None:
            missing += 1
    return missing


def direct_for():
    missing = 0
    for _ in range(N):
        if ForInit() is None:
            missing += 1
    return missing


def wrapped_for():
    missing = 0
    for _ in range(N):
        if make_for() is None:
            missing += 1
    return missing


def main():
    cases = [
        ("while init, direct", direct_while),
        ("while init, wrapper", wrapped_while),
        ("for init, direct", direct_for),
        ("for init, wrapper", wrapped_for),
    ]

    failures = []
    for label, fn in cases:
        missing = fn()
        if missing:
            failures.append(f"{label}: {missing} of {N} constructor calls evaluated to None")

    if failures:
        for line in failures:
            print("FAIL", line)
        return 1
    print("PASS a call-assembler constructor inline still evaluates to the instance")
    return 0


sys.exit(main())
