# pyre-check: max-pypy-ratio=2.7
# main CI 10-24..10-26 (6 runs, all native legs): 1.2x-2.0x; ceiling = max x 1.35.
# CALL_KW and CALL_FUNCTION_EX inside a callee that is inlined into the hot
# loop.  The residual port compiles both opcodes rather than emitting
# abort_permanent and declining the callee's jitcode.  Guards output
# correctness and the demonstrable inline (loops_aborted is 0).
#
# `call_kw_hot_loop.py` is the contrasting shape: a CALL_KW directly in the
# hot loop body rather than nested inside an inlined callee.
N = 232258065


def add(a, b, c):
    return a + b + c


def helper_kw(i):
    # `add(a, c=..., b=...)` compiles to CALL_KW.
    return add(i, c=2, b=1)


def helper_ex(i):
    # `add(*args)` compiles to CALL_FUNCTION_EX.
    args = (i, 1, 2)
    return add(*args)


def main_kw():
    total = 0
    for i in range(N):
        total += helper_kw(i)
    print(total)


def main_ex():
    total = 0
    for i in range(N):
        total += helper_ex(i)
    print(total)


main_kw()
main_ex()
