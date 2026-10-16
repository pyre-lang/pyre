# pyre-check: max-pypy-ratio=55
# A callee catches a residual-call exception in-frame and RETURNS a sentinel
# into a JIT-hot outer loop.  When the callee gets hot enough that a bridge is
# compiled for the exception-guard (GUARD_NO_EXCEPTION) failure on the
# caught-KeyError path, the bridge must not be built from the guard's
# no-exception fallthrough resume_pc (which would record `Finish` of the NULL
# raised-call result and hand a NULL back to the caller — "call failed").  The
# bridge tracer declines the caught-in-frame exception-guard case so the
# blackhole resume routes the exception to the handler and returns the sentinel.
N = 360000


def lookup(d, k):
    try:
        return d[k]
    except KeyError:
        return -1


def run():
    d = {1: 100, 2: 200, 3: 300}
    acc = 0
    hits = 0
    for i in range(N):
        k = i % 5          # 0 and 4 miss → KeyError caught → -1
        v = lookup(d, k)
        if v == -1:
            acc -= 1
        else:
            acc += v
            hits += 1
    return acc, hits


print(run())
