# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=entry-bridge:inner
# A GuardException(class A) bridge compiled after a CallMayForceR that
# returned normally must take GUARD_NO_EXCEPTION, even while an outer
# `except A` is active so `ec.sys_exc_value` is an A instance.
#
# `pyjitpl.py _prepare_exception_resumption` takes the exception only from
# `cpu.grab_exc_value`. A null grab that inherited the handled A used the
# SAVE_EXCEPTION box as the caught exception and dereferenced null.

try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass

N = 8000
SPLIT = 4000


def lookup(d, k):
    try:
        return d[k]
    except KeyError:
        return -1


def inner(d, k):
    return lookup(d, k) + 1


def run():
    d = {}
    for j in range(SPLIT, N):
        d[j] = j
    total = 0
    try:
        raise KeyError("outer")
    except KeyError:
        for i in range(N):
            total += inner(d, i)
    return total


# Misses return 0 from `inner`; hits add j+1 for j in [SPLIT, N).
expected = (SPLIT + 1 + N) * (N - SPLIT) // 2
got = run()
assert got == expected, got
print("PASS")
