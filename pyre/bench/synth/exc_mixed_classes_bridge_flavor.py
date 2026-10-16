# pyre-check: max-pypy-ratio=16
# A callee raises TWO different exception classes into a hot try/except loop.
# The loop trace records the raising iteration's GUARD_EXCEPTION(class-A); the
# no-raise iterations chronically fail that guard WITHOUT a pending exception,
# so a bridge is compiled for the no-exception continuation.  When class B
# arrives, the same guard fails WITH a pending exception and enters the same
# bridge — the bridge's entry flavor guard (GUARD_NO_EXCEPTION,
# prepare_resume_from_failure) must deopt that entry to the blackhole instead
# of running the recorded continuation on the NULL raised-call result.
N = 4800000


def f(i):
    if i % 3 == 1:
        raise ValueError(i)
    if i % 3 == 2:
        raise TypeError(i)
    return i


def run():
    acc = 0
    for i in range(N):
        try:
            acc += f(i)
        except ValueError:
            acc += i
        except TypeError:
            acc += i * 2
    return acc


print(run())
