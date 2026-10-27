# pyre-check: max-pypy-ratio=88
# An exception raised by a CALLEE (a helper function or an exhausted generator)
# with a per-iteration computed argument, caught in a hot JIT-traced caller's
# `except ... as e`. walker_record_guard_exception discarded the GuardException
# result box and left last_exc_value as the recording-time const_ref, so once the
# loop compiled, e.args[0] / e.value read back the value frozen at trace-record
# time instead of the live raised instance. A same-frame raise never showed this
# because its raised value stays live. Deterministic, terminating, int checksum;
# jit == nojit once the guard result box carries the live exception.
M = 1000003


def boom(w):
    raise ValueError(w)


def echo(n):
    total = 0
    i = 0
    while i < n:
        got = yield total
        total = (total + got) % M
        i += 1
    return total


def run():
    h = 0
    # Callee helper raise: e.args[0] must track w each iteration.
    for k in range(24000):
        w = (k * 7 + 3) % M
        try:
            boom(w)
        except ValueError as e:
            h = (h * 31 + e.args[0]) & 0xFFFFFFFFFF
    # Exhausted-generator StopIteration: e.value must track the returned total.
    for k in range(24000):
        gg = echo(9)
        gg.send(None)
        i = 0
        while i < 9:
            try:
                gg.send(k + i)
            except StopIteration as e:
                h = (h * 31 + (e.value or 0)) & 0xFFFFFFFFFF
                break
            i += 1
    return h


print(run())
