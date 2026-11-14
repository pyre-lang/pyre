# An except-as-return callee (`except E as e: return e`) is first compiled
# with no handled exception, then reused inside `except KeyError`. The
# inner ValueError's `__context__` must be that KeyError
# (`error.py record_context` / `chain_context`). A recording-time skip of
# the chaining hook leaves `__context__` None on the reused body.
#
# Warmup contribution is 1 when the first compile sees no context; reuse
# contribution is 1 when the chain is the KeyError. A mismatch jumps by
# 1000.
#
# Expected: (N, N)
# No `max-pypy-ratio`: this is a shape oracle, not a workload.
N = 20000


def swallow():
    try:
        raise ValueError("inner")
    except ValueError as e:
        return e


def main():
    i = 0
    warm = 0
    while i < N:
        e = swallow()
        if e.__context__ is None:
            warm = warm + 1
        else:
            warm = warm + 1000
        i = i + 1
    i = 0
    hits = 0
    while i < N:
        try:
            raise KeyError("outer")
        except KeyError:
            e = swallow()
            ctx = e.__context__
            if type(ctx) is KeyError and ctx.args == ("outer",):
                hits = hits + 1
            else:
                hits = hits + 1000
        i = i + 1
    print((warm, hits))


main()
