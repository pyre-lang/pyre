# pyre-check: max-pypy-ratio=17
# Multi-except dispatch + counter-only handler bodies on the blackhole
# guard-failure resume path. The loop warms up exception-free (so the trace
# compiles the no-exception path), then raises ZeroDivisionError and
# IndexError after warm-up. Each `except` body only bumps a counter and the
# `try` result is dead — the shape that exposed two resume bugs:
#   - every exception matched the FIRST `except` unconditionally (the type
#     dispatch fell through to a blanket match), so the second clause was
#     never reached, and
#   - a single matched exception ESCAPED because the CHECK_EXC_MATCH bool
#     was pinned to a scratch register the following truth test never read,
#     so the handler was skipped.
# A trace that raises from iteration 1 records the handler and never needs
# the resume path; only warm-up-then-raise reaches it.
N = 864000


def work(n):
    data = [3, 1, 4, 1, 5]
    zdiv = 0
    idxe = 0
    i = 0
    while i < n:
        if i < 2000:
            d = 1
            j = 0
        else:
            d = i % 2  # 0 on even i -> ZeroDivisionError
            j = i % 9  # > 4 -> IndexError (data has 5 elements)
        try:
            q = (i + 1) // d
            v = data[j]
        except ZeroDivisionError:
            zdiv += 1
        except IndexError:
            idxe += 1
        i += 1
    return zdiv, idxe


def main():
    print(work(N))


main()
