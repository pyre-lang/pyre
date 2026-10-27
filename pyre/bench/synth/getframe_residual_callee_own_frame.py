# pyre-check: max-pypy-ratio=32
# Regression guard: a residual (may-force) callee that inspects its OWN frame
# via sys._getframe() must not clear the traced CALLER's virtualizable tracing
# token. Clearing it raised a spurious frame-escape with no committed resume pc,
# replaying the loop body from entry and double-applying the callee's
# non-journaled STORE_ATTR side effect -- a JIT-only wrong answer (c.n > loops).
import sys


class Counter:
    pass


c = Counter()
c.n = 0


def bump(x):
    c.n += 1                  # STORE_ATTR: non-journaled body effect
    frame = sys._getframe(0)  # may-force residual inspecting the callee's own frame
    return x if frame is not None else -1


def main():
    total = 0
    for i in range(6000000):
        total += bump(i)
    print(total, c.n)


main()
