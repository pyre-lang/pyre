# pyre-check: max-pypy-ratio=167
# Regression guard for the per-call-created nested closure JIT hang.
#
# `add` is defined fresh on every `make` call, and `make` is driven from a
# hot outer loop.  Each nested code constant must realize to one shared code
# object (stable `__code__` identity / JIT green key), so the trace give-up
# counter accumulates and the JIT stops re-tracing the dead closure.  When the
# code object is re-materialized per call instead, every closure instance gets
# a fresh green key, the give-up never sticks, and tracing thrashes
# super-linearly until this bench times out.
N = 120000


def make(n):
    acc = [0]

    def add(x):
        acc[0] += x
        return acc[0]

    r = 0
    for i in range(n):
        r = add(i)
    return r


def main():
    s = 0
    for k in range(N):
        s += make(k % 12)
    print(s)


main()
