# pyre-check: trace-shape=entry-bridge:push:absent=CallMayForceR,append_first:absent=CallMayForceN
#
# `push` binds `a.append` inside an inlined callee.  The folds that shape a
# bound-method load decline only where a guard would collapse its resume to
# the caller's CALL, rather than for the whole of such a sub-walk, so the
# binding must still fold here.
# Inlined-callee shared-heap mutation parity, in both helper orderings.
#
# A tiny helper mutates a caller-owned list/instance inside a hot while-loop,
# so the tracer inlines the call and the mutating opcode (LIST_APPEND /
# STORE_ATTR) lands on a SHARED heap object. The final counts must match the
# iteration count exactly: a doubled side effect over-counts, a dropped one
# under-counts.
#
# `append_first` appends before bumping.  `store_first` reverses the order so
# the attribute store commits during recording BEFORE the deliberate
# list.append abort discards the trace: bump(c) traces through (its STORE_ATTR
# side effect commits during the inline concrete step), then push(acc, i) hits
# the append abort and the interpreter restarts the iteration from the
# unadvanced frame, re-running bump.  A recording attempt that aborts after
# the committed store must not leave a doubled side effect.
# Both orderings compile and the two folds are censused directly.  Keeping
# millions of list entries alive after those paths are established only turns
# the regression guard into an allocation benchmark.
N = 1200000


def push(a, v):
    a.append(v)


class Counter:
    def __init__(self):
        self.n = 0


def bump(c):
    c.n = c.n + 1


def append_first():
    acc = []
    c = Counter()
    i = 0
    while i < N:
        push(acc, i)
        bump(c)
        i = i + 1
    print(len(acc))
    print(c.n)
    print(acc[0], acc[N // 2], acc[N - 1])
    print(sum(acc))


def store_first():
    acc = []
    c = Counter()
    i = 0
    while i < N:
        bump(c)
        push(acc, i)
        i = i + 1
    print(len(acc))
    print(c.n)
    print(acc[0], acc[N // 2], acc[N - 1])


append_first()
store_first()
