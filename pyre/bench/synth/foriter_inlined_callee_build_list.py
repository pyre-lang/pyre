# A FOR_ITER trace must keep compiling when an inlined callee allocates a
# fresh list and copies it into a tuple.  Both fresh allocations are replay-safe
# and have no reason to abort the caller's loop trace.


def chain(item):
    out = []
    return len(tuple(out)) + (item - item)


def drive(n):
    last = -1
    for i in range(n):
        last = chain(i)
    return last


print(drive(1800000))
