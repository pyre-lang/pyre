# pyre-check: max-pypy-ratio=1.0
# Restoring `MIFrame._create_segmented_trace_and_blackhole` parity makes this
# deliberately tiny trace limit exercise repeated segment compilation, like
# PyPy (whose oracle reports trace-too-long and segmenting aborts here).  Local
# post-parity runs are 0.4-0.5x; the ceiling is twice the slowest, rounded up to
# one decimal place.
try:
    import pypyjit
except ImportError:
    pypyjit = None


def set_trace_limit(n):
    if pypyjit is not None:
        pypyjit.set_param("trace_limit=%d,threshold=1,function_threshold=1" % n)


# The limit has to leave room for the loop header before the body's first
# store: `for e in items` records the list cursor and its element load
# directly (~20 operations), where the opaque `for_iter_next` residual it
# replaced was one.  At 40 the walk reached its ceiling inside the header and
# never saw the `STORE_ATTR` this fixture is about.
set_trace_limit(60)

errs = list(map(lambda i: OSError(1, str(i)), range(20000)))


def mutate_all(items):
    total = 0
    for e in items:
        e.errno = e.errno + 1
        tmp = total
        del tmp
        a = total + 1
        b = a + 2
        c = b + 3
        d = c + 4
        f = d + 5
        g = f + 6
        h = g + 7
        i = h + 8
        j = i + 9
        k = j + 10
        m = k + 11
        n = m + 12
        o = n + 13
        p = o + 14
        q = p + 15
        r = q + 16
        s = r + 17
        t = s + 18
        total = t & 1023
    return total


total = mutate_all(errs)
print(sum(map(lambda e: e.errno, errs)), total)
