# pyre-check: max-pypy-ratio=8.6
# Exact, non-specialized tuples use the ordinary wrapped-items array.  Keep
# fixed-length unpack on that storage visible to the trace while preserving
# the generic path for tuple subclasses at the same call site.


class TupleSubclass(tuple):
    pass


def unpack_sum(value, count):
    total = 0
    for _ in range(count):
        a, b, c = value
        total += a + b + c
    return total


print(unpack_sum((1, 2, 3), 21600000))
print(unpack_sum(TupleSubclass((4, 5, 6)), 3))
