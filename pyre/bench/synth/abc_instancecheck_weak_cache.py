# `_abc_instancecheck` answers from two per-class weak-set caches, and a
# question that hits one is meant to cost the probe and nothing else.  This
# drives both answers in a hot loop: a class that matches (positive cache) and
# one that never will (negative cache), plus the stdlib `collections.abc`
# checks, whose caches are shared with everything else that asks them.
#
# A `register` partway through discards every negative cache by bumping the
# invalidation counter, so the second half re-walks and refills rather than
# reading the entries the first half recorded.  Deterministic.
import collections.abc as cabc
from abc import ABCMeta

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass


class Shape(metaclass=ABCMeta):
    pass


class Circle(Shape):
    pass


class Square(Shape):
    pass


class Blob:
    pass


class LateBlob:
    pass


# The lowered threshold compiles both cache probes before registration; later
# iterations only repeat the same populated-cache states.
N = 60000
SWITCH = N // 2


def main():
    circle = Circle()
    square = Square()
    blob = Blob()
    late = LateBlob()
    seq = [1, 2, 3]
    mapping = {"k": 1}
    hits = 0
    misses = 0
    for i in range(N):
        # Positive cache: two classes so the probe is not answered by a single
        # resident entry.
        if isinstance(circle, Shape):
            hits += 1
        if isinstance(square, Shape):
            hits += 1
        # Negative cache, and the class the mid-loop `register` moves across.
        if isinstance(blob, Shape):
            hits += 1
        else:
            misses += 1
        if isinstance(late, Shape):
            hits += 1
        else:
            misses += 1
        # The stdlib ABCs, asked against a matching and a non-matching type.
        if isinstance(seq, cabc.Sequence):
            hits += 1
        if isinstance(mapping, cabc.Mapping):
            hits += 1
        if isinstance(seq, cabc.Mapping):
            hits += 1
        else:
            misses += 1
        if i == SWITCH:
            Shape.register(LateBlob)
    return hits, misses


hits, misses = main()
# `late` misses until the registration and hits after it; `blob` never hits.
expected_hits = N * 4 + (N - SWITCH - 1)
expected_misses = N * 2 + (SWITCH + 1)
assert hits == expected_hits, "hits %r != %r" % (hits, expected_hits)
assert misses == expected_misses, "misses %r != %r" % (misses, expected_misses)
print(hits, misses)
