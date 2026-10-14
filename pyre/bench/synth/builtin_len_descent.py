# pyre-check: spec-folds=builtin_len_descent
# pyre-check: trace-shape=hot:absent=CallMayForceR
# PyPy installs the builtin implementations reached by
# descroperation.py `_len` through typedef.py
# `use_special_method_shortcut('__len__')`.  Keep every layout formerly emitted
# by the hand-written builtin_len fold on the generated gateway descent: list
# strategies, ordinary and specialised pairs, set/frozenset, range, unicode,
# bytes, and mutable bytearray.  The shared inner loop deliberately retraces
# across layouts; every result contributes to one checksum.


def pair(a, b):
    return (a, b)


def hot(n, objects):
    total = 0
    for obj in objects:
        for _ in range(n):
            total += len(obj)
    return total


def main():
    try:
        import pypyjit

        pypyjit.set_param("threshold=20,function_threshold=20")
    except ImportError:
        pass

    marker = object()
    objects = (
        [1, 2, 3, 4],
        [1.25, 2.5],
        # Mixed numbers take the IntOrFloat strategy, whose length read is the
        # Integer strategy's own.
        [1, 2.5],
        [marker],
        [],
        (1, 2, 3),
        pair(1, 2),
        pair(1.0, 2.0),
        pair(marker, marker),
        {1, 2, 3},
        frozenset((1, 2)),
        range(11),
        "abc",
        b"ab",
        bytearray(b"abcd"),
    )
    print(hot(2181819, objects))


main()
