# pyre-check: max-pypy-ratio=9.8
# Ubuntu run 33279264115: 3.4-4.9x; the ceiling is twice the slowest,
# rounded up to one decimal place.
# N/ITERS are sized to prove the opcode compiles and to drive the compiled
# loop thousands of times, not to race pypy.
# A hot exact `dict.get` loop rides along; it traces through the
# `dict.get` gateway.
N = 300
ITERS = 500


def run(n, extra):
    # `{"k": i, **extra}` compiles to BUILD_MAP + DICT_UPDATE in a
    # while-loop body.  Before DICT_UPDATE was lowered, its abort_permanent
    # marker declined the whole loop.
    total = 0
    i = 0
    while i < n:
        d = {"k": i, **extra}
        total += d["k"] + d["x"]
        i += 1
    return total


def main():
    extra = {"x": 7}
    total = 0
    for _ in range(ITERS):
        total += run(N, extra)
    print(total)


main()


GETD = {"a": 1, "b": 2, "c": 3}


def hot_dict_get(n):
    """Hot exact `dict.get`."""
    s = 0
    i = 0
    while i < n:
        s += GETD.get("a", 0)
        i += 1
    return s


print(hot_dict_get(20000000))
