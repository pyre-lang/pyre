# pyre-check: max-pypy-ratio=6.8
N = 1200000


def build(n):
    # An inlined list comprehension compiles to LOAD_FAST_AND_CLEAR
    # (isolating the `j` iteration variable) around a hot FOR_ITER +
    # LIST_APPEND body.  Before LOAD_FAST_AND_CLEAR was lowered, its
    # abort_permanent marker declined the whole comprehension loop.
    return [j & 3 for j in range(n)]


def main():
    total = 0
    for _ in range(20):
        total += sum(build(N))
    print(total)


main()
