# A `with` block in a hot FOR_ITER body, whose handler runs every tenth
# iteration. The frame is traced: 2 loops and 2 bridges, and the 401 guard
# failures are the two bridges' warm-up (trace_eagerness=200 each), not a
# per-iteration bailout. `exception_with_exit_self_null_slot` is the same shape
# written as a `while` loop.
N = 160000


class Context:
    def __init__(self):
        self.exits = 0

    def __enter__(self):
        return 3

    def __exit__(self, exc_type, exc_value, traceback):
        self.exits += 1
        return exc_type is ValueError


def main():
    context = Context()
    total = 0
    for i in range(N):
        with context as value:
            total += value
            if i % 10 == 0:
                raise ValueError
            total += i
    print(total, context.exits)


main()
# The line carries 3*N + sum(i for i in range(N) if i % 10), then N.
# Expected: 11520480000 160000
