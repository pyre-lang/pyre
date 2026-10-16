# pyre-check: max-pypy-ratio=16
# A hot loop must call its producer exactly once per iteration when the
# consumer is a bound method whose trailing defaults the binder fills.
#
# `SimpleQueue.put(item, block=True, timeout=None)` is the builtin shape;
# `Sink.put` is the same signature on a plain Python method.  Both are
# called as `put(counter.next())` so the two defaults stay with the binder.
# `Counter.next` is side-effecting: a guard / resume / declined wrapper
# walk that re-enters at or before that call shows up as a producer count
# past the iteration count (`cut_declined_subwalk` / `do_residual_call` /
# `make_result_of_lastop`).
import queue

LIMIT = 1500


class Counter:
    def __init__(self):
        self.n = 0

    def next(self):
        value = self.n
        self.n += 1
        return value


class Sink:
    def __init__(self):
        self.items = []

    def put(self, item, block=True, timeout=None):
        self.items.append(item)

    def get(self):
        return self.items.pop(0)


def run(put, get, limit):
    counter = Counter()
    results = []
    while True:
        put(counter.next())
        results.append(get())
        if results[-1] >= limit:
            break
    assert counter.n == len(results), (
        "the producer ran more often than the loop body",
        counter.n,
        len(results),
    )
    expected = list(range(limit + 1))
    assert results == expected, (
        "a produced value never reached the sink",
        next(
            (i, results[i], expected[i])
            for i in range(len(results))
            if results[i] != expected[i]
        ),
    )
    return counter.n, len(results), results[0], results[-1]


def main():
    q = queue.SimpleQueue()
    print(*run(q.put, q.get, LIMIT))
    sink = Sink()
    print(*run(sink.put, sink.get, LIMIT))


main()
