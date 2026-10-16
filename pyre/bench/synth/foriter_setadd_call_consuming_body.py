# pyre-check: max-pypy-ratio=8
# Thick baselines read 3.4-4.1x. A thin baseline prints 7.8x and still passes
# this ceiling.
# A set comprehension whose element call consumes the very heap it is draining,
# so a body execution replayed or dropped by the tracer changes the heap, not
# just the result.
#
# The set is the weaker of the two checks here: it absorbs a duplicated pop
# silently, so `len(heap) == 0` is what carries the detection — a replayed body
# leaves the heap short, a dropped one leaves it long.
#
# The fixture is a call-bearing set comprehension: SET_ADD plus a consuming
# call in the same body, so a tracer abort that replayed or dropped the body
# would change the heap, not just the result.
import heapq

N = 600
SIZE = 128


def lcg(seed, n):
    # Deterministic data, so the fixture's output is a constant rather than a
    # property of the host's hash seed or clock.
    out = []
    x = seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(x % 100000)
    return out


total = 0
for round_ in range(N):
    data = lcg(round_ + 1, SIZE)
    heap = list(data)
    heapq.heapify(heap)
    size = len(heap)
    # `range(size)` is fixed before the body runs, so the body executes exactly
    # `size` times unless the tracer replays one.
    drained = {heapq.heappop(heap) for _ in range(size)}
    assert len(heap) == 0, 'heap not drained: %d left' % len(heap)
    assert drained == set(data), 'set comprehension lost or invented an item'
    total += len(drained)

print(total)
