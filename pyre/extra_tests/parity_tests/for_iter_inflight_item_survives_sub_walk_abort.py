# CPython-suite gap: the loss is a jit walk-abort path, invisible to an interpreter-level suite.
# parity-tests reason: pins a consumed FOR_ITER item against an abort inside a nested inline callee, and against a root-trace too-long abort whose delivery is refused.

"""A FOR_ITER item consumed before an aborted sub-walk still reaches the body.

`step` is a bound method that mutates before it calls, so the walk admits it as
an inline callee and the append commits; `rec` is self-recursive, so descending
into it is refused and the walk aborts with the item already taken off the
iterator. Redelivering it would replay the append, so delivery is refused
whenever a body effect has committed -- which leaves the abort itself owing the
item a carrier. Without one the item is neither delivered nor rolled back, and
the loop body runs for it while its value never lands in `res`: the failure
signature is `len(res) < len(w.seen)`.

The accumulator is a statement loop, so no LIST_APPEND is involved and what is
pinned is the item rather than the opcode that accumulates it. The trip counts
are the ones the loss was observed at -- the walk has to warm up before the
abort is reached, and a shorter sweep never reaches it.
"""


def ck(seq):
    h = 7
    for v in seq:
        h = (h * 1000003 + v) & 0xFFFFFFFFFF
    return h


def rec(x):
    if x <= 0:
        # `id` is opaque, so the descent stops here rather than folding away.
        return 1 if id(x) else 0
    return rec(x - 1) + 1


class W:
    def __init__(self):
        self.seen = []

    def step(self, x):
        # Committed before the call below, and no rollback undoes it.
        self.seen.append(x)
        return x * 1000003 + rec(x % 4)


hh = 7
bad = []
for trial in range(1000):
    n = trial % 20
    w = W()
    res = []
    for i in range(n):
        res.append(w.step(i))
    if len(res) != n or len(w.seen) != n:
        bad.append((trial, n, len(res), len(w.seen)))
    # Consuming both lists every trial is part of the shape the loss needs.
    hh = (hh * 1000003 + ck(res) + ck(w.seen) + len(w.seen)) & 0xFFFFFFFFFF

assert not bad, f"dropped items, (trial, n, len(res), len(seen)): {bad}"
assert hh == 950649862511, hh


# A list FOR_ITER whose consume the walk descended, in a root loop trace that
# runs out of length before the iteration's body completes.  The trace limit
# puts that abort inside `sum`'s loop; its FOR_ITER carries an EXTENDED_ARG,
# so the frame is parked one opcode before the FOR_ITER and the delivery is
# refused.  The refusal has to put the iterator cursor back: without the
# snapshot it resumes on the next element and the sum comes out short
# (19898 for 19900).
try:
    import pypyjit

    pypyjit.set_param("trace_limit=70,threshold=1,function_threshold=1")
except ImportError:
    pass

values = list(range(200))
short = []
for trial in range(30):
    s = sum(values)
    if s != 19900:
        short.append((trial, s))
assert not short, f"sum over list(range(200)) came out wrong, (trial, sum): {short}"
print("OK")
