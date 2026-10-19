# pyre-check: pypy-diverges: pins the PEP 667 write-through f_locals proxy, which pypy3 does not have -- its f_locals is a snapshot dict, so the proxy write never reaches the fast local and reading the key back raises KeyError
# CPython-suite gap: the suite exercises FrameLocalsProxy access only against
# cold frames, never against one whose surrounding loop the JIT has compiled.
# parity-tests reason: this guards the coherence of the running frame's
# `f_locals` against the values a compiled loop holds for those locals.
#
# framelocalsproxy_gc_roots:
# no suite test collects inside the key comparison a FrameLocalsProxy operation performs.
# this is a pyre/PyPy moving-GC root-liveness regression. pins PEP 667 write-through
# f_locals (3.13); a write through pypy3's proxy does not reach the fast local
#
# flocals_read_through_bound_proxy:
# the suite reads FrameLocalsProxy only against cold frames, never against a frame whose
# surrounding loop is being traced at the moment of the read. the read is served by a call
# the tracer runs for real, against the frame the tracer is stepping a copy of, so it is
# the tracer's handling of that call -- not the proxy -- that decides which value comes
# back. pins the PEP 667 f_locals proxy as a LIVE view, which pypy3 does not have -- its

"""A write through the running frame's ``f_locals`` must be visible to a later
read of the same local, including after the loop around the write is compiled.

The proxy is a live view over the frame's fast locals, so ``proxy["x"] = i`` has
to reach the storage a subsequent ``LOAD_FAST x`` reads.  A compiled loop that
keeps ``x`` in its virtualizable shadow answers the value the local held when
the loop was compiled, which makes the result depend on the compilation
threshold rather than on the program.

``write_through_proxy`` puts its write in the loop *body*.  That is what makes
it a witness: a write placed before the loop happens while the frame is still
interpreted, so the two sides are still equal at the moment the loop compiles
and the arm passes whether or not the defect is present.  The two controls keep
shapes that agree either way, each for its own reason.

Only the write direction is covered.  The read direction -- a ``STORE_FAST`` in
the body followed by a proxy read -- diverges the same way and is not fixed
here: forcing on a read would also force the reads a trace has inlined, which
upstream never does, and there is no way to tell those apart at the accessor
yet.
"""

import sys

ROUNDS = 4000


def write_through_proxy(rounds):
    x = -1
    proxy = sys._getframe(0).f_locals
    for i in range(rounds):
        proxy["x"] = i
    return x


def store_then_read_proxy(rounds):
    # Control: the store precedes the compile, so this agrees either way.
    proxy = sys._getframe(0).f_locals
    x = -1
    for i in range(rounds):
        x = i
    return proxy["x"]


def write_through_caller_proxy(rounds):
    # Second control: this one agrees either way because `setter` builds a
    # fresh proxy per iteration, so the getter's own force fires every time.
    # What the arms above need is a proxy that OUTLIVES that force.
    def setter(value):
        sys._getframe(1).f_locals["x"] = value

    x = -1
    for i in range(rounds):
        setter(i)
    return x


def check(name, got):
    # Every arm drives the same local to ``ROUNDS - 1``; a stale answer reports
    # the value it held when the loop compiled, which is well short of that.
    if got != ROUNDS - 1:
        print(f"{name}: got {got!r}, want {ROUNDS - 1!r}")
        return False
    return True


results = [
    check("write_through_proxy", write_through_proxy(ROUNDS)),
    check("store_then_read_proxy", store_then_read_proxy(ROUNDS)),
    check("write_through_caller_proxy", write_through_caller_proxy(ROUNDS)),
]

def framelocalsproxy_gc_roots():
    """``frame.f_locals`` operations must survive a collection mid-lookup.

    Every proxy operation materializes a fresh snapshot dict, or scans the code
    object's names allocating a string per candidate, before it reads the key and
    value it was handed.  A key whose ``__eq__`` collects relocates those arguments
    while the operation still holds them.  The payloads below are lists and dicts
    because those relocate; an instance of a Python class is born at a stable
    address and would never exercise this.
    """

    import gc
    import sys


    class CollectingName(str):
        """A name that forces a moving collection before answering a compare."""

        def __eq__(self, other):
            garbage = [[index] for index in range(4000)]
            assert len(garbage) == 4000
            gc.collect()
            return str.__eq__(str(self), other)

        def __ne__(self, other):
            return not self.__eq__(other)

        def __hash__(self):
            return str.__hash__(self)


    def write_fast_local():
        """A proxy store resolving to a fast slot must store the live value."""
        alpha = None
        proxy = sys._getframe().f_locals
        replacement = ["replacement"]
        proxy[CollectingName("alpha")] = replacement
        assert alpha is replacement, alpha
        assert alpha == ["replacement"], alpha
        return alpha


    def write_extra_key():
        """A key naming no fast local lands in the frame's extras dict."""
        beta = ["beta"]
        proxy = sys._getframe().f_locals
        stored = {"stored": True}
        proxy[CollectingName("absent")] = stored
        assert beta == ["beta"], beta
        assert proxy[CollectingName("absent")] is stored
        assert CollectingName("absent") in proxy
        assert proxy.get(CollectingName("absent")) is stored
        return stored


    def read_and_default():
        """``get`` / ``setdefault`` route through the same snapshot and scan."""
        gamma = ["gamma"]
        proxy = sys._getframe().f_locals
        fallback = ["fallback"]
        assert proxy.get(CollectingName("gamma")) is gamma
        assert proxy.setdefault(CollectingName("gamma"), fallback) is gamma
        assert proxy.setdefault(CollectingName("fresh"), fallback) is fallback
        assert proxy[CollectingName("fresh")] is fallback
        assert gamma == ["gamma"], gamma
        return gamma


    def merge_and_snapshot():
        """``update`` and ``items`` rebuild their pairs across allocations."""
        delta = None
        proxy = sys._getframe().f_locals
        incoming = ["incoming"]
        proxy.update({CollectingName("delta"): incoming})
        assert delta is incoming, delta
        pairs = dict(proxy.items())
        assert pairs["delta"] is incoming
        assert pairs["incoming"] is incoming
        return delta


    for _ in range(5):
        assert write_fast_local() == ["replacement"]
        assert write_extra_key() == {"stored": True}
        assert read_and_default() == ["gamma"]
        assert merge_and_snapshot() == ["incoming"]
framelocalsproxy_gc_roots()

def flocals_read_through_bound_proxy():
    """A proxy bound before a loop reads every subsequent ``STORE_FAST``.

    The trace keeps the local in a box while the proxy reads
    ``locals_cells_stack_w``. The residual subscript and bound-``get`` paths must
    write the live box back first; otherwise both freeze at the recording value and
    the accumulated lag becomes nonzero.
    """

    import sys

    ROUNDS = 4000


    def through_subscript(rounds):
        x = -1
        p = sys._getframe(0).f_locals
        lag = 0
        for i in range(rounds):
            x = i
            lag += i - p["x"]
        return lag


    def through_get(rounds):
        x = -1
        p = sys._getframe(0).f_locals
        lag = 0
        for i in range(rounds):
            x = i
            lag += i - p.get("x")
        return lag


    for read in (through_subscript, through_get):
        lag = read(ROUNDS)
        assert lag == 0, f"{read.__name__}: {lag} behind over {ROUNDS} iterations"
flocals_read_through_bound_proxy()

if all(results):
    print("OK")
