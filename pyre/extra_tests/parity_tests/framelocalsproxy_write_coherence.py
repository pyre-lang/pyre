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
import types

try:
    from inspect import GEN_CLOSED, GEN_CREATED, GEN_SUSPENDED, getgeneratorstate
except ImportError:
    # `inspect.getgeneratorstate`. The cranelift core binary has no `_tokenize`,
    # so `inspect` cannot import; the checks are the same four `gi_*` reads.
    GEN_CREATED = "GEN_CREATED"
    GEN_RUNNING = "GEN_RUNNING"
    GEN_SUSPENDED = "GEN_SUSPENDED"
    GEN_CLOSED = "GEN_CLOSED"

    def getgeneratorstate(generator):
        if generator.gi_running:
            return GEN_RUNNING
        if generator.gi_suspended:
            return GEN_SUSPENDED
        if generator.gi_frame is None:
            return GEN_CLOSED
        return GEN_CREATED

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


def unstarted_generator_cellvar_argument():
    # A generator is created at the CALL after the prologue `MAKE_CELL` has
    # run (`f_lasti` past that instruction). `framelocalsproxy_getval` /
    # `framelocalsproxy_setitem` unwrap / write-through the Cell, so both
    # print 42, write 43 into the cell, and `next(g)` yields 43.
    def gen(x):
        def capture():
            return x

        yield capture()

    g = gen(42)
    before = g.gi_frame.f_locals["x"]
    g.gi_frame.f_locals["x"] = 43
    after = g.gi_frame.f_locals["x"]
    yielded = next(g)
    return before, after, yielded


def unstarted_generator_celltype_argument():
    # The argument *is* a cell object. `MAKE_CELL` wraps it, so
    # `f_locals["x"]` is that cell (identity), a write stores into the
    # wrapper, and `next(g)` is 7 while the original cell still holds 5.
    c = types.CellType(5)

    def gen(x):
        def capture():
            return x

        yield capture()

    g = gen(c)
    before = g.gi_frame.f_locals["x"]
    g.gi_frame.f_locals["x"] = 7
    after = g.gi_frame.f_locals["x"]
    yielded = next(g)
    return before, after, yielded, c.cell_contents, before is c


def check_unstarted_generator_cellvar():
    got = unstarted_generator_cellvar_argument()
    want = (42, 43, 43)
    if got != want:
        print(f"unstarted_generator_cellvar_argument: got {got!r}, want {want!r}")
        return False
    return True


def check_unstarted_generator_celltype():
    before, after, yielded, orig, before_is_c = unstarted_generator_celltype_argument()
    if not before_is_c or after != 7 or yielded != 7 or orig != 5:
        print(
            "unstarted_generator_celltype_argument: "
            f"got before={before!r} after={after!r} yielded={yielded!r} "
            f"orig={orig!r} before_is_c={before_is_c!r}"
        )
        return False
    return True


def check_unstarted_generator_state_and_lineno():
    def gen(x):
        yield x

    g = gen(1)
    before = getgeneratorstate(g)
    lineno = g.gi_frame.f_lineno
    first = gen.__code__.co_firstlineno
    yielded = next(g)
    after = getgeneratorstate(g)
    if before != GEN_CREATED or after != GEN_SUSPENDED:
        print(
            "unstarted_generator_state: "
            f"got before={before!r} after={after!r}"
        )
        return False
    if lineno != first:
        print(
            "unstarted_generator_f_lineno: "
            f"got {lineno!r}, want {first!r}"
        )
        return False
    if yielded != 1:
        print(f"unstarted_generator_state yield: got {yielded!r}")
        return False
    return True


def check_unstarted_closure_generator_f_locals():
    def outer():
        n = 1

        def inner():
            yield n

        return inner()

    g = outer()
    got = g.gi_frame.f_locals
    want = {"n": 1}
    if got != want:
        print(f"unstarted_closure_generator_f_locals: got {got!r}, want {want!r}")
        return False
    return True


def check_unstarted_generator_throw():
    def gen():
        yield 1

    g = gen()
    raised = None
    try:
        g.throw(ValueError)
    except ValueError as e:
        raised = type(e)
    if raised is not ValueError:
        print(f"unstarted_generator_throw: got {raised!r}, want ValueError")
        return False
    if getgeneratorstate(g) != GEN_CLOSED or g.gi_frame is not None:
        print(
            "unstarted_generator_throw closed: "
            f"state={getgeneratorstate(g)!r} gi_frame={g.gi_frame!r}"
        )
        return False
    return True


def check_unstarted_coroutine_celltype_f_locals():
    c = types.CellType(5)

    async def coro(x):
        def capture():
            return x

        return capture()

    co = coro(c)
    got = co.cr_frame.f_locals["x"]
    if got is not c:
        print(
            "unstarted_coroutine_celltype_f_locals: "
            f"got {got!r}, want the argument cell"
        )
        return False
    return True


def check_unstarted_asyncgen_celltype_f_locals():
    c = types.CellType(5)

    async def agen(x):
        def capture():
            return x

        yield capture()

    ag = agen(c)
    got = ag.ag_frame.f_locals["x"]
    if got is not c:
        print(
            "unstarted_asyncgen_celltype_f_locals: "
            f"got {got!r}, want the argument cell"
        )
        return False
    return True


results = [
    check("write_through_proxy", write_through_proxy(ROUNDS)),
    check("store_then_read_proxy", store_then_read_proxy(ROUNDS)),
    check("write_through_caller_proxy", write_through_caller_proxy(ROUNDS)),
    check_unstarted_generator_cellvar(),
    check_unstarted_generator_celltype(),
    check_unstarted_generator_state_and_lineno(),
    check_unstarted_closure_generator_f_locals(),
    check_unstarted_generator_throw(),
    check_unstarted_coroutine_celltype_f_locals(),
    check_unstarted_asyncgen_celltype_f_locals(),
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
