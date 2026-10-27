# pyre-check: max-pypy-ratio=2.2
# A `BINARY_OP` / `COMPARE_OP` whose dunder is a call-free Python body.
#
# The walker used to refuse every one of these.  The refusal was not about the
# operand or the class: it was that a `NotImplemented` result has to go back to
# the full binary protocol, and with no rewind at this entry the walk discarded
# it with a bare `Err`, which leaves the driver replaying the loop with the
# body's effects already applied.  So admission asked for a whole-body `Clean`
# verdict -- and `LoadAttr` and `BinaryOp` are both deferred helpers, which
# makes `return self.x + o` `DeferredCall` and refuses the simplest dunder
# there is.  The residual that stood in its place runs a whole interpreter
# frame per iteration.
#
# What the loops below cover is the shape that IS admitted: a body that reaches
# its result without committing anything on the way.  That is not read off the
# code object -- no static scan can separate `return self.v + o.v` over ints
# from one whose `+` dispatches to a mutating dunder -- so the entry refuses
# the first commit, before it runs, and cuts.  `fast` below is the admitted
# side of that: its `+`, `*` and `<` all reduce to integer arithmetic, so
# there is nothing to refuse and the body inlines whole.
# `slow` is the declined side, delegating through a call that can; it must stay
# residual and is here so the two sit side by side, not because it gates.
# `synth/binop_dunder_commit_then_notimplemented` is what gates the refusal.
#
# The ceiling gates admission of `fast`.
# main CI 10-24..10-26 (6 runs, all native legs): 1.1x-1.6x; ceiling = max x 1.35.
# Losing admission puts
# `fast`'s `+`, `*` and `<` back on a residual frame apiece, and that reading
# stays above the ceiling.
#
# Deterministic, terminating, prints an int checksum; jit == nojit.
import operator

M = 1000000007
# Sized so pypy's own execution clears `FLOOR_GATE_MIN_BASELINE_S`: below it
# the floor gate declines the baseline as too small to judge and the ratio
# reports startup rather than these loops.
N = 58181819


class Leaf:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = x

    def __add__(self, o):
        return self.x + o

    def __mul__(self, o):
        return self.x * o + 1

    def __lt__(self, o):
        return self.x < o


class Delegating:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = x

    def _cmp(self, o, op):
        return op(self.x, o)

    def __lt__(self, o):
        return self._cmp(o, operator.lt)


def fast(n):
    """The admitted shape: every dunder body reads a slot and calls nothing."""
    p = Leaf(0)
    acc = 0
    hits = 0
    for i in range(n):
        p.x = i % 9173
        acc = (acc + (p + i) + (p * 3)) % M
        if p < 4096:
            hits += 1
    return acc, hits


def slow(n):
    """The delegating shape, which stays residual."""
    q = Delegating(0)
    hits = 0
    for i in range(n):
        q.x = i % 9173
        if q < 4096:
            hits += 1
    return hits


a, h = fast(N)
print(a, h, slow(20000))
