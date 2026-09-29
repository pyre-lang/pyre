# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=main
# pyre-check: trace-shape=entry-bridge:tag:absent=CallMayForceR,entry-bridge:val:absent=CallMayForceR
# Self-checking guard for zero-argument `super()` bound to a name, which is the
# spelling that reaches the frame-escape path.
#
# The two spellings do not share a route.  `super().m()` compiles to
# LOAD_SUPER_ATTR, which the JIT lowers to a frame-explicit residual: the red
# frame travels as an operand and `builtin_super_from_frame` reads it, so the
# force happens through the may-force channel the walker already models.
# `s = super()` compiles to LOAD_GLOBAL super + CALL, which reaches
# `builtin_super`'s zero-argument tail, and that calls
# `ExecutionContext::gettopframe()` -- whose `force_frame` runs INSIDE an opaque
# residual and clears TOKEN_TRACING_RESCALL, which the walker reads back as
# `VableEscapedDuringResidualCall`.
#
# Measured on this shape, 200k iterations, dynasm: `super().val()` reports
# loops_aborted=0 abrt_escape=0 fbw_force_by_portal=0, while `s = super()`
# reports 5/5/5 -- same answer, different route.
#
# So this fixture is coverage of a route, not of a wrong answer: the escape
# aborts a recording walk and the interpreter finishes the iteration, which is
# why the answer stays correct while the loop stops compiling.  A regression
# that made the escape return a WRONG proxy -- the outer portal frame's `self`
# and `__class__` instead of the executing frame's -- would be invisible to a
# counter and is exactly what the sites below pin.
#
# Sites:
#   A  a method that binds `super()` to a name and calls through it, so the
#      proxy must resolve against A's own class, not its caller's.
#   B  the same in a callee the loop can inline, where the frame the proxy is
#      built from is the callee's own and not the portal's.  A proxy built
#      from the wrong frame answers with the caller's `__class__` cell and so
#      returns the caller's override.
#   C  a three-deep chain, so a proxy built one level off is still wrong but
#      would not be caught by a two-level shape alone.
#   D  a receiver `super_check` rejects by walking MROs, so the fold declines
#      before running anything and the generic residual owns the TypeError.
#   E  a receiver `super_check` can only classify by asking Python, which the
#      fold also declines.  E carries per-iteration data in its exception on
#      purpose: D's message is identical every iteration, so D alone cannot
#      tell a correctly-declined call from a folded one that answers with the
#      RECORDING-time exception object.  E can — that answer collapses the
#      distinct-message count, and it is what an earlier version of this fold
#      did (20000 iterations, 1662 distinct).  E also pins that the `__class__`
#      property runs exactly once per iteration, which a fold that executed the
#      call and then declined would double.
#   F  the loop INSIDE the super-bearing method, so the frame the proxy is
#      built from is the portal's own rather than an inlined callee's.  The
#      two frames reach `walker_bare_super_frame_slots` on different channels
#      -- A/B/C through the callee slot shadow, F through the standard
#      virtualizable -- and a fold that read only one of them would answer for
#      the wrong frame on the other.
#   G  a method whose own `self` is also a cellvar, because a nested function
#      closes over it.  Slot zero then holds a `Cell` rather than the receiver;
#      the virtual fold must perform the same guarded live dereference as
#      LOAD_DEREF.
#   H  a method that has already rebound its `__class__` cell.  The virtual
#      fold must decline once `CellFamily.ever_mutated` is set, leaving the
#      frame-explicit `bare_super_call` re-route live.  Together G and H keep
#      both labels in the `spec-folds` header above covered.
N = 20000


class Base:
    def val(self):
        return 1

    def tag(self):
        return "base"


class Middle(Base):
    def val(self):
        s = super()
        return s.val() + 10

    def tag(self):
        s = super()
        return "middle-" + s.tag()


class Leaf(Middle):
    def val(self):
        s = super()
        return s.val() + 100

    def tag(self):
        s = super()
        return "leaf-" + s.tag()


class Tricky:
    """A receiver `super_check` can only classify by asking Python."""

    hits = 0

    @property
    def __class__(self):
        Tricky.hits += 1
        raise ValueError(str(Tricky.hits))


class Portal(Base):
    """Site F's own class: the loop is inside the method that calls `super()`.

    A method whose own body carries the loop is the frame the portal traces,
    not a callee the trace inlines, so there is no callee slot shadow here and
    `walker_bare_super_frame_slots` reads the standard virtualizable instead.
    Both channels reach the virtual fold -- `PYRE_FBW_SPEC_CENSUS=1` reads
    `bare_super_virtual fired=1` on this shape as well as on A/B/C -- which is
    what a fold reading only the shadow would get wrong here.
    """

    def run_own_loop(self, n):
        acc = 0
        for _ in range(n):
            s = super()
            acc += s.val()
        return acc


class Captured(Base):
    """Site G's own class: `self` is a cellvar, shared with slot zero.

    `_get_self_location`'s cellvar branch reads the receiver out of that cell,
    and the virtual fold performs the same guarded live dereference as
    LOAD_DEREF before `_super_check` sees the receiver: `PYRE_FBW_SPEC_CENSUS=1`
    reads `bare_super_virtual fired=1` here.  `Rebound` below is the shape that
    keeps the re-route, so the two together cover both labels in the
    `spec-folds` header.
    """

    def val(self):
        def again():
            return self.tag()

        s = super()
        return s.val() + len(again())


class Rebound(Base):
    """Site H: a live but no-longer-quasi-immutable `__class__` cell."""

    def val(self):
        def touch_class_cell():
            nonlocal __class__
            __class__ = Rebound

        touch_class_cell()
        s = super()
        return s.val() + 20


class ETrap(Base):
    """Site E's own method, reached only with a `Tricky` receiver.

    Sharing `Leaf.val` does not work: the sites above compile it against a real
    instance first, so the raising call side-exits on those guards and runs
    interpreted, where the defect cannot appear.
    """

    def val(self):
        s = super()
        return s.val()


def main():
    leaf = Leaf()
    middle = Middle()
    site_a = set()
    site_b = set()
    site_c = set()
    site_d = set()
    site_e = set()
    site_g = set()
    site_h = set()
    tricky = Tricky()
    captured = Captured()
    rebound = Rebound()
    total = 0
    for _ in range(N):
        site_a.add(middle.val())
        site_b.add(leaf.val())
        site_c.add(leaf.tag())
        site_g.add(captured.val())
        site_h.add(rebound.val())
        try:
            Leaf.val(42)
        except TypeError as exc:
            site_d.add(str(exc))
        total += 1

    # Site F owns the loop itself, which is what puts `super()` in the portal
    # frame rather than in an inlined callee.
    site_f = Portal().run_own_loop(N)

    # Site E gets its own loop on purpose.  Sharing the loop above leaves this
    # call on a path the backend never compiles, and an uncompiled site cannot
    # witness a compiled-iteration defect.
    for _ in range(N):
        try:
            ETrap.val(tricky)
        except ValueError as exc:
            site_e.add(str(exc))

    for label, seen, want in (
        ("A", site_a, 11),
        ("B", site_b, 111),
        ("C", site_c, "leaf-middle-base"),
        ("G", site_g, 1 + len("base")),
        ("H", site_h, 21),
        (
            "D",
            site_d,
            "super(type, obj): obj (instance of int) is not an instance "
            "or subtype of type (Leaf).",
        ),
    ):
        if len(seen) != 1:
            print(f"FAIL site {label} diverged across iterations: {sorted(seen)}")
            return 1
        got = next(iter(seen))
        if got != want:
            print(f"FAIL site {label} {got!r} != {want!r}")
            return 1
    if total != N:
        print(f"FAIL dropped iteration: total={total}")
        return 1
    # `Base.val` answers 1, so the sum is the iteration count.  A proxy built
    # from the wrong frame reaches `Portal.val` — which does not exist, so the
    # MRO walk would find `Base.val` anyway; what it would get wrong is the
    # class the walk starts after, and a `super()` resolved against the CALLER
    # of `run_own_loop` raises instead of answering.
    if site_f != N:
        print(f"FAIL site F {site_f} != {N}")
        return 1
    # Site E's message is the raise count, so one distinct message per
    # iteration is the only answer that has the property running every
    # iteration AND each raise reporting its own value.  A compiled iteration
    # answering with the recording-time exception object collapses this.
    if Tricky.hits != N:
        print(f"FAIL site E ran the property {Tricky.hits} times, want {N}")
        return 1
    if len(site_e) != N:
        print(f"FAIL site E saw {len(site_e)} distinct messages, want {N}")
        return 1
    print("PASS bare super frame escape")
    return 0


import sys

sys.exit(main())
