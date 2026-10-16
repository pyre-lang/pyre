# `sys._current_frames()` reached from a hot loop, alongside `sys._getframe()`
# on the same source line.
#
# `_current_frames` (`module/thread/mod.rs`) forces and `mark_as_escaped`s the
# top frame of every thread under a stop-the-world, so it is a second producer
# of the frame-escape shape `sys._getframe` produces on its own
# (`module/sys/vm.rs getframe`).  Nothing in the corpus reached it: the escape
# fixtures all arrive through `_getframe`, `f_back`, or a traceback, leaving
# the whole-thread-roster route uncovered and its abort count ungated.
#
# The discriminator is the SINGLE source line carrying both `f_lineno` reads.
# Both name this frame, so the answers must agree; a values-only check would
# accept a stale line number from either route, because each looks plausible
# on its own.  Identity is asserted too -- the two routes must return the same
# frame object, and a route answering with a copy would still report a
# believable line.
#
# The roster's own read comes FIRST, and `sys._getframe()` -- the reference
# route, which forces -- is only reached after it, on the same source line.
# The two routes hand back ONE object, so a `_getframe()` evaluated ahead of
# the roster read would force that object and repair a stale roster image
# before it was ever inspected: `_current_frames` could stop forcing entirely
# and this loop would still see agreeing line numbers.  The identity check
# runs after both reads for the same reason.
#
# The roster is unpacked rather than indexed by thread id: the wasm guest ships
# no `threading` module, so `get_ident` is unavailable there, and the roster's
# own length is the stronger assertion anyway.
#
# `sys._getframemodulename` belongs to this family and is deliberately NOT
# here: PyPy does not implement it, and a synthetic fixture is also run under
# PyPy for the ratio.  It has its own self-checking fixture,
# `getframemodulename_hot_loop`, which runs on pyre alone.
#
# The shape compiles no loop -- the escape aborts the recording walk each time.
# That is recorded rather than asserted: the jitstats baselines gate
# `loops_aborted`, so a move in either direction becomes a review question
# instead of silent drift.
#
# The zero is the oracle's answer too, not a gap left open.  `PYPYLOG=jit-summary:-`
# on this shape reports `Total # of loops: 0` and a nonzero `abort: vable escape`,
# so the roster read aborts the recording walk upstream for the same reason: it
# escapes the virtualizable and `jit_force_virtualizable` has no fold to offer.
# A future change that compiles this loop is therefore a divergence to justify,
# not a win -- unless it also names what upstream hint makes the escape foldable.
import sys

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

# Recording aborts and blacklists this shape after five attempts.  This count
# clears the lowered threshold and the blacklist sequence without spending the
# rest of the fixture on an already-interpreted path.
N = 2800


def hot(n):
    bad_ident = 0
    bad_line = 0
    roster_sizes = set()
    for _ in range(n):
        roster = sys._current_frames()
        roster_sizes.add(len(roster))
        (frame_roster,) = roster.values()
        line_roster, line_direct = frame_roster.f_lineno, sys._getframe().f_lineno
        if frame_roster is not sys._getframe():
            bad_ident += 1
        elif line_roster != line_direct:
            bad_line += 1
    return bad_ident, bad_line, sorted(roster_sizes)


bad_ident, bad_line, roster_sizes = hot(N)
print("identity mismatches:", bad_ident)
print("f_lineno mismatches:", bad_line)
print("roster sizes:", roster_sizes)
