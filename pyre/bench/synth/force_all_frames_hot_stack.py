# pyre-check: selfcheck
# pyre-check: trace-shape=hot:absent=CallMayForceR,present=GuardNoException
# pyre-check: selfcheck-compiles=hot
# Self-checking regression guard for `force_all_frames`
# (`executioncontext.rs`), the only frame-materializing consumer in the tree
# that no fixture reached.
#
# Both of its callers install a hook that will be handed frames the JIT may
# still be holding virtual, so both force the whole stack first:
# `settrace` calls it with `is_being_profiled=False` and `setllprofile` -- the
# implementation behind `setprofile` -- with `True`.  This fixture drives both
# arms, because they are separate call sites and only the second one also
# stamps `is_being_profiled` on every frame it walks.
#
# The discriminator is a caller local read back through the forced frame.
# `witness` is computed in `middle` and never read by `middle` itself, so a
# stale materialization returns an earlier round's number, and every round uses
# a different one.
#
# What this fixture does NOT establish, stated plainly because the wording it
# replaces claimed otherwise: `middle`'s frame is not an inlined virtual frame
# here.  Measured on this shape -- `loops_compiled=1`, `loops_aborted=3`, and
# `sys_getframe` / `builtin_locals` both `consulted=3 fired=0` -- the tracer
# does walk into `leaf` three times, but every one of those walks aborts, and
# the loop that does compile is the inner `for j` loop, which reaches `middle`
# only through a guard failure into the interpreter.  Calling `middle` on every
# iteration instead does not help: the `install` on the hot path drops the
# workload to `loops_compiled=0`.  That is inherent to the subject rather than
# a gap in the fixture -- installing a hook is precisely what stops a frame
# being compiled -- so the value check below is a correctness guard on
# `force_all_frames`, not evidence that a virtual frame was materialized.
#
# Three checks keep it from passing while nothing happened.  `entered` fails
# the fixture if the inner loop stops being entered and `hook_calls` fails it
# if the hook was never installed -- either would make the value check pass
# without the forced read ever happening.  Neither looks at the JIT at all, so
# the `spec-folds` header above adds the one signal that does: check.py censuses
# the trace-time folds and fails the fixture unless both named folds fire, which
# they cannot if the workload stops being traced.
import sys

OUTER = 400
INNER = 300

hook_calls = [0]


def trace_hook(frame, event, arg):
    # Returning None declines local tracing, so only the global events fire.
    hook_calls[0] += 1
    return None


def profile_hook(frame, event, arg):
    hook_calls[0] += 1


def probe():
    # A Python-level call, so a `call` event fires while the hook is installed.
    # Without one the hook is live but never invoked, and neither arm's install
    # would be observable from inside the fixture.
    return 0


def leaf(install, hook):
    # The install forces every frame on the stack, this one and `middle`'s and
    # the compiled loop's, before the hook can be handed any of them.
    install(hook)
    try:
        seen = sys._getframe(1).f_locals.get("witness")
        probe()
        return seen
    finally:
        install(None)


def middle(round_no, install, hook):
    witness = round_no * 7 + 3
    return leaf(install, hook), witness


def hot(install, hook):
    bad = []
    entered = 0
    for i in range(OUTER):
        for j in range(INNER):
            if j == INNER - 1:
                entered += 1
                seen, witness = middle(i, install, hook)
                if seen != witness:
                    bad.append((i, seen, witness))
    return entered, bad


def report(label, entered, bad, calls_before):
    if entered != OUTER:
        print(f"FAIL {label}: inner loop entered {entered} times, expected {OUTER}")
        return 1
    if hook_calls[0] == calls_before:
        print(f"FAIL {label}: hook never fired, so nothing was ever installed")
        return 1
    if bad:
        print(f"FAIL {label}: caller local read through the forced frame was stale")
        print("  first five (round, seen, expected):", bad[:5])
        return 1
    return 0


def main():
    rc = 0

    before = hook_calls[0]
    entered, bad = hot(sys.settrace, trace_hook)
    rc |= report("settrace", entered, bad, before)

    before = hook_calls[0]
    entered, bad = hot(sys.setprofile, profile_hook)
    rc |= report("setprofile", entered, bad, before)

    if rc:
        return rc
    print("PASS force_all_frames materializes a compiled caller frame")
    return 0


sys.exit(main())
