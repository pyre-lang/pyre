# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot,entry-bridge:rec
# The `root:` arm is the premise, not a relaxation: the activations this
# fixture counts are owed precisely because `rec` reaches the JIT and its
# recursive calls leave through the portal runner. A `rec` that stopped
# compiling would make every count below pass without testing anything.
# A profile hook is installed over a loop whose callee calls ITSELF, and the
# recursive activations are the ones a compiled caller performs through the
# portal runner rather than through the interpreter's call door.
#
# `pyframe.py execute_frame` brackets every activation with `ec.call_trace` and
# `leave`'s `_trace('leaveframe')`. pyre's portal sits at `execute_frame` level
# rather than at `dispatch` where upstream's merge point is, so the portal
# carries that bracket itself — and only the ROOT portal entry carried it. The
# recursive entries (`bhimpl_recursive_call_*` -> `bh_portal_runner_c`, the
# CALL_ASSEMBLER force leg, `compile_tmp_callback`'s callback loop, and the
# `jit_force_*_recursive_call_*` helpers) all reach the same portal body with a
# callee frame they built a line earlier, and reported nothing.
#
# THE SHAPE IS THE TEST: `rec(DEPTH)` is exactly DEPTH + 1 activations, so the
# owed count is a multiple of the tail that no partial reporting can reach by
# accident. Measured before the fix, `call/rec` read exactly `tail` from tail
# 10 000 upward — one activation per iteration, the outermost, which is the one
# that still goes through the interpreter — while cpython 3.14.6 reports
# `(DEPTH + 1) * tail`.
#
# THE TAIL IS ALSO A TEST: the loss was total and permanent from the moment
# compiled code took the caller over, so each count is checked at TWO tails and
# must track the difference between them. A count that saturates fails however
# large it is.
#
# `settrace` is checked alongside `setprofile` because they fail differently:
# `w_tracefunc` is guarded where the loop calls a Python callee, so arming a
# tracer exits compiled code, while `is_being_profiled` is a portal green, so
# arming a profiler mints a different cell that KEEPS the JIT and leaves the
# call a residual. Only the
# second reaches the portal entries this fixture is about, so a fixture that
# armed only a tracer would pass without testing anything.
import sys

WARM = 3000  # past the loop threshold (1039) without redundant warmup
TAILS = (250, 1000)
DEPTH = 3


def rec(n):
    if n <= 0:
        return 0
    return rec(n - 1) + 1


def hot(n):
    total = 0
    for _ in range(n):
        total = (total + rec(DEPTH)) % 1000003
    return total


def measure(tail, install):
    counts = {}

    def hook(frame, event, arg):
        key = (event, frame.f_code.co_name)
        counts[key] = counts.get(key, 0) + 1
        return hook

    # Compile the loop first, with nothing installed, so the arming below has
    # compiled code to interrupt rather than merely a cold frame to decline.
    hot(WARM)
    install(hook)
    try:
        hot(tail)
    finally:
        install(None)
    return counts


def exact(counts, key, expected, arm, failures):
    got = counts.get(key, 0)
    if got != expected:
        failures.append('%s: %s = %d, expected %d' % (arm, key, got, expected))


def tracks_the_tail(short, long_, key, per_iteration, arm, failures):
    grew = long_.get(key, 0) - short.get(key, 0)
    owed = (TAILS[1] - TAILS[0]) * per_iteration
    if grew < owed:
        failures.append(
            '%s: %s grew by %d from tail %d to tail %d, owed %d — the count '
            'does not track the tail, so the recursive activations are '
            'unreported from some point onward'
            % (arm, key, grew, TAILS[0], TAILS[1], owed)
        )


def main():
    failures = []
    for arm, install in (('profile', sys.setprofile), ('trace', sys.settrace)):
        measured = [measure(tail, install) for tail in TAILS]
        short, long_ = measured
        for tail, counts in zip(TAILS, measured):
            # One activation of the loop frame; DEPTH + 1 of `rec` per
            # iteration, of which DEPTH are recursive.
            exact(counts, ('call', 'hot'), 1, arm, failures)
            exact(counts, ('return', 'hot'), 1, arm, failures)
            exact(counts, ('call', 'rec'), (DEPTH + 1) * tail, arm, failures)
            exact(counts, ('return', 'rec'), (DEPTH + 1) * tail, arm, failures)
        for key in (('call', 'rec'), ('return', 'rec')):
            tracks_the_tail(short, long_, key, DEPTH + 1, arm, failures)
    if failures:
        for line in failures:
            print('FAIL', line)
        return 1
    print('PASS a recursive portal activation reports to an installed hook')
    return 0


sys.exit(main())
