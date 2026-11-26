# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot
# A global trace function installs `sys.setprofile()` from the loop frame's own
# `call` event, so the frame is already past every gate that decides how it will
# run by the time it becomes profiled.
#
# `executioncontext.py call_trace` runs the callback first and only then writes
# `getorcreatedebug().is_being_profiled = True`. That write lands before
# `dispatch_frame` / `ll_portal_runner`, and `ll_portal_runner` does not refuse
# a profiled frame -- `is_being_profiled` is an `interp_jit.py` portal green,
# not a decline.  The profiled answer cannot come from refusing the frame.
# It comes from the green key instead:
# `is_being_profiled` is an `interp_jit.py` portal green, so a profiled
# activation names a different cell from the one the warm-up loop filled, and a
# trace recorded under that key folds no call
# (`walker_foldable_runtime_helper`), leaving every one of them a residual that
# fires its own `c_call` / `c_return` (`residual_call_c_profile_frame`).
#
# THE TAIL IS THE TEST.  A `c_call` count that stops at the entry threshold and
# then holds is what a green key read once, or read before the write, produces;
# the counts below must instead grow by the whole difference between the two
# tails.  Both are exact rather than floors: cpython 3.14.6 and pypy3 agree on
# `c_call` and `c_return` per builtin call at every tail.
import sys

WARM = 20000  # past the loop threshold (1039) many times over
TAILS = (2500, 10000)


def hot(n):
    total = 0
    for _ in range(n):
        total = (total + len('abc')) % 1000003
    return total


def measure(tail):
    counts = {}

    def profiler(frame, event, arg):
        name = getattr(arg, '__name__', None) or frame.f_code.co_name
        key = (event, name)
        counts[key] = counts.get(key, 0) + 1

    armed = []

    def tracer(frame, event, arg):
        # Arm the profiler from inside the loop frame's `call` event, once, and
        # return None so this frame never gets an `f_trace` of its own -- an
        # armed `f_trace` declines the frame outright and would answer the
        # question a different way.
        if event == 'call' and frame.f_code.co_name == 'hot' and not armed:
            armed.append(1)
            sys.setprofile(profiler)
        return None

    # Compile the loop first, unprofiled, so the arming below has compiled code
    # to interrupt and a filled cell whose green key it must not match.
    hot(WARM)
    sys.settrace(tracer)
    try:
        hot(tail)
    finally:
        sys.settrace(None)
        sys.setprofile(None)
    assert armed, 'the trace callback never reached the loop frame'
    return counts


def main():
    failures = []
    measured = [measure(tail) for tail in TAILS]
    for tail, counts in zip(TAILS, measured):
        for event in ('c_call', 'c_return'):
            key = (event, 'len')
            got = counts.get(key, 0)
            if got != tail:
                failures.append('%s = %d at tail %d, expected %d' % (key, got, tail, tail))
    short, long_ = measured
    owed = TAILS[1] - TAILS[0]
    for event in ('c_call', 'c_return'):
        key = (event, 'len')
        grew = long_.get(key, 0) - short.get(key, 0)
        if grew < owed:
            failures.append(
                '%s grew by %d from tail %d to tail %d, owed %d — the count does '
                'not track the tail, so the profiled activation is running '
                'compiled code recorded without it'
                % (key, grew, TAILS[0], TAILS[1], owed)
            )
    if failures:
        for line in failures:
            print('FAIL', line)
        return 1
    print('PASS a profiler armed from a call event keeps its c_call/c_return')
    return 0


sys.exit(main())
