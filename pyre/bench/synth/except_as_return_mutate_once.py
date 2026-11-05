# A seeded `except E as e: return` callee whose try body mutates and then
# hits a residual that raises into that arm. The except-as-return admit
# walks the Dirty happy path (`ResidualCallWritesLiveHeap` on `append`,
# `UnprovableStoreOrCallForm` on `type()`). Each mutation must apply once
# per iteration — the same count pypy3 prints.
#
# The second callee is the unrelated-handler variant: `append` is poison
# on the happy path and the `except ValueError as e: return` arm is never
# taken. Same once-per-iteration count.
#
# Abort-during-tracing cannot re-execute the outer CALL after those
# writes. `fbw_bump_executed_effect` moves the odometer, the Entry
# carrier rewind is the zero-delta gate, and `blackhole_if_trace_too_long`
# (`pyjitpl.py`) continues forward (`fbw_blackhole_adopted_single_frame`).
# `fbw_rolled_back_with_effects` stays 0. Forced with
# `pypyjit.set_param("trace_limit=40")` the counts still equal N.
#
# N is past the tracing threshold so the admit actually walks the callees.
# No `max-pypy-ratio`: this is a shape oracle, not a workload.
#
# Expected: (N, N, N, N)
N = 20000
LONE = "\udcff"

log_raise = []
log_unrelated = []


def mutate_then_raise(x):
    try:
        log_raise.append(x)
        return type(LONE, (), {}).__name__
    except UnicodeEncodeError as e:
        return ("E", e.start, x)


def mutate_unrelated_handler(x):
    try:
        log_unrelated.append(x)
        if x < 0:
            raise ValueError
        return x
    except ValueError as e:
        return ("E", x)


def main():
    i = 0
    acc_raise = 0
    acc_unrelated = 0
    while i < N:
        r = mutate_then_raise(i)
        if r[0] == "E":
            acc_raise = acc_raise + 1
        u = mutate_unrelated_handler(i)
        if u == i:
            acc_unrelated = acc_unrelated + 1
        i = i + 1
    print((len(log_raise), len(log_unrelated), acc_raise, acc_unrelated))


main()
